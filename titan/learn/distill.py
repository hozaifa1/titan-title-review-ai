"""LLM-as-judge rule distillation and YAML rule storage.

Given the last N :class:`EditEvent` rows for a section, asks Gemini 2.0 Flash
to propose up to 7 reusable rules and persists them as a versioned YAML file
under ``rules/{section}.yaml``. When no Gemini key is configured (or the call
fails), a deterministic offline distiller produces a minimal viable RuleSet
from the edits so the loop still works end-to-end in tests and demos.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from titan.config import get_settings
from titan.schemas import EditEvent, EditType, Rule, RuleSet, SupersededRule
from titan.telemetry import get_logger

logger = get_logger(__name__)

DEFAULT_RULES_DIR = Path("rules")
DEFAULT_DISTILL_MODEL = "gemini-2.0-flash"
MAX_RULES = 7
DEFAULT_EDIT_WINDOW = 20


class RuleStore:
    """Read/write versioned ``rules/{section}.yaml`` files."""

    def __init__(self, rules_dir: Path | str = DEFAULT_RULES_DIR) -> None:
        self.rules_dir = Path(rules_dir)

    def path_for(self, section: str) -> Path:
        return self.rules_dir / f"{section}.yaml"

    def load(self, section: str) -> RuleSet | None:
        path = self.path_for(section)
        if not path.exists():
            return None
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            logger.warning("rule_store.read_failed", path=str(path), error=str(exc))
            return None
        data.setdefault("section", section)
        try:
            return RuleSet.model_validate(data)
        except Exception as exc:
            logger.warning("rule_store.invalid_yaml", path=str(path), error=str(exc))
            return None

    def save(self, rule_set: RuleSet) -> Path:
        path = self.path_for(rule_set.section)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(rule_set.model_dump_json())
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path

    def next_version(self, section: str) -> int:
        existing = self.load(section)
        if existing is None:
            return 1
        return existing.version + 1

    def aggregated_version_tag(self) -> str | None:
        if not self.rules_dir.exists():
            return None
        versions: list[int] = []
        for path in sorted(self.rules_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            version = data.get("version")
            if isinstance(version, int):
                versions.append(version)
        if not versions:
            return None
        return f"rules-v{max(versions)}"


@dataclass(frozen=True)
class DistillationResult:
    rule_set: RuleSet
    used_gemini: bool
    raw_response: str | None = None


async def distill_rules_for_section(
    section: str,
    events: Iterable[EditEvent],
    rule_store: RuleStore,
    model_name: str = DEFAULT_DISTILL_MODEL,
    window: int = DEFAULT_EDIT_WINDOW,
    use_gemini: bool | None = None,
) -> DistillationResult:
    """Run the LLM-as-judge pass and persist a new RuleSet for ``section``."""

    edit_window = list(events)[-window:]
    if not edit_window:
        raise ValueError(f"No edits supplied for section {section!r}")

    existing = rule_store.load(section)
    next_version = (existing.version + 1) if existing else 1
    should_use_llm = get_settings().has_any_llm if use_gemini is None else use_gemini

    rules: list[Rule] = []
    superseded: list[SupersededRule] = []
    raw_response: str | None = None
    used_gemini = False

    if should_use_llm:
        try:
            rules, superseded, raw_response = await _call_llm_judge(
                section=section,
                events=edit_window,
                existing_rules=existing.rules if existing else [],
            )
            used_gemini = True  # name kept for backwards-compat; means "LLM-driven path"
        except Exception as exc:
            logger.warning("rule_distill.llm_failed", section=section, error=str(exc))

    if not rules:
        rules = _offline_distill(edit_window)

    rule_set = RuleSet(
        section=section,
        version=next_version,
        created_at=datetime.now(timezone.utc),
        generated_from_edit_ids=[event.edit_id for event in edit_window],
        rules=rules[:MAX_RULES],
        superseded=superseded,
    )
    rule_store.save(rule_set)
    return DistillationResult(rule_set=rule_set, used_gemini=used_gemini, raw_response=raw_response)


def _has_gemini_key() -> bool:
    return get_settings().has_gemini


async def _call_llm_judge(
    section: str,
    events: list[EditEvent],
    existing_rules: list[Rule],
) -> tuple[list[Rule], list[SupersededRule], str]:
    """Route through the unified LLM client so Groq/OpenRouter can serve when Gemini is rate-limited."""

    from titan.llm_client import get_llm_client

    prompt = _judge_prompt(section, events, existing_rules)
    parsed, result = await get_llm_client().generate_json(prompt, temperature=0.1)
    logger.info("rule_distill.llm_provider_used", section=section, provider=result.provider, model=result.model)
    rules = _coerce_rules(parsed.get("rules") or [])
    superseded = _coerce_superseded(parsed.get("superseded") or [])
    return rules, superseded, result.text


def _judge_prompt(section: str, events: list[EditEvent], existing_rules: list[Rule]) -> str:
    examples = [
        {
            "edit_id": event.edit_id,
            "field_path": event.field_path,
            "before": event.before,
            "after": event.after,
            "edit_type": event.edit_type,
            "operator_note": event.operator_note,
        }
        for event in events
    ]
    existing_payload = [rule.model_dump(mode="json") for rule in existing_rules]
    schema_hint = {
        "rules": [
            {
                "id": "r1",
                "text": "Always cite Deed Book and Page when describing chain of title.",
                "trigger_edit_types": ["citation_fix"],
                "confidence": 0.9,
            }
        ],
        "superseded": [
            {"id": "r0", "text": "Old rule text", "superseded_by": "r1", "reason": "..."}
        ],
    }
    return (
        "You are a meta-reviewer for a real-estate title-review AI.\n"
        "You are given the last edits the senior partner made to drafts of section "
        f"'{section}'. Each edit shows the model's BEFORE text and the partner's AFTER text.\n\n"
        "Task: extract up to 7 REUSABLE RULES that, if followed by the drafter from the start, "
        "would have prevented these edits. Each rule MUST be:\n"
        "  - generalizable (not tied to one specific property/person/parcel),\n"
        "  - testable (clear pass/fail),\n"
        "  - short (<=20 words),\n"
        "  - labelled with trigger_edit_types from the edit list.\n\n"
        "If any existing rule is contradicted by new edits, list it under `superseded` with a reason.\n\n"
        f"Return strict JSON matching this shape:\n{json.dumps(schema_hint, indent=2)}\n\n"
        f"Existing rules ({len(existing_rules)}):\n{json.dumps(existing_payload, indent=2)}\n\n"
        f"Edits ({len(events)}):\n{json.dumps(examples, indent=2, default=str)}"
    )


def _parse_judge_response(raw_text: str) -> dict[str, Any]:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _coerce_rules(items: list[Any]) -> list[Rule]:
    rules: list[Rule] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        rule_id = str(item.get("id") or f"r{index + 1}")
        triggers = _coerce_triggers(item.get("trigger_edit_types"))
        confidence_raw = item.get("confidence", 0.7)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.7
        rules.append(
            Rule(
                id=rule_id,
                text=text[:240],
                trigger_edit_types=triggers,
                confidence=max(0.0, min(1.0, confidence)),
            )
        )
    return rules[:MAX_RULES]


def _coerce_superseded(items: list[Any]) -> list[SupersededRule]:
    out: list[SupersededRule] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            SupersededRule(
                id=str(item.get("id") or ""),
                text=str(item.get("text") or ""),
                superseded_by=str(item.get("superseded_by") or ""),
                reason=str(item.get("reason") or ""),
            )
        )
    return out


_VALID_EDIT_TYPES: set[EditType] = {
    "wording",
    "fact_correction",
    "addition",
    "deletion",
    "citation_fix",
    "style",
    "house_format",
    "risk_rating",
}


def _coerce_triggers(value: Any) -> list[EditType]:
    if not isinstance(value, list):
        return []
    out: list[EditType] = []
    for item in value:
        text = str(item).strip().lower().replace("-", "_")
        if text in _VALID_EDIT_TYPES:
            out.append(text)  # type: ignore[arg-type]
    return out


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        parts = getattr(getattr(candidates[0], "content", None), "parts", []) or []
        return "".join(str(getattr(part, "text", "")) for part in parts)
    return ""


def _offline_distill(events: list[EditEvent]) -> list[Rule]:
    """Deterministic distillation used when Gemini is unavailable.

    Mines surface signals: common new tokens introduced by the partner, the
    dominant edit_type, and operator notes. Yields at most :data:`MAX_RULES`
    short, generalizable rules.
    """

    edit_type_counts: Counter[str] = Counter(event.edit_type for event in events)
    added_tokens: Counter[str] = Counter()
    for event in events:
        before_tokens = set(_words(event.before))
        after_tokens = set(_words(event.after))
        for token in after_tokens - before_tokens:
            if len(token) >= 4 and not _is_stopword(token):
                added_tokens[token] += 1

    rules: list[Rule] = []
    if added_tokens:
        top_tokens = [token for token, _ in added_tokens.most_common(3)]
        rules.append(
            Rule(
                id="r1",
                text=f"Prefer terminology added by reviewers: {', '.join(top_tokens)}.",
                trigger_edit_types=["wording", "style"],
                confidence=0.6,
            )
        )

    if edit_type_counts.get("citation_fix", 0) > 0:
        rules.append(
            Rule(
                id="r2",
                text="Cite Book, Page, and Instrument Number for every recorded reference.",
                trigger_edit_types=["citation_fix", "fact_correction"],
                confidence=0.85,
            )
        )

    if edit_type_counts.get("fact_correction", 0) > 0:
        rules.append(
            Rule(
                id="r3",
                text="Verify named parties, dates, and dollar amounts against the source chunk before asserting them.",
                trigger_edit_types=["fact_correction"],
                confidence=0.8,
            )
        )

    if edit_type_counts.get("addition", 0) > 0:
        rules.append(
            Rule(
                id="r4",
                text="Surface gaps and missing facts as explicit bullet findings instead of omitting them.",
                trigger_edit_types=["addition"],
                confidence=0.7,
            )
        )

    if edit_type_counts.get("risk_rating", 0) > 0:
        rules.append(
            Rule(
                id="r5",
                text="Escalate flags to yellow/red when liens or extraction gaps exist; never leave green by default.",
                trigger_edit_types=["risk_rating"],
                confidence=0.75,
            )
        )

    if not rules:
        rules.append(
            Rule(
                id="r1",
                text="Adopt the operator's phrasing for this section's standard wording.",
                trigger_edit_types=["wording"],
                confidence=0.5,
            )
        )

    return rules[:MAX_RULES]


def _words(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text)]


_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "have", "has", "was",
    "were", "are", "any", "all", "into", "their", "there", "these", "those",
    "section", "policy", "document", "extracted", "extraction", "structured",
    "subject", "based", "confirmation",
}


def _is_stopword(token: str) -> bool:
    return token in _STOPWORDS
