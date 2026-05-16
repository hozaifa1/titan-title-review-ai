"""Field-level structured diff between a baseline draft and an operator-edited draft.

Emits :class:`EditEvent` rows. Operates section-by-section over a
:class:`TitleReviewSummary`, only descending into the citation-bearing fields
that operators actually edit (``summary[*].text``, ``bullet_findings[*].text``,
``gaps[*]``, ``flags``). Whole-section additions/deletions are also captured.
"""

from __future__ import annotations

from typing import Any, Iterable

from titan.schemas import (
    CitedSentence,
    EditEvent,
    EditType,
    TitleReviewSection,
    TitleReviewSummary,
)

# Single source of truth — :mod:`titan.sections.SECTION_REGISTRY`.
from titan.sections import section_field_names

SECTION_FIELDS = section_field_names()


def diff_summaries(
    baseline: TitleReviewSummary,
    edited: TitleReviewSummary,
    operator_id: str | None = None,
    operator_note: str | None = None,
) -> list[EditEvent]:
    """Compute the list of operator edits between two summaries.

    Both summaries must reference the same ``matter_id`` (or no edits are
    captured). Citations are intentionally ignored — the operator-edited
    rendering of a sentence is the signal, not its provenance tags.
    """

    if baseline.matter_id != edited.matter_id:
        return []

    events: list[EditEvent] = []
    for field_name in SECTION_FIELDS:
        before_section: TitleReviewSection = getattr(baseline, field_name)
        after_section: TitleReviewSection = getattr(edited, field_name)
        events.extend(
            _diff_section(
                section_name=field_name,
                before=before_section,
                after=after_section,
                matter_id=baseline.matter_id,
                source_draft_version=baseline.generator_version,
                operator_id=operator_id,
                operator_note=operator_note,
            )
        )
    return events


def _diff_section(
    section_name: str,
    before: TitleReviewSection,
    after: TitleReviewSection,
    matter_id: str,
    source_draft_version: str,
    operator_id: str | None,
    operator_note: str | None,
) -> list[EditEvent]:
    events: list[EditEvent] = []

    events.extend(
        _diff_sentence_list(
            section_name=section_name,
            field_root="summary",
            before=before.summary,
            after=after.summary,
            matter_id=matter_id,
            source_draft_version=source_draft_version,
            operator_id=operator_id,
            operator_note=operator_note,
        )
    )
    events.extend(
        _diff_sentence_list(
            section_name=section_name,
            field_root="bullet_findings",
            before=before.bullet_findings,
            after=after.bullet_findings,
            matter_id=matter_id,
            source_draft_version=source_draft_version,
            operator_id=operator_id,
            operator_note=operator_note,
        )
    )
    events.extend(
        _diff_string_list(
            section_name=section_name,
            field_root="gaps",
            before=before.gaps,
            after=after.gaps,
            matter_id=matter_id,
            source_draft_version=source_draft_version,
            operator_id=operator_id,
            operator_note=operator_note,
        )
    )
    events.extend(
        _diff_string_list(
            section_name=section_name,
            field_root="flags",
            before=list(before.flags),
            after=list(after.flags),
            matter_id=matter_id,
            source_draft_version=source_draft_version,
            operator_id=operator_id,
            operator_note=operator_note,
            edit_type_override="risk_rating",
        )
    )
    return events


def _diff_sentence_list(
    section_name: str,
    field_root: str,
    before: list[CitedSentence],
    after: list[CitedSentence],
    matter_id: str,
    source_draft_version: str,
    operator_id: str | None,
    operator_note: str | None,
) -> list[EditEvent]:
    events: list[EditEvent] = []
    pairs = _align_sentences(before, after)
    for index, (before_sentence, after_sentence) in enumerate(pairs):
        before_text = before_sentence.text if before_sentence else ""
        after_text = after_sentence.text if after_sentence else ""
        if before_text == after_text:
            continue
        events.append(
            EditEvent(
                matter_id=matter_id,
                section_name=section_name,
                field_path=f"{field_root}[{index}].text",
                before=before_text,
                after=after_text,
                edit_type=_classify_text_edit(before_text, after_text),
                operator_id=operator_id,
                operator_note=operator_note,
                source_draft_version=source_draft_version,
            )
        )
    return events


def _diff_string_list(
    section_name: str,
    field_root: str,
    before: list[str],
    after: list[str],
    matter_id: str,
    source_draft_version: str,
    operator_id: str | None,
    operator_note: str | None,
    edit_type_override: EditType | None = None,
) -> list[EditEvent]:
    events: list[EditEvent] = []
    width = max(len(before), len(after))
    for index in range(width):
        before_text = before[index] if index < len(before) else ""
        after_text = after[index] if index < len(after) else ""
        if before_text == after_text:
            continue
        events.append(
            EditEvent(
                matter_id=matter_id,
                section_name=section_name,
                field_path=f"{field_root}[{index}]",
                before=before_text,
                after=after_text,
                edit_type=edit_type_override or _classify_text_edit(before_text, after_text),
                operator_id=operator_id,
                operator_note=operator_note,
                source_draft_version=source_draft_version,
            )
        )
    return events


def _align_sentences(
    before: Iterable[CitedSentence],
    after: Iterable[CitedSentence],
) -> list[tuple[CitedSentence | None, CitedSentence | None]]:
    before_list = list(before)
    after_list = list(after)
    width = max(len(before_list), len(after_list))
    return [
        (
            before_list[index] if index < len(before_list) else None,
            after_list[index] if index < len(after_list) else None,
        )
        for index in range(width)
    ]


def _classify_text_edit(before: str, after: str) -> EditType:
    if not before and after:
        return "addition"
    if before and not after:
        return "deletion"
    before_lower = before.lower()
    after_lower = after.lower()
    if _looks_like_citation_change(before_lower, after_lower):
        return "citation_fix"
    if _shares_skeleton(before_lower, after_lower):
        return "wording"
    return "fact_correction"


_CITATION_TOKENS = ("book", "page", "instrument", "schedule", "exhibit", "deed")


def _looks_like_citation_change(before: str, after: str) -> bool:
    return any(token in after and token not in before for token in _CITATION_TOKENS) or any(
        token in before and token not in after for token in _CITATION_TOKENS
    )


def _shares_skeleton(before: str, after: str) -> bool:
    before_tokens = set(_word_tokens(before))
    after_tokens = set(_word_tokens(after))
    if not before_tokens or not after_tokens:
        return False
    overlap = len(before_tokens & after_tokens)
    smallest = min(len(before_tokens), len(after_tokens))
    return overlap / max(smallest, 1) >= 0.5


def _word_tokens(text: str) -> list[str]:
    return [token for token in _split_words(text) if len(token) >= 3]


def _split_words(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for char in text:
        if char.isalnum():
            buf.append(char)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


def summarize_for_embedding(event: EditEvent) -> str:
    """Render an EditEvent as a single string used for BGE-M3 embedding."""

    note = f" | note: {event.operator_note}" if event.operator_note else ""
    return (
        f"[{event.section_name} / {event.edit_type}] "
        f"BEFORE: {event.before}\nAFTER: {event.after}{note}"
    )


def _decimal_repr(value: Any) -> str:  # pragma: no cover - small helper
    return str(value)
