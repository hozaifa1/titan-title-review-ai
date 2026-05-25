"""Two-pane edit / regenerate UI for Titan.

Left pane:  the default draft (read-only, pretty-printed).
Right pane: the same JSON, editable inline.
Buttons:    Capture Edits  ·  Distill Rules  ·  Regenerate.

The intent is to give an operator a one-screen demo of the learning loop:
load a draft, change a field, click through capture → distill → regenerate,
and watch the next draft pick up the pattern.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from titan.config import get_settings
from titan.draft.orchestrator import DraftOrchestrator
from titan.index.chunker import chunk_title_document
from titan.index.embed import embed_chunks
from titan.index.qdrant_store import HybridChunkStore
from titan.learn.diff import diff_summaries
from titan.learn.distill import RuleStore, distill_rules_for_section
from titan.learn.memory import EditMemory
from titan.persist.sqlite import load_edit_events, persist_edit_events
from titan.retrieve.hybrid import HybridRetriever
from titan.schemas import TitleDocument, TitleReviewSummary


REPO_ROOT = Path(__file__).resolve().parent
DATA_OUT = REPO_ROOT / "data" / "out"
DATA_GOLD = REPO_ROOT / "data" / "gold"
EXAMPLES = REPO_ROOT / "examples"
RULES_DIR = REPO_ROOT / "rules"
SQLITE_PATH = REPO_ROOT / "data" / "titan.db"


def _list_drafts() -> list[Path]:
    """Find every TitleReviewSummary on disk that the operator can load."""

    candidates: list[Path] = []
    for directory in (DATA_OUT, EXAMPLES):
        if directory.exists():
            candidates.extend(sorted(directory.glob("*.json")))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        if path.stem in seen:
            continue
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(blob, dict) and "s1_vesting_and_estate" in blob:
            unique.append(path)
            seen.add(path.stem)
    return unique


def _load_title_document_for(matter_id: str) -> TitleDocument | None:
    """Pair a draft with its TitleDocument so we can rerun retrieval/drafting."""

    for directory in (DATA_OUT, DATA_GOLD):
        candidate = directory / f"{matter_id}.title_document.json"
        if candidate.exists():
            return TitleDocument.model_validate_json(candidate.read_text(encoding="utf-8"))
    return None


def _read_markdown_for(document: TitleDocument) -> str:
    """Reproduce the CLI's PDF → markdown helper without circular imports."""

    path = Path(document.file_path)
    if path.exists() and path.suffix.lower() == ".pdf":
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                pages = [
                    f"## Page {idx}\n\n{page.extract_text() or ''}"
                    for idx, page in enumerate(pdf.pages, start=1)
                ]
                return "\n\n".join(pages)
        except Exception:
            pass
    return "\n".join(
        [
            f"# {document.doc_id}",
            f"Document type: {document.doc_type}",
            "Vesting: " + ", ".join(p.name for p in document.vesting),
            "Parties: " + ", ".join(f"{p.name} ({p.role})" for p in document.parties),
        ]
    )


async def _regenerate(
    title_document: TitleDocument,
    with_learning: bool,
) -> TitleReviewSummary:
    markdown = _read_markdown_for(title_document)
    chunks = await chunk_title_document(title_document, markdown)
    embedded, bm25, dense_embedder = embed_chunks(chunks)
    settings = get_settings()
    store = HybridChunkStore(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key or None,
    )
    store.upsert(embedded, bm25)
    retriever = HybridRetriever(store, dense_embedder, bm25)

    edit_memory: EditMemory | None = None
    rule_store: RuleStore | None = None
    if with_learning:
        rule_store = RuleStore(RULES_DIR)
        edit_memory = EditMemory(
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key or None,
            embedder=dense_embedder,
        )
        events = load_edit_events(SQLITE_PATH)
        if events:
            edit_memory.add_many(events)

    summary = await DraftOrchestrator(
        retriever,
        edit_memory=edit_memory,
        rule_store=rule_store,
    ).generate(title_document)
    return summary


def _safe_parse(raw: str) -> tuple[TitleReviewSummary | None, str | None]:
    try:
        return TitleReviewSummary.model_validate_json(raw), None
    except Exception as exc:  # noqa: BLE001 — Streamlit surfaces the message verbatim
        return None, str(exc)


def main() -> None:
    st.set_page_config(page_title="Titan — Title Review", layout="wide")
    st.title("Titan — Title Review")
    st.caption("Left: the default draft.  Right: edit JSON in place, then capture, distill, regenerate.")

    drafts = _list_drafts()
    if not drafts:
        st.warning(
            "No drafts found in `data/out/` or `examples/`. "
            "Run `python -m titan.cli draft <pdf>` to produce one first."
        )
        st.stop()

    selected = st.sidebar.selectbox(
        "Draft to load",
        options=drafts,
        format_func=lambda p: p.name,
    )
    raw_text = selected.read_text(encoding="utf-8")
    baseline, baseline_err = _safe_parse(raw_text)
    if baseline_err:
        st.error(f"Could not parse {selected.name}: {baseline_err}")
        st.stop()

    # Persistent state: keep the operator's edits across reruns and surface the
    # next regenerated draft alongside the original baseline.
    if st.session_state.get("loaded_path") != str(selected):
        st.session_state["loaded_path"] = str(selected)
        st.session_state["edited_text"] = baseline.model_dump_json(indent=2)
        st.session_state["regenerated"] = None

    left, right = st.columns(2)

    with left:
        st.subheader("Default draft")
        st.json(json.loads(raw_text))

    with right:
        st.subheader("Edit JSON")
        edited_text = st.text_area(
            "Editable TitleReviewSummary JSON",
            value=st.session_state["edited_text"],
            height=600,
            key="editor",
        )
        st.session_state["edited_text"] = edited_text

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("Capture edits", help="Diff edited JSON against baseline, persist EditEvents"):
            edited, err = _safe_parse(edited_text)
            if err or edited is None:
                st.error(f"Invalid JSON: {err}")
            else:
                events = diff_summaries(baseline, edited, operator_id="streamlit")
                if not events:
                    st.info("No field-level changes detected.")
                else:
                    persist_edit_events(events, SQLITE_PATH)
                    settings = get_settings()
                    memory = EditMemory(
                        qdrant_url=settings.qdrant_url,
                        qdrant_api_key=settings.qdrant_api_key or None,
                    )
                    memory.add_many(events)
                    st.success(f"Captured {len(events)} edits to SQLite + edit_memory.")

    with col_b:
        if st.button("Distill rules", help="Run LLM-as-judge to convert edits into rules/*.yaml"):
            events = load_edit_events(SQLITE_PATH)
            if not events:
                st.info("No edits in SQLite yet.")
            else:
                sections = sorted({e.section_name for e in events})
                rule_store = RuleStore(RULES_DIR)
                outputs: list[dict[str, Any]] = []
                for section in sections:
                    section_events = [e for e in events if e.section_name == section]
                    result = asyncio.run(
                        distill_rules_for_section(section, section_events, rule_store)
                    )
                    outputs.append(
                        {
                            "section": section,
                            "version": result.rule_set.version,
                            "rules": len(result.rule_set.rules),
                            "used_gemini": result.used_gemini,
                        }
                    )
                st.success(f"Distilled {len(outputs)} section(s).")
                st.json(outputs)

    with col_c:
        learning_on = st.checkbox("With learning", value=True, help="Inject rules + few-shot edits")
        if st.button("Regenerate", type="primary"):
            title_document = _load_title_document_for(baseline.matter_id)
            if title_document is None:
                st.error(
                    f"No TitleDocument found for matter_id={baseline.matter_id}. "
                    "Run `python -m titan.cli demo-ingest` first."
                )
            else:
                with st.spinner("Regenerating draft..."):
                    new_summary = asyncio.run(_regenerate(title_document, learning_on))
                st.session_state["regenerated"] = new_summary.model_dump_json(indent=2)
                st.success(f"Regenerated. rules_version={new_summary.rules_version}")

    regenerated_text = st.session_state.get("regenerated")
    if regenerated_text:
        st.divider()
        st.subheader("Regenerated draft")
        st.json(json.loads(regenerated_text))
        st.download_button(
            "Download regenerated JSON",
            data=regenerated_text,
            file_name=f"{baseline.matter_id}.v2.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
