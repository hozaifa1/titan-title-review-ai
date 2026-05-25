# Titan — Title Review AI

Titan ingests messy real-estate title documents, pulls structured facts out of them, retrieves the relevant passages, and drafts a cited ALTA-style **Title Review Summary** that an operator can edit. The system learns from those edits and applies the patterns to the next draft.

Author: S. M. Hozaifa Hossain

---

## What it does

You hand it a PDF — a title commitment, a deed, a mortgage, a handwritten conveyance, a lis pendens, an ALTA/NSPS survey, a tax certificate, whatever. It does five things:

1. **Parses the file.** Text PDFs go through pdfplumber (~1 s/page). Scanned/image-only PDFs fall back to Docling for OCR. Handwritten pages route to a Qwen2.5-VL hook (transcript fixtures included so the demo runs without paying for the VLM). Parsed markdown is cached to `data/.md_cache/` with a per-page timeout so a pathological survey page can't stall the pipeline.
2. **Extracts a typed `TitleDocument` schema** with character-level provenance via BAML + a multi-provider LLM client (Cerebras, SambaNova, Groq, Gemini, GitHub Models, OpenRouter — first to respond wins). A deterministic regex/heuristic fallback runs underneath so the pipeline still works with no API key.
3. **Builds a hybrid index** — BM25 sparse + BGE-M3 dense embeddings in Qdrant, fused with Reciprocal Rank Fusion, reranked by bge-reranker-v2-m3. Each chunk keeps its `{doc_id, page, char_span}` so citations are inspectable end-to-end.
4. **Drafts a `TitleReviewSummary`** — eight ALTA sections (Vesting, Legal Description, Chain of Title, Open Encumbrances, Easements, Schedule B-I, Schedule B-II, Taxes/Survey). All eight section calls fan out in parallel under per-provider locks. Every section is generated against retrieved evidence with inline `[doc_id:page:span]` citations. The ALTA section list lives in **one place** ([titan/sections.py](titan/sections.py)) — adding a ninth section is a single edit.
5. **Learns from operator edits.** A field-level diff is logged to SQLite, the edit pair is embedded into Qdrant as future few-shot fodder, and an LLM-as-judge pass distills the recent edits per-section into a versioned YAML rule set that gets injected into the prompt next time.

There's a Streamlit UI on top that lets you load a draft, edit the JSON in-place, and regenerate.

---

## Quick start

```bash
# 1. Install (uv is recommended; plain pip works too)
uv pip install -e .
#   or:  pip install -e .

# 2. Bring up Qdrant
docker compose up -d qdrant
#   Windows users can run  ./scripts/start_docker_and_qdrant.ps1
#   which also boots Docker Desktop if it isn't already running.

# 3. Run the demo on three sample docs
python -m titan.cli demo-ingest

# 4. Generate a cited draft
python -m titan.cli draft data/raw/commitment/wayne_county_commitment_0.pdf \
    --with-learning
```

The full eval (paired pre/post-learning):

```bash
python -m titan.cli eval-run
```

The Streamlit edit-and-regenerate UI:

```bash
streamlit run streamlit_app.py
```

API keys are optional. The pipeline works offline using local models and a heuristic extractor that pulls real party names, instrument types, and amounts from the structured extraction. Copy `.env.example` to `.env` and drop in any combination of provider keys (Cerebras, SambaNova, Groq, Gemini, GitHub Models, OpenRouter) — the LLM client walks the configured chain in order and uses whichever one responds first.

---

## Architecture (10-second tour)

```mermaid
flowchart TB
    A[PDF / TIFF / JPG] --> B{Page classifier}
    B -->|typed| C1[Docling]
    B -->|low-conf| C2[pdfplumber]
    B -->|handwritten| C3[Qwen2.5-VL hook]
    C1 & C2 & C3 --> D[Page markdown + bbox spans]
    D --> E[BAML extract → TitleDocument]
    E --> F[(SQLite)]
    D --> G[Semantic chunker + heuristic context]
    G --> H[BGE-M3 dense + BM25 sparse]
    H --> I[(Qdrant hybrid)]
    Q[Section prompt] --> R[Dense + BM25 → RRF → reranker → top-5]
    I --> R
    R --> S[Multi-provider LLM chain + citations + rules + few-shot]
    F --> S
    S --> T[TitleReviewSummary JSON]
    T --> U[Operator edit]
    U --> V[Field-level diff → EditEvent]
    V --> W[(SQLite edit_events)]
    V --> X[BGE-M3 embed → Qdrant edit_memory]
    W --> Y[LLM-as-judge distillation → rules/*.yaml]
    X -.few-shot.-> S
    Y -.system prompt.-> S
```

Full architectural notes live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Repository layout

```
titan/                      Source code
  config.py                 Pydantic Settings, env loading
  errors.py                 Domain exceptions
  sections.py               Single source of truth for the ALTA section list
  schemas/                  TitleDocument, TitleReviewSummary, EditEvent
  ingest/                   pdfplumber + Docling OCR, VLM hook, page classifier, BAML extraction
    markdown_cache.py       Disk-cached per-page extraction with timeout
    vlm.py                  Qwen2.5-VL hook (production seam + offline transcript fallback)
  index/                    Chunker, BGE-M3 embed, Qdrant client
  retrieve/                 Hybrid BM25+dense, RRF fusion, BGE reranker
  draft/                    Section orchestrator + multi-provider LLM calls
  learn/                    Edit diff, embedding memory, rule distillation
  eval/                     Paired pre/post eval harness
  persist/                  SQLModel + SQLite operations
  telemetry/                Langfuse tracing, structlog config
  llm_client.py             Multi-provider LLM client with per-provider locks
  llm_cache.py              Disk-backed response cache
  cli.py                    `python -m titan.cli ...`

baml_src/                   BAML extraction prompts + types
rules/                      Distilled YAML rules per section
data/
  raw/                      Source PDFs (sample fixtures committed)
  gold/                     Hand-labelled gold JSONs + transcript fixtures
  out/                      Generated drafts (gitignored)
  .md_cache/                Cached parsed markdown (gitignored)
  .llm_cache/               Cached LLM responses (gitignored)
examples/                   Sample v1/v2/edited artifacts
eval/                       results_pre.json, results_post.json
docs/                       Architecture, evaluation, assumptions
tests/                      Pytest suite
scripts/                    Setup helpers (incl. Docker bring-up for Windows)
streamlit_app.py            Edit / regenerate UI
docker-compose.yml          Qdrant container
Dockerfile                  Reproducible runtime image
```

---

## Sample input & output

The `examples/` folder has sample artifacts you can inspect without running anything:

- `examples/input.pdf` — Wayne County commitment (clean digital PDF).
- `examples/output_v1.json` — first-pass draft (no learning).
- `examples/edited_v1.json` — simulated operator edits applied.
- `examples/edits.json` — the captured `EditEvent` records.
- `examples/rules_s4_open_encumbrances_and_liens.yaml` — rules distilled from those edits.
- `examples/output_v2.json` — regenerated draft using the rules + few-shot edits.

For a side-by-side with the human gold, see `data/gold/wayne_county_commitment_0.TitleReviewSummary.gold.json`.

---

## Evaluation

Five held-out docs (Wayne County commitment, OSMRE deed of trust, 1875 handwritten deed, Orlando ALTA/NSPS survey, Bartlesville lis pendens), paired conditions, run with `python -m titan.cli eval-run`:

| Metric | No learning | With learning | Δ |
|---|---:|---:|---:|
| **Field edit distance** (lower is better) | 0.913 | 0.829 | **−9.2 %** |
| **Faithfulness** (claim ↔ retrieved chunk) | 0.786 | 0.910 | **+0.123** |
| **Answer relevancy** (produced vs gold) | 0.455 | 0.539 | **+0.085** |
| **Retrieval recall@5** (gold spans) | 1.000 | 1.000 |  0.000 |
| **Citation accuracy** | 0.324 | 0.460 | **+0.136** |
| **Rule application rate** | 0.000 | 0.700 | **+0.700** |
| **Edit memory size** | 0 | 24 | +24 |

Edit distance is the headline number — the same documents get noticeably closer to gold after the system has seen 24 simulated operator edits and run one rule-distillation pass. The full per-doc breakdown is in `eval/results_pre.json` and `eval/results_post.json`. More detail on methodology, why these metrics, and what's still weak in [docs/EVALUATION.md](docs/EVALUATION.md).

---

## Assumptions and tradeoffs

The compressed version (full version in [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md)):

- **Output is grounded, not self-certifying.** Every claim in a draft carries a citation; the operator decides if it's right.
- **Learning is retrieval, not fine-tuning.** Past edits become few-shot examples and distilled rules, keeping the learning loop inspectable and auditable.
- **Offline parity.** Every external call has a fallback: pdfplumber → Docling → Qwen2.5-VL hook (transcript fixture in offline mode); LLM provider chain walks 6 providers before degrading to a structured-extraction heuristic that pulls real names, instrument types, and amounts. Users without keys still get a working pipeline.
- **VLM hook, not VLM call.** Handwritten pages route through `titan/ingest/vlm.py`. Default implementation returns `None`; with `TITAN_VLM_ENABLED=1` and a real `call_vlm` body it wires straight into a hosted Qwen2.5-VL (or any other VLM). The transcript fixture is a clearly-labelled offline stub — never consulted when the real VLM is enabled.
- **SQLite for persistence.** One file, zero infra. Scales fine for tens of thousands of edits.
- **PDFs in `data/raw/`** are committed for reproducibility. The sample corpus is available out of the box.
- **A real Qwen2.5-VL call is a wiring change, not a re-architecture.** The page classifier already routes to it.
- **Sections live in one place.** [`titan/sections.py`](titan/sections.py) is the canonical ALTA section list. The schema, orchestrator, metrics, and edit-diff all derive from it — adding a ninth section is one edit, not three.

---

## Code quality

- Pydantic v2 throughout — no raw dicts crossing module boundaries.
- Async where it pays (OCR, embed, retrieval); sync everywhere else.
- `tenacity` retries on every external call, exponential backoff, capped at 3.
- Per-provider asyncio locks on the LLM client so a single 429 marks the provider in cooldown without burning seven concurrent quota slots.
- Per-provider cooldown (capped 60 s) instead of permanent-dead on rate-limit; permanent-dead reserved for auth / 4xx misconfiguration.
- Domain errors (`OCRFailedError`, `LowConfidenceError`) defined in `titan/errors.py`.
- `structlog` with per-request `trace_id` correlation; Langfuse `@observe` on every LLM-touching function.
- `ruff check` is clean.
- `pytest` suite covers ingest, index/chunker, index/embed, retrieve, draft, learn, eval, persist, and schemas — at least one happy-path test per module.

---

## What I'd do with more time

- Migrate from `google-generativeai` to the newer `google.genai` SDK once it stabilises in BAML.
- Plug in Patronus Lynx for sentence-level hallucination detection. The citation tag pattern catches most of it, but Lynx would tighten the unsupported-generation control.
- Run the eval on 15–20 docs to get a real confidence interval on edit-distance reduction. Five is enough to show the loop works; it isn't enough to claim a number.
- Build a hosted Qwen2.5-VL endpoint and remove the transcript fixture path for handwriting.

---

## Repository

- Repo: `github.com/hozaifa1/titan-title-review-ai`

---

## License

MIT.
