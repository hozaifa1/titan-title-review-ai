# Pearson Specter Litt — Title Review AI (Take-Home Submission)

Author: S. M. Hozaifa Hossain · Submitted: May 15, 2026

> **TL;DR** — Drop a stack of scanned title docs in, get back a grounded, ALTA-style
> Title Review Summary with inline citations to the exact source span. Edit it,
> regenerate; the system learns from your edits and the next draft is measurably
> closer to your house style.

## Demo (90 seconds)
![demo](docs/demo.gif)

## Quick start (3 commands)
```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"

# 2. Start Qdrant
docker compose up -d qdrant

# 3. Run the demo
python -m titan.cli demo
```

## What this is
A hybrid (hosted-API + local) pipeline that:
1. Parses scanned/handwritten/messy title documents (commitments, deeds, mortgages, judgments, surveys, tax certs).
2. Extracts a strict typed schema with character-level provenance using **BAML**.
3. Retrieves with hybrid BM25 + dense + reranking + contextual chunks in **Qdrant**.
4. Generates a section-by-section ALTA-aligned Title Review Summary with inline citations using **Gemini 2.0 Flash**.
5. Captures operator edits and **demonstrably** improves the next draft via a learning loop.

## Architecture
```mermaid
flowchart TB
    subgraph Ingestion["1 · INGESTION"]
        A[PDF / TIFF / JPG upload] --> B{Doc Classifier<br/>Gemini Flash}
        B -->|Commitment| C1[Docling]
        B -->|Deed/Mortgage| C1
        B -->|Handwritten note| C2[Qwen2.5-VL OCR]
        B -->|Low-conf page| C3[Docling local]
        C1 --> D[Page Markdown + bbox spans]
        C2 --> D
        C3 --> D
    end

    subgraph Extraction["2 · STRUCTURED EXTRACTION"]
        D --> E[BAML extract: TitleDocument schema]
        E --> F[(SQLite: title_documents)]
    end

    subgraph Indexing["3 · INDEX BUILD"]
        D --> G[Semantic chunker 600 tok overlap 80]
        G --> H[Contextual chunk:<br/>Gemini Flash writes 1-sent context<br/>per chunk vs full doc]
        H --> I1[BGE-M3 dense]
        H --> I2[BM25 sparse]
        I1 --> J[(Qdrant: hybrid index)]
        I2 --> J
    end

    subgraph Retrieval["4 · RETRIEVAL & RERANK"]
        Q[User query / draft section prompt] --> K[Query rewrite + HyDE optional]
        K --> L1[Dense top-30]
        K --> L2[BM25 top-30]
        J --> L1
        J --> L2
        L1 --> M[RRF k=60 → top-20]
        L2 --> M
        M --> N[bge-reranker-v2-m3 → top-5]
    end

    subgraph Generation["5 · GROUNDED DRAFT"]
        F --> O[Section orchestrator]
        N --> O
        O --> P[Gemini 2.0 Flash<br/>+ Citations API<br/>+ EditMemory + Rules]
        P --> R[Structured TitleReviewSummary JSON<br/>with citations per sentence]
    end

    subgraph Loop["6 · LEARNING LOOP"]
        R --> S[Operator UI / API edit]
        S --> T[Field-level Diff Capturer]
        T --> U[(SQLite: edit_events)]
        U --> V[Edit embedder BGE-M3]
        V --> W[(Qdrant: edit_memory)]
        U --> X[Nightly LLM-as-judge<br/>distills rules]
        X --> Y[(rules.yaml versioned)]
        W -.dynamic few-shot.-> P
        Y -.system prompt injection.-> P
    end

    subgraph Eval["7 · EVAL"]
        P --> Z1[RAGAS faithfulness]
        P --> Z2[Citation accuracy]
        P --> Z3[Field edit-distance before/after]
        Z1 --> Z4[Langfuse dashboard]
        Z2 --> Z4
        Z3 --> Z4
    end
```

## Tech stack
| # | Purpose | Tool / Model |
|---|---|---|
| 1 | **Primary OCR** | **Docling** (Mistral/IBM) |
| 2 | **Handwriting Fallback** | **Qwen2.5-VL-7B** |
| 3 | **Structured Extraction** | **BAML** + **Gemini 2.0 Flash** |
| 4 | **Embeddings** | **BGE-M3** (Dense + Sparse) |
| 5 | **Vector DB** | **Qdrant** |
| 6 | **Reranker** | **bge-reranker-v2-m3** |
| 7 | **Draft Generation** | **Gemini 2.0 Flash** (Citations API) |
| 8 | **Tracing** | **Langfuse Cloud** |
| 9 | **Evaluation** | **RAGAS** + Custom Edit-Distance |
| 10 | **Persistence** | **SQLite** via `sqlmodel` |

## The edit-learning loop
The system features a three-layer learning loop to capture firm-specific "house style" and correct recurring errors:
1. **EditEvent Log:** Every operator change is captured as a structured diff in SQLite.
2. **Dynamic Few-Shot:** Top-k similar past edits are retrieved from Qdrant and injected into the draft prompt.
3. **Distilled Rules:** An LLM-as-judge periodic pass distills recent edits into reusable YAML rules (e.g., "Always use TX terminology for Texas deeds").

## Sample input & output
- **Input:** `data/raw/wayne_county_commitment_0.pdf`
- **v1 Draft:** `data/out/wayne_county_commitment_0.v1.json`
- **v2 Draft (Learned):** `data/out/wayne_county_commitment_0.v2.json`
- **Rules:** `rules/s4_open_encumbrances_and_liens.yaml`

## Evaluation
*(Placeholder: Results from `titan.eval.run`)*

| Metric | v1 (no learning) | v2 (with learning) | Δ |
|---|---|---|---|
| RAGAS Faithfulness | [TBD] | [TBD] | [TBD] |
| Citation accuracy | [TBD] | [TBD] | [TBD] |
| Field-level edit distance | [TBD] | [TBD] | [TBD] |

## Repository layout
```
titan/
  ingest/             # Docling OCR, BAML extraction
  index/              # Contextual chunking, BGE-M3 embedding
  retrieve/           # Hybrid search + BGE reranking
  draft/              # Section-by-section orchestration
  learn/              # Edit diffing, memory, and rule distillation
  eval/               # RAGAS and edit-distance metrics
  schemas/            # Pydantic models for TitleDocument and Summary
  persist/            # SQLite storage
baml_src/             # BAML extraction prompts/schemas
rules/                # Distilled YAML rules
data/                 # raw, gold, and out directories
tests/                # Pytest suite
```

## Assumptions & tradeoffs
- **ALTA 2021 Alignment:** The extraction and summary schemas are aligned with ALTA 2021 Title Commitment standards.
- **Hybrid RAG:** Uses a blend of hosted APIs (Gemini) for high-reasoning tasks and local models (BGE) for cost-effective retrieval.
- **Learning vs Fine-tuning:** Chose RAG-based learning (few-shot + rules) over fine-tuning for immediate auditability and low-data efficiency.

## License
MIT
