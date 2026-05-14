## Titan Title Review AI

Hybrid indexing and retrieval for title-review source documents.

### Local Setup

```powershell
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python -m pytest tests/test_hybrid_retrieval.py -q
python -m titan.cli index-query --query "Who is the vested owner?" --top-k 5
```

The local CLI will use BGE-M3 / bge-reranker-v2-m3 when
`TITAN_LOCAL_MODELS=1` is set. Keep the default `TITAN_LOCAL_MODELS=0` for fast
smoke tests; it uses deterministic local embeddings/reranking and avoids pulling
multi-GB Hugging Face weights.

To exercise the real Transformer path:

```powershell
$env:TITAN_LOCAL_MODELS="1"
python -m titan.cli index-query --query "Who is the vested owner?" --top-k 5
```

### Qdrant With Docker

```powershell
docker compose up -d qdrant
python -m titan.cli index-query --query "Who is the vested owner?" --top-k 5 --qdrant-url http://localhost:6333
```

If Docker is not running, verify the same named-vector mirror path with
Qdrant's in-memory backend:

```powershell
python -m titan.cli index-query --query "Who is the vested owner?" --top-k 5 --qdrant-url :memory:
```

The `title_chunks` collection is created with two named vectors:

- `dense`
- `sparse`

Each payload includes `doc_id`, `doc_type`, `provenance`, source text, and
contextual text.

### Full Docker Flow

```powershell
docker compose --profile app build titan
docker compose --profile app run --rm titan index-query --query "Who is the vested owner?" --top-k 5 --qdrant-url http://qdrant:6333
```

### Environment

Copy `.env.example` to `.env` and fill keys as available:

```powershell
Copy-Item .env.example .env
```

`GOOGLE_API_KEY` enables Gemini contextual chunk sentences. Without it, the
chunker uses a deterministic section/page-context fallback.

`TITAN_LOCAL_MODELS=1` enables Hugging Face downloads for `BAAI/bge-m3` and
`BAAI/bge-reranker-v2-m3`.
