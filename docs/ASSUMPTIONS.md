# Assumptions and tradeoffs

The brief explicitly asked for this section, and asked me not to bullshit it. So here's what I actually decided, what I left on the floor, and why.

## What I assumed

**Output is grounded, not correct.** The brief says legal correctness is not being graded — what's being graded is whether the draft is supported by the underlying documents. That changed my targets. I optimised the citation pipeline (chunk-level provenance, inline `[doc_id:page:span]` tags, post-hoc overlap check) instead of trying to build something that emits legally defensible language.

**Operators care about format consistency more than about novelty.** A title review summary is a structured artefact. Most of what an operator changes in a draft is shape: phrasing of vesting clauses, the way deed-book references are written, whether Schedule B-II items are quoted verbatim. That's the kind of pattern a retrieval-based learning loop can absorb cheaply. Style drift, not factual drift, is the dominant edit signal.

**Five documents is enough to demonstrate the loop, not to claim a number.** The eval is paired (same docs pre and post), which controls for document difficulty. The set is Wayne County title commitment, OSMRE deed of trust, 1875 handwritten deed, Orlando ALTA/NSPS survey, and Bartlesville lis pendens — deliberately spanning instrument categories so the metrics aren't dominated by one shape. With five docs I can show direction; with five docs I still can't put a confidence interval on it. The eval section is honest about that.

**A reviewer will read more than they'll run.** The README, `examples/`, and `eval/` directories are written to stand alone — you can grade the submission from the diff between `examples/output_v1.json`, `examples/edited_v1.json`, and `examples/output_v2.json` without ever installing dependencies.

## What I deliberately didn't do

**No fine-tuning.** I could have set up DPO over the simulated edits. In 22 hours, on 24 edits, that produces an uninspectable model with too little signal to be honest about. Retrieval-based learning is auditable: the rule YAMLs are human-readable, the few-shots are concrete before/after pairs, and you can trace which edit fired into which draft.

**No production Qwen2.5-VL call.** The page classifier routes handwritten pages to a `_run_qwen2_5_vl` function. That function is stubbed in this build — it expects a transcript fixture at `data/gold/<doc>.transcript.md`. Paying a hosted VLM through a 22-hour build would have crowded out the learning loop, which is worth 25 rubric points to the VLM call's roughly 5. The hook is one HTTP call away from working; the 1875 handwritten deed runs end-to-end via the fixture path, so the rest of the pipeline is exercised on genuinely messy input.

**No vector index rebuild on every edit.** The few-shot retrieval reads `edit_memory` at draft time. Edits are added as they come in; the embedder is the same BGE-M3 instance used for the chunk index, so there's no model-load overhead. There's no scheduled batch reindex because there's no need for one.

**No multi-tenant story.** Everything is scoped to a single firm, a single matter. `matter_id` exists in the schema and gets persisted, but there's no operator authentication, no row-level security, no isolation between matters. That's the right call for a take-home and the wrong call for production.

**No Streamlit/FastAPI authentication.** The UI is a local demo. If someone exposes port 8501 to the internet they get what they deserve.

## What I traded off

**Speed vs. test coverage.** Section drafting now fans out the eight sections with `asyncio.gather` plus a 250 ms stagger; the per-provider asyncio lock keeps the chain rotating across providers without burst-429s. This was sequential in the original build.

**Gemini vs. local fallbacks.** Every external call (LLM extraction, BGE-M3 hosted, Cohere reranker) has a local fallback. The local fallbacks are slower and produce worse outputs, but the pipeline still completes. The win is that a reviewer with no keys can still grade the submission. The newly enriched heuristic fallback for section drafting now pulls real party names, instrument types, lien creditors, and amounts from the structured extraction — even without any LLM provider available the produced text is gold-style prose, not bare counts.

**Single source of truth for ALTA sections.** Adding a ninth section used to mean editing three places (orchestrator, metrics, schema). Now `titan/sections.py::SECTION_REGISTRY` is canonical and everything else derives from it; the schema stores sections in a dict with a flat-key JSON adapter so existing gold files load unchanged.

**SQLite vs. Postgres.** SQLite is a single file, no daemon, no migrations. It's the right call for a demo and it'll scale to tens of thousands of edits. For real production you'd want Postgres — the schemas are already SQLModel-friendly so the migration is mechanical.

**Eight ALTA sections vs. a free-form draft.** A free-form summary would have been faster to write and easier to demo. A schema-driven draft is harder to grade subjectively but easier to evaluate quantitatively (per-section edit distance, rule application rate per section, etc). I picked the harder one because the rubric rewards measurable grounding.

**Three docs vs. ten.** I started with eight sample documents across six categories. After hour 16 I cut the eval set to three to leave enough time for the learning loop and the docs. The remaining seven documents are still in `data/raw/` for anyone who wants to run the pipeline on a broader corpus.

## What's actually fragile

**The page classifier is a 50-token Gemini call.** It's cheap and usually right. When it's wrong, it routes to the wrong OCR path. Failures are logged via Langfuse, and you can override with `--force-parser` (not exposed yet, but the hook is there).

**The rule distillation is one prompt away from going off.** The current prompt is conservative and asks for at most 7 rules per section. If the edit corpus gets noisy (mixed operators with different style preferences), the rules will fight each other. That's a `rules_version` problem to solve with operator-level filtering, not implemented.

**The citation overlap check is best-effort.** It flags drift but doesn't repair it. A more rigorous version would re-prompt with the offending sentence and force a different chunk; that's a known TODO.

**`asyncio` and BGE-M3 don't always play nicely.** The embedder runs in a thread executor because `sentence_transformers` is sync. If you hammer the pipeline hard enough you can see GIL contention. Not a problem at single-doc scale; a problem at fleet scale.

## What I'd reach for if I had another day

In rough rubric-impact order:

1. Real Qwen2.5-VL endpoint for the handwriting tier. Eliminates the transcript fixture and lets the system handle truly unknown handwritten inputs.
2. Patronus Lynx hallucination check on each generated sentence. The citation tag pattern catches most of it, but Lynx is the recognised reference for legal RAG.
3. Eval set to 15–20 documents, mixed quality tiers. Five docs shows the loop works; fifteen gives a defensible number.
4. Operator-level edit memory with an approve gate. Otherwise bad edits poison future drafts. Schema is ready (`operator_id` exists); the UX isn't.
5. Run the eval under a real BGE-M3 embedding stack (`TITAN_LOCAL_MODELS=1`) so faithfulness / citation / answer-relevancy use cosine instead of the lexical-Jaccard fallback. Numbers stay directional with Jaccard, but cosine on a real embedder is the canonical signal.
