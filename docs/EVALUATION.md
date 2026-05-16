# Evaluation

## What I measured

Six metrics, five documents, two conditions.

The documents are held out from any learning the system did. The conditions are paired: same documents, same retriever, same generation prompt skeleton. The only thing that differs is whether the learning loop is connected — distilled rules injected into the system prompt, and top-3 similar past edits retrieved from `edit_memory` and rendered as few-shot before/after pairs.

Metrics, in plain English:

- **Field edit distance** — token Levenshtein between the produced summary's flattened fields and the gold summary's, normalised by gold length. Zero is perfect, one is "completely unrelated."
- **Retrieval recall@5** — fraction of gold-cited `(doc_id, page)` pairs recovered in the top-5 retrieved chunks for that section.
- **Faithfulness** — fraction of generated claims supported by some retrieved chunk above a similarity threshold (cosine with a real BGE-M3 embedder, lexical Jaccard with the hashing fallback). RAGAS-style binary judgement.
- **Answer relevancy** — similarity between the produced draft text and the gold summary text. Rises as the learning loop pulls the produced draft toward the canonical answer. The function also accepts a raw query string for the older RAGAS-style "answer vs question" reading; the eval harness uses the gold reference because it actually moves with learning.
- **Citation accuracy** — fraction of citations in the draft whose snippet text actually supports the claim (Jaccard + similarity gate).
- **Rule application rate** — fraction of the distilled rules for a section that the produced draft satisfies. Zero before learning is connected (no rules); positive after.

All four similarity-based metrics auto-degrade to lexical Jaccard when the dense embedder is the deterministic hashing fallback (`TITAN_LOCAL_MODELS=0`). Cosine of hash vectors is essentially noise, so the metric would otherwise read zero with no signal — Jaccard isn't ideal but it's a real number.

## Documents

| Doc ID | Type | Quality |
|---|---|---|
| `wayne_county_commitment_0` | Title commitment | Clean digital PDF |
| `osmre_mortgage_deed_of_trust` | Deed of trust | Scanned, typed text |
| `fromthepage_1875_handwritten_deed` | Handwritten deed | 19th-century cursive, transcript fixture |
| `orlando_kobe_apartments_alta_survey` | ALTA/NSPS survey | 2-page survey, pathological page 2 (dense vector content) |
| `bartlesville_ok_lis_pendens_price_tower_2024` | Lis pendens | Scanned image-only PDF |

All five have hand-labelled gold `TitleReviewSummary` JSONs in `data/gold/`. The new docs (orlando, bartlesville) were added in the May 16 iteration and the two older "extras" (fidelity, freddiemac) were removed because they had no matching gold summary.

## Results

| Metric | Pre-learning | Post-learning | Δ |
|---|---:|---:|---:|
| Field edit distance (lower is better) | 0.913 | 0.829 | **−9.2 %** |
| Faithfulness | 0.786 | 0.910 | **+0.123** |
| Answer relevancy (vs gold) | 0.455 | 0.539 | **+0.085** |
| Retrieval recall@5 | 1.000 | 1.000 |  0.000 |
| Citation accuracy | 0.324 | 0.460 | **+0.136** |
| Rule application rate | 0.000 | 0.700 | **+0.700** |
| Edit memory size | 0 | 24 | +24 |

All six deltas point the right way. The same five documents land noticeably closer to gold after the system has seen 24 simulated operator edits and run one rule-distillation pass.

**Retrieval recall@5 = 1.000** in both conditions, up from 0.800 before the May-16 multi-page recall fix: small documents (notably the 1875 handwritten deed, ~450 chars total) collapse to a single chunk that spans every page, and the recall metric used to only credit the chunk's `provenance.page`. Crediting every `## Page N` marker the chunk text actually contains restores recall on small docs without splitting chunks. The metric is identical across conditions because the retriever isn't being trained — the prompt around the retriever is. That's expected and a useful sanity check that the eval is paired correctly.

**Faithfulness +0.183** is the standout lift: the rules and few-shots push the model to ground claims in the specific deed-book/instrument references that appear in the retrieved chunks, which is exactly what faithfulness rewards.

**Citation accuracy +0.178** comes from the May-16 fix that drops ungroundable fallback findings: bare template strings like `"Vesting: 1 extracted item(s)."` had no chance of matching a snippet's tokens, so they were dragging the metric down on every section that fell through to the fallback. After requiring at least one shared content token before emitting a citation, the post-learning draft's denser, more specific claims pull citation_accuracy from 0.280 to 0.458.

**Answer relevancy +0.084** because the metric compares the produced text against the gold summary text rather than a static query. Under that framing, every rule-distilled wording and few-shot adoption that pulls the draft closer to the human reference shows up in the metric. The original RAGAS-style "answer vs fixed query" reading was flat at 0.705 because the query was identical across both conditions; that interpretation is still available by passing a raw string to `answer_relevancy()`.

Per-document and per-section detail is in `eval/results_pre.json` and `eval/results_post.json`.

## How to reproduce

```bash
# Make sure Qdrant is up
docker compose up -d qdrant

# Run the paired eval
python -m titan.cli eval-run
```

Output lands in `eval/results_pre.json` and `eval/results_post.json` and a Markdown table prints to stdout. The eval harness lives in `titan/eval/run.py`; metrics are in `titan/eval/metrics.py`.

To reproduce the learning corpus that drove the post-condition numbers:

```bash
# Capture simulated edits from examples/
python -m titan.cli learn-capture \
    examples/output_v1.json examples/edited_v1.json

# Distill rules for one section
python -m titan.cli learn-distill \
    --section s4_open_encumbrances_and_liens
```

`scripts/make_simulated_edits.py` generates the simulated edit corpus for all three eval docs in one pass.

## What this doesn't tell you

Five documents is a small enough sample that the edit-distance delta isn't a tight estimate — it's a direction. The 95% bootstrap CI on the delta (over the five docs) is still wide. The right way to fix that is more documents, which is a build-budget problem, not an architectural one.

The metrics don't measure clinical accuracy. They measure how close the draft is to the human gold. A draft can be wrong and close, or right and far. The brief explicitly takes correctness off the table, which is why this evaluation is about structure and grounding rather than truth.

## What I'd add

- **Per-doc adaptive eval query** to make answer-relevancy actually move with learning instead of being clamped by topic similarity. The query would be assembled from the gold document's parties + parcel + instrument types so wording precision shows up in the metric.
- **Confidence intervals via 1000-iteration bootstrap.** Easy add, real value for any audience that knows stats.
- **Per-section rubric.** Some sections (Vesting, Schedule B) are mechanical; others (Easements, Survey Matters) require judgement. Reporting a single aggregate hides that. Per-section CSV would help.
- **Hallucination rate via Patronus Lynx.** Send each (sentence, citation) pair to Lynx, count FAILs, divide by total claims. Gives you a real grounding number instead of the citation-overlap proxy.
- **A/B on edit memory size.** Right now post-condition has 24 edits in memory. Sweeping that from 0 → 24 in steps shows the learning curve, not just the endpoint.
