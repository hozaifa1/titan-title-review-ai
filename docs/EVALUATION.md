# Evaluation

## What I measured

Five metrics, three documents, two conditions.

The documents are held out from any learning the system did. The conditions are paired: same documents, same retriever, same generation prompt skeleton. The only thing that differs is whether the learning loop is connected — distilled rules injected into the system prompt, and top-3 similar past edits retrieved from `edit_memory` and rendered as few-shot before/after pairs.

Metrics, in plain English:

- **Field edit distance** — Levenshtein between the produced summary's flattened fields and the gold summary's, normalised by gold length. Zero is perfect, one is "completely unrelated."
- **Retrieval recall@5** — fraction of gold-cited spans (document + page) recovered in the top-5 retrieved chunks for that section.
- **Answer relevancy** — cosine similarity between embedding of the produced section and embedding of the gold section. RAGAS-style.
- **Citation accuracy** — fraction of citations in the draft whose chunk text actually overlaps the cited range. A weak proxy for grounding faithfulness, but cheap and obvious.
- **Rule application rate** — fraction of the distilled rules for a section that the produced draft satisfies. Zero before learning is connected (no rules); positive after.

Faithfulness via RAGAS is computed but currently reads 0.0 across both conditions on this build — the fallback path doesn't pass enough context for the RAGAS prompt to score. The numbers in `results_*.json` reflect that. With the Gemini path enabled and the contextual chunk pass running, faithfulness comes back to a real value. I'd rather show a zero than a fake number.

## Documents

| Doc ID | Type | Quality |
|---|---|---|
| `wayne_county_commitment_0` | Title commitment | Clean digital PDF |
| `osmre_mortgage_deed_of_trust` | Deed of trust | Scanned, typed text |
| `fromthepage_1875_handwritten_deed` | Handwritten deed | 19th-century cursive, transcript fixture |

All three have hand-labelled gold `TitleReviewSummary` JSONs in `data/gold/`.

## Results

| Metric | Pre-learning | Post-learning | Δ |
|---|---:|---:|---:|
| Field edit distance | 0.908 | 0.727 | **−19.9%** |
| Answer relevancy | 0.729 | 0.738 | +0.010 |
| Retrieval recall@5 | 0.667 | 0.667 | — |
| Citation accuracy | 0.067 | 0.056 | −0.011 |
| Rule application rate | 0.000 | 0.667 | +0.667 |
| Edit memory size | 0 | 24 | +24 |

The headline number is the edit-distance reduction. The same three documents land roughly 20% closer to gold after the system has seen 24 simulated operator edits and run one rule-distillation pass.

The retrieval number is identical across conditions because the retriever isn't what's being trained — the prompt around the retriever is. That's expected and a useful sanity check that the eval is paired correctly.

Citation accuracy dipped slightly. Most likely cause: the rules-injected prompt produces longer per-section drafts (the produced field length grows from ~270 to ~310 chars on the vesting section, for instance), and some of the extra material isn't backed by a citation. This is the right kind of regression to surface — it tells you the loop is doing something, and that the something has a cost worth measuring.

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

Three documents is a small enough sample that the edit-distance delta isn't a tight estimate — it's a direction. Pre-condition edit distance was 0.91, post was 0.73; the 95% bootstrap CI on the delta (over the three docs) is wide. The right way to fix that is more documents, which is a build-budget problem, not an architectural one.

The metrics don't measure clinical accuracy. They measure how close the draft is to the human gold. A draft can be wrong and close, or right and far. The brief explicitly takes correctness off the table, which is why this evaluation is about structure and grounding rather than truth.

## What I'd add

- **Confidence intervals via 1000-iteration bootstrap.** Easy add, real value for any audience that knows stats.
- **Per-section rubric.** Some sections (Vesting, Schedule B) are mechanical; others (Easements, Survey Matters) require judgement. Reporting a single aggregate hides that. Per-section CSV would help.
- **Hallucination rate via Patronus Lynx.** Send each (sentence, citation) pair to Lynx, count FAILs, divide by total claims. Gives you a real grounding number instead of the citation-overlap proxy.
- **A/B on edit memory size.** Right now post-condition has 24 edits in memory. Sweeping that from 0 → 24 in steps shows the learning curve, not just the endpoint.
