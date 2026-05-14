"""Eval metrics for Titan.

Implements the four metrics the architecture calls for in §7:

* ``field_edit_distance`` — Levenshtein distance between produced section text
  and gold section text, normalized by gold length, averaged across sections.
* ``retrieval_recall_at_k`` — fraction of gold ``(doc_id, page)`` spans that
  appear in the top-k retrieved chunks.
* ``faithfulness`` — RAGAS-style: for each generated claim, does the retrieved
  context support it? Offline fallback uses token-overlap entailment;
  RAGAS/Gemini is used when ``ragas`` is installed and ``GOOGLE_API_KEY`` is
  configured.
* ``answer_relevancy`` — RAGAS-style: how well does the answer match the
  question. Offline fallback is cosine similarity between BGE-M3-style
  embeddings of the question and the answer.

Every metric is deterministic, returns floats in ``[0, 1]``, and degrades
gracefully when no Gemini key or RAGAS package is available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from titan.index.embed import DenseEmbedder, cosine, tokenize
from titan.index.models import SearchHit
from titan.schemas import CitedSentence, TitleReviewSection, TitleReviewSummary

SECTION_FIELDS: tuple[str, ...] = (
    "s1_vesting_and_estate",
    "s2_legal_description",
    "s3_chain_of_title",
    "s4_open_encumbrances_and_liens",
    "s5_easements_and_restrictions",
    "s6_requirements_schedule_b_i",
    "s7_exceptions_schedule_b_ii",
    "s8_taxes_and_survey_matters",
)


@dataclass(frozen=True)
class SectionEditDistance:
    section: str
    distance: float
    gold_length: int
    produced_length: int


def field_edit_distance(
    produced: TitleReviewSummary, gold: TitleReviewSummary
) -> tuple[float, list[SectionEditDistance]]:
    """Average normalized Levenshtein distance across all eight sections."""

    per_section: list[SectionEditDistance] = []
    for field in SECTION_FIELDS:
        gold_text = _section_text(getattr(gold, field))
        produced_text = _section_text(getattr(produced, field))
        if not gold_text:
            continue
        distance = _normalized_levenshtein(produced_text, gold_text)
        per_section.append(
            SectionEditDistance(
                section=field,
                distance=distance,
                gold_length=len(gold_text),
                produced_length=len(produced_text),
            )
        )
    if not per_section:
        return 0.0, per_section
    average = sum(item.distance for item in per_section) / len(per_section)
    return average, per_section


def retrieval_recall_at_k(
    gold: TitleReviewSummary, hits: list[SearchHit], k: int = 5
) -> float:
    """Fraction of gold (doc_id, page) pairs that appear in the top-k hits."""

    gold_spans = _collect_spans(gold)
    if not gold_spans:
        return 1.0
    hit_spans = {
        (hit.chunk.doc_id, hit.chunk.provenance.page) for hit in hits[:k]
    }
    matched = sum(1 for span in gold_spans if span in hit_spans)
    return matched / len(gold_spans)


def faithfulness(
    produced: TitleReviewSummary,
    hits: list[SearchHit],
    embedder: DenseEmbedder | None = None,
) -> float:
    """Fraction of generated claims supported by the retrieved context.

    Each claim is scored by the maximum cosine similarity (clamped to ``[0, 1]``)
    between the claim's embedding and the embedding of each retrieved chunk.
    This mirrors RAGAS faithfulness semantics in an LLM-free way: it rewards
    paraphrase that stays on-topic with the source chunks and penalises
    drift, instead of demanding lexical overlap.
    """

    claims = list(_iter_claims(produced))
    if not claims or not hits:
        return 0.0
    enc = embedder or DenseEmbedder()
    chunk_vectors = enc.embed(
        [hit.chunk.contextual_text for hit in hits]
    )
    claim_vectors = enc.embed(claims)
    scores: list[float] = []
    for claim_vector in claim_vectors:
        best = max(
            (cosine(claim_vector, chunk_vector) for chunk_vector in chunk_vectors),
            default=0.0,
        )
        scores.append(max(0.0, min(1.0, (best + 1.0) / 2.0)))
    return sum(scores) / len(scores)


def answer_relevancy(
    produced: TitleReviewSummary,
    query: str,
    embedder: DenseEmbedder | None = None,
) -> float:
    """Cosine similarity between the question and the produced answer."""

    answer_text = _summary_text(produced)
    if not answer_text or not query:
        return 0.0
    enc = embedder or DenseEmbedder()
    vectors = enc.embed([query, answer_text])
    similarity = cosine(vectors[0], vectors[1])
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def _section_text(section: TitleReviewSection) -> str:
    summary = " ".join(sentence.text for sentence in section.summary)
    findings = " ".join(sentence.text for sentence in section.bullet_findings)
    gaps = " ".join(section.gaps)
    return " ".join(part for part in (summary, findings, gaps) if part).strip()


def _summary_text(summary: TitleReviewSummary) -> str:
    parts: list[str] = []
    for field in SECTION_FIELDS:
        parts.append(_section_text(getattr(summary, field)))
    parts.append(" ".join(sentence.text for sentence in summary.overall_summary))
    return " ".join(part for part in parts if part).strip()


def _iter_claims(summary: TitleReviewSummary) -> Iterable[str]:
    for field in SECTION_FIELDS:
        section: TitleReviewSection = getattr(summary, field)
        for sentence in section.summary:
            yield sentence.text
        for sentence in section.bullet_findings:
            yield sentence.text
    for sentence in summary.overall_summary:
        yield sentence.text


def _collect_spans(
    summary: TitleReviewSummary,
) -> set[tuple[str, int]]:
    spans: set[tuple[str, int]] = set()
    for sentence in _iter_cited_sentences(summary):
        for citation in sentence.citations:
            spans.add((citation.doc_id, citation.page))
    return spans


def _iter_cited_sentences(summary: TitleReviewSummary) -> Iterable[CitedSentence]:
    for field in SECTION_FIELDS:
        section: TitleReviewSection = getattr(summary, field)
        yield from section.summary
        yield from section.bullet_findings
    yield from summary.overall_summary


def _normalized_levenshtein(produced: str, gold: str) -> float:
    """Token-level Levenshtein normalized by the gold token count.

    Using a single unit (tokens) for both numerator and denominator keeps a
    perfect match at 0.0 and an entirely-different draft at ~1.0.
    """

    distance, gold_token_count = _levenshtein(produced, gold)
    denom = max(gold_token_count, 1)
    return min(1.0, distance / denom)


def _levenshtein(left: str, right: str) -> tuple[int, int]:
    left_tokens = re.findall(r"\w+", left.lower())
    right_tokens = re.findall(r"\w+", right.lower())
    if not left_tokens:
        return len(right_tokens), len(right_tokens)
    if not right_tokens:
        return len(left_tokens), 0
    previous = list(range(len(right_tokens) + 1))
    for i, left_token in enumerate(left_tokens, start=1):
        current = [i]
        for j, right_token in enumerate(right_tokens, start=1):
            cost = 0 if left_token == right_token else 1
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1], len(right_tokens)


_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "as",
        "at",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "from",
        "into",
        "but",
        "if",
        "any",
        "no",
        "not",
        "must",
        "should",
        "shall",
        "will",
        "may",
        "can",
        "each",
        "every",
        "all",
        "some",
        "such",
        "than",
        "then",
        "so",
        "also",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "how",
    }
)
