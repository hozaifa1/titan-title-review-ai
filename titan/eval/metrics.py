"""Eval metrics for Titan.

Implements the metrics the architecture calls for in §7 and §11:

* ``field_edit_distance`` — token-level Levenshtein between produced section
  text and gold section text, normalized by gold length, averaged across
  sections.
* ``retrieval_recall_at_k`` — fraction of gold ``(doc_id, page)`` spans that
  appear in the top-k retrieved chunks.
* ``faithfulness`` — RAGAS-style: for each generated claim, is it supported
  by the retrieved context? Discrete binary judge (cosine threshold),
  averaged across claims. Tighter than the prior ``(cos+1)/2`` mapping.
* ``answer_relevancy`` — RAGAS-style: how well does the answer match the
  question. Offline fallback is cosine similarity between BGE-M3 embeddings
  of the question and the answer.
* ``citation_accuracy`` (§11.1) — fraction of cited sentences whose citation
  snippet actually contains/supports the sentence's claim.
* ``rule_application_rate`` (§11.1) — fraction of active rules with which
  the produced draft visibly complies.

Every metric is deterministic, returns floats in ``[0, 1]``, and degrades
gracefully when no Gemini key or RAGAS package is available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from titan.index.embed import DenseEmbedder, cosine
from titan.index.models import SearchHit
from titan.schemas import (
    CitedSentence,
    Rule,
    TitleReviewSection,
    TitleReviewSummary,
)

# Single source of truth — derived from :mod:`titan.sections`. Adding a new
# section means editing SECTION_REGISTRY there; metrics iterate automatically.
from titan.sections import section_field_names

SECTION_FIELDS: tuple[str, ...] = section_field_names()


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


FAITHFULNESS_THRESHOLD = 0.55  # cosine cutoff for "claim supported by chunk"
CITATION_ACCURACY_THRESHOLD = 0.45  # cosine cutoff for "snippet supports claim"
# Hashing-fallback embeddings produce near-random cosine values, so we
# bypass the cosine path entirely and use token overlap when the embedder
# isn't backed by a real model. Threshold tuned for title-review prose.
FAITHFULNESS_LEXICAL_OVERLAP = 0.18
CITATION_LEXICAL_OVERLAP = 0.15


def _embedder_is_lexical(enc: DenseEmbedder) -> bool:
    """True when the embedder is the hashing fallback (no real model loaded)."""
    return getattr(enc, "backend", "") == "hashing-fallback"


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower())
        if len(token) >= 4 and token not in _STOPWORDS
    }


def faithfulness(
    produced: TitleReviewSummary,
    hits: list[SearchHit],
    embedder: DenseEmbedder | None = None,
) -> float:
    """Fraction of generated claims supported by the retrieved context.

    Discrete RAGAS-style judge: a claim is "supported" iff its best cosine
    similarity against any retrieved chunk meets ``FAITHFULNESS_THRESHOLD``.
    The aggregate is the supported fraction. This matches RAGAS faithfulness
    semantics (binary per-claim verdict, not a soft average) without an LLM.

    When the only embedder available is the deterministic hashing fallback
    we substitute token-overlap (cosine of hash vectors is meaningless), so
    the metric remains a usable signal in CI without a downloaded model.
    """

    claims = list(_iter_claims(produced))
    if not claims or not hits:
        return 0.0
    enc = embedder or DenseEmbedder()
    if _embedder_is_lexical(enc):
        chunk_token_sets = [_content_tokens(hit.chunk.contextual_text) for hit in hits]
        supported = 0
        for claim in claims:
            claim_tokens = _content_tokens(claim)
            if not claim_tokens:
                continue
            best = 0.0
            for chunk_tokens in chunk_token_sets:
                if not chunk_tokens:
                    continue
                overlap = len(claim_tokens & chunk_tokens) / len(claim_tokens)
                if overlap > best:
                    best = overlap
            if best >= FAITHFULNESS_LEXICAL_OVERLAP:
                supported += 1
        return supported / len(claims)

    chunk_vectors = enc.embed([hit.chunk.contextual_text for hit in hits])
    claim_vectors = enc.embed(claims)
    supported = 0
    for claim_vector in claim_vectors:
        best = max(
            (cosine(claim_vector, chunk_vector) for chunk_vector in chunk_vectors),
            default=-1.0,
        )
        if best >= FAITHFULNESS_THRESHOLD:
            supported += 1
    return supported / len(claims)


def citation_accuracy(
    produced: TitleReviewSummary,
    embedder: DenseEmbedder | None = None,
) -> tuple[float, dict[str, float]]:
    """Fraction of cited sentences whose snippet actually supports the claim.

    For each ``CitedSentence`` with at least one citation that carries a
    snippet, we check whether the snippet text supports the sentence text.
    Support is judged by both lexical overlap (Jaccard of significant tokens
    >= 0.10) AND embedding similarity above ``CITATION_ACCURACY_THRESHOLD``.
    Either signal alone is too lenient (lexical alone misses paraphrase;
    embedding alone rewards topical drift); requiring both raises the bar.

    Returns ``(overall_score, per_section_scores)``.
    """

    enc = embedder or DenseEmbedder()
    per_section: dict[str, float] = {}
    overall_supported = 0
    overall_total = 0

    for field in SECTION_FIELDS:
        section: TitleReviewSection = getattr(produced, field)
        section_supported = 0
        section_total = 0
        for sentence in section.summary + section.bullet_findings:
            verdict = _citation_supports(sentence, enc)
            if verdict is None:
                continue
            section_total += 1
            overall_total += 1
            if verdict:
                section_supported += 1
                overall_supported += 1
        if section_total:
            per_section[field] = round(section_supported / section_total, 4)
    score = overall_supported / overall_total if overall_total else 0.0
    return score, per_section


def rule_application_rate(
    produced: TitleReviewSummary,
    rules: list[Rule],
) -> tuple[float, dict[str, str]]:
    """Fraction of active rules with which the produced draft visibly complies.

    Heuristic per rule: extract content tokens (>=4 chars, non-stopword) from
    the rule text and check whether at least 30% appear anywhere in the
    produced draft. This rewards rules whose distinctive vocabulary the
    drafter has actually adopted, without an LLM judge.

    Returns ``(score, per_rule_verdicts)`` where verdicts are one of
    ``compliant``, ``non_compliant``, ``not_applicable``.
    """

    if not rules:
        return 0.0, {}
    draft_tokens = {
        token
        for token in re.findall(r"\w+", _summary_text(produced).lower())
        if len(token) >= 4
    }
    verdicts: dict[str, str] = {}
    compliant = 0
    applicable = 0
    for rule in rules:
        rule_tokens = {
            token
            for token in re.findall(r"\w+", rule.text.lower())
            if len(token) >= 4 and token not in _STOPWORDS
        }
        if not rule_tokens:
            verdicts[rule.id] = "not_applicable"
            continue
        applicable += 1
        overlap = len(rule_tokens & draft_tokens) / len(rule_tokens)
        if overlap >= 0.30:
            verdicts[rule.id] = "compliant"
            compliant += 1
        else:
            verdicts[rule.id] = "non_compliant"
    score = compliant / applicable if applicable else 0.0
    return score, verdicts


def _citation_supports(sentence: CitedSentence, enc: DenseEmbedder) -> bool | None:
    """Evaluate whether at least one of a sentence's citations supports it.

    Returns ``None`` when there is nothing to score (no citation or empty
    snippet) — those sentences are excluded from the citation-accuracy
    denominator.

    With a real embedding model: jaccard >= 0.10 AND cosine >= threshold.
    With the hashing fallback: jaccard >= ``CITATION_LEXICAL_OVERLAP`` only,
    since hash-cosine is uninformative.
    """

    candidate_snippets = [
        citation.snippet for citation in sentence.citations if citation.snippet
    ]
    if not candidate_snippets or not sentence.text.strip():
        return None
    claim_tokens = _content_tokens(sentence.text)
    if not claim_tokens:
        return None
    lexical_only = _embedder_is_lexical(enc)
    if lexical_only:
        for snippet in candidate_snippets:
            snippet_tokens = _content_tokens(snippet)
            if not snippet_tokens:
                continue
            jaccard = len(claim_tokens & snippet_tokens) / max(
                len(claim_tokens | snippet_tokens), 1
            )
            if jaccard >= CITATION_LEXICAL_OVERLAP:
                return True
        return False
    claim_vec = enc.embed([sentence.text])[0]
    snippet_vecs = enc.embed(candidate_snippets)
    for snippet, snippet_vec in zip(candidate_snippets, snippet_vecs):
        snippet_tokens = _content_tokens(snippet)
        if not snippet_tokens:
            continue
        jaccard = len(claim_tokens & snippet_tokens) / max(
            len(claim_tokens | snippet_tokens), 1
        )
        sim = cosine(claim_vec, snippet_vec)
        if jaccard >= 0.10 and sim >= CITATION_ACCURACY_THRESHOLD:
            return True
    return False


def answer_relevancy(
    produced: TitleReviewSummary,
    query: str,
    embedder: DenseEmbedder | None = None,
) -> float:
    """Cosine similarity between the question and the produced answer.

    Falls back to token Jaccard with hashing-fallback embedders so the score
    is meaningful without a downloaded BGE-M3 model.
    """

    answer_text = _summary_text(produced)
    if not answer_text or not query:
        return 0.0
    enc = embedder or DenseEmbedder()
    if _embedder_is_lexical(enc):
        q_tokens = _content_tokens(query)
        a_tokens = _content_tokens(answer_text)
        if not q_tokens:
            return 0.0
        return len(q_tokens & a_tokens) / max(len(q_tokens), 1)
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
