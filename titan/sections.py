"""Single source of truth for the ALTA review sections.

Historically the eight ALTA sections were declared in three coupled places:
``orchestrator.SECTION_SPECS``, ``metrics.SECTION_FIELDS``, and the explicit
``s1_..._s8_`` fields on ``TitleReviewSummary``. Adding a ninth section
meant editing all three. Defining them once here and consuming the registry
elsewhere lets a new ALTA section drop in with a single edit.

``TitleReviewSummary`` keeps backward-compatible ``s1_..._s8_`` JSON accessors
via a custom validator/serializer in ``titan/schemas/title.py`` — the model
stores sections in a ``dict[str, TitleReviewSection]`` keyed by ``field_name``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectionSpec:
    """One ALTA review section's identity and retrieval/structured hints."""

    field_name: str
    section_name: str
    query: str
    structured_fields: tuple[str, ...]


SECTION_REGISTRY: tuple[SectionSpec, ...] = (
    SectionSpec(
        "s1_vesting_and_estate",
        "Vesting and Estate",
        "Schedule A vested owner, estate or interest, proposed insured, effective date",
        ("vesting", "estate_or_interest", "proposed_insured", "effective_date"),
    ),
    SectionSpec(
        "s2_legal_description",
        "Legal Description",
        "legal description parcel lot block subdivision metes bounds APN",
        ("legal_description",),
    ),
    SectionSpec(
        "s3_chain_of_title",
        "Chain of Title",
        "deeds chain of title grantor grantee recording book page instrument number",
        ("chain_of_title", "parties"),
    ),
    SectionSpec(
        "s4_open_encumbrances_and_liens",
        "Open Encumbrances and Liens",
        "open liens mortgages deeds of trust judgments unreleased encumbrances",
        ("open_liens", "released_liens"),
    ),
    SectionSpec(
        "s5_easements_and_restrictions",
        "Easements and Restrictions",
        "easements restrictions covenants rights of way utility access declarations",
        ("easements", "restrictions"),
    ),
    SectionSpec(
        "s6_requirements_schedule_b_i",
        "Requirements - Schedule B-I",
        "Schedule B part I requirements payoff releases execution recordation title company",
        ("schedule_b_requirements",),
    ),
    SectionSpec(
        "s7_exceptions_schedule_b_ii",
        "Exceptions - Schedule B-II",
        "Schedule B part II exceptions standard exceptions taxes survey matters mineral rights",
        ("schedule_b_exceptions",),
    ),
    SectionSpec(
        "s8_taxes_and_survey_matters",
        "Taxes and Survey Matters",
        "tax certificate assessed taxes delinquent taxes survey encroachments boundary matters",
        ("taxes", "survey_matters"),
    ),
)


def section_field_names() -> tuple[str, ...]:
    """Stable order of ``field_name`` ids for indexing."""

    return tuple(spec.field_name for spec in SECTION_REGISTRY)


def section_by_field(field_name: str) -> SectionSpec:
    """Look up a section spec by its ``field_name``; raises ``KeyError`` if unknown."""

    for spec in SECTION_REGISTRY:
        if spec.field_name == field_name:
            return spec
    raise KeyError(f"Unknown section field_name: {field_name}")


__all__ = ["SectionSpec", "SECTION_REGISTRY", "section_field_names", "section_by_field"]
