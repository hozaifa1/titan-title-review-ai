"""Pydantic contracts for extracted title documents and review summaries."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator

T = TypeVar("T")


class Provenance(BaseModel):
    doc_id: str
    page: int
    char_span: tuple[int, int] | None = None
    snippet: str | None = Field(None, description="<=200 char excerpt for human verification")


class FieldWithProvenance(BaseModel, Generic[T]):
    value: T
    confidence: float = Field(ge=0.0, le=1.0)
    source: Provenance


class PartyParty(BaseModel):
    name: str
    role: Literal[
        "grantor",
        "grantee",
        "trustor",
        "trustee",
        "beneficiary",
        "lender",
        "borrower",
        "lienholder",
        "owner",
        "buyer",
        "seller",
    ]
    is_entity: bool = False
    capacity: Optional[str] = Field(None, description="e.g. 'a single man', 'as joint tenants'")


class LegalDescription(BaseModel):
    description_type: Literal["metes_and_bounds", "platted", "condominium", "section_township_range"]
    text: str
    plat_book: Optional[str] = None
    plat_page: Optional[str] = None
    lot: Optional[str] = None
    block: Optional[str] = None
    subdivision: Optional[str] = None
    parcel_id_apn: Optional[str] = None
    acreage: Optional[Decimal] = None


class ChainOfTitleLink(BaseModel):
    instrument_type: Literal[
        "warranty_deed",
        "grant_deed",
        "quitclaim_deed",
        "special_warranty_deed",
        "trustees_deed",
        "tax_deed",
        "deed_of_trust",
        "mortgage",
        "release",
        "assignment",
        "affidavit",
        "court_order",
    ]
    grantor: list[str]
    grantee: list[str]
    instrument_date: Optional[date] = None
    recorded_date: Optional[date] = None
    book: Optional[str] = None
    page: Optional[str] = None
    instrument_number: Optional[str] = None
    consideration: Optional[Decimal] = None
    source: Provenance


class Lien(BaseModel):
    lien_type: Literal[
        "mortgage",
        "deed_of_trust",
        "tax_lien",
        "judgment_lien",
        "mechanics_lien",
        "hoa_lien",
        "ucc_filing",
        "child_support",
        "irs_lien",
        "state_tax_lien",
        "assessment",
    ]
    creditor: str
    debtor: str
    original_amount: Optional[Decimal] = None
    current_balance: Optional[Decimal] = None
    recorded_date: Optional[date] = None
    book: Optional[str] = None
    page: Optional[str] = None
    instrument_number: Optional[str] = None
    status: Literal["open", "released", "satisfied", "disputed", "unknown"]
    release_reference: Optional[str] = None
    source: Provenance


class Easement(BaseModel):
    easement_type: Literal["utility", "access", "drainage", "conservation", "right_of_way", "ingress_egress", "party_wall", "other"]
    holder: Optional[str] = None
    description: str
    recorded_date: Optional[date] = None
    instrument_number: Optional[str] = None
    appurtenant_or_in_gross: Optional[Literal["appurtenant", "in_gross"]] = None
    source: Provenance


class Restriction(BaseModel):
    restriction_type: Literal["ccr", "hoa_declaration", "deed_restriction", "zoning_overlay", "historic_designation", "other"]
    description: str
    recorded_date: Optional[date] = None
    instrument_number: Optional[str] = None
    discriminatory_redacted: bool = False
    source: Provenance


class TaxInfo(BaseModel):
    tax_year: int
    tax_amount: Optional[Decimal] = None
    paid: Optional[bool] = None
    delinquent_amount: Optional[Decimal] = None
    parcel_id: Optional[str] = None
    taxing_authority: Optional[str] = None
    source: Provenance


class ScheduleBRequirement(BaseModel):
    """ALTA Schedule B Part I - must be satisfied before policy issues."""

    requirement_id: str
    text: str
    category: Literal[
        "payment",
        "execution_recordation",
        "release_of_lien",
        "satisfaction_of_judgment",
        "death_administration",
        "entity_authority",
        "survey",
        "other",
    ]
    addressed_to: Literal["seller", "buyer", "lender", "title_co", "other"]
    source: Provenance


class ScheduleBException(BaseModel):
    """ALTA Schedule B Part II - what the policy will NOT insure against."""

    exception_id: str
    text: str
    category: Literal["standard", "tax", "lien", "easement", "restriction", "survey_matter", "mineral_rights", "lease", "other"]
    is_standard: bool
    can_be_removed: Optional[bool] = None
    removal_action: Optional[str] = None
    source: Provenance


class SurveyMatter(BaseModel):
    issue_type: Literal["encroachment", "boundary_dispute", "missing_survey", "improvement_outside_lot", "easement_violation", "other"]
    description: str
    source: Provenance


class TitleDocument(BaseModel):
    """Canonical extracted representation of ONE source document."""

    doc_id: str
    doc_type: Literal[
        "title_commitment",
        "warranty_deed",
        "grant_deed",
        "quitclaim_deed",
        "deed_of_trust",
        "mortgage",
        "release",
        "judgment",
        "tax_certificate",
        "survey",
        "plat",
        "affidavit",
        "ucc_filing",
        "court_order",
        "other",
    ]
    file_path: str
    page_count: int
    parsed_at: date

    effective_date: Optional[FieldWithProvenance[date]] = None
    proposed_insured: Optional[FieldWithProvenance[str]] = None
    estate_or_interest: Optional[FieldWithProvenance[Literal["fee_simple", "leasehold", "easement", "life_estate", "other"]]] = None
    vesting: list[PartyParty] = Field(default_factory=list)
    policy_amount: Optional[FieldWithProvenance[Decimal]] = None

    legal_description: Optional[LegalDescription] = None
    parties: list[PartyParty] = Field(default_factory=list)

    chain_of_title: list[ChainOfTitleLink] = Field(default_factory=list)
    open_liens: list[Lien] = Field(default_factory=list)
    released_liens: list[Lien] = Field(default_factory=list)
    easements: list[Easement] = Field(default_factory=list)
    restrictions: list[Restriction] = Field(default_factory=list)
    taxes: list[TaxInfo] = Field(default_factory=list)
    schedule_b_requirements: list[ScheduleBRequirement] = Field(default_factory=list)
    schedule_b_exceptions: list[ScheduleBException] = Field(default_factory=list)
    survey_matters: list[SurveyMatter] = Field(default_factory=list)

    has_recording_stamp: bool = False
    notarized: Optional[bool] = None
    extraction_warnings: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    doc_id: str
    page: int
    char_span: tuple[int, int]
    snippet: str


class CitedSentence(BaseModel):
    text: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"] = "high"


class TitleReviewSection(BaseModel):
    section_name: str
    summary: list[CitedSentence]
    bullet_findings: list[CitedSentence] = Field(default_factory=list)
    gaps: list[str] = Field(
        default_factory=list,
        description="Information missing from source docs that an operator should obtain",
    )
    flags: list[Literal["red", "yellow", "green"]] = Field(default_factory=list)

    @field_validator("flags", mode="before")
    @classmethod
    def _coerce_flag_list(cls, value: object) -> object:
        """Accept ``"green"`` as well as ``["green"]`` — Llama collapses single-item lists."""

        if isinstance(value, str):
            return [value]
        return value

    @field_validator("gaps", mode="before")
    @classmethod
    def _coerce_gaps(cls, value: object) -> object:
        """Accept dicts and nested structures the model emits, flatten to strings.

        Llama/qwen sometimes emits `gaps` as ``[{"field": "x", "issue": "..."}]``
        or other shapes instead of plain strings. Flatten so the section
        validates instead of losing it to the offline fallback.
        """

        if isinstance(value, str):
            return [value]
        if not isinstance(value, list):
            return value
        flattened: list[str] = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    flattened.append(item.strip())
                continue
            if isinstance(item, dict):
                preferred_keys = ("text", "gap", "issue", "description", "message", "missing")
                picked = next((str(item[key]) for key in preferred_keys if key in item and item[key]), None)
                if picked is None and item:
                    picked = "; ".join(f"{k}: {v}" for k, v in item.items() if v)
                if picked:
                    flattened.append(picked)
                continue
            if item is not None:
                flattened.append(str(item))
        return flattened

    @field_validator("summary", "bullet_findings", mode="before")
    @classmethod
    def _coerce_cited_sentences(cls, value: object) -> object:
        """Accept loose LLM outputs and normalize them to CitedSentence shape.

        Llama 3.3 and other models frequently emit a bullet finding as a bare
        string ("Defects, liens, …") or a partial dict missing ``citations``.
        Reject-on-mismatch loses the entire section to the offline fallback —
        we'd rather salvage the model's work and let downstream re-anchor the
        citations from real retrieved chunks.
        """

        if not isinstance(value, list):
            return value
        normalized: list[object] = []
        for item in value:
            if isinstance(item, str):
                normalized.append({"text": item, "citations": [], "confidence": "medium"})
                continue
            if isinstance(item, dict):
                item.setdefault("citations", [])
                item.setdefault("confidence", "medium")
                if not isinstance(item["citations"], list):
                    item["citations"] = []
            normalized.append(item)
        return normalized


import re as _re_sections


class TitleReviewSummary(BaseModel):
    """The deliverable. ALTA-aligned review sections.

    Sections are stored in ``sections: dict[str, TitleReviewSection]`` keyed by
    each section's stable ``field_name`` (``s1_vesting_and_estate``, etc.).
    For backward-compat with gold JSONs we accept top-level ``s<N>_...`` keys
    on input AND emit them flat on output via a ``model_serializer``. Adding a
    new ALTA section is therefore a single edit in
    :mod:`titan.sections.SECTION_REGISTRY`; the schema picks it up
    automatically.

    Tests and callers can still reach a section with attribute syntax
    (``summary.s4_open_encumbrances_and_liens``) — the ``__getattr__`` shim
    forwards unknown attribute access into the sections dict.
    """

    matter_id: str
    property_address: Optional[str] = None
    parcel_id: Optional[str] = None
    effective_date: Optional[date] = None
    proposed_insured: Optional[str] = None
    policy_amount: Optional[Decimal] = None
    generated_at: date
    generator_version: str
    model: str
    rules_version: Optional[str] = None

    sections: dict[str, TitleReviewSection] = Field(default_factory=dict)

    overall_risk: Literal["clear_to_close", "curable_issues", "material_issues", "uninsurable"]
    overall_summary: list[CitedSentence]
    open_questions_for_client: list[str] = Field(default_factory=list)

    # ── backward-compat shims ───────────────────────────────────────────────

    @model_validator(mode="before")
    @classmethod
    def _absorb_flat_section_keys(cls, value: object) -> object:
        """Promote any top-level ``s<int>_*`` keys into the ``sections`` dict.

        Lets gold JSONs that store ``s1_vesting_and_estate`` etc. as
        top-level keys load straight into the new dict-backed model.
        """

        if not isinstance(value, dict):
            return value
        data: dict[str, object] = dict(value)
        sections = data.get("sections")
        if sections is None or not isinstance(sections, dict):
            sections = {}
        else:
            sections = dict(sections)
        for key in list(data.keys()):
            if key == "sections":
                continue
            if _re_sections.fullmatch(r"s\d+_[A-Za-z0-9_]+", key):
                sections.setdefault(key, data.pop(key))
        data["sections"] = sections
        return data

    @model_serializer(mode="wrap")
    def _flatten_sections_for_output(self, handler):  # type: ignore[no-untyped-def]
        """Flatten ``sections`` back to top-level ``s<N>_*`` keys on dump.

        Preserves the canonical on-disk JSON shape so previously-generated
        gold and produced JSONs remain comparable byte-for-byte.
        """

        raw = handler(self)
        sections = raw.pop("sections", {}) or {}
        # Maintain insertion order: dump sections first so they appear after
        # the matter-level fields and before the trailing summary fields, as
        # in the legacy layout.
        flat: dict[str, object] = {}
        for key, value in raw.items():
            if key == "overall_risk":
                # Insert section keys just before the trailing summary fields.
                for section_key, section_value in sections.items():
                    flat[section_key] = section_value
                sections = {}  # mark as flushed
            flat[key] = value
        # If overall_risk wasn't present (shouldn't happen) append remaining.
        for section_key, section_value in sections.items():
            flat[section_key] = section_value
        return flat

    def __getattr__(self, item: str):
        # __getattr__ only runs when normal lookup fails, so we never shadow
        # real fields. Forward s1_..._sN_ attribute access into the dict.
        if _re_sections.fullmatch(r"s\d+_[A-Za-z0-9_]+", item):
            sections = self.__dict__.get("sections") or {}
            if item in sections:
                return sections[item]
        raise AttributeError(item)
