"""Pydantic contracts for extracted title documents and review summaries."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

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
        description="Information missing from source docs that a reviewer should obtain",
    )
    flags: list[Literal["red", "yellow", "green"]] = Field(default_factory=list)


class TitleReviewSummary(BaseModel):
    """The deliverable. 8 sections, ALTA-aligned."""

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

    s1_vesting_and_estate: TitleReviewSection
    s2_legal_description: TitleReviewSection
    s3_chain_of_title: TitleReviewSection
    s4_open_encumbrances_and_liens: TitleReviewSection
    s5_easements_and_restrictions: TitleReviewSection
    s6_requirements_schedule_b_i: TitleReviewSection
    s7_exceptions_schedule_b_ii: TitleReviewSection
    s8_taxes_and_survey_matters: TitleReviewSection

    overall_risk: Literal["clear_to_close", "curable_issues", "material_issues", "uninsurable"]
    overall_summary: list[CitedSentence]
    open_questions_for_client: list[str] = Field(default_factory=list)
