"""Schemas for the operator-edit learning loop."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

EditType = Literal[
    "wording",
    "fact_correction",
    "addition",
    "deletion",
    "citation_fix",
    "style",
    "house_format",
    "risk_rating",
]


class EditEvent(BaseModel):
    """A structured before/after diff of an operator change to a draft."""

    edit_id: str = Field(default_factory=lambda: f"e_{uuid4().hex[:12]}")
    matter_id: str
    section_name: str
    field_path: str
    before: str
    after: str
    edit_type: EditType = "wording"
    operator_id: Optional[str] = None
    operator_note: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_draft_version: str = "draft-v1"


class Rule(BaseModel):
    id: str
    text: str = Field(..., max_length=240)
    trigger_edit_types: list[EditType] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class SupersededRule(BaseModel):
    id: str
    text: str
    superseded_by: str
    reason: str


class RuleSet(BaseModel):
    """Versioned, distilled-from-edits rule pack for one section."""

    section: str
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    generated_from_edit_ids: list[str] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    superseded: list[SupersededRule] = Field(default_factory=list)
