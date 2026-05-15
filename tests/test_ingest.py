"""Ingest module tests — extraction + OCR helpers without external calls."""

from __future__ import annotations

from datetime import date

import pytest

from titan.ingest.extract import _heuristic_extract, infer_doc_type
from titan.ingest.models import ParsedDoc, ParsedPage


SAMPLE_COMMITMENT_MD = """
SCHEDULE A
1. Effective Date: January 15, 2024
2. Policy Amount: $250,000.00
3. The estate or interest in the land described is FEE SIMPLE.
4. Title to the fee simple estate or interest in the land is at the Effective
   Date vested in: Sam Seller, a single man.
5. Proposed Insured: Patricia Purchaser.
"""


def test_infer_doc_type_classifies_commitment() -> None:
    assert infer_doc_type("wayne_county_commitment_0.pdf", SAMPLE_COMMITMENT_MD) == "title_commitment"


def test_infer_doc_type_classifies_deed_of_trust() -> None:
    assert infer_doc_type("osmre_deed_of_trust.pdf", "this is a Deed of Trust") == "deed_of_trust"


def test_infer_doc_type_classifies_judgment() -> None:
    assert infer_doc_type("ca_abstract_of_judgment.pdf", "Abstract of Judgment") == "judgment"


def test_infer_doc_type_falls_back_to_other() -> None:
    assert infer_doc_type("random.pdf", "totally unrelated content") == "other"


def test_heuristic_extract_pulls_vesting_and_amount() -> None:
    parsed = ParsedDoc(
        doc_id="sample",
        file_path="sample.pdf",
        markdown=SAMPLE_COMMITMENT_MD,
        pages=[
            ParsedPage(
                page_number=1,
                markdown=SAMPLE_COMMITMENT_MD,
                parser="pdfplumber",
                confidence=0.9,
                classification="clean_text",
            )
        ],
        parser_chain=["pdfplumber"],
    )
    extracted = _heuristic_extract(SAMPLE_COMMITMENT_MD, "title_commitment", parsed)
    assert extracted.doc_id == "sample"
    assert extracted.doc_type == "title_commitment"
    # Heuristic should at least know who the named parties are.
    names = {p.name.lower() for p in extracted.parties + extracted.vesting}
    assert any("sam seller" in n for n in names)


def test_parsed_page_validates_confidence_range() -> None:
    with pytest.raises(ValueError):
        ParsedPage(
            page_number=1,
            markdown="x",
            parser="pdfplumber",
            confidence=1.5,
            classification="clean_text",
        )


def test_parsed_doc_page_count_property() -> None:
    parsed = ParsedDoc(
        doc_id="d",
        file_path="d.pdf",
        markdown="",
        pages=[
            ParsedPage(
                page_number=i,
                markdown="",
                parser="pdfplumber",
                confidence=1.0,
                classification="clean_text",
            )
            for i in (1, 2, 3)
        ],
        parser_chain=["pdfplumber"],
    )
    assert parsed.page_count == 3
