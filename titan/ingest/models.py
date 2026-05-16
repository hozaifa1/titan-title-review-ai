"""OCR-layer data contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


ParserName = Literal["docling", "pdfplumber", "qwen2_5_vl", "fixture_transcript", "vlm"]
PageClass = Literal["clean_text", "scanned_typed", "handwritten", "mixed_low_quality"]


class ParsedPage(BaseModel):
    page_number: int = Field(ge=1)
    markdown: str
    parser: ParserName
    confidence: float = Field(ge=0.0, le=1.0)
    classification: PageClass
    warnings: list[str] = Field(default_factory=list)


class ParsedDoc(BaseModel):
    doc_id: str
    file_path: str
    markdown: str
    pages: list[ParsedPage]
    parser_chain: list[ParserName] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @classmethod
    def from_pages(
        cls,
        path: str | Path,
        pages: list[ParsedPage],
        parser_chain: list[ParserName],
        warnings: list[str] | None = None,
    ) -> "ParsedDoc":
        source_path = Path(path)
        markdown = "\n\n".join(f"## Page {page.page_number}\n\n{page.markdown}" for page in pages)
        return cls(
            doc_id=source_path.stem,
            file_path=str(source_path),
            markdown=markdown,
            pages=pages,
            parser_chain=parser_chain,
            warnings=warnings or [],
        )
