"""SQLite persistence for ingest artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Column, JSON
from sqlmodel import Field, Session, SQLModel, create_engine, select

from titan.ingest.models import ParsedDoc
from titan.schemas import TitleDocument


class ParsedDocumentRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    doc_id: str = Field(index=True, unique=True)
    file_path: str
    parser_chain: list[str] = Field(sa_column=Column(JSON))
    markdown: str
    payload: dict = Field(sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TitleDocumentRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    doc_id: str = Field(index=True, unique=True)
    doc_type: str = Field(index=True)
    file_path: str
    payload: dict = Field(sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def open_engine(sqlite_path: str | Path = "data/titan.db"):
    db_path = Path(sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    SQLModel.metadata.create_all(engine)
    return engine


def persist_parsed_doc(parsed_doc: ParsedDoc, sqlite_path: str | Path = "data/titan.db") -> ParsedDocumentRecord:
    engine = open_engine(sqlite_path)
    record = ParsedDocumentRecord(
        doc_id=parsed_doc.doc_id,
        file_path=parsed_doc.file_path,
        parser_chain=list(parsed_doc.parser_chain),
        markdown=parsed_doc.markdown,
        payload=parsed_doc.model_dump(mode="json"),
    )
    with Session(engine) as session:
        existing = session.exec(select(ParsedDocumentRecord).where(ParsedDocumentRecord.doc_id == parsed_doc.doc_id)).first()
        if existing:
            record.id = existing.id
            record = session.merge(record)
        else:
            session.add(record)
        session.commit()
        session.refresh(record)
    return record


def persist_title_document(title_document: TitleDocument, sqlite_path: str | Path = "data/titan.db") -> TitleDocumentRecord:
    engine = open_engine(sqlite_path)
    record = TitleDocumentRecord(
        doc_id=title_document.doc_id,
        doc_type=title_document.doc_type,
        file_path=title_document.file_path,
        payload=title_document.model_dump(mode="json"),
    )
    with Session(engine) as session:
        existing = session.exec(select(TitleDocumentRecord).where(TitleDocumentRecord.doc_id == title_document.doc_id)).first()
        if existing:
            record.id = existing.id
            record = session.merge(record)
        else:
            session.add(record)
        session.commit()
        session.refresh(record)
    return record
