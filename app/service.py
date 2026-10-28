import hashlib
import re
import uuid
from pathlib import Path

from sqlalchemy import select, func, text
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.config import MAX_CHUNKS, UPLOAD_DIR
from app.models import Document, DocumentChunk
from app.nvidia import embed
from app.pdf_processing import InvalidPDF, extract_pages, prepare_chunks

MAX_DOCUMENTS = 8


class DocumentRejected(ValueError):
    pass


def safe_filename(name: str | None) -> str:
    filename = (name or 'document.pdf').replace('\\', '/').split('/')[-1]
    filename = re.sub(r'[^\w.() -]', '_', filename)[:200]
    return filename or 'document.pdf'


def ingest_document(db: Session, filename: str, raw_pdf: bytes) -> tuple[Document, bool]:
    filename = safe_filename(filename)
    if not filename.lower().endswith('.pdf'):
        raise DocumentRejected('Only .pdf files are supported.')
    fingerprint = hashlib.sha256(raw_pdf).hexdigest()
    existing = db.scalar(select(Document).where(Document.sha256 == fingerprint))
    if existing:
        return existing, False

    # Serialize concurrent uploads so the document limit is respected.
    db.execute(text("SELECT pg_advisory_xact_lock(81008008)"))

    document_count = db.scalar(
        select(func.count(Document.id))
    )

    if document_count >= MAX_DOCUMENTS:
        raise HTTPException(
            status_code=409,
            detail="Document limit reached. Maximum 8 PDFs allowed."
        )

    try:
        page_count, pages = extract_pages(raw_pdf)
    except InvalidPDF as exc:
        raise DocumentRejected(str(exc)) from exc
    if not pages:
        raise DocumentRejected('No extractable text found. Scanned PDFs need OCR, which is not in this MVP.')
    chunks = prepare_chunks(pages)
    if len(chunks) > MAX_CHUNKS:
        raise DocumentRejected(f'Too many text chunks (max {MAX_CHUNKS}).')

    # Avoid committing a half-indexed document when the API fails.
    vectors = []
    for start in range(0, len(chunks), 8):
        batch = chunks[start:start + 8]
        vectors.extend(embed([text for _, _, text in batch], input_type='passage'))
    if len(vectors) != len(chunks):
        raise RuntimeError('Did not receive all requested embeddings.')

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = uuid.uuid4().hex + '.pdf'
    target = UPLOAD_DIR / stored_name
    target.write_bytes(raw_pdf)
    try:
        document = Document(
            filename=filename,
            storage_name=stored_name,
            sha256=fingerprint,
            page_count=page_count,
        )
        db.add(document)
        db.flush()
        for (page, index, content), vector in zip(chunks, vectors):
            db.add(DocumentChunk(
                document_id=document.id,
                page_number=page,
                chunk_index=index,
                content=content,
                embedding=vector,
            ))
        db.commit()
        db.refresh(document)
        return document, True
    except Exception:
        db.rollback()
        target.unlink(missing_ok=True)
        raise
