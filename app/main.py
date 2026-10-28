from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import MAX_UPLOAD_BYTES, UPLOAD_DIR
from app.db import get_db
from app.models import Document, DocumentChunk
from app.rag import ask
from app.service import DocumentRejected, ingest_document

app = FastAPI(title='earagmvp', version='0.1.0')
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / 'templates')


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=1500)


@app.get('/')
def home():
    return RedirectResponse('/chat', status_code=302)


@app.get('/chat', response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(request=request, name='chat.html')


@app.get('/documents', response_class=HTMLResponse)
def documents_page(request: Request):
    return templates.TemplateResponse(request=request, name='documents.html')


@app.get('/api/health')
def health(db: Session = Depends(get_db)):
    db.execute(text('SELECT 1'))
    return {'status': 'ok'}


@app.get('/api/documents')
def list_documents(db: Session = Depends(get_db)):
    docs = db.execute(
        select(Document.id, Document.filename, Document.page_count, func.count(DocumentChunk.id))
        .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
        .group_by(Document.id)
        .order_by(Document.id.desc())
    ).all()
    return [
        {'id': doc_id, 'filename': filename, 'pages': pages, 'chunks': count}
        for doc_id, filename, pages, count in docs
    ]


@app.post('/api/documents')
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    raw_pdf = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw_pdf) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, 'PDF exceeds the 10 MB upload limit.')
    try:
        document, created = ingest_document(db, file.filename or 'document.pdf', raw_pdf)
    except DocumentRejected as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        'id': document.id,
        'filename': document.filename,
        'page_count': document.page_count,
        'created': created,
    }


@app.get('/api/documents/{document_id}/file')
def view_pdf(document_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, 'Document not found.')
    path = UPLOAD_DIR / doc.storage_name
    if not path.is_file():
        raise HTTPException(404, 'Stored PDF not found.')
    return FileResponse(path, media_type='application/pdf', filename=doc.filename, content_disposition_type='inline')


@app.post('/api/chat')
def chat(payload: Question, db: Session = Depends(get_db)):
    return ask(db, payload.question)
