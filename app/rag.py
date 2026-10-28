from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.nvidia import embed, generate_answer


def ask(db: Session, question: str) -> dict:
    if not db.scalar(select(Document.id).limit(1)):
        return {'answer': 'Upload a PDF first; the knowledge base is empty.', 'sources': []}

    #embed user's question
    query_vector = embed([question], input_type='query')[0]

    #measures distances to stored embeddings
    distance = DocumentChunk.embedding.cosine_distance(query_vector).label('distance')
    stmt = (
        select(DocumentChunk, Document.filename, distance)
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(distance)
        .limit(8)
    )

    #get closest chunks
    rows = db.execute(stmt).all()
    context = []
    sources = []
    for i, (chunk, filename, similarity_distance) in enumerate(rows, start=1):
        ref = f'S{i}'
        context.append(f'[{ref}] {filename}, page {chunk.page_number}:\n{chunk.content}')
        sources.append({
            'reference': ref,
            'document_id': chunk.document_id,
            'filename': filename,
            'page': chunk.page_number,
            'chunk_id': chunk.id,
            'distance': round(float(similarity_distance), 4),
        })
    answer = generate_answer(question, '\n\n'.join(context))
    return {'answer': answer, 'sources': sources}
