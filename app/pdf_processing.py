import re

import pymupdf

from app.config import MAX_PAGES


class InvalidPDF(ValueError):
    pass


def extract_pages(raw_pdf: bytes) -> tuple[int, list[tuple[int, str]]]:
    if not raw_pdf.startswith(b'%PDF-'):
        raise InvalidPDF('File is not a valid PDF.')
    try:
        with pymupdf.open(stream=raw_pdf, filetype='pdf') as document:
            if document.needs_pass:
                raise InvalidPDF('Password-protected PDFs are not supported.')
            if document.page_count > MAX_PAGES:
                raise InvalidPDF(f'PDF has too many pages (maximum {MAX_PAGES}).')
            pages = []
            for index, page in enumerate(document):
                text = page.get_text('text', sort=False)
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    pages.append((index + 1, text))
            return document.page_count, pages
    except InvalidPDF:
        raise
    except Exception as exc:
        raise InvalidPDF('Could not read this PDF.') from exc


def split_text(text: str, chunk_size: int = 1100, overlap: int = 180) -> list[str]:
    """Character chunks with overlap. Keeps each chunk on a single PDF page."""
    if chunk_size <= overlap or overlap < 0:
        raise ValueError('chunk_size must be greater than overlap')
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    result = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer a word boundary rather than splitting in the middle of a word.
            boundary = text.rfind(' ', start + chunk_size // 2, end)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            result.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return result


def prepare_chunks(pages: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    prepared = []
    for page_number, text in pages:
        for index, chunk in enumerate(split_text(text)):
            prepared.append((page_number, index, chunk))
    return prepared
