"""
app/services/document_processor.py
====================================
Phase 2: Document Upload, Parsing & Chunking

CONCEPT: Text extraction pipeline
    Raw file → extract text per page → clean text → split into chunks → List[Chunk]

WHY PAGE-AWARE EXTRACTION?
We track which page each chunk came from so citations can say
"Answer found on page 4 of report.pdf" — this is what impresses recruiters.

WHY RECURSIVE CHARACTER SPLITTER?
LangChain's RecursiveCharacterTextSplitter tries to split on:
  "\n\n" (paragraphs) → "\n" (lines) → " " (words) → "" (chars)
It always tries the highest-level boundary first, so chunks stay
semantically coherent — a paragraph isn't split mid-sentence if avoidable.
"""

import io
import time
import uuid
from pathlib import Path
from typing import Optional

try:
    import pymupdf as fitz              # PyMuPDF (new name)
except ImportError:
    import fitz                         # PyMuPDF (legacy name)
import docx                             # python-docx
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # older langchain
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.models import Document, Chunk, DocumentStatus
from app.core.config import get_config
from app.core.exceptions import (
    UnsupportedFileTypeError, FileTooLargeError,
    DocumentProcessingError, EmptyDocumentError
)
from app.utils.logger import get_logger
from app.utils.helpers import (
    generate_doc_id, hash_file_content,
    sanitize_filename, get_file_extension, timer
)

logger = get_logger(__name__)
cfg    = get_config()


class DocumentProcessor:
    """
    Handles the full pipeline from raw uploaded bytes → List[Chunk].

    Single Responsibility: This class does ONE thing — turn files into chunks.
    It doesn't embed, doesn't store, doesn't call the LLM. Clean separation.
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size      = cfg.CHUNK_SIZE,
            chunk_overlap   = cfg.CHUNK_OVERLAP,
            length_function = len,
            separators      = ["\n\n", "\n", ". ", " ", ""],
            # Try paragraph breaks first, then lines, then sentences,
            # then words, then characters — preserves semantic meaning
        )
        logger.info(
            f"DocumentProcessor ready | chunk_size={cfg.CHUNK_SIZE} "
            f"overlap={cfg.CHUNK_OVERLAP}"
        )

    @timer
    def process_uploaded_file(
        self,
        file_bytes : bytes,
        filename   : str
    ) -> tuple[Document, list[Chunk]]:
        """
        Main entry point. Takes raw bytes from Streamlit file uploader.

        Args:
            file_bytes: raw bytes from st.file_uploader.read()
            filename:   original filename as string

        Returns:
            (Document, List[Chunk]) — the metadata object + all text chunks

        Raises:
            UnsupportedFileTypeError, FileTooLargeError, DocumentProcessingError
        """
        logger.info(f"Processing file: {filename} ({len(file_bytes):,} bytes)")

        # ── Validation ──────────────────────────────────────────────────────
        self._validate_file(file_bytes, filename)

        # ── Create Document metadata object ─────────────────────────────────
        file_ext     = get_file_extension(filename)
        safe_name    = sanitize_filename(filename)
        content_hash = hash_file_content(file_bytes)
        doc_id       = generate_doc_id(safe_name, content_hash)

        save_path = cfg.UPLOADS_DIR / f"{doc_id}_{safe_name}"

        document = Document(
            doc_id    = doc_id,
            filename  = filename,           # show original name in UI
            file_path = str(save_path),
            file_type = file_ext,
            file_size = len(file_bytes),
            status    = DocumentStatus.PROCESSING,
        )

        # ── Extract text per page ───────────────────────────────────────────
        try:
            pages: list[tuple[int, str]] = self._extract_pages(file_bytes, file_ext, filename)
        except (UnsupportedFileTypeError, EmptyDocumentError):
            document.status = DocumentStatus.FAILED
            raise
        except Exception as e:
            document.status = DocumentStatus.FAILED
            raise DocumentProcessingError(filename, str(e)) from e

        # ── Chunk all pages ─────────────────────────────────────────────────
        chunks = self._chunk_pages(pages, doc_id, filename)

        if not chunks:
            document.status = DocumentStatus.FAILED
            raise EmptyDocumentError(filename)

        # ── Save original to disk (only after parsing succeeded) ─────────────
        cfg.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(file_bytes)

        # ── Update document metadata ─────────────────────────────────────────
        document.page_count  = len(pages)
        document.chunk_count = len(chunks)
        document.status      = DocumentStatus.INDEXED  # will be set properly after embedding

        logger.info(
            f"✅ Processed '{filename}': "
            f"{len(pages)} pages → {len(chunks)} chunks"
        )
        return document, chunks

    # ── Private: Validation ──────────────────────────────────────────────────

    def _validate_file(self, file_bytes: bytes, filename: str) -> None:
        """Raise descriptive errors before doing any real work."""
        ext = get_file_extension(filename)

        if ext not in cfg.ALLOWED_EXTENSIONS:
            raise UnsupportedFileTypeError(filename, ext, cfg.ALLOWED_EXTENSIONS)

        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > cfg.MAX_FILE_SIZE_MB:
            raise FileTooLargeError(filename, size_mb, cfg.MAX_FILE_SIZE_MB)

    # ── Private: Extraction ──────────────────────────────────────────────────

    def _extract_pages(
        self, file_bytes: bytes, file_ext: str, filename: str
    ) -> list[tuple[int, str]]:
        """
        Dispatch to the correct parser based on file type.
        Returns a list of (page_number, text) tuples (1-indexed pages).
        """
        extractors = {
            "pdf"  : self._extract_pdf,
            "docx" : self._extract_docx,
            "txt"  : self._extract_txt,
        }
        return extractors[file_ext](file_bytes, filename)

    def _extract_pdf(
        self, file_bytes: bytes, filename: str
    ) -> list[tuple[int, str]]:
        """
        Extract text from PDF using PyMuPDF (fitz).

        PyMuPDF is faster and more accurate than pdfplumber for text-heavy PDFs.
        It handles multi-column layouts and extracts text in reading order.
        """
        pages = []
        try:
            pdf = fitz.open(stream=file_bytes, filetype="pdf")

            if pdf.is_encrypted:
                raise DocumentProcessingError(filename, "PDF is password-protected.")

            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text("text")   # "text" = plain text mode
                text = self._clean_text(text)
                if text:
                    pages.append((page_num + 1, text))   # 1-indexed

            pdf.close()

        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(filename, f"PDF parsing error: {e}") from e

        if not pages:
            raise EmptyDocumentError(filename)

        return pages

    def _extract_docx(
        self, file_bytes: bytes, filename: str
    ) -> list[tuple[int, str]]:
        """
        Extract text from DOCX using python-docx.

        DOCX has no native "pages" concept — pages depend on rendering.
        We simulate pages by grouping ~500 words per virtual page.
        This gives meaningful page citations even for Word documents.
        """
        try:
            doc      = docx.Document(io.BytesIO(file_bytes))
            all_text = []

            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    all_text.append(text)

            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(
                        cell.text.strip() for cell in row.cells if cell.text.strip()
                    )
                    if row_text:
                        all_text.append(row_text)

            full_text = "\n".join(all_text)
            full_text = self._clean_text(full_text)

            if not full_text:
                raise EmptyDocumentError(filename)

            # Simulate pages: ~500 words per page
            words       = full_text.split()
            words_per_page = 500
            pages = []
            for i in range(0, len(words), words_per_page):
                page_words = words[i:i + words_per_page]
                page_text  = " ".join(page_words)
                page_num   = (i // words_per_page) + 1
                pages.append((page_num, page_text))

            return pages

        except EmptyDocumentError:
            raise
        except Exception as e:
            raise DocumentProcessingError(filename, f"DOCX parsing error: {e}") from e

    def _extract_txt(
        self, file_bytes: bytes, filename: str
    ) -> list[tuple[int, str]]:
        """
        Extract text from plain text files.
        Tries UTF-8 first, falls back to latin-1 for legacy files.
        """
        try:
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1")

            text = self._clean_text(text)
            if not text:
                raise EmptyDocumentError(filename)

            # Split into virtual pages by newlines (~50 lines per page)
            lines          = text.split("\n")
            lines_per_page = 50
            pages          = []
            for i in range(0, len(lines), lines_per_page):
                page_text = "\n".join(lines[i:i + lines_per_page]).strip()
                if page_text:
                    page_num = (i // lines_per_page) + 1
                    pages.append((page_num, page_text))

            return pages

        except EmptyDocumentError:
            raise
        except Exception as e:
            raise DocumentProcessingError(filename, f"TXT parsing error: {e}") from e

    # ── Private: Chunking ─────────────────────────────────────────────────────

    def _chunk_pages(
        self,
        pages    : list[tuple[int, str]],
        doc_id   : str,
        filename : str
    ) -> list[Chunk]:
        """
        Convert extracted pages into overlapping Chunk objects.

        IMPORTANT: We chunk per-page, not the whole document at once.
        This ensures each chunk KNOWS its page number for accurate citations.
        If we chunked the whole doc, we'd lose page information at boundaries.
        """
        chunks      = []
        chunk_index = 0

        for page_num, page_text in pages:
            # Split this page's text into sub-chunks
            page_chunks = self.splitter.split_text(page_text)

            for chunk_text in page_chunks:
                chunk_text = chunk_text.strip()
                if len(chunk_text) < 50:   # skip micro-chunks (headers, footers)
                    continue

                chunk = Chunk(
                    chunk_id    = f"{doc_id}_p{page_num}_c{chunk_index}",
                    doc_id      = doc_id,
                    content     = chunk_text,
                    page_number = page_num,
                    chunk_index = chunk_index,
                    filename    = filename,
                )
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    # ── Private: Text Cleaning ────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """
        Normalise extracted text.
        - Collapse 3+ newlines into 2 (preserve paragraph breaks)
        - Strip leading/trailing whitespace
        - Remove null bytes that sometimes appear in PDFs
        """
        import re
        text = text.replace("\x00", "")          # remove null bytes
        text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excess newlines
        text = re.sub(r" {3,}", " ", text)        # collapse excess spaces
        return text.strip()
