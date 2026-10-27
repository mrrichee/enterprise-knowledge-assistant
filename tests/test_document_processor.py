"""
tests/test_document_processor.py
===================================
Unit tests for the DocumentProcessor.

CONCEPT: Unit Testing
Unit tests verify one piece of logic in isolation.
We test DocumentProcessor without FAISS, without Gemini, without Streamlit.
Each test is fast (< 1 second) and tells you exactly what broke.

RUN TESTS:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov-report=html   # with coverage
"""

import pytest
from unittest.mock import patch, MagicMock

from app.core.exceptions import (
    UnsupportedFileTypeError, FileTooLargeError, EmptyDocumentError
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def processor():
    """Create a DocumentProcessor with test config."""
    with patch("app.services.document_processor.get_config") as mock_cfg:
        cfg              = MagicMock()
        cfg.CHUNK_SIZE   = 500
        cfg.CHUNK_OVERLAP= 100
        cfg.ALLOWED_EXTENSIONS = ["pdf", "docx", "txt"]
        cfg.MAX_FILE_SIZE_MB   = 10
        cfg.UPLOADS_DIR        = MagicMock()
        cfg.UPLOADS_DIR.__truediv__ = lambda self, x: MagicMock(write_bytes=MagicMock())
        mock_cfg.return_value  = cfg

        from app.services.document_processor import DocumentProcessor
        return DocumentProcessor()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestValidation:
    def test_rejects_unsupported_extension(self, processor):
        """Should raise UnsupportedFileTypeError for .xlsx files."""
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            processor._validate_file(b"data", "report.xlsx")
        assert "xlsx" in exc_info.value.message

    def test_rejects_oversized_file(self, processor):
        """Should raise FileTooLargeError when file exceeds limit."""
        from app.core.config import get_config
        big_bytes = b"x" * ((get_config().MAX_FILE_SIZE_MB + 1) * 1024 * 1024)  # just over the limit
        with pytest.raises(FileTooLargeError):
            processor._validate_file(big_bytes, "big.pdf")

    def test_accepts_valid_pdf(self, processor):
        """Should not raise for a small .pdf file."""
        processor._validate_file(b"small content", "valid.pdf")  # no exception


class TestTextCleaning:
    def test_removes_null_bytes(self, processor):
        dirty = "Hello\x00World\x00!"
        clean = processor._clean_text(dirty)
        assert "\x00" not in clean
        assert "HelloWorld!" in clean

    def test_collapses_excess_newlines(self, processor):
        text  = "Para 1\n\n\n\n\nPara 2"
        clean = processor._clean_text(text)
        assert "\n\n\n" not in clean

    def test_strips_whitespace(self, processor):
        text  = "  Hello World  "
        clean = processor._clean_text(text)
        assert clean == "Hello World"


class TestTxtExtraction:
    def test_extracts_plain_text(self, processor):
        content  = b"This is a test document.\nIt has multiple lines."
        pages    = processor._extract_txt(content, "test.txt")
        assert len(pages) > 0
        assert all(isinstance(p[0], int) for p in pages)   # page numbers are ints
        assert all(isinstance(p[1], str) for p in pages)   # content is str

    def test_handles_latin1_encoding(self, processor):
        content = "Caf\xe9 au lait".encode("latin-1")
        pages   = processor._extract_txt(content, "latin.txt")
        assert len(pages) > 0

    def test_raises_on_empty_content(self, processor):
        with pytest.raises(EmptyDocumentError):
            processor._extract_txt(b"   \n\n   ", "empty.txt")


class TestChunking:
    def test_produces_chunks_from_pages(self, processor):
        pages  = [(1, "A" * 600), (2, "B" * 600)]   # two pages
        chunks = processor._chunk_pages(pages, "doc1", "test.txt")
        assert len(chunks) > 0
        assert all(c.doc_id == "doc1" for c in chunks)

    def test_chunks_preserve_page_numbers(self, processor):
        pages  = [(3, "C" * 600)]
        chunks = processor._chunk_pages(pages, "doc2", "test.txt")
        assert all(c.page_number == 3 for c in chunks)

    def test_skips_micro_chunks(self, processor):
        """Chunks with < 50 characters should be discarded."""
        pages  = [(1, "Short")]   # way below 50 char minimum
        chunks = processor._chunk_pages(pages, "doc3", "test.txt")
        assert len(chunks) == 0


class TestHelpers:
    def test_generate_id_is_unique(self):
        from app.utils.helpers import generate_id
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100   # all 100 should be unique

    def test_sanitize_filename_removes_special_chars(self):
        from app.utils.helpers import sanitize_filename
        result = sanitize_filename("my file (1).pdf")
        assert "/" not in result
        assert result.endswith(".pdf")

    def test_chunk_list_splits_correctly(self):
        from app.utils.helpers import chunk_list
        result = chunk_list([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_truncate_text(self):
        from app.utils.helpers import truncate_text
        long_text = "A" * 500
        result    = truncate_text(long_text, 100)
        assert len(result) <= 103   # 100 + "..." (3)
        assert result.endswith("...")
