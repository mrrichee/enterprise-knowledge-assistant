"""
app/core/exceptions.py
=======================
Custom Exception Hierarchy

CONCEPT: Why custom exceptions instead of generic ones?
- raise Exception("file too large") gives no structure — hard to catch selectively
- raise FileTooLargeError(filename, size) is self-documenting and catchable precisely
- Custom exceptions let the UI layer show friendly messages while the
  service layer raises meaningful errors

HIERARCHY:
    AppBaseError
    ├── DocumentError
    │   ├── UnsupportedFileTypeError
    │   ├── FileTooLargeError
    │   └── DocumentProcessingError
    ├── EmbeddingError
    ├── VectorStoreError
    ├── LLMError
    │   ├── APIKeyError
    │   └── RateLimitError
    └── DatabaseError
"""


class AppBaseError(Exception):
    """Base class for all application errors. Never raise this directly."""
    def __init__(self, message: str, details: str = ""):
        self.message = message
        self.details = details
        super().__init__(message)

    def user_message(self) -> str:
        """Human-friendly message safe to show in the UI."""
        return self.message


# ── Document Errors ──────────────────────────────────────────────────────────

class DocumentError(AppBaseError):
    """Base for all document-related errors."""
    pass


class UnsupportedFileTypeError(DocumentError):
    def __init__(self, filename: str, file_type: str, allowed: list[str]):
        self.filename  = filename
        self.file_type = file_type
        self.allowed   = allowed
        super().__init__(
            message=f"Unsupported file type '.{file_type}' for '{filename}'.",
            details=f"Allowed types: {', '.join(allowed)}"
        )

    def user_message(self) -> str:
        return f"❌ '{self.filename}' is a .{self.file_type} file. Please upload PDF, DOCX, or TXT."


class FileTooLargeError(DocumentError):
    def __init__(self, filename: str, size_mb: float, max_mb: int):
        self.filename = filename
        self.size_mb  = size_mb
        self.max_mb   = max_mb
        super().__init__(
            message=f"'{filename}' is {size_mb:.1f} MB, exceeds limit of {max_mb} MB.",
        )

    def user_message(self) -> str:
        return f"❌ '{self.filename}' is too large ({self.size_mb:.1f} MB). Maximum allowed: {self.max_mb} MB."


class DocumentProcessingError(DocumentError):
    def __init__(self, filename: str, reason: str):
        self.filename = filename
        super().__init__(
            message=f"Failed to process '{filename}': {reason}",
            details=reason
        )

    def user_message(self) -> str:
        return f"❌ Could not process '{self.filename}'. The file may be corrupted or password-protected."


class EmptyDocumentError(DocumentError):
    def __init__(self, filename: str):
        self.filename = filename
        super().__init__(message=f"'{filename}' contains no extractable text.")

    def user_message(self) -> str:
        return f"❌ '{self.filename}' appears to be empty or contains only images. No text could be extracted."


# ── Embedding & Vector Store Errors ─────────────────────────────────────────

class EmbeddingError(AppBaseError):
    def __init__(self, reason: str):
        super().__init__(message=f"Embedding generation failed: {reason}", details=reason)

    def user_message(self) -> str:
        return "❌ Failed to generate embeddings. Please try again."


class VectorStoreError(AppBaseError):
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        super().__init__(message=f"Vector store '{operation}' failed: {reason}", details=reason)

    def user_message(self) -> str:
        return f"❌ Index operation failed during '{self.operation}'. Please re-upload your documents."


# ── LLM Errors ───────────────────────────────────────────────────────────────

class LLMError(AppBaseError):
    """Base for LLM/API errors."""
    pass


class APIKeyError(LLMError):
    def __init__(self):
        super().__init__(message="No valid LLM API key (Groq or Gemini).")

    def user_message(self) -> str:
        return "❌ API key error. Please check GROQ_API_KEY / GEMINI_API_KEY in your .env file (run check_models.py)."


class RateLimitError(LLMError):
    def __init__(self):
        super().__init__(message="LLM API rate limit exceeded.")

    def user_message(self) -> str:
        return "⏳ Too many requests. Please wait a moment and try again."


class LLMResponseError(LLMError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(message=f"LLM returned an invalid response: {reason}", details=reason)

    def user_message(self) -> str:
        # Show the real cause (e.g. Groq's error text) so it can be fixed
        return f"❌ The AI request failed: {self.reason[:300]}"


# ── Database Errors ───────────────────────────────────────────────────────────

class DatabaseError(AppBaseError):
    def __init__(self, operation: str, reason: str):
        super().__init__(message=f"Database '{operation}' failed: {reason}", details=reason)

    def user_message(self) -> str:
        return "❌ A database error occurred. Your session data may not be saved."


# ── Query Errors ──────────────────────────────────────────────────────────────

class NoDocumentsIndexedError(AppBaseError):
    def __init__(self):
        super().__init__(message="No documents are indexed. Upload and process documents first.")

    def user_message(self) -> str:
        return "📂 Please upload at least one document before asking questions."


class NoRelevantChunksError(AppBaseError):
    def __init__(self, query: str):
        self.query = query
        super().__init__(message=f"No relevant chunks found for query: '{query}'")

    def user_message(self) -> str:
        return "🔍 No relevant information found in your documents for that question. Try rephrasing."
