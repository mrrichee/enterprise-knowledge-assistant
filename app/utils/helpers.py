"""
app/utils/helpers.py
=====================
Shared utility functions — small, pure, reusable helpers.

CONCEPT: Pure functions
A pure function has no side effects and always returns the same
output for the same input. They're the easiest code to test and reuse.
"""

import uuid
import time
import hashlib
from pathlib import Path
from functools import wraps
from typing import Callable, Any

from app.utils.logger import get_logger

logger = get_logger(__name__)


def generate_id() -> str:
    """Generate a short unique ID using UUID4 (random)."""
    return str(uuid.uuid4())


def generate_doc_id(filename: str, content_hash: str) -> str:
    """
    Deterministic ID based on filename + content.
    Same file uploaded twice → same doc_id → we can detect duplicates.
    """
    raw = f"{filename}:{content_hash}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def hash_file_content(file_bytes: bytes) -> str:
    """SHA-256 hash of file bytes — for duplicate detection."""
    return hashlib.sha256(file_bytes).hexdigest()


def sanitize_filename(filename: str) -> str:
    """
    Make a filename safe for disk storage.
    Removes path traversal characters, spaces → underscores.
    """
    name = Path(filename).name                     # strip any path prefix
    name = "".join(c for c in name if c.isalnum() or c in "._- ")
    name = name.replace(" ", "_")
    return name[:200]                              # max 200 chars


def get_file_extension(filename: str) -> str:
    """Return lowercase extension without the dot: 'pdf', 'docx', 'txt'."""
    return Path(filename).suffix.lstrip(".").lower()


def format_duration(ms: float) -> str:
    """Convert milliseconds to a human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def timer(func: Callable) -> Callable:
    """
    Decorator that logs function execution time.

    CONCEPT: Decorators
    A decorator wraps a function to add behaviour before/after it runs.
    @timer logs how long any function takes without touching its logic.

    Usage:
        @timer
        def process_document(path): ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"{func.__qualname__} completed in {format_duration(elapsed_ms)}")
            return result
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(f"{func.__qualname__} failed after {format_duration(elapsed_ms)}: {e}")
            raise
    return wrapper


def truncate_text(text: str, max_chars: int = 300, suffix: str = "...") -> str:
    """Truncate text to max_chars, appending suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def chunk_list(lst: list, size: int) -> list[list]:
    """
    Split a list into batches of given size.
    Useful for batched embedding generation.
    Example: chunk_list([1,2,3,4,5], 2) → [[1,2], [3,4], [5]]
    """
    return [lst[i:i + size] for i in range(0, len(lst), size)]
