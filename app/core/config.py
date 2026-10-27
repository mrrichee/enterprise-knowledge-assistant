"""
app/core/config.py
==================
Centralised Configuration

CONCEPT: Why not just use os.getenv() everywhere?
- Scattered os.getenv() calls mean you don't know all your config
  requirements without reading every file
- No validation — a missing API key fails at runtime deep in your code,
  not at startup where it's obvious
- pydantic-settings reads .env automatically, validates types,
  and raises clear errors at startup if something is missing

CONCEPT: Singleton pattern via @lru_cache
- We only want ONE config object for the entire app lifetime
- @lru_cache(maxsize=1) on get_config() ensures it's instantiated once
  and reused everywhere — like a global but testable

USAGE:
    from app.core.config import get_config
    cfg = get_config()
    print(cfg.GEMINI_API_KEY)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


# ── Path Constants ───────────────────────────────────────────────────────────
# Using pathlib.Path is better than string concatenation:
#   Path("data") / "uploads"  vs  "data" + os.sep + "uploads"
# Path objects are cross-platform and have useful methods like .exists()

ROOT_DIR        = Path(__file__).parent.parent.parent   # project root
DATA_DIR        = ROOT_DIR / "data"
UPLOADS_DIR     = DATA_DIR / "uploads"
FAISS_DIR       = DATA_DIR / "faiss_index"
DB_DIR          = DATA_DIR / "db"
LOGS_DIR        = ROOT_DIR / "logs"


class Settings(BaseSettings):
    """
    All application settings in one place.

    pydantic-settings reads values in this priority order:
        1. Environment variables (highest priority)
        2. .env file
        3. Default values defined here (lowest priority)

    The type annotations (str, int, float) are enforced — if CHUNK_SIZE
    is set to "abc" in .env, pydantic raises a clear validation error.
    """

    # ── LLM / API ──────────────────────────────────────────────────────────
    GEMINI_API_KEY      : str   = Field("", description="Google Gemini API key (optional)")
    LLM_PROVIDER        : str   = "auto"
    # "auto" = use whichever key works (Groq first), or force "groq" / "gemini"
    GROQ_API_KEY        : str   = Field("", description="Groq API key (required for answers)")
    GROQ_MODEL          : str   = ""      # empty = auto-detect a working model
    GEMINI_MODEL        : str   = ""      # empty = auto-detect a working model
    # Using flash for speed + cost. Swap to "gemini-1.5-pro" for quality.
    GEMINI_TEMPERATURE  : float = 0.1
    # Low temperature (0.0–0.3) = focused, factual answers
    # High temperature (0.7–1.0) = creative, varied answers
    # For a knowledge assistant we want factual — keep this low.
    GEMINI_MAX_TOKENS   : int   = 2048

    # ── Embedding Model ────────────────────────────────────────────────────
    EMBEDDING_MODEL     : str   = "all-MiniLM-L6-v2"
    # This model creates 384-dimensional vectors
    # Good balance of speed, memory, and quality for a student project
    EMBEDDING_DIMENSION : int   = 384
    # Must match the model above. If you switch models, update this.

    # ── Chunking Strategy ─────────────────────────────────────────────────
    CHUNK_SIZE          : int   = 1000
    # How many characters per chunk. This is a critical hyperparameter:
    # - Too small (< 200): each chunk lacks context, answers are shallow
    # - Too large (> 2000): too much noise, retrieval is less precise
    # 800-1200 is the sweet spot for most documents.
    CHUNK_OVERLAP       : int   = 200
    # How many characters overlap between consecutive chunks.
    # Overlap prevents losing context at chunk boundaries.
    # Rule of thumb: ~20% of CHUNK_SIZE.

    # ── Retrieval Settings ─────────────────────────────────────────────────
    RETRIEVAL_TOP_K     : int   = 5
    # How many chunks to retrieve for each query.
    # More chunks = more context for LLM but more tokens = higher cost.
    # 3-6 is the typical production range.
    SIMILARITY_THRESHOLD: float = 0.15
    # Minimum cosine similarity to include a chunk.
    # Below this, the chunk is probably irrelevant to the query.
    # 0.0 = include everything, 1.0 = exact match only

    # ── Database ──────────────────────────────────────────────────────────
    DB_PATH             : str   = str(DB_DIR / "knowledge.db")

    # ── File Upload Limits ─────────────────────────────────────────────────
    MAX_FILE_SIZE_MB    : int   = 50
    ALLOWED_EXTENSIONS  : list  = ["pdf", "docx", "txt"]

    # ── App Metadata ──────────────────────────────────────────────────────
    APP_NAME            : str   = "Enterprise Knowledge Assistant"
    APP_VERSION         : str   = "1.0.0"
    DEBUG               : bool  = False
    LOG_LEVEL           : str   = "INFO"

    # ── Paths (computed, not from env) ─────────────────────────────────────
    UPLOADS_DIR         : Path  = UPLOADS_DIR
    FAISS_DIR           : Path  = FAISS_DIR
    DB_DIR              : Path  = DB_DIR

    @field_validator("GEMINI_TEMPERATURE")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("GEMINI_TEMPERATURE must be between 0.0 and 1.0")
        return v

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        # Overlap must be smaller than chunk size to make sense
        # We access chunk_size from the partially-validated data
        chunk_size = info.data.get("CHUNK_SIZE", 1000)
        if v >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        return v

    def ensure_directories(self) -> None:
        """
        Create all required data directories if they don't exist.
        Call this once at app startup.
        exist_ok=True means no error if it already exists.
        """
        for directory in [UPLOADS_DIR, FAISS_DIR, DB_DIR, LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    class Config:
        """pydantic-settings configuration."""
        env_file        = str(ROOT_DIR / ".env")   # absolute: works from any cwd
        env_file_encoding = "utf-8"
        case_sensitive  = False           # GEMINI_API_KEY == gemini_api_key
        extra           = "ignore"        # ignore unknown env vars


@lru_cache(maxsize=1)
def get_config() -> Settings:
    """
    Returns the singleton Settings instance.

    CONCEPT: @lru_cache(maxsize=1)
    - lru_cache memoises function results
    - maxsize=1 means cache the single result forever
    - First call creates the Settings object (reads .env)
    - Every subsequent call returns the SAME cached object
    - This is cheaper than reading the .env file on every access

    In tests, call get_config.cache_clear() to reset between test cases.
    """
    return Settings()
