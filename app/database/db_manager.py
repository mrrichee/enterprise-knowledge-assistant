"""
app/database/db_manager.py
===========================
Phase 6: SQLite Database Layer

CONCEPT: Repository Pattern
We don't scatter raw SQL everywhere. Instead, we have Repository classes
that own all DB queries for their entity. The rest of the app calls
chat_repo.save_message() and never writes SQL.

WHY SQLITE?
- Zero setup — just a file, no server needed
- Python has built-in sqlite3 support
- Perfect for single-user apps and demos
- Easy to migrate to PostgreSQL later by swapping just this file

SCHEMA DESIGN:
    documents   — one row per uploaded document
    chat_sessions — one row per browser session
    chat_messages — one row per message (links to sessions)
    query_analytics — one row per RAG query (for the dashboard)
"""

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from app.core.config import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
cfg    = get_config()


class DatabaseManager:
    """Manages SQLite connection and schema creation."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or cfg.DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()
        logger.info(f"✅ SQLite DB ready at: {self.db_path}")

    @contextmanager
    def get_connection(self):
        """
        Context manager for DB connections.

        CONCEPT: Context managers (with statement)
        Using 'with' ensures the connection is always closed properly,
        even if an exception occurs — prevents connection leaks.
        conn.row_factory = sqlite3.Row lets us access columns by name.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row      # rows behave like dicts
        conn.execute("PRAGMA journal_mode=WAL")  # better concurrent reads
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_tables(self) -> None:
        """Create all tables if they don't exist. Safe to call multiple times."""
        schema = """
        -- Documents table
        CREATE TABLE IF NOT EXISTS documents (
            doc_id       TEXT PRIMARY KEY,
            filename     TEXT NOT NULL,
            file_path    TEXT NOT NULL,
            file_type    TEXT NOT NULL,
            file_size    INTEGER NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending',
            page_count   INTEGER,
            chunk_count  INTEGER DEFAULT 0,
            error_msg    TEXT,
            uploaded_at  TEXT NOT NULL
        );

        -- Chat sessions table
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id   TEXT PRIMARY KEY,
            created_at   TEXT NOT NULL,
            last_active  TEXT NOT NULL,
            message_count INTEGER DEFAULT 0
        );

        -- Chat messages table
        CREATE TABLE IF NOT EXISTS chat_messages (
            message_id   TEXT PRIMARY KEY,
            session_id   TEXT NOT NULL,
            role         TEXT NOT NULL,
            content      TEXT NOT NULL,
            citations    TEXT,           -- JSON array
            tokens_used  INTEGER,
            timestamp    TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
        );

        -- Query analytics table
        CREATE TABLE IF NOT EXISTS query_analytics (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT NOT NULL,
            query_text       TEXT NOT NULL,
            response_time_ms REAL,
            tokens_used      INTEGER,
            citation_count   INTEGER,
            timestamp        TEXT NOT NULL
        );

        -- Indexes for fast lookups
        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON chat_messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_analytics_timestamp
            ON query_analytics(timestamp);
        """
        with self.get_connection() as conn:
            conn.executescript(schema)
        logger.debug("Database schema verified")
