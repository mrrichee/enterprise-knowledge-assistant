"""
app/database/repositories.py
=============================
Repository Pattern — CRUD operations for all entities.

Each repository owns all database operations for one entity.
The rest of the app imports these repositories, never raw SQL.
"""

import json
from datetime import datetime

from app.core.models import Document, ChatMessage, Citation, DocumentStatus, MessageRole
from app.database.db_manager import DatabaseManager
from app.core.exceptions import DatabaseError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentRepository:
    """CRUD operations for the documents table."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def save(self, doc: Document) -> None:
        sql = """
        INSERT OR REPLACE INTO documents
            (doc_id, filename, file_path, file_type, file_size,
             status, page_count, chunk_count, error_msg, uploaded_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        try:
            with self.db.get_connection() as conn:
                conn.execute(sql, (
                    doc.doc_id, doc.filename, doc.file_path, doc.file_type,
                    doc.file_size, doc.status.value, doc.page_count,
                    doc.chunk_count, doc.error_msg,
                    doc.uploaded_at.isoformat()
                ))
            logger.debug(f"Saved document: {doc.doc_id}")
        except Exception as e:
            raise DatabaseError("save_document", str(e)) from e

    def get_all(self) -> list[Document]:
        try:
            with self.db.get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM documents ORDER BY uploaded_at DESC"
                ).fetchall()
            return [self._row_to_doc(r) for r in rows]
        except Exception as e:
            raise DatabaseError("get_all_documents", str(e)) from e

    def update_status(self, doc_id: str, status: DocumentStatus, error_msg: str = None) -> None:
        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    "UPDATE documents SET status=?, error_msg=? WHERE doc_id=?",
                    (status.value, error_msg, doc_id)
                )
        except Exception as e:
            raise DatabaseError("update_status", str(e)) from e

    def delete(self, doc_id: str) -> None:
        try:
            with self.db.get_connection() as conn:
                conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        except Exception as e:
            raise DatabaseError("delete_document", str(e)) from e

    def _row_to_doc(self, row) -> Document:
        return Document(
            doc_id      = row["doc_id"],
            filename    = row["filename"],
            file_path   = row["file_path"],
            file_type   = row["file_type"],
            file_size   = row["file_size"],
            status      = DocumentStatus(row["status"]),
            page_count  = row["page_count"],
            chunk_count = row["chunk_count"] or 0,
            error_msg   = row["error_msg"],
            uploaded_at = datetime.fromisoformat(row["uploaded_at"]),
        )


class ChatRepository:
    """CRUD operations for chat_sessions and chat_messages tables."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def create_session(self, session_id: str) -> None:
        now = datetime.now().isoformat()
        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO chat_sessions (session_id, created_at, last_active) VALUES (?,?,?)",
                    (session_id, now, now)
                )
        except Exception as e:
            raise DatabaseError("create_session", str(e)) from e

    def save_message(self, msg: ChatMessage) -> None:
        citations_json = json.dumps([
            {"filename": c.filename, "page_number": c.page_number,
             "relevance": c.relevance, "excerpt": c.excerpt}
            for c in msg.citations
        ])
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO chat_messages
                        (message_id, session_id, role, content, citations, tokens_used, timestamp)
                    VALUES (?,?,?,?,?,?,?)
                """, (
                    msg.message_id, msg.session_id, msg.role.value,
                    msg.content, citations_json, msg.tokens_used,
                    msg.timestamp.isoformat()
                ))
                conn.execute("""
                    UPDATE chat_sessions
                    SET last_active=?, message_count=message_count+1
                    WHERE session_id=?
                """, (datetime.now().isoformat(), msg.session_id))
        except Exception as e:
            raise DatabaseError("save_message", str(e)) from e

    def get_session_messages(self, session_id: str) -> list[ChatMessage]:
        try:
            with self.db.get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM chat_messages WHERE session_id=? ORDER BY timestamp",
                    (session_id,)
                ).fetchall()
            return [self._row_to_message(r) for r in rows]
        except Exception as e:
            raise DatabaseError("get_session_messages", str(e)) from e

    def save_analytics(self, session_id: str, query: str,
                       response_time_ms: float, tokens: int, citation_count: int) -> None:
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO query_analytics
                        (session_id, query_text, response_time_ms, tokens_used, citation_count, timestamp)
                    VALUES (?,?,?,?,?,?)
                """, (session_id, query, response_time_ms, tokens, citation_count,
                      datetime.now().isoformat()))
        except Exception as e:
            logger.warning(f"Analytics save failed (non-critical): {e}")

    def get_analytics_summary(self) -> dict:
        """For the analytics dashboard in Phase 6."""
        try:
            with self.db.get_connection() as conn:
                total    = conn.execute("SELECT COUNT(*) FROM query_analytics").fetchone()[0]
                avg_time = conn.execute(
                    "SELECT AVG(response_time_ms) FROM query_analytics"
                ).fetchone()[0] or 0
                avg_tok  = conn.execute(
                    "SELECT AVG(tokens_used) FROM query_analytics"
                ).fetchone()[0] or 0
                recent   = conn.execute("""
                    SELECT query_text, response_time_ms, citation_count, timestamp
                    FROM query_analytics ORDER BY timestamp DESC LIMIT 10
                """).fetchall()
            return {
                "total_queries"    : total,
                "avg_response_ms"  : round(avg_time, 1),
                "avg_tokens"       : round(avg_tok, 1),
                "recent_queries"   : [dict(r) for r in recent],
            }
        except Exception as e:
            logger.error(f"Analytics fetch failed: {e}")
            return {}

    def _row_to_message(self, row) -> ChatMessage:
        citations_data = json.loads(row["citations"] or "[]")
        citations = [
            Citation(
                filename    = c["filename"],
                page_number = c["page_number"],
                chunk_index = 0,
                relevance   = c["relevance"],
                excerpt     = c["excerpt"],
            )
            for c in citations_data
        ]
        return ChatMessage(
            message_id  = row["message_id"],
            session_id  = row["session_id"],
            role        = MessageRole(row["role"]),
            content     = row["content"],
            citations   = citations,
            tokens_used = row["tokens_used"],
            timestamp   = datetime.fromisoformat(row["timestamp"]),
        )
