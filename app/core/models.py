"""
app/core/models.py
==================
Domain Models — the "language" of this entire system.

CONCEPT: Why dataclasses?
- Plain Python classes require __init__, __repr__, __eq__ boilerplate
- @dataclass auto-generates all of that from field declarations
- These models are PURE DATA — no logic, no dependencies, no imports from
  other parts of the app. This is what "domain layer" means.

CONCEPT: Why frozen=True on some models?
- frozen=True makes the object immutable (like a read-only struct)
- Immutable objects are safer to pass around — nothing can accidentally
  mutate shared state. Use it for value objects that represent facts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


# ── Enumerations ────────────────────────────────────────────────────────────

class DocumentStatus(Enum):
    """
    Tracks where a document is in the processing pipeline.
    Using an Enum (not plain strings) means typos cause errors at
    definition time, not silently at runtime.
    """
    PENDING    = "pending"      # just uploaded, not yet processed
    PROCESSING = "processing"   # currently being chunked / embedded
    INDEXED    = "indexed"      # in FAISS, ready to query
    FAILED     = "failed"       # something went wrong


class MessageRole(Enum):
    """Who sent a chat message."""
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


# ── Core Domain Models ───────────────────────────────────────────────────────

@dataclass
class Document:
    """
    Represents an uploaded file.

    CONCEPT: This is an 'Entity' in Domain-Driven Design (DDD).
    An entity has an identity (doc_id) and a lifecycle (status changes).

    Fields:
        doc_id      : unique identifier (we'll use UUID strings)
        filename    : original file name as uploaded by the user
        file_path   : where we saved it on disk under data/uploads/
        file_type   : 'pdf', 'docx', or 'txt'
        file_size   : bytes, for display purposes
        status      : current pipeline status (see DocumentStatus)
        uploaded_at : when the user uploaded it
        page_count  : how many pages (for PDFs) or None
        chunk_count : how many chunks were created from this doc
        error_msg   : populated if status == FAILED
    """
    doc_id      : str
    filename    : str
    file_path   : str
    file_type   : str
    file_size   : int
    status      : DocumentStatus = DocumentStatus.PENDING
    uploaded_at : datetime       = field(default_factory=datetime.now)
    page_count  : Optional[int]  = None
    chunk_count : int            = 0
    error_msg   : Optional[str]  = None

    def is_ready(self) -> bool:
        """Convenience method — is this doc queryable?"""
        return self.status == DocumentStatus.INDEXED

    def size_display(self) -> str:
        """Human-readable file size: 1.2 MB, 340 KB, etc."""
        if self.file_size < 1024:
            return f"{self.file_size} B"
        elif self.file_size < 1024 ** 2:
            return f"{self.file_size / 1024:.1f} KB"
        else:
            return f"{self.file_size / (1024 ** 2):.1f} MB"


@dataclass(frozen=True)
class Chunk:
    """
    Represents one piece of text extracted from a document.

    CONCEPT: This is a 'Value Object' in DDD.
    It has no identity of its own — two chunks with identical content
    and metadata ARE the same thing. That's why frozen=True.

    The chunk is the unit of retrieval — FAISS finds chunks, not documents.

    Fields:
        chunk_id    : unique id, usually "{doc_id}_chunk_{index}"
        doc_id      : which document this came from
        content     : the actual text content
        page_number : page it came from (for citations)
        chunk_index : position within the document (0-based)
        filename    : denormalised here so retrieval results are self-contained
    """
    chunk_id    : str
    doc_id      : str
    content     : str
    page_number : int
    chunk_index : int
    filename    : str

    def preview(self, length: int = 100) -> str:
        """Short preview for display/debugging."""
        return self.content[:length] + "..." if len(self.content) > length else self.content


@dataclass(frozen=True)
class Citation:
    """
    A source reference attached to an answer.

    When the LLM answers a question, we show exactly which chunk(s)
    it used and where they came from. This is what separates a
    production RAG app from a basic chatbot.

    Fields:
        filename    : document name to display to the user
        page_number : page in the original document
        chunk_index : position of the chunk (for ordering multiple citations)
        relevance   : cosine similarity score (0.0–1.0). Higher = more relevant.
        excerpt     : the actual chunk text (shortened for display)
    """
    filename    : str
    page_number : int
    chunk_index : int
    relevance   : float
    excerpt     : str

    def relevance_percent(self) -> str:
        return f"{self.relevance * 100:.0f}%"


@dataclass
class ChatMessage:
    """
    One message in the conversation history.

    CONCEPT: We store both role and content — this mirrors the format
    that LLM APIs (including Gemini) expect for multi-turn conversations.
    The citations list is only populated for ASSISTANT messages.

    Fields:
        message_id  : unique id
        session_id  : which chat session this belongs to
        role        : user / assistant / system
        content     : the message text
        citations   : list of sources (only for assistant messages)
        timestamp   : when this message was created
        tokens_used : how many LLM tokens this response consumed (Phase 6)
    """
    message_id  : str
    session_id  : str
    role        : MessageRole
    content     : str
    citations   : list[Citation]    = field(default_factory=list)
    timestamp   : datetime          = field(default_factory=datetime.now)
    tokens_used : Optional[int]     = None

    def has_citations(self) -> bool:
        return len(self.citations) > 0


@dataclass
class QueryResult:
    """
    The complete output of one RAG pipeline run.

    This is what flows from the pipeline back to the UI.
    Bundling everything in one object makes the interface clean —
    the UI asks one question, gets one structured result.

    Fields:
        answer      : the LLM-generated answer text
        citations   : ranked list of source chunks used
        chat_message: the full ChatMessage to persist
        processing_time_ms: how long the full pipeline took
    """
    answer             : str
    citations          : list[Citation]
    chat_message       : ChatMessage
    processing_time_ms : float

    def top_citation(self) -> Optional[Citation]:
        """Most relevant source, if any."""
        return self.citations[0] if self.citations else None


@dataclass
class SessionState:
    """
    Tracks the current UI session.

    Streamlit reruns the entire script on every interaction.
    We persist session state in st.session_state using this model.

    Fields:
        session_id      : unique id for this browser session
        active_documents: list of indexed Document objects
        chat_history    : list of ChatMessage objects in order
        created_at      : when the session started
    """
    session_id       : str
    active_documents : list[Document]  = field(default_factory=list)
    chat_history     : list[ChatMessage] = field(default_factory=list)
    created_at       : datetime        = field(default_factory=datetime.now)

    def add_message(self, msg: ChatMessage) -> None:
        self.chat_history.append(msg)

    def get_indexed_docs(self) -> list[Document]:
        return [d for d in self.active_documents if d.is_ready()]

    def clear_history(self) -> None:
        self.chat_history.clear()
