"""
app/services/rag_pipeline.py
==============================
Phase 4: The RAG Pipeline — Orchestrator

CONCEPT: Orchestrator Pattern
The RAG pipeline doesn't DO anything itself — it coordinates other services.
Think of it as a conductor: it tells each musician (service) when to play.

FULL FLOW:
    User query
        ↓
    [EmbeddingService] embed the query → query_vector
        ↓
    [VectorStore] search for top-K similar chunks → List[Chunk]
        ↓
    [LLMService] build prompt with context → generate answer
        ↓
    Build QueryResult with citations
        ↓
    Return to UI

This is the heart of the application.
"""

import time
import uuid
from datetime import datetime

from app.core.models import (
    Chunk, Citation, ChatMessage, QueryResult,
    MessageRole, SessionState
)
from app.core.exceptions import (
    NoDocumentsIndexedError, NoRelevantChunksError, LLMError
)
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.utils.logger import get_logger
from app.utils.helpers import timer

logger = get_logger(__name__)


class RAGPipeline:
    """
    End-to-end query pipeline: question → answer with citations.

    This is a stateless service — it holds no document state itself.
    All state (indexed documents, chat history) lives in the session
    or is fetched from VectorStore/SQLite.
    """

    def __init__(
        self,
        embedding_service : EmbeddingService,
        vector_store      : VectorStore,
        llm_service       : LLMService,
    ):
        self.embedder     = embedding_service
        self.vector_store = vector_store
        self.llm          = llm_service
        logger.info("✅ RAG Pipeline initialised")

    @timer
    def query(
        self,
        question      : str,
        session       : SessionState,
        doc_ids_filter: list[str] | None = None,
    ) -> tuple[QueryResult, ChatMessage]:
        """
        Process a user question through the full RAG pipeline.

        Args:
            question:       The user's natural language question
            session:        Current session (for chat history + doc list)
            doc_ids_filter: Restrict search to specific doc_ids (None = all)

        Returns:
            (QueryResult, user ChatMessage) — the caller persists both
        """
        start_time = time.perf_counter()
        logger.info(f"RAG query: '{question[:80]}...' " if len(question) > 80 else f"RAG query: '{question}'")

        # ── Guard: need at least one indexed document ─────────────────────
        indexed = session.get_indexed_docs()
        if not indexed:
            raise NoDocumentsIndexedError()

        # ── Step 1: Embed the query ────────────────────────────────────────
        logger.debug("Step 1: Embedding query")
        query_vector = self.embedder.embed_query(question)

        # ── Step 2: Retrieve top-K relevant chunks (single search) ─────────
        # Restrict to documents that are still in this session, so deleted
        # documents can never leak into answers.
        if doc_ids_filter is None:
            doc_ids_filter = [d.doc_id for d in indexed]
        logger.debug("Step 2: Searching vector store")
        scored = self.vector_store.search_with_scores(
            query_embedding = query_vector,
            doc_ids_filter  = doc_ids_filter,
            min_fallback    = 3,   # still give the LLM something for vague questions
        )
        relevant_chunks = [c for c, _ in scored]
        logger.info(f"Retrieved {len(relevant_chunks)} chunks")

        # ── Step 3: Build citation objects from the same results ───────────
        citations = [self.vector_store.to_citation(c, sc) for c, sc in scored]

        # ── Step 4: Generate answer ────────────────────────────────────────
        logger.debug("Step 4: Generating answer")
        history = self.llm.build_history_for_api(session.chat_history)
        answer, tokens = self.llm.generate_answer(
            query          = question,
            context_chunks = relevant_chunks,
            chat_history   = history,
        )

        # ── Step 5: Build result objects ───────────────────────────────────
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # User message
        user_message = ChatMessage(
            message_id = str(uuid.uuid4()),
            session_id = session.session_id,
            role       = MessageRole.USER,
            content    = question,
        )

        # Assistant message with citations
        assistant_message = ChatMessage(
            message_id  = str(uuid.uuid4()),
            session_id  = session.session_id,
            role        = MessageRole.ASSISTANT,
            content     = answer,
            citations   = citations,
            tokens_used = tokens,
        )

        result = QueryResult(
            answer             = answer,
            citations          = citations,
            chat_message       = assistant_message,
            processing_time_ms = elapsed_ms,
        )

        logger.info(
            f"✅ Query complete in {elapsed_ms:.0f}ms | "
            f"{len(citations)} citations | {tokens} tokens"
        )
        return result, user_message

    def index_document(
        self,
        chunks     : list[Chunk],
    ) -> None:
        """
        Embed chunks and add them to the vector store.
        Called after document processing in Phase 2.

        Args:
            chunks: the Chunk objects from DocumentProcessor
        """
        logger.info(f"Indexing {len(chunks)} chunks")
        embeddings = self.embedder.embed_chunks(chunks)
        self.vector_store.add_chunks(chunks, embeddings)
        logger.info(f"✅ {len(chunks)} chunks indexed in FAISS")
