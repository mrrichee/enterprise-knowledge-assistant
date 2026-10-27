"""
app/ui/chat_interface.py
=========================
Chat Interface — Main conversation window

Handles:
- Displaying full chat history
- Query input box
- Calling RAG pipeline on new messages
- Streaming-style response display
"""

import time
import streamlit as st

from app.core.models import SessionState, MessageRole
from app.ui.components import (
    render_message, render_empty_state, render_error
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def render_chat_interface(
    session     : SessionState,
    rag_pipeline,
    chat_repo,
) -> None:
    """
    Render the main chat area.

    Args:
        session:      Current SessionState
        rag_pipeline: RAGPipeline for answering questions
        chat_repo:    ChatRepository for persisting messages
    """
    # ── Chat history area ────────────────────────────────────────────────
    chat_container = st.container()

    with chat_container:
        if not session.chat_history:
            _render_welcome(session)
        else:
            for msg in session.chat_history:
                render_message(msg)

    # ── Query input (pinned to bottom via Streamlit's chat_input) ────────
    if not session.get_indexed_docs():
        st.info("👈 Upload documents in the sidebar to start asking questions.", icon="💡")
        return

    query = st.chat_input(
        placeholder = "Ask anything about your documents...",
        key         = "chat_input",
    )

    # A suggestion button stores its text here; consume it as if typed
    prefill = st.session_state.pop("_prefill_query", None)
    query = query or prefill

    if query:
        _handle_query(query, session, rag_pipeline, chat_repo)


def _handle_query(query: str, session: SessionState, rag_pipeline, chat_repo) -> None:
    """Process a user query through the RAG pipeline and display results."""

    # Optimistically display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(query)

    # Run pipeline and stream the response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                result, user_message = rag_pipeline.query(
                    question = query,
                    session  = session,
                )

                # Display answer
                st.markdown(result.answer)

                # Display citations
                from app.ui.components import render_citations
                if result.citations:
                    render_citations(result.citations)

                # Show timing info
                st.caption(
                    f"⚡ {result.processing_time_ms:.0f}ms · "
                    f"📚 {len(result.citations)} source(s) · "
                    f"🔤 {result.chat_message.tokens_used or 0} tokens"
                )

                # Persist to session and DB
                session.add_message(user_message)
                session.add_message(result.chat_message)

                try:
                    chat_repo.save_message(user_message)
                    chat_repo.save_message(result.chat_message)
                    chat_repo.save_analytics(
                        session_id       = session.session_id,
                        query            = query,
                        response_time_ms = result.processing_time_ms,
                        tokens           = result.chat_message.tokens_used or 0,
                        citation_count   = len(result.citations),
                    )
                except Exception as db_err:
                    logger.warning(f"DB save failed (non-critical): {db_err}")

                st.rerun()   # refresh to show message in history properly

            except Exception as e:
                from app.core.exceptions import AppBaseError
                msg = e.user_message() if isinstance(e, AppBaseError) else f"An error occurred: {e}"
                render_error(msg)
                logger.error(f"Query failed: {e}", exc_info=True)


def _render_welcome(session: SessionState) -> None:
    """Show a welcome screen when no messages exist yet."""
    doc_count = len(session.get_indexed_docs())

    st.markdown(
        """
        <div style="text-align: center; padding: 4rem 2rem;">
            <h1>🧠 Enterprise Knowledge Assistant</h1>
            <p style="font-size: 1.1rem; opacity: 0.7;">
                Ask questions about your uploaded documents.<br>
                Get precise answers with source citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if doc_count > 0:
        st.markdown("---")
        st.markdown("### 💡 Try asking:")
        cols = st.columns(2)
        suggestions = [
            "What is the main topic of this document?",
            "Summarize the key points",
            "What are the most important findings?",
            "List all dates or deadlines mentioned",
        ]
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, width="stretch", key=f"suggest_{i}"):
                    # Pre-fill the chat input by storing in session state
                    st.session_state["_prefill_query"] = suggestion
                    st.rerun()
