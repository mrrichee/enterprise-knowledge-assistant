"""
app/ui/components.py
=====================
Reusable UI Components

CONCEPT: Component-based UI
Instead of writing raw st.markdown() everywhere, we create functions
that encapsulate styling + logic. This means:
- Change the look of ALL message bubbles in one place
- Consistent styling across the app
- Readable, self-documenting UI code
"""

import streamlit as st
from app.core.models import ChatMessage, Citation, Document, MessageRole, DocumentStatus


def render_message(msg: ChatMessage) -> None:
    """Render a single chat message with appropriate styling."""
    is_user = msg.role == MessageRole.USER

    if is_user:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg.content)
            if msg.has_citations():
                render_citations(msg.citations)


def render_citations(citations: list[Citation]) -> None:
    """Render the source citations in a collapsed expander."""
    if not citations:
        return

    with st.expander(f"📚 {len(citations)} source(s) used", expanded=False):
        for i, cite in enumerate(citations, 1):
            relevance_color = (
                "🟢" if cite.relevance > 0.7
                else "🟡" if cite.relevance > 0.5
                else "🔴"
            )
            st.markdown(
                f"**{relevance_color} Source {i}:** `{cite.filename}` — "
                f"Page **{cite.page_number}** "
                f"*(relevance: {cite.relevance_percent()})*"
            )
            st.caption(f"> {cite.excerpt[:200]}...")
            if i < len(citations):
                st.divider()


def render_document_card(doc: Document, on_delete=None) -> None:
    """Render a document card in the sidebar."""
    status_icon = {
        DocumentStatus.PENDING    : "⏳",
        DocumentStatus.PROCESSING : "⚙️",
        DocumentStatus.INDEXED    : "✅",
        DocumentStatus.FAILED     : "❌",
    }.get(doc.status, "❓")

    file_icon = {
        "pdf" : "📄", "docx": "📝", "txt": "📃"
    }.get(doc.file_type, "📁")

    with st.container():
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(
                f"{file_icon} **{doc.filename[:30]}{'...' if len(doc.filename) > 30 else ''}**"
            )
            st.caption(
                f"{status_icon} {doc.status.value.title()} · "
                f"{doc.size_display()} · "
                f"{doc.chunk_count} chunks"
                + (f" · {doc.page_count}p" if doc.page_count else "")
            )
        with col2:
            if on_delete and st.button("🗑️", key=f"del_{doc.doc_id}", help="Remove document"):
                on_delete(doc.doc_id)

        if doc.status == DocumentStatus.FAILED and doc.error_msg:
            st.error(f"Error: {doc.error_msg}", icon="⚠️")


def render_metric_card(label: str, value: str, delta: str = None, icon: str = "") -> None:
    """Render a styled metric card for the analytics dashboard."""
    st.metric(label=f"{icon} {label}", value=value, delta=delta)


def render_empty_state(icon: str, title: str, subtitle: str) -> None:
    """Render a friendly empty state when there's nothing to show."""
    st.markdown(
        f"""
        <div style="text-align: center; padding: 3rem 1rem; opacity: 0.6;">
            <div style="font-size: 3rem;">{icon}</div>
            <h3>{title}</h3>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_processing_status(filename: str, step: str) -> None:
    """Show a live processing status for document upload."""
    st.info(f"⚙️ Processing **{filename}**: {step}", icon="ℹ️")


def render_error(message: str) -> None:
    """Render a styled error message."""
    st.error(message, icon="🚨")


def render_success(message: str) -> None:
    """Render a styled success message."""
    st.success(message, icon="✅")
