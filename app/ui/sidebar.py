"""
app/ui/sidebar.py
==================
Sidebar — Document Management Panel

Handles:
- File upload widget
- Document list with status
- Delete documents
- Session controls (clear chat)
"""

from pathlib import Path

import streamlit as st
from app.core.models import SessionState, DocumentStatus
from app.ui.components import (
    render_document_card, render_empty_state,
    render_error, render_success
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def render_sidebar(
    session       : SessionState,
    doc_processor,
    rag_pipeline,
    doc_repo,
    chat_repo,
) -> None:
    """
    Render the full sidebar.

    Args:
        session:       Current SessionState
        doc_processor: DocumentProcessor instance
        rag_pipeline:  RAGPipeline instance
        doc_repo:      DocumentRepository instance
        chat_repo:     ChatRepository instance
    """
    with st.sidebar:
        # Header
        st.markdown("## 🧠 Knowledge Assistant")
        st.caption("Upload documents and start asking questions")
        st.divider()
        _show_flash()

        # ── Upload Section ───────────────────────────────────────────────
        st.markdown("### 📤 Upload Documents")
        uploaded_files = st.file_uploader(
            label       = "Drag & drop files here",
            type        = ["pdf", "docx", "txt"],
            accept_multiple_files = True,
            help        = "Supported formats: PDF, DOCX, TXT (max 50MB each)",
            key         = "file_uploader",
        )

        if uploaded_files:
            _process_uploads(
                uploaded_files, session, doc_processor,
                rag_pipeline, doc_repo
            )
        else:
            st.session_state["_processed_uploads"] = set()

        st.divider()

        # ── Document List ────────────────────────────────────────────────
        st.markdown("### 📚 Uploaded Documents")

        if not session.active_documents:
            render_empty_state(
                icon     = "📂",
                title    = "No documents yet",
                subtitle = "Upload PDFs, DOCX, or TXT files above"
            )
        else:
            indexed_count = len(session.get_indexed_docs())
            st.caption(f"{indexed_count}/{len(session.active_documents)} documents ready")

            for doc in session.active_documents:
                render_document_card(
                    doc,
                    on_delete=lambda doc_id: _delete_document(
                        doc_id, session, doc_repo, rag_pipeline
                    )
                )
                st.markdown("")   # spacing

        st.divider()

        # ── Session Controls ─────────────────────────────────────────────
        st.markdown("### ⚙️ Session")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", width="stretch", help="Clear conversation history"):
                session.clear_history()
                st.rerun()

        with col2:
            msg_count  = len(session.chat_history)
            doc_count  = len(session.active_documents)
            st.caption(f"💬 {msg_count} messages\n📄 {doc_count} docs")

        st.divider()

        # ── Stats Footer ─────────────────────────────────────────────────
        stats = _get_session_stats(session)
        st.caption(
            f"Session: `{session.session_id[:8]}...`\n\n"
            f"Chunks indexed: **{stats['total_chunks']}**"
        )


def _process_uploads(uploaded_files, session, doc_processor, rag_pipeline, doc_repo):
    """
    Process newly uploaded files.

    Streamlit keeps files in the uploader across reruns, so we remember which
    ones we've already handled. Without this, deleting a document would
    re-index it on the very next rerun.
    """
    processed = st.session_state.setdefault("_processed_uploads", set())

    # Forget files the user removed from the uploader (so they can re-upload later)
    current_ids = {f"{f.name}:{f.size}" for f in uploaded_files}
    processed.intersection_update(current_ids)

    existing_names = {d.filename for d in session.active_documents}
    new_files = []
    for f in uploaded_files:
        fid = f"{f.name}:{f.size}"
        if fid in processed:
            continue
        processed.add(fid)
        if f.name not in existing_names:
            new_files.append(f)

    if not new_files:
        return

    progress_bar = st.progress(0, text="Starting upload...")
    flash = st.session_state.setdefault("_flash", [])
    vector_store = rag_pipeline.vector_store

    for i, uploaded_file in enumerate(new_files):
        progress = (i + 1) / len(new_files)
        progress_bar.progress(progress, text=f"Processing {uploaded_file.name}...")
        document = None

        try:
            file_bytes = uploaded_file.getvalue()
            document, chunks = doc_processor.process_uploaded_file(
                file_bytes = file_bytes,
                filename   = uploaded_file.name,
            )

            # Embed and index
            progress_bar.progress(progress, text=f"Indexing {uploaded_file.name}...")
            rag_pipeline.index_document(chunks)

            document.status = DocumentStatus.INDEXED

            # Save to DB and session (roll back the index if this fails)
            try:
                doc_repo.save(document)
            except Exception:
                vector_store.remove_document(document.doc_id)
                raise
            session.active_documents.append(document)

            flash.append(("success", f"'{uploaded_file.name}' indexed — {len(chunks)} chunks ready"))
            logger.info(f"Document uploaded: {uploaded_file.name}")

        except Exception as e:
            from app.core.exceptions import AppBaseError
            msg = e.user_message() if isinstance(e, AppBaseError) else f"Failed: {uploaded_file.name} ({e})"
            flash.append(("error", msg))
            logger.error(f"Upload failed for {uploaded_file.name}: {e}", exc_info=True)
            
    progress_bar.empty()
    st.rerun()


def _show_flash() -> None:
    """Show (and clear) messages queued before a st.rerun()."""
    for kind, msg in st.session_state.pop("_flash", []):
        (render_success if kind == "success" else render_error)(msg)


def _delete_document(doc_id: str, session: SessionState, doc_repo, rag_pipeline) -> None:
    """Remove a document from the session, DB, vector index and disk."""
    doc = next((d for d in session.active_documents if d.doc_id == doc_id), None)
    session.active_documents = [d for d in session.active_documents if d.doc_id != doc_id]

    try:
        rag_pipeline.vector_store.remove_document(doc_id)
    except Exception as e:
        logger.error(f"Failed to remove vectors for {doc_id}: {e}")
    doc_repo.delete(doc_id)

    if doc and doc.file_path:
        try:
            Path(doc.file_path).unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Could not delete file {doc.file_path}: {e}")
    st.rerun()


def _get_session_stats(session: SessionState) -> dict:
    """Compute simple session stats for display."""
    total_chunks = sum(d.chunk_count for d in session.active_documents)
    return {"total_chunks": total_chunks}
