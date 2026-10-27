"""
main.py
========
Streamlit Entry Point — Application Bootstrap

This is the ONLY file Streamlit runs: `streamlit run main.py`

RESPONSIBILITIES:
1. Initialise all services (once, cached by Streamlit)
2. Initialise session state (once per browser session)
3. Render the page layout (tabs + sidebar)
4. Wire everything together

CONCEPT: st.cache_resource
Services like EmbeddingService load a 90MB model.
@st.cache_resource tells Streamlit to create this object ONCE
and reuse it across all reruns and all users — like a singleton.
Without this, the model would reload on every interaction.

CONCEPT: st.session_state
Streamlit reruns the entire script on every user interaction.
st.session_state is a dict that persists across reruns for the
same browser session. We store our SessionState object there.
"""

import uuid
import streamlit as st

# ── Page config must be the FIRST Streamlit call ────────────────────────────
st.set_page_config(
    page_title     = "Enterprise Knowledge Assistant",
    page_icon      = "🧠",
    layout         = "wide",
    initial_sidebar_state = "expanded",
    menu_items     = {
        "Get Help"    : "https://github.com/yourusername/enterprise-knowledge-assistant",
        "Report a bug": "https://github.com/yourusername/enterprise-knowledge-assistant/issues",
        "About"       : "# Enterprise Knowledge Assistant\nRAG-powered document intelligence.",
    }
)


# ── Service Initialisation (cached — runs ONCE) ──────────────────────────────

@st.cache_resource(show_spinner="Loading AI models...")
def init_services():
    """
    Initialise all services. Cached globally — runs once on first load.
    Returns a dict of all service instances.
    """
    from app.core.config import get_config
    from app.services.document_processor import DocumentProcessor
    from app.services.embedding_service import EmbeddingService
    from app.services.vector_store import VectorStore
    from app.services.llm_service import LLMService
    from app.services.rag_pipeline import RAGPipeline
    from app.database.db_manager import DatabaseManager
    from app.database.repositories import DocumentRepository, ChatRepository

    cfg = get_config()
    cfg.ensure_directories()

    embedding_svc  = EmbeddingService()
    vector_store   = VectorStore()
    llm_svc        = LLMService()
    rag_pipeline   = RAGPipeline(embedding_svc, vector_store, llm_svc)
    doc_processor  = DocumentProcessor()
    db_manager     = DatabaseManager()
    doc_repo       = DocumentRepository(db_manager)
    chat_repo      = ChatRepository(db_manager)

    return {
        "doc_processor" : doc_processor,
        "rag_pipeline"  : rag_pipeline,
        "vector_store"  : vector_store,
        "doc_repo"      : doc_repo,
        "chat_repo"     : chat_repo,
    }


def init_session_state(doc_repo, vector_store) -> None:
    """
    Initialise per-browser session state.
    Called on every rerun but only does work on the first run.
    """
    from app.core.models import SessionState

    if "session" not in st.session_state:
        session_id = str(uuid.uuid4())
        session    = SessionState(session_id=session_id)

        # Reload documents from DB for this new session
        try:
            all_docs = doc_repo.get_all()
            from app.core.models import DocumentStatus
            # Only keep docs whose vectors really exist in the index, so the
            # DB and FAISS can never disagree about what is queryable.
            indexed_ids = vector_store.doc_ids()
            session.active_documents = [
                d for d in all_docs
                if d.status == DocumentStatus.INDEXED and d.doc_id in indexed_ids
            ]
        except Exception as e:
            from app.utils.logger import get_logger
            get_logger(__name__).warning(f"Could not reload documents: {e}")

        st.session_state["session"] = session


# ── Main Application ──────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    # Initialise services
    try:
        services = init_services()
    except Exception as e:
        st.error(
            f"❌ Failed to initialise services: {e}\n\n"
            "Check that your `.env` has a valid `GROQ_API_KEY` or `GEMINI_API_KEY` (run `python check_models.py`).",
            icon="🚨"
        )
        st.stop()

    # Initialise session
    init_session_state(services["doc_repo"], services["vector_store"])
    session = st.session_state["session"]

    # Register session in DB
    try:
        services["chat_repo"].create_session(session.session_id)
    except Exception:
        pass

    # ── Render layout ─────────────────────────────────────────────────────
    from app.ui.sidebar import render_sidebar
    from app.ui.chat_interface import render_chat_interface
    from app.ui.analytics import render_analytics_dashboard

    # Sidebar
    render_sidebar(
        session       = session,
        doc_processor = services["doc_processor"],
        rag_pipeline  = services["rag_pipeline"],
        doc_repo      = services["doc_repo"],
        chat_repo     = services["chat_repo"],
    )

    # Main area with tabs
    tab_chat, tab_analytics = st.tabs(["💬 Chat", "📊 Analytics"])

    with tab_chat:
        render_chat_interface(
            session      = session,
            rag_pipeline = services["rag_pipeline"],
            chat_repo    = services["chat_repo"],
        )

    with tab_analytics:
        render_analytics_dashboard(
            chat_repo    = services["chat_repo"],
            vector_store = services["vector_store"],
        )


if __name__ == "__main__":
    main()
