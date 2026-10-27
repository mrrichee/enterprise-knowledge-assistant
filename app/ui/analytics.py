"""
app/ui/analytics.py
====================
Phase 6: Analytics Dashboard

Shows query statistics, token usage, and recent queries.
Accessible via a tab in the main interface.
"""

import streamlit as st
from app.utils.logger import get_logger

logger = get_logger(__name__)


def render_analytics_dashboard(chat_repo, vector_store) -> None:
    """Render the analytics dashboard tab."""
    st.markdown("## 📊 Usage Analytics")

    try:
        summary = chat_repo.get_analytics_summary()
        vs_stats = vector_store.stats()
    except Exception as e:
        st.error(f"Failed to load analytics: {e}")
        return

    # ── Top metrics row ─────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", summary.get("total_queries", 0), help="Total questions asked")
    with col2:
        avg_ms = summary.get("avg_response_ms", 0)
        st.metric("Avg Response Time", f"{avg_ms:.0f}ms")
    with col3:
        st.metric("Avg Tokens / Query", f"{summary.get('avg_tokens', 0):.0f}")
    with col4:
        st.metric("Vectors Indexed", vs_stats.get("total_vectors", 0))

    st.divider()

    # ── Vector store stats ──────────────────────────────────────────────
    st.markdown("### 🗄️ Vector Store")
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", vs_stats.get("total_documents", 0))
    col2.metric("Total Chunks", vs_stats.get("total_chunks", 0))
    col3.metric("Index Dimensions", vs_stats.get("dimension", 384))

    st.divider()

    # ── Recent queries table ────────────────────────────────────────────
    st.markdown("### 🕐 Recent Queries")
    recent = summary.get("recent_queries", [])

    if not recent:
        st.info("No queries yet. Start asking questions!")
    else:
        import pandas as pd
        df = pd.DataFrame(recent)
        if not df.empty:
            df = df.rename(columns={
                "query_text"      : "Question",
                "response_time_ms": "Time (ms)",
                "citation_count"  : "Citations",
                "timestamp"       : "When",
            })
            df["Time (ms)"] = df["Time (ms)"].apply(lambda x: f"{x:.0f}" if x else "-")
            df["When"]      = df["When"].apply(lambda x: x[:16] if x else "-")
            st.dataframe(df, width="stretch", hide_index=True)
