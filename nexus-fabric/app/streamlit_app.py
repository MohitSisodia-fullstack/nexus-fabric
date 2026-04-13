# app/streamlit_app.py
# Nexus-Fabric — Recruiter Demo UI
# Run locally: streamlit run streamlit_app.py
# Deploy:      https://share.streamlit.io

import os
import time
import streamlit as st
from rag_engine import get_engine, DEMO_MODE

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nexus-Fabric — Enterprise RAG Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/Microsoft%20Fabric-0078D4?logo=microsoft&logoColor=white")
    st.markdown("## 🧠 Nexus-Fabric")
    st.markdown("**Enterprise Multi-Modal RAG Platform**")
    st.divider()

    st.markdown("### Architecture")
    st.markdown("""
- 🟫 **Bronze** — Raw ingest (SharePoint, SQL, PDFs)
- ⬜ **Silver** — PySpark chunking + Azure OpenAI embeddings
- 🟨 **Gold** — Azure AI Search hybrid index
- 🟪 **App** — FastAPI + Streamlit + GPT-4o
    """)
    st.divider()

    st.markdown("### Guardrails Active")
    st.success("✅ PII detection (Microsoft Presidio)")
    st.success("✅ Hybrid search (Vector + BM25 + RRF)")
    if DEMO_MODE:
        st.info("ℹ️ Demo mode — no Azure credentials needed")

    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("`Microsoft Fabric` `Azure OpenAI` `Azure AI Search`  \n`PySpark` `Delta Lake` `Presidio` `GPT-4o`")
    st.markdown("[View on GitHub](https://github.com/YOUR_USERNAME/nexus-fabric)")


# ── Main content ─────────────────────────────────────────────────────────────
st.title("🧠 Nexus-Fabric")
st.subheader("Enterprise Multi-Modal RAG Platform with LLMOps & Governance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Architecture", "Medallion", "Bronze→Silver→Gold")
col2.metric("Search Mode", "Hybrid", "Vector + BM25 + RRF")
col3.metric("Guardrails", "PII Active", "Presidio NER")
col4.metric("LLM", "GPT-4o", "Azure OpenAI")

st.divider()

# ── Chat interface ────────────────────────────────────────────────────────────
st.markdown("### 💬 Ask the Knowledge Base")
st.caption("Try: 'What is the chunking strategy?' / 'How does PII detection work?' / 'Explain the hybrid search approach'")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"📄 Sources ({len(msg['sources'])} chunks retrieved)"):
                for src in msg["sources"]:
                    st.markdown(f"**{src.get('source_path','Unknown')}** — score: `{src.get('score', 0):.3f}`")
                    st.code(src.get("chunk_text", "")[:300] + "...", language=None)

# Input
if prompt := st.chat_input("Ask anything about the knowledge base..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving + generating..."):
            try:
                engine  = get_engine()
                t0      = time.time()
                result  = engine.query(prompt)
                elapsed = time.time() - t0

                if result.retrieval_mode == "blocked":
                    st.error(result.answer)
                    response_msg = result.answer
                    sources = []
                else:
                    st.markdown(result.answer)
                    if result.pii_found:
                        st.warning(f"⚠️ PII detected and redacted in query: `{', '.join(result.pii_found)}`")

                    st.caption(
                        f"⚡ {elapsed:.1f}s · Mode: `{result.retrieval_mode}` · "
                        f"Sources: `{len(result.sources)}` chunks"
                    )
                    response_msg = result.answer
                    sources = result.sources

                    with st.expander(f"📄 Sources ({len(sources)} chunks retrieved)"):
                        for src in sources:
                            st.markdown(f"**{src.get('source_path','Unknown')}** — score: `{src.get('score',0):.3f}`")
                            st.code(src.get("chunk_text","")[:300] + "...", language=None)

            except Exception as e:
                response_msg = f"❌ Error: {str(e)}"
                sources = []
                st.error(response_msg)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_msg,
        "sources": sources
    })

# ── Architecture diagram tab ──────────────────────────────────────────────────
st.divider()
with st.expander("🏗️ Full Architecture Diagram"):
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                     DATA SOURCES                                │
    │  SharePoint ──┐                                                 │
    │  SQL / DB  ───┼──► Fabric Data Factory ──► OneLake (ADLS Gen2) │
    │  PDFs      ──┘                                                  │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
                         ┌───────────▼───────────┐
                         │  BRONZE (Delta Table) │
                         │  Raw docs + PII scan  │
                         └───────────┬───────────┘
                                     │ PySpark (Fabric Notebook)
                         ┌───────────▼───────────┐
                         │  SILVER (Delta Table) │
                         │  Semantic chunks      │
                         │  + 1536-dim vectors   │
                         └───────────┬───────────┘
                                     │
                         ┌───────────▼───────────┐
                         │  GOLD (AI Search)     │
                         │  Hybrid index         │
                         │  BM25 + HNSW + RRF    │
                         └───────────┬───────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │        APP LAYER                │
                    │  FastAPI /query endpoint        │
                    │  Streamlit UI (this page)       │
                    │  PII guardrail → GPT-4o answer  │
                    └─────────────────────────────────┘
    ```
    """)

st.markdown("---")
st.caption("Built by [Your Name] · Nexus-Fabric · [GitHub](https://github.com/YOUR_USERNAME/nexus-fabric)")
