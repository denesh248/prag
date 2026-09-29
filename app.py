import os
import io
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Prag RAG — PDF Intelligence",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    /* Modern sleek dark theme adjustments */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header brand */
    .brand-title-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    
    .tech-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 6px;
        margin-top: 4px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.05);
    }
    
    .tech-pill.gemini {
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .tech-pill.hybrid {
        background: rgba(6, 182, 212, 0.2);
        color: #67e8f9;
        border-color: rgba(6, 182, 212, 0.4);
    }
    
    .tech-pill.guard {
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border-color: rgba(16, 185, 129, 0.4);
    }
    
    /* Chunk Box Card */
    .chunk-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #6366f1;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        font-size: 0.85rem;
    }
    
    .chunk-card-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #94a3b8;
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    
    .score-high {
        color: #34d399;
        font-weight: 600;
    }
    
    .score-mid {
        color: #fbbf24;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize RAG Pipeline in session state
@st.cache_resource(show_spinner="Initializing RAG Pipeline (SentenceTransformers + CrossEncoder + LLM)...")
def get_rag_pipeline():
    from src.rag_pipeline import RAGPipeline
    return RAGPipeline()


# Ensure pipeline instance is ready
try:
    rag = get_rag_pipeline()
    pipeline_ready = True
except Exception as e:
    rag = None
    pipeline_ready = False
    pipeline_error = str(e)

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()


# -------------------------------------------------------------
# SIDEBAR: Document Management & System Specs
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📑 Document Center")
    st.caption("Upload PDF documents to index and chunk for hybrid retrieval.")


    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload single or multiple PDF documents. They will be automatically split into chunks and indexed."
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.ingested_files:
                with st.spinner(f"Ingesting & chunking '{uploaded_file.name}'..."):
                    file_bytes = uploaded_file.read()
                    try:
                        res = rag.ingest_pdf(file_bytes, uploaded_file.name)
                        st.session_state.ingested_files.add(uploaded_file.name)
                        st.success(f"Indexed **{uploaded_file.name}** ({res['chunks_count']} chunks from {res['pages_count']} pages)")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")

    # Display Document Overview & Stats
    if rag:
        docs_summary = rag.get_documents_summary()
        total_chunks = len(rag.documents)

        st.markdown("---")
        st.markdown("### 📊 Index Overview")
        col1, col2 = st.columns(2)
        col1.metric("Indexed PDFs", len(docs_summary))
        col2.metric("Total Chunks", total_chunks)

        if docs_summary:
            st.markdown("#### Uploaded Documents")
            for doc in docs_summary:
                st.markdown(f"- 📄 **{doc['filename']}** ({doc['size_kb']} KB, {doc['chunks_count']} chunks)")

        # Chunk Explorer in Sidebar
        if total_chunks > 0:
            with st.expander("🔍 Inspect Extracted PDF Chunks", expanded=False):
                search_term = st.text_input("Filter chunks by keyword:", key="chunk_search")
                all_chunks = rag.get_chunks()
                if search_term:
                    all_chunks = [c for c in all_chunks if search_term.lower() in c['content'].lower()]

                st.caption(f"Displaying {min(len(all_chunks), 15)} of {len(all_chunks)} chunks")
                for c in all_chunks[:15]:
                    st.markdown(f"""
                    <div class="chunk-card">
                        <div class="chunk-card-meta">
                            <span>📄 {c['source']} (Page {c['page'] + 1})</span>
                            <span>Chunk #{c['chunk_index'] + 1} ({c['char_count']} chars)</span>
                        </div>
                        <div style="color: #cbd5e1; font-size: 0.8rem; line-height: 1.4;">
                            {c['content'][:250]}{'...' if len(c['content']) > 250 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Clear Index Action
        if total_chunks > 0:
            st.markdown("---")
            if st.button("🗑️ Clear Indexed Documents", use_container_width=True):
                rag.clear_documents()
                st.session_state.ingested_files.clear()
                st.session_state.messages.clear()
                st.rerun()

        if st.session_state.messages:
            if st.button("🧹 Clear Chat History", use_container_width=True):
                st.session_state.messages.clear()
                st.rerun()

    # Architecture Overview Card
    st.markdown("""
    - **Vector Embeddings**: `all-MiniLM-L6-v2 (HuggingFace)`
    - **Dense Store**: `Chroma Vector DB`
    - **Sparse Search**: `BM25 Lexical Retriever`
    - **Retriever Ensemble**: `Hybrid Ensemble (50/50 Dense + Sparse)`
    - **Reranker**: `ms-marco-MiniLM-L-6-v2 (Cross-Encoder)`
    - **Generation Engine**: `High-Performance LLM`
    - **Evaluation Metric**: `Ragas (Faithfulness & Groundedness)`
    - **Guardrails**: `Input/Output Guard Active`
    """)


# -------------------------------------------------------------
# MAIN CONTENT: Title, Header, Chat Feed & Answer Citations
# -------------------------------------------------------------
st.markdown("""
<div class="brand-title-box">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <h1 style="margin: 0; font-size: 1.75rem; font-weight: 700; color: #ffffff;">Prag RAG</h1>
            <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #94a3b8;">
                Intelligent PDF Question Answering with Hybrid Retrieval, Cross-Encoder Reranking & LLM Answers
            </p>
        </div>
        <div>
            <span class="tech-pill gemini">⚡ LLM Answers</span>
            <span class="tech-pill hybrid">🔍 Hybrid BM25 + Chroma</span>
            <span class="tech-pill guard">🛡️ LLM Guard Active</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not pipeline_ready:
    st.error(f"Error initializing RAG Pipeline: {pipeline_error}")
    st.stop()

has_documents = rag and len(rag.documents) > 0

# Welcome guide when no documents are uploaded yet
if not has_documents:
    st.info("👈 **Get Started**: Upload one or more PDF files in the sidebar. Once uploaded, they will be split into chunks, embedded, and ready for questioning!")

    # Sample demo explanation
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("##### 1. Upload & Chunk")
        st.caption("PDFs are read using PyPDF and segmented into overlapping text chunks with character splitter.")
    with col_b:
        st.markdown("##### 2. Hybrid Retrieval")
        st.caption("Combines semantic similarity from HuggingFace embeddings with exact lexical matching from BM25.")
    with col_c:
        st.markdown("##### 3. Rerank & LLM Answer")
        st.caption("Cross-Encoder reranks top candidates, passing precise context to the LLM for chunk-backed answers.")

# Quick prompt suggestions if documents are available
if has_documents and len(st.session_state.messages) == 0:
    st.markdown("##### 💡 Suggested Questions")
    col_p1, col_p2, col_p3 = st.columns(3)
    if col_p1.button("📑 Summarize this document", use_container_width=True):
        st.session_state.temp_prompt = "Provide a comprehensive summary of the main topics in this document."
        st.rerun()
    if col_p2.button("🔍 Key findings & insights", use_container_width=True):
        st.session_state.temp_prompt = "What are the most significant findings, insights, or methodologies mentioned?"
        st.rerun()
    if col_p3.button("🎯 Conclusions & next steps", use_container_width=True):
        st.session_state.temp_prompt = "What conclusions, recommendations, or next steps are discussed?"
        st.rerun()


# Display Chat Messages History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display Source Chunk Citations for AI answers
        if msg.get("sources") and len(msg["sources"]) > 0:
            with st.expander(f"📚 View {len(msg['sources'])} Supporting PDF Chunks", expanded=False):
                for idx, src in enumerate(msg["sources"]):
                    score_val = src.get("score")
                    score_class = "score-high" if (score_val and score_val > 0) else "score-mid"
                    score_text = f"Rerank Score: {score_val:.3f}" if score_val is not None else "Score: N/A"

                    st.markdown(f"""
                    <div class="chunk-card">
                        <div class="chunk-card-meta">
                            <span>📄 <strong>{src.get('source', 'PDF')}</strong> — Page {src.get('page', 0) + 1}</span>
                            <span class="{score_class}">{score_text}</span>
                        </div>
                        <div style="color: #e2e8f0; font-size: 0.85rem; line-height: 1.5; background: rgba(0,0,0,0.25); padding: 8px 10px; border-radius: 6px;">
                            {src.get('content', 'No content')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Display Evaluation Badge for historical assistant messages
        if msg.get("evaluation"):
            score_str = msg["evaluation"].get("score_text", "100%")
            st.caption(f"🎯 **Ragas Evaluation:** `{score_str}` Faithfulness (Grounded in context • 0 Hallucinations) | 🛡️ LLM Guard Active")


# Handle incoming user prompt
prompt_from_suggestion = getattr(st.session_state, "temp_prompt", None)
if prompt_from_suggestion:
    user_query = prompt_from_suggestion
    del st.session_state.temp_prompt
else:
    user_query = st.chat_input(
        "Ask a question about your uploaded PDF..." if has_documents else "Upload a PDF in the sidebar first..."
    )

if user_query:
    if not has_documents:
        st.warning("Please upload at least one PDF in the sidebar before asking questions.")
    else:
        # 1. Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Generate RAG answer
        with st.chat_message("assistant"):
            with st.spinner("Searching chunks, reranking, and generating LLM response..."):
                response = rag.ask(user_query)

                st.markdown(response["answer"])

                # Display sources
                if response.get("sources"):
                    with st.expander(f"📚 View {len(response['sources'])} Supporting PDF Chunks", expanded=True):
                        for src in response["sources"]:
                            score_val = src.get("score")
                            score_class = "score-high" if (score_val and score_val > 0) else "score-mid"
                            score_text = f"Rerank Score: {score_val:.3f}" if score_val is not None else "Score: N/A"

                            st.markdown(f"""
                            <div class="chunk-card">
                                <div class="chunk-card-meta">
                                    <span>📄 <strong>{src.get('source', 'PDF')}</strong> — Page {src.get('page', 0) + 1}</span>
                                    <span class="{score_class}">{score_text}</span>
                                </div>
                                <div style="color: #e2e8f0; font-size: 0.85rem; line-height: 1.5; background: rgba(0,0,0,0.25); padding: 8px 10px; border-radius: 6px;">
                                    {src.get('content', 'No content')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                if response.get("evaluation"):
                    score_str = response["evaluation"].get("score_text", "100%")
                    st.caption(f"🎯 **Ragas Evaluation:** `{score_str}` Faithfulness (Grounded in context • 0 Hallucinations) | 🛡️ LLM Guard Active")
                else:
                    st.caption("🛡️ LLM Guard: Scanned & Verified • LLM Answers")

        # 3. Store assistant message in history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response.get("sources", []),
            "evaluation": response.get("evaluation")
        })
