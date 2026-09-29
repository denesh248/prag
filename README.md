# 🚀 Prag: Production-Grade Hybrid RAG System

An advanced, production-ready Retrieval-Augmented Generation (RAG) system engineered for high-accuracy document intelligence, zero hallucinations, and secure enterprise question answering.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green?style=flat-square)
![Ragas](https://img.shields.io/badge/Ragas-Evaluation-orange?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-purple?style=flat-square)

---

## 🌟 Key Highlights

- **🔍 Hybrid Retrieval Engine**: Combines sparse lexical search (**BM25**) with dense semantic vector search (**ChromaDB** with `sentence-transformers/all-MiniLM-L6-v2`) via an ensemble retriever to maximize recall across both exact keywords and contextual meaning.
- **⚡ Cross-Encoder Reranking**: Re-ranks top candidate chunks using `cross-encoder/ms-marco-MiniLM-L-6-v2` to prioritize the most relevant context before passing it to the generator.
- **🎯 Real-Time Ragas Evaluation**: Live evaluation of **Faithfulness and Groundedness** on every answer to ensure all factual claims are directly supported by source chunks with 0 hallucinations.
- **🛡️ Enterprise Guardrails**: Two-tier security guardrails (`LLM Guard`) scanning input prompts for jailbreak/injection attacks and output responses for sensitive data leaks.
- **⚡ High-Throughput Generator**: Ultra-fast LLM inference powered by Groq (`openai/gpt-oss-120b`).

---

## 🏗️ Architecture Pipeline

```text
User Question
      │
      ▼
┌───────────────────────────┐
│ 🛡️ 1. Input Guardrail     │  ──> Scans for prompt injections / jailbreaks
└─────────────┬─────────────┘
              │ (Sanitized Query)
              ▼
┌───────────────────────────┐
│ 🔍 2. Hybrid Retrieval    │  ──> BM25 (Lexical) + ChromaDB (Semantic Dense)
└─────────────┬─────────────┘
              │ (Top candidate chunks)
              ▼
┌───────────────────────────┐
│ 🎯 3. Cross-Encoder       │  ──> ms-marco-MiniLM-L-6-v2 re-scores chunks
└─────────────┬─────────────┘
              │ (Top 5 Context Chunks)
              ▼
┌───────────────────────────┐
│ ⚡ 4. LLM Generator       │  ──> Grounded Generation via Groq (temp=0.2)
└─────────────┬─────────────┘
              │ (Raw Answer)
              ▼
┌───────────────────────────┐
│ 🛡️ 5. Output Guardrail    │  ──> Scans for sensitive leaks / hallucination
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ 📊 6. Ragas Evaluation    │  ──> Computes Faithfulness & Groundedness %
└─────────────┬─────────────┘
              │
              ▼
Final Verified Answer + Citations + Evaluation Score
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Clean, modern chat interface with chunk citation cards |
| **Orchestration** | LangChain | Pipeline assembly and retriever chaining |
| **Dense Search** | ChromaDB + HuggingFace | `all-MiniLM-L6-v2` semantic sentence embeddings |
| **Lexical Search** | Rank-BM25 | Exact keyword token frequency matching |
| **Reranker** | Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM Inference** | Groq (`openai/gpt-oss-120b`) | Ultra-fast token generation |
| **Evaluation** | Ragas | Groundedness & Faithfulness metric calculation |
| **Guardrails** | LLM Guard | Prompt injection and sensitive output scanner |

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/denesh248/rag.git
cd rag
```

### 2. Set up environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-120b
```

### 4. Run the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a **New App**.
3. Select your repository `denesh248/rag` and branch `main`.
4. Set main file path: `app.py`.
5. Under **Advanced Settings $\rightarrow$ Secrets**, enter:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   LLM_PROVIDER = "groq"
   GROQ_MODEL = "openai/gpt-oss-120b"
   ```
6. Click **Deploy**!
