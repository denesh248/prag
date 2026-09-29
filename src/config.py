import os
from dotenv import load_dotenv

load_dotenv()


def get_secret(key, default=None):
    """
    Reads configuration from environment variables (.env)
    or Streamlit Cloud secrets (st.secrets).
    """
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default


GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

LLM_PROVIDER = get_secret("LLM_PROVIDER", "groq")
GROQ_MODEL = get_secret("GROQ_MODEL", "openai/gpt-oss-120b")

DATA_PATH = "data/documents"
CHROMA_PATH = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "gemini-3.8-flash"