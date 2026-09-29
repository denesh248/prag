import os
from src.config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    GOOGLE_API_KEY,
    LLM_MODEL
)


class LLMGenerator:
    """
    RAG Generation Module.
    Initializes the LLM (Groq or Google) to generate grounded answers from retrieved context.
    """

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER)
        self._init_model()

    def _init_model(self):
        if self.provider == "groq":
            from langchain_groq import ChatGroq
            api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
            model = os.getenv("GROQ_MODEL", GROQ_MODEL)
            self.llm = ChatGroq(
                model_name=model,
                temperature=0.2,
                groq_api_key=api_key
            ) if api_key else None
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
            model = os.getenv("LLM_MODEL", LLM_MODEL)
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=0.2,
                google_api_key=api_key
            ) if api_key else None

    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""You are a helpful RAG assistant.

Answer the question using only the provided context.
If the answer cannot be found in the context, say:
"I could not find this information in the documents."

Context:
{context}

Question:
{query}

Answer:"""
        if not self.llm:
            return "LLM is not configured. Please check your API key."

        response = self.llm.invoke(prompt)
        content = response.content

        # Extract text from response content
        if isinstance(content, list):
            return "\n".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            ).strip()
        return str(content).strip()