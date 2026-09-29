import os
from pathlib import Path
import shutil

from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader

from src.config import DATA_PATH, CHROMA_PATH
from src.document_loader import (
    load_documents,
    split_documents
)
from src.vector_store import (
    create_vector_store
)
from src.hybrid_search import (
    create_hybrid_retriever
)
from src.reranker import Reranker
from src.generator import LLMGenerator
from src.evaluator import RAGEvaluator
try:
    from src.guardrails import (
        check_input,
        check_output
    )
except ImportError:
    from src.guardraills import (
        check_input,
        check_output
    )


class RAGPipeline:

    def __init__(self):
        Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
        self.documents = []
        self.vector_store = None
        self.bm25_retriever = None
        self.hybrid_retriever = None

        print("Initializing Reranker and Generator...")
        self.reranker = Reranker()
        self.generator = LLMGenerator()
        self.evaluator = RAGEvaluator(llm=getattr(self.generator, "llm", None))

        # Check if existing documents exist in DATA_PATH
        existing_pdfs = list(Path(DATA_PATH).glob("*.pdf"))
        if existing_pdfs:
            print(f"Found {len(existing_pdfs)} existing PDF(s) in {DATA_PATH}. Indexing...")
            self.reindex_from_directory()
        else:
            print("No documents found in DATA_PATH yet. Ready for dynamic uploads.")

    def reindex_from_directory(self):
        raw_documents = load_documents(DATA_PATH)
        if not raw_documents:
            self.documents = []
            self.vector_store = None
            self.hybrid_retriever = None
            return

        self.documents = split_documents(raw_documents)
        print(f"Created {len(self.documents)} chunks from {len(raw_documents)} pages.")

        self.vector_store = create_vector_store(self.documents)
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 10}
        )

        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = 10
        self.bm25_retriever = bm25_retriever

        self.hybrid_retriever = create_hybrid_retriever(
            bm25_retriever,
            vector_retriever
        )
        print("Hybrid retriever and vector store ready.")

    def ingest_pdf(self, file_path_or_bytes, filename=None):
        """
        Accepts either a file path (str or Path) or (bytes, filename).
        Saves the file to DATA_PATH, splits into chunks, and updates the vector store + BM25 retriever.
        """
        Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

        if isinstance(file_path_or_bytes, (bytes, bytearray)):
            if not filename:
                raise ValueError("Filename must be provided when passing raw bytes.")
            target_path = Path(DATA_PATH) / filename
            with open(target_path, "wb") as f:
                f.write(file_path_or_bytes)
        else:
            source_path = Path(file_path_or_bytes)
            filename = source_path.name
            target_path = Path(DATA_PATH) / filename
            if source_path.resolve() != target_path.resolve():
                shutil.copy2(source_path, target_path)

        # Load pages with PyPDFLoader
        loader = PyPDFLoader(str(target_path))
        pages = loader.load()

        # Split into chunks
        new_chunks = split_documents(pages)

        # Normalize chunk metadata
        for i, chunk in enumerate(new_chunks):
            chunk.metadata["source"] = filename
            chunk.metadata["page"] = chunk.metadata.get("page", 0)
            chunk.metadata["chunk_index"] = i

        # Re-index all documents in DATA_PATH to maintain clean BM25 and Chroma state
        self.reindex_from_directory()

        return {
            "filename": filename,
            "pages_count": len(pages),
            "chunks_count": len(new_chunks),
            "total_chunks": len(self.documents)
        }

    def get_chunks(self, filename=None):
        """
        Returns all chunks with content and metadata for UI inspection.
        """
        chunks_list = []
        for i, doc in enumerate(self.documents):
            source_file = Path(doc.metadata.get("source", "")).name
            if filename and source_file.lower() != filename.lower():
                continue
            chunks_list.append({
                "chunk_index": doc.metadata.get("chunk_index", i),
                "page": doc.metadata.get("page", 0),
                "source": source_file,
                "content": doc.page_content,
                "char_count": len(doc.page_content)
            })
        return chunks_list

    def get_documents_summary(self):
        """
        Returns summary info for all PDFs in DATA_PATH.
        """
        pdf_files = list(Path(DATA_PATH).glob("*.pdf"))
        summary = []
        for p in pdf_files:
            file_chunks = [d for d in self.documents if Path(d.metadata.get("source", "")).name.lower() == p.name.lower()]
            size_kb = round(p.stat().st_size / 1024, 1)
            summary.append({
                "filename": p.name,
                "size_kb": size_kb,
                "chunks_count": len(file_chunks)
            })
        return summary

    def clear_documents(self):
        """
        Clears all stored PDFs and resets retrievers.
        """
        pdf_files = list(Path(DATA_PATH).glob("*.pdf"))
        for p in pdf_files:
            try:
                p.unlink()
            except Exception as e:
                print(f"Error removing {p}: {e}")

        # Reset chroma directory if exists
        if Path(CHROMA_PATH).exists():
            try:
                shutil.rmtree(CHROMA_PATH)
            except Exception as e:
                print(f"Error removing chroma_db: {e}")

        self.documents = []
        self.vector_store = None
        self.bm25_retriever = None
        self.hybrid_retriever = None

    def ask(self, query):
        if not self.documents or not self.hybrid_retriever:
            return {
                "answer": "No documents have been indexed yet. Please upload a PDF document first.",
                "sources": [],
                "safe": True,
                "evaluation": None
            }

        # -----------------------------
        # 1. Input Guardrail
        # -----------------------------
        input_result = check_input(query)

        if not input_result["is_safe"]:
            return {
                "answer": "Unsafe query detected by input guardrail.",
                "sources": [],
                "safe": False,
                "evaluation": None
            }

        safe_query = input_result["prompt"]

        # -----------------------------
        # 2. Hybrid Retrieval
        # -----------------------------
        documents = self.hybrid_retriever.invoke(safe_query)

        # -----------------------------
        # 3. Reranking
        # -----------------------------
        documents = self.reranker.rerank(
            safe_query,
            documents,
            top_k=5
        )

        # -----------------------------
        # 4. Create Context
        # -----------------------------
        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        # -----------------------------
        # 5. Generate Answer
        # -----------------------------
        answer = self.generator.generate_answer(
            safe_query,
            context
        )

        # -----------------------------
        # 6. Output Guardrail
        # -----------------------------
        output_result = check_output(
            safe_query,
            answer
        )

        if not output_result["is_safe"]:
            return {
                "answer": "Unsafe output detected by output guardrail.",
                "sources": [],
                "safe": False,
                "evaluation": None
            }

        final_answer = output_result["answer"]

        # -----------------------------
        # 7. Sources with Chunk Content
        # -----------------------------
        sources = [
            {
                "source": Path(doc.metadata.get("source", "Document")).name,
                "page": doc.metadata.get("page", 0),
                "score": doc.metadata.get("rerank_score", 0.0),
                "content": doc.page_content
            }
            for doc in documents
        ]

        # -----------------------------
        # 8. Ragas Evaluation (Faithfulness & Groundedness)
        # -----------------------------
        self.evaluator.llm = getattr(self.generator, "llm", None)
        evaluation = self.evaluator.evaluate(
            question=safe_query,
            answer=final_answer,
            context_chunks=documents
        )

        return {
            "answer": final_answer,
            "sources": sources,
            "safe": True,
            "evaluation": evaluation
        }