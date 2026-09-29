from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            RERANKER_MODEL
        )

    def rerank(
        self,
        query,
        documents,
        top_k=5
    ):

        pairs = [
            (query, document.page_content)
            for document in documents
        ]

        scores = self.model.predict(pairs)

        for document, score in zip(
            documents,
            scores
        ):
            document.metadata["rerank_score"] = float(score)

        documents.sort(
            key=lambda document:
            document.metadata["rerank_score"],
            reverse=True
        )

        return documents[:top_k]