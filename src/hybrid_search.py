try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers import EnsembleRetriever


def create_hybrid_retriever(
    bm25_retriever,
    vector_retriever
):

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_retriever
        ],
        weights=[
            0.5,
            0.5
        ]
    )

    return hybrid_retriever