from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def create_compression_retriever(
    retriever,
    llm
):

    compressor = LLMChainExtractor.from_llm(
        llm
    )

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=compressor
    )

    return compression_retriever