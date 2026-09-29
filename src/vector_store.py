from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_PATH, EMBEDDING_MODEL


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def create_vector_store(documents):

    embeddings = create_embeddings()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vector_store


def load_vector_store():

    embeddings = create_embeddings()

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return vector_store