from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(data_path):
    documents = []

    for file_path in Path(data_path).glob("*.pdf"):
        loader = PyPDFLoader(str(file_path))
        documents.extend(loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)