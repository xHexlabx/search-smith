# search_smith/text_splitter.py
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs_into_chunks(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    แบ่งเอกสารเป็นส่วนย่อยๆ (chunks)
    Splits documents into smaller chunks.

    Args:
        documents (List[Document]): The list of documents to split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[Document]: A list of document chunks.
    """
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")
    return docs
