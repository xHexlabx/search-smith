# search_smith/data_loader.py
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader


def load_text_documents(directory: Path) -> List[Document]:
    """
    โหลดเอกสาร Text ทั้งหมดจากโฟลเดอร์ที่กำหนด
    Loads all Text documents from a specified directory.

    Args:
        directory (Path): The path to the directory containing .txt files.

    Returns:
        List[Document]: A list of loaded documents.
    """
    print(f"Loading documents from: {directory}")
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return []

    loader = DirectoryLoader(
        str(directory),
        glob="**/*.txt",
        show_progress=True
    )

    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents
