# search_smith/data_loader.py
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader

def load_markdown_documents(directory: Path) -> List[Document]:
    """
    โหลดเอกสาร Markdown ทั้งหมดจากโฟลเดอร์ที่กำหนด
    Loads all Markdown documents from a specified directory.
    
    Args:
        directory (Path): The path to the directory containing .md files.

    Returns:
        List[Document]: A list of loaded documents.
    """
    print(f"Loading documents from: {directory}")
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return []
        
    loader = DirectoryLoader(str(directory), glob="**/*.md", show_progress=True)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents
