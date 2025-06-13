# search_smith/vector_store_handler.py
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_huggingface_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """
    สร้าง instance ของ embedding model จาก Hugging Face
    Instantiates the embedding model from Hugging Face.
    """
    print(f"Initializing Hugging Face embedding model: {model_name}")
    print("Note: The first time, this will download the model, which may take time.")
    model_kwargs = {'device': 'cpu'} # Change to 'cuda' if GPU is available
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def build_and_persist_vector_store(docs: List[Document], embeddings: HuggingFaceEmbeddings, store_path: Path):
    """
    สร้างและบันทึก Vector Database โดยใช้ ChromaDB
    Creates and saves the Vector Database using ChromaDB.
    """
    print("Building and persisting the vector database...")
    store_path.mkdir(parents=True, exist_ok=True)
    
    vector_db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory=str(store_path)
    )
    
    print("-" * 50)
    print("Vector database created successfully!")
    print(f"Number of vectors in DB: {vector_db._collection.count()}")
    print(f"Database saved at: {store_path}")
    print("-" * 50)

def load_existing_vector_store(store_path: Path, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    โหลด Vector Store ที่มีอยู่แล้วจาก disk
    Loads an existing vector store from disk.
    """
    if not store_path.exists() or not os.listdir(str(store_path)):
        print(f"Error: Vector store not found at '{store_path}'")
        print("Please run 'vector_database_setup.py' first to create the database.")
        return None
        
    print(f"Loading vector database from: {store_path}")
    db = Chroma(
        persist_directory=str(store_path),
        embedding_function=embeddings
    )
    return db

def perform_similarity_search(db: Chroma, query: str, k: int) -> List[Document]:
    """
    ดำเนินการค้นหาความคล้ายคลึงบน Vector Store
    Performs a similarity search on the vector store.
    """
    if not db:
        print("Vector database is not loaded.")
        return []
    
    return db.similarity_search(query, k=k)
