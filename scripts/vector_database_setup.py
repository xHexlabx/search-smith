# scripts/vector_database_setup.py

import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# กำหนดค่าคงที่
PDF_DIR = "pdfs"
DB_PATH = "chroma_db_langchain" # สร้าง DB ใหม่สำหรับ LangChain
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B" # หรือ 'intfloat/multilingual-e5-large' สำหรับไทย

def setup():
    print("🚀 Starting vector database setup...")

    # 1. Load Documents (โหลด PDF ทั้งหมดในโฟลเดอร์)
    # ใช้ PyPDFLoader สำหรับแต่ละไฟล์ PDF ใน Directory
    loader = DirectoryLoader(
        PDF_DIR,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    if not documents:
        print("No PDF documents found. Exiting.")
        return

    print(f"✅ Loaded {len(documents)} document(s).")

    # 2. Split Documents into Chunks
    # RecursiveCharacterTextSplitter เป็นวิธีที่ฉลาดกว่าในการตัดคำ
    # มันพยายามรักษาประโยคและย่อหน้าไว้ด้วยกัน
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📄 Split documents into {len(chunks)} chunks.")

    # 3. Create Embeddings Model
    # LangChain มี wrapper สำหรับ sentence-transformers คือ HuggingFaceEmbeddings
    print(f"🤖 Loading embedding model: {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # 4. Create and Persist Vector Store (ChromaDB)
    # .from_documents คือคำสั่งมหัศจรรย์ของ LangChain
    # มันจะทำการสร้าง embedding และเก็บลง DB ให้เราในขั้นตอนเดียว
    print(f"📦 Creating and persisting vector store at: {DB_PATH}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("\n✅ Vector database setup complete!")

if __name__ == "__main__":
    setup()
