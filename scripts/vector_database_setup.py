# scripts/vector_database_setup.py

import time
from pathlib import Path

# --- นำเข้าฟังก์ชันจากโมดูลต่างๆ ---
# --- Import functions from various modules ---
from search_smith.data_loader import load_markdown_documents
from search_smith.text_splitter import split_docs_into_chunks
from search_smith.vector_store_handler import create_huggingface_embeddings, build_and_persist_vector_store

# --- การตั้งค่าหลัก (Main Configuration) ---

# 1. กำหนด Path หลักของโปรเจกต์
BASE_DIR = Path(__file__).resolve().parent.parent
TEXTS_DIR = BASE_DIR / "databases" / "solutions"
VECTOR_STORE_DIR = BASE_DIR / "databases" / "vector_store"

# 2. กำหนดค่าสำหรับการแบ่งข้อความ
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 3. กำหนดชื่อโมเดล Embeddings
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

def main():
    """
    ฟังก์ชันหลักสำหรับรันกระบวนการสร้าง Vector Database โดยเรียกใช้โมดูลต่างๆ
    Main function to run the Vector Database creation process by calling various modules.
    """
    start_time = time.time()
    
    # Step 1: โหลดเอกสาร .md จากโมดูล data_loader
    documents = load_markdown_documents(TEXTS_DIR)
    if not documents:
        print("No documents to process. Exiting.")
        return
        
    # Step 2: แบ่งเอกสารเป็น Chunks จากโมดูล text_splitter
    docs_splitted = split_docs_into_chunks(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Step 3: สร้าง Embeddings จากโมดูล vector_store_handler
    embeddings = create_huggingface_embeddings(EMBED_MODEL_NAME)
    
    # Step 4: สร้างและบันทึก Vector Database จากโมดูล vector_store_handler
    build_and_persist_vector_store(docs_splitted, embeddings, VECTOR_STORE_DIR)
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
