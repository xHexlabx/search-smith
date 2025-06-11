# scripts/main.py

import os
import time
from pathlib import Path

# --- นำเข้าฟังก์ชันจากโมดูล vector_store_handler ---
# --- Import functions from the vector_store_handler module ---
from search_smith.vector_store_handler import (
    create_huggingface_embeddings,
    load_existing_vector_store,
    perform_similarity_search,
)

# --- การตั้งค่าหลัก (Main Configuration) ---

# 1. กำหนด Path หลัก
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = BASE_DIR / "databases" / "vector_store"

# 2. กำหนดชื่อโมเดล Embeddings (ต้องเป็นตัวเดียวกับตอนที่สร้าง DB)
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

# 3. กำหนดจำนวนผลลัพธ์
K_RESULTS = 5

def main():
    """
    ฟังก์ชันหลักสำหรับรันระบบแนะนำโจทย์โดยเรียกใช้โมดูลต่างๆ
    Main function to run the recommendation system by calling various modules.
    """
    # Step 1: สร้าง Embedding Model
    print("Initializing the recommendation system...")
    try:
        embeddings = create_huggingface_embeddings(EMBED_MODEL_NAME)
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return

    # Step 2: โหลด Vector Database
    db = load_existing_vector_store(VECTOR_STORE_DIR, embeddings)
    if not db:
        return # ออกจากโปรแกรมถ้าโหลด DB ไม่สำเร็จ

    print("System is ready. You can start asking now.")
    print("-" * 50)

    # Step 3: Loop รับ Input และค้นหา
    while True:
        query = input("โจทย์ที่คุณกำลังเจอ (หรือพิมพ์ 'exit' เพื่อจบการทำงาน): \n> ")
        if query.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break
        if not query.strip():
            print("Please enter a description.")
            continue

        print("\nSearching for similar problems...")
        start_time = time.time()
        
        similar_docs = perform_similarity_search(db, query, k=K_RESULTS)
        
        end_time = time.time()
        print(f"Search completed in {end_time - start_time:.2f} seconds.")

        # Step 4: แสดงผลลัพธ์
        if not similar_docs:
            print("Sorry, no similar problems found.")
        else:
            recommended_problems = {Path(doc.metadata.get("source", "Unknown")).name for doc in similar_docs}
            
            print(f"\n--- โจทย์ที่ใกล้เคียงที่สุด {len(recommended_problems)} ข้อ ---")
            for i, problem_name in enumerate(recommended_problems):
                print(f"{i+1}. {problem_name}")
        
        print("-" * 50)


if __name__ == "__main__":
    main()
