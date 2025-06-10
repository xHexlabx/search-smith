# scripts/main.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# กำหนดค่าคงที่ให้ตรงกับตอน setup
DB_PATH = "chroma_db_langchain"
MODEL_NAME = "Qwen/Qwen3-Embedding-8B" # หรือ 'intfloat/multilingual-e5-large'

def main():
    # 1. Load Embeddings Modelz
    print("🤖 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # 2. Load Existing Vector Store
    # โหลด DB ที่เราสร้างและบันทึกไว้จากสคริปต์ setup
    print(f"📂 Loading vector store from: {DB_PATH}")
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    print("\n✅ Ready to search!")

    while True:
        keyword = input("\nEnter your search keyword (or 'q' to quit): ")
        if keyword.lower() == 'q':
            break

        if not keyword.strip():
            continue

        print(f"\n🔍 Searching for '{keyword}'...")

        # 3. Perform Search
        # ใช้ similarity_search_with_score เพื่อเอาผลลัพธ์พร้อมคะแนน
        # ผลลัพธ์ที่ได้จะเป็น list ของ (Document, score)
        results_with_scores = vector_store.similarity_search_with_score(
            keyword,
            k=3 # จำนวนผลลัพธ์ที่ต้องการ
        )

        if not results_with_scores:
            print("No matching results found.")
        else:
            print(f"Found {len(results_with_scores)} results:\n")
            for i, (doc, score) in enumerate(results_with_scores):
                print(f"--- Result {i+1} ---")
                # score ที่ได้จาก LangChain สำหรับ Chroma จะเป็น cosine distance
                # ค่ายิ่งน้อยยิ่งเหมือน
                print(f"✅ File: {doc.metadata.get('source', 'N/A')}")
                print(f"📊 Score (Distance): {score:.4f}")
        print("-" * 40 + "\n")

if __name__ == "__main__":
    main()
