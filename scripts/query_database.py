# file: query_database.py
import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def recommend_problems(retriever, query: str):
    """
    ฟังก์ชันสำหรับรับคำค้นหาและแสดงโจทย์ที่เกี่ยวข้อง
    """
    print(f"\n🔎 กำลังค้นหาโจทย์สำหรับ query: '{query}'")
    relevant_docs = retriever.invoke(query)
    
    if not relevant_docs:
        print("ไม่พบโจทย์ที่ตรงกับคำค้นหาของคุณ")
        return

    print(f"\n✨ พบโจทย์ที่แนะนำ {len(relevant_docs)} ข้อ:\n")
    for i, doc in enumerate(relevant_docs):
        problem_name = doc.metadata.get('problem_name', 'N/A')
        tags = doc.metadata.get('tags', 'N/A')
        full_content = doc.page_content
        display_content = full_content.split("\n---\n", 1)[1] if "\n---\n" in full_content else full_content
        
        print(f"--- ข้อที่ {i+1}: {problem_name} ---")
        print(f"  Tags: {tags}")
        print(f"  เนื้อหา: {display_content.strip()[:200]}...")
        print("-" * (len(problem_name) + 12))

def main():
    """
    ฟังก์ชันหลักสำหรับโหลด Database และรอรับคำสั่งจากผู้ใช้
    """
    # --- 1. การตั้งค่าและโหลด Database ---
    load_dotenv()
    
    # !ปรับปรุง: ใช้ Path ของฐานข้อมูลที่สร้างจากโมเดล Qwen
    VECTOR_STORE_PATH = "databases/chroma_db_problems_qwen"
    
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"❌ ไม่พบฐานข้อมูลที่ '{VECTOR_STORE_PATH}'")
        print("กรุณารันไฟล์ 'create_database.py' ก่อนเพื่อสร้างฐานข้อมูล")
        return

    print("กำลังโหลด Vector Store และ Local Embedding Model (Qwen)...")
    try:
        # !ปรับปรุง: ใช้โมเดล Qwen ตัวเดียวกัน
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        print("✅ โหลดสำเร็จ!")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดระหว่างการโหลด Vector Store: {e}")
        return

    # --- 2. ส่วนของการรับ Input จากผู้ใช้ ---
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        recommend_problems(retriever, user_query)
    else:
        print("\nเข้าสู่โหมดค้นหาโจทย์ (พิมพ์ 'exit' หรือ 'quit' เพื่อออก)")
        while True:
            user_query = input("\nกรอกคำค้นหาของคุณ: ")
            if user_query.lower() in ['exit', 'quit']:
                break
            recommend_problems(retriever, user_query)

if __name__ == "__main__":
    main()