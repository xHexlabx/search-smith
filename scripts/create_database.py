# file: create_database.py
import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def main():
    """
    ฟังก์ชันหลักสำหรับสร้าง Vector Database โดยใช้ Local Model Qwen/Qwen3-Embedding-0.6B
    """
    # --- 1. การตั้งค่าเริ่มต้น ---
    print("กำลังโหลด Environment Variables...")
    load_dotenv()
    print("โหมด Local Embedding: ไม่จำเป็นต้องใช้ API Token")

    # --- 2. โหลดและประมวลผลข้อมูล ---
    DOCUMENTS_JSON_PATH = 'databases/documents/documents.json'
    print(f"กำลังโหลดไฟล์ '{DOCUMENTS_JSON_PATH}'...")
    try:
        with open(DOCUMENTS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ไม่พบไฟล์ '{DOCUMENTS_JSON_PATH}'")
        return

    print("กำลังประมวลผลข้อมูลเพื่อเพิ่มน้ำหนักให้ Tags...")
    documents_for_db = []
    for item in data:
        metadata = item['metadata']
        original_content = item['page_content']
        tags_as_string = ", ".join(metadata['tags']) if isinstance(metadata.get('tags'), list) else metadata.get('tags', '')
        metadata['tags'] = tags_as_string
        enhanced_content = f"TAGS: {tags_as_string}\n---\n{original_content}"
        documents_for_db.append(Document(page_content=enhanced_content, metadata=metadata))
    print(f"✅ ประมวลผลเอกสารเรียบร้อยแล้ว {len(documents_for_db)} ข้อ")

    # --- 3. สร้าง Vector Store ด้วย Local Qwen Model ---
    
    # !ปรับปรุง: เปลี่ยนไปใช้โมเดล Qwen ตามที่คุณต้องการ
    print("กำลังโหลด Local Embedding Model (Qwen)...")
    print("(ครั้งแรกอาจใช้เวลาดาวน์โหลดโมเดลนานพอสมควร)")
    
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    model_kwargs = {'device': 'cpu'}
    # การ normalize embeddings เป็นขั้นตอนที่แนะนำสำหรับงานค้นหาความเหมือน
    encode_kwargs = {'normalize_embeddings': True} 
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print(f"✅ โหลดโมเดล '{model_name}' สำเร็จ")

        # สร้าง DB ใหม่สำหรับโมเดล Qwen โดยเฉพาะ
        VECTOR_STORE_PATH = "databases/chroma_db_problems_qwen"
        print(f"กำลังสร้าง Vector Store และบันทึกไปยัง '{VECTOR_STORE_PATH}'...")
        
        Chroma.from_documents(
            documents=documents_for_db,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_PATH
        )
        print("\n🎉 สร้างและบันทึก Vector Store ด้วย Local Model (Qwen) สำเร็จ!")

    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาดระหว่างการสร้าง Vector Store: {e}")
        print("💡 คำแนะนำ: ตรวจสอบว่าติดตั้ง library ครบถ้วนและเป็นเวอร์ชันล่าสุด")

if __name__ == "__main__":
    main()