# Search Smith

## Search Smith

Search Smith คือเครื่องมือค้นหาอัจฉริยะสำหรับเอกสาร ที่ใช้การค้นหาความคล้ายคลึงของเวกเตอร์ (Vector Similarity Search) เพื่อค้นหาข้อมูลที่เกี่ยวข้องในเอกสารของคุณได้อย่างรวดเร็วและแม่นยำ

### การติดตั้ง

1.  **Clone a repository:**
    ```bash
    git clone <URL ของ repository>
    cd search-smith
    ```
2.  **สร้างและ Activate Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # สำหรับ macOS/Linux
    venv\Scripts\activate  # สำหรับ Windows
    ```
3.  **ติดตั้ง Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### การตั้งค่า (Configuration)

โปรเจกต์นี้จำเป็นต้องใช้ Environment Variables ในการตั้งค่า API Token ต่างๆ

1.  **สร้างไฟล์ `.env`:**
    สร้างไฟล์ชื่อ `.env` ขึ้นมาใน root directory ของโปรเจกต์
2.  **เพิ่ม Tokens ในไฟล์ `.env`:**
    เปิดไฟล์ `.env` และเพิ่ม Token ที่จำเป็นดังนี้:
    ```
    HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
    ```
    (หมายเหตุ: จากการตรวจสอบไฟล์ `scripts/create_database.py` พบว่ามีการใช้งาน Local Embedding Model จึงไม่จำเป็นต้องใช้ API Token สำหรับส่วนนี้ แต่ในไฟล์ `search_smith/llm_handler.py` มีการเรียกใช้งาน `get_huggingface_llm` ซึ่งจำเป็นต้องใช้ `HF_TOKEN`)

### การใช้งาน (Usage)

#### 1. การสร้างฐานข้อมูล (Vector Database)

สคริปต์ `create_database.py` จะทำการประมวลผลเอกสารของคุณและสร้าง Vector Database เพื่อใช้ในการค้นหา

**การรันสคริปต์:**
```bash
python scripts/create_database.py
```
สคริปต์นี้จะ:
* โหลดข้อมูลจาก `databases/documents/documents.json`
* ประมวลผลและเพิ่มน้ำหนักให้กับ Tags
* ใช้ Local Embedding Model (`Qwen/Qwen3-Embedding-0.6B`) ในการสร้าง Vector Embeddings
* บันทึก Vector Store ลงใน `databases/chroma_db_problems_qwen`

#### 2. การค้นหาข้อมูลในฐานข้อมูล

สคริปต์ `query_database.py` ใช้สำหรับค้นหาข้อมูลจาก Vector Database ที่สร้างไว้

**การรันสคริปต์:**
* **แบบโต้ตอบ (Interactive Mode):**
    ```bash
    python scripts/query_database.py
    ```
    จากนั้นคุณสามารถพิมพ์คำค้นหาที่ต้องการได้
* **แบบส่ง Query ผ่าน Command-line:**
    ```bash
    python scripts/query_database.py "คำค้นหาของคุณ"
    ```
สคริปต์นี้จะแสดงผลโจทย์ที่เกี่ยวข้อง 5 ข้อพร้อม Tags และเนื้อหาบางส่วน
