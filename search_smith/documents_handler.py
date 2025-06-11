# search_smith/documents_handler.py

import os
import asyncio
import configparser
from pathlib import Path
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from google import genai

def initialize_genai(config: configparser.ConfigParser) -> str:
    """
    ตั้งค่า Google API Key และดึงชื่อโมเดลจากไฟล์คอนฟิก
    Configures the Google API Key and retrieves the model name from the config file.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")
    
    genai.configure(api_key=api_key)
    
    # ดึงชื่อโมเดลที่จะใช้
    # Get the model name to be used
    model_name = config.get('GeminiParser', 'model_name', fallback='gemini-1.5-flash')
    print(f"Gemini API configured. Using model: {model_name}")
    return model_name


async def parse_and_save_worker(
    model: genai, 
    parsing_instruction: str,
    pdf_path: Path, 
    text_dir: Path, 
    semaphore: asyncio.Semaphore
):
    """
    Worker ที่อัปโหลดไฟล์ PDF ไปยัง File API, ประมวลผลด้วย Gemini, และบันทึกผลลัพธ์
    A worker that uploads a PDF file to the File API, processes it with Gemini, and saves the result.
    """
    uploaded_file = None
    async with semaphore: # รอขอสิทธิ์เข้าทำงานจาก Semaphore
        try:
            # Step 1: อัปโหลดไฟล์ PDF ไปยัง Google File API
            # Note: The native SDK's upload/delete functions are not async, so we run them in a separate thread.
            print(f"Uploading {pdf_path.name}...")
            uploaded_file = await asyncio.to_thread(
                genai.upload_file, path=pdf_path, display_name=pdf_path.name
            )

            # Step 2: สร้าง Prompt และเรียก Gemini API
            # The prompt now consists of the File object and the instruction text.
            prompt = [uploaded_file, parsing_instruction]
            
            # Use the async version of generate_content
            response = await model.generate_content_async(prompt)

            # Step 3: บันทึกไฟล์ Markdown
            md_filename = pdf_path.with_suffix(".md").name
            output_path = text_dir / md_filename
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            
            return f"Processed: {pdf_path.name}"
        
        except Exception as e:
            return f"Error on {pdf_path.name}: {e}"
        
        finally:
            # Step 4: ลบไฟล์ออกจากเซิร์ฟเวอร์ของ Google เพื่อจัดการพื้นที่
            # Step 4: Delete the file from Google's servers to manage storage
            if uploaded_file:
                await asyncio.to_thread(genai.delete_file, name=uploaded_file.name)
                # print(f"Cleaned up file: {uploaded_file.name}") # Optional: for debugging


def documents_setup():
    """
    ดำเนินการแปลงไฟล์ PDF ทั้งหมดโดยใช้ Gemini File API
    Processes all PDF files using the Gemini File API.
    """
    base_dir = Path(__file__).resolve().parent.parent
    
    config_path = base_dir / "config.ini"
    if not config_path.exists():
        raise FileNotFoundError("config.ini not found in the root directory. Please create one.")
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    concurrency_limit = config.getint('General', 'concurrency_limit', fallback=2)
    parsing_instruction = config.get('GeminiParser', 'parsing_instruction', fallback="Please reformat the following text into clean markdown.")

    print("Starting document setup with Gemini File API...")
    
    model_name = initialize_genai(config)
    model = genai.GenerativeModel(model_name)

    pdf_dir = base_dir / "databases" / "pdfs"
    text_dir = base_dir / "databases" / "texts"
    text_dir.mkdir(parents=True, exist_ok=True)
    
    all_pdf_files = list(pdf_dir.glob("*.pdf"))
    if not all_pdf_files:
        print("No PDF files found.")
        return

    print(f"Found {len(all_pdf_files)} PDF files. All will be processed, overwriting existing markdown files.")

    semaphore = asyncio.Semaphore(concurrency_limit)
    
    tasks = [
        parse_and_save_worker(model, parsing_instruction, pdf_path, text_dir, semaphore) 
        for pdf_path in all_pdf_files
    ]

    async def run_tasks():
        results = await tqdm_asyncio.gather(*tasks, desc="Uploading & Parsing with Gemini")
        for res in results:
            if "Error" in res or "Failed" in res:
                print(res)

    asyncio.run(run_tasks())

    print("\nDocument setup completed.")