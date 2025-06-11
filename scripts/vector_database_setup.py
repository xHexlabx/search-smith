# scripts/vector_database_setup.py

import os
import re
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
TEXT_DIR = "texts"  # Directory containing the text files
DB_PATH = "chroma_db_langchain"
MODEL_NAME = "Qwen/Qwen3-Embedding-4B" # Or "intfloat/multilingual-e5-large"

# --- Chunking Strategy ---
# Adjust chunk size and overlap based on the nature of your text documents.
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500

def clean_text(text: str) -> str:
    """
    Helper function to clean extracted text.
    - Removes excessive newlines and whitespace.
    - Fixes common text extraction artifacts.
    """
    # Replace multiple newlines with a single one
    text = re.sub(r'\n\s*\n', '\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    # Remove leading/trailing whitespace from each line
    text = "\n".join([line.strip() for line in text.split('\n')])
    return text.strip()

def load_and_process_texts(directory_path: str) -> list:
    """
    Loads text files from a directory, cleans their content, and enhances metadata.
    - Iterates through files, handling potential errors.
    - Skips files that cannot be read.
    """
    all_docs = []
    # Find all files ending with .txt in the specified directory
    text_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".txt")]
    
    if not text_files:
        print(f"⚠️ No text files found in '{directory_path}'.")
        return []

    print(f"📂 Found {len(text_files)} text files. Processing each file...")
    
    for filename in tqdm(text_files, desc="Processing Text Files"):
        file_path = os.path.join(directory_path, filename)
        try:
            # Load a single text file
            # Specify UTF-8 encoding for broad compatibility
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
            # Process the loaded document (TextLoader loads one file as one document)
            for doc in documents:
                # Clean the page content
                doc.page_content = clean_text(doc.page_content)
                
                # Enhance metadata
                # Use the filename as the primary source identifier
                doc.metadata['source'] = filename
                
            all_docs.extend(documents)
        except Exception as e:
            # If a file is unreadable, skip it and report the error
            print(f"\n❌ Error processing file '{filename}': {e}. Skipping this file.")
            
    return all_docs


def setup():
    """
    Main function to set up the vector database.
    """
    print("🚀 Starting vector database setup...")

    # 1. Load, Clean, and Process Documents
    documents = load_and_process_texts(TEXT_DIR)
    
    if not documents:
        print("No documents were successfully loaded. Exiting.")
        return

    print(f"\n✅ Loaded and processed {len(documents)} documents in total.")

    # 2. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📄 Split documents into {len(chunks)} chunks.")

    # 3. Create Embeddings Model
    print(f"🤖 Loading embedding model: {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # 4. Create and Persist Vector Store (ChromaDB)
    print(f"📦 Creating and persisting vector store at: {DB_PATH}")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("\n✅ Vector database setup complete!")

if __name__ == "__main__":
    setup()