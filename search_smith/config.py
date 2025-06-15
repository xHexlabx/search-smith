# search_smith/config.py
from pathlib import Path

# --- Project Root ---
# This helps in creating absolute paths, making the project more robust.
PROJECT_ROOT = Path(__file__).parent.parent

# --- Directory Paths ---
# All paths are now relative to the project root.
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DATABASES_DIR = PROJECT_ROOT / "databases"
PROBLEMS_DIR = DATABASES_DIR / "texts"
SOLUTIONS_DIR = DATABASES_DIR / "solutions"
VECTOR_STORE_DIR = DATABASES_DIR / "chroma_db" 
JSON_OUTPUT_DIR = DATABASES_DIR / "documents" / "documents.json"
# --- File Paths ---
PROMPT_FILE_PATH = PROMPTS_DIR / "tagger.txt"

# --- LLM and Embedding Model Settings ---
GEMINI_MODEL_NAME = "gemini-2.0-flash"  # Changed to a common, stable version
HF_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
# HF_MODEL_NAME = "Qwen/Qwen3-8B"

# --- Text Splitter Settings ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


