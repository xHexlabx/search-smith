# search_smith/config.py
from pathlib import Path

# --- Project Root ---
# This helps in creating absolute paths, making the project more robust.
PROJECT_ROOT = Path(__file__).parent.parent

# --- Directory Paths ---
# All paths are now relative to the project root.
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DATABASES_DIR = PROJECT_ROOT / "databases"
SOLUTIONS_DIR = DATABASES_DIR / "solutions"
VECTOR_STORE_DIR = DATABASES_DIR / "chroma_db" 

# --- File Paths ---
PROMPT_FILE_PATH = PROMPTS_DIR / "tagger.txt"

# --- LLM and Embedding Model Settings ---
GEMINI_MODEL_NAME = "gemini-2.0-flash"  # Changed to a common, stable version

# --- Text Splitter Settings ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Vector Store Settings ---
SIMILARITY_SEARCH_K = 3 # Number of results to return from similarity search

