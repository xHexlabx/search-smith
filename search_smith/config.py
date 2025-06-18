# search_smith/config.py
from pathlib import Path

# --- Project Root ---
PROJECT_ROOT = Path(__file__).parent.parent

# --- Directory Paths ---
DATABASES_DIR = PROJECT_ROOT / "databases"
DOCUMENTS_JSON_PATH = DATABASES_DIR / "documents" / "documents.json"
VECTOR_STORE_PATH = DATABASES_DIR / "chroma_db_problems_qwen"

# --- Embedding Model Settings ---
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}
EMBEDDING_ENCODE_KWARGS = {'normalize_embeddings': True}

# --- Retriever Settings ---
SEARCH_KWARGS = {"k": 5}