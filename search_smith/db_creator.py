# search_smith/db_creator.py
import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from . import config

def create_vector_database():
    """
    Creates a Vector Database from the documents JSON file using the specified embedding model.
    """
    print("--- 1. Loading and Processing Data ---")
    print(f"Loading documents from '{config.DOCUMENTS_JSON_PATH}'...")
    try:
        with open(config.DOCUMENTS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: '{config.DOCUMENTS_JSON_PATH}'")
        return

    print("Processing documents for the database...")
    documents_for_db = []
    for item in data:
        metadata = item['metadata']
        original_content = item['page_content']
        tags_as_string = ", ".join(metadata['tags']) if isinstance(metadata.get('tags'), list) else metadata.get('tags', '')
        metadata['tags'] = tags_as_string
        enhanced_content = f"TAGS: {tags_as_string}\n---\n{original_content}"
        documents_for_db.append(Document(page_content=enhanced_content, metadata=metadata))
    print(f"✅ Processed {len(documents_for_db)} documents.")

    print("\n--- 2. Creating Vector Store ---")
    print(f"Loading embedding model: '{config.EMBEDDING_MODEL_NAME}'...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs=config.EMBEDDING_MODEL_KWARGS,
            encode_kwargs=config.EMBEDDING_ENCODE_KWARGS
        )
        print("✅ Embedding model loaded.")

        print(f"Creating and persisting Vector Store to '{config.VECTOR_STORE_PATH}'...")
        Chroma.from_documents(
            documents=documents_for_db,
            embedding=embeddings,
            persist_directory=str(config.VECTOR_STORE_PATH)
        )
        print("\n🎉 Vector Store created successfully!")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")