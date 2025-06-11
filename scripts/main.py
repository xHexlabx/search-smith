#  scripts/main.py

import sys
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

# --- Configuration ---
DB_PATH = "chroma_db_langchain"
MODEL_NAME = "Qwen/Qwen3-Embedding-4B" # Or "intfloat/multilingual-e5-large"
K_RESULTS = 5 # Number of results to return

def main():
    # 1. Load Embeddings Model
    print("🤖 Loading embedding model...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please ensure you have the correct model name and an internet connection.")
        sys.exit(1)

    # 2. Load Existing Vector Store
    print(f"📂 Loading vector store from: {DB_PATH}")
    try:
        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )
    except Exception as e:
        print(f"❌ Error loading vector store from '{DB_PATH}': {e}")
        print("Please ensure the path is correct and you have run the setup script first.")
        sys.exit(1)

    print("\n✅ Ready to search using MMR (Maximal Marginal Relevance).")

    while True:
        # Prompt for user input, allowing for optional filtering
        user_input = input("\nEnter query (or 'q' to quit) [e.g., 'dijkstra filter:problemA.pdf']: ")
        
        if user_input.lower() == 'q':
            break

        if not user_input.strip():
            continue

        # ✨ IMPROVEMENT 1: Parse keyword and filter from input
        keyword = user_input
        filter_dict = None

        # Check if the user wants to filter by source
        match = re.search(r'filter:([\w\._-]+)', user_input, re.IGNORECASE)
        if match:
            source_filter = match.group(1).strip()
            keyword = re.sub(r'filter:[\w\._-]+', '', user_input, flags=re.IGNORECASE).strip()
            filter_dict = {"source": source_filter}
            print(f"Applying filter -> source: '{source_filter}'")

        print(f"\n🔍 Searching for '{keyword}'...")

        # ✨ IMPROVEMENT 2: Use the Retriever interface as requested
        # This is the standard, modern way to perform retrieval in LangChain.
        search_kwargs = {
            "k": K_RESULTS,
            "fetch_k": 20,      # Number of documents to fetch to apply MMR on.
            "lambda_mult": 0.7  # 0 for max similarity, 1 for max diversity.
        }
        
        # Add the filter to search_kwargs if it exists
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        retriever: VectorStoreRetriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )

        # The invoke method returns a list of Document objects.
        # Note: The standard retriever interface does not return scores directly.
        results = retriever.invoke(keyword)

        if not results:
            print("No matching results found.")
        else:
            print(f"Found {len(results)} results:\n")
            for i, doc in enumerate(results):
                # ✨ IMPROVEMENT 3: Simplified output for the retriever pattern
                print(f"--- Result {i+1} ---")
                print(f"✅ File: {doc.metadata.get('source', 'N/A')}")
                
                content_snippet = doc.page_content.strip().replace('\n', ' ')
                print(f"📄 Content Snippet: {content_snippet[:300]}...")
                print("-" * 50)
        print("\n")

if __name__ == "__main__":
    main()

