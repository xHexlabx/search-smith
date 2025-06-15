# search_smith/document_processor.py
import os
from pathlib import Path
from langchain_core.runnables import Runnable
from langchain_community.document_loaders import TextLoader

def tag_documents(directory: Path, chain: Runnable, file_limit: int = None):
    """
    Processes each text document in a directory, applies a LangChain chain
    to tag it, and prints the results.

    Args:
        directory (Path): The directory containing .txt files.
        chain (Runnable): The LangChain (LCEL) chain to invoke for tagging.
        file_limit (int, optional): The maximum number of files to process. Defaults to None (no limit).
    """
    print(f"\n🔎 Processing text documents in '{directory}'...")
    if not directory.exists():
        print(f"⚠️ Warning: Directory not found: '{directory}'.")
        return

    # Get a list of all .txt files
    files_to_process = sorted([f for f in os.listdir(directory) if f.endswith(".txt")])
    
    # Apply the file limit if specified
    if file_limit is not None:
        print(f"ℹ️  Processing a limit of {file_limit} files.")
        files_to_process = files_to_process[:file_limit]


    for filename in files_to_process:
        file_path = directory / filename
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()

            if not docs:
                print(f"    ⚠️ Could not extract content from '{filename}'. Skipping.")
                continue

            # Assuming the prompt expects a variable named 'question_markdown'
            content = docs[0].page_content
            tags = chain.invoke({"question_markdown": content})

            print(f"🏷️  {filename}: {tags.strip()}")

        except Exception as e:
            print(f"    ❌ Error processing file {filename}: {e}")

    print("\n✅ Document tagging process complete.")

