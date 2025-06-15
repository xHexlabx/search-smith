# search_smith/document_processor.py
import os
import json
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

def create_langchain_json(
    problems_dir: Path,
    solutions_dir: Path,
    output_path: Path,
    chain: Runnable,
    file_limit: int = None
):
    """
    Processes problem and solution files, tags them using a LangChain chain,
    and generates a single JSON file in the LangChain Document format.

    Args:
        problems_dir (Path): The directory containing problem description .md files.
        solutions_dir (Path): The directory containing corresponding solution .txt files.
        output_path (Path): The path to the output JSON file.
        chain (Runnable): The LangChain (LCEL) chain to invoke for tagging.
        file_limit (int, optional): Max number of files to process. Defaults to None.
    """
    print(f"\n🔎 Processing documents from '{problems_dir}' and '{solutions_dir}'...")
    if not problems_dir.is_dir() or not solutions_dir.is_dir():
        print(f"⚠️ Error: One or more directories not found.")
        return

    # Get a list of all .md problem files
    files_to_process = sorted([f for f in os.listdir(problems_dir) if f.endswith(".md")])

    if file_limit is not None:
        print(f"ℹ️  Processing a limit of {file_limit} files.")
        files_to_process = files_to_process[:file_limit]

    all_documents = []
    for filename in files_to_process:
        problem_id = Path(filename).stem
        problem_file_path = problems_dir / filename
        # Solution files have a .txt extension
        solution_file_path = solutions_dir / f"{problem_id}.txt"

        try:
            # 1. Load problem content from .md file
            # TextLoader works fine for Markdown files as they are plain text.
            problem_loader = TextLoader(str(problem_file_path), encoding="utf-8")
            problem_docs = problem_loader.load()
            if not problem_docs:
                print(f"    ⚠️ Could not extract content from '{filename}'. Skipping.")
                continue
            problem_content = problem_docs[0].page_content

            # 2. Load solution code from .txt file
            solution_code = ""
            if solution_file_path.exists():
                with open(solution_file_path, 'r', encoding='utf-8') as f:
                    solution_code = f.read()
            else:
                print(f"    ⚠️ No solution file found for '{problem_id}' at '{solution_file_path}'.")

            # 3. Invoke chain to get tags
            raw_tags = chain.invoke({"question_markdown": problem_content})
            # Parse comma-separated tags into a list
            tag_list = [tag.strip() for tag in raw_tags.strip().split(',') if tag.strip()]

            # 4. Construct the document object
            document_data = {
                "page_content": problem_content,
                "metadata": {
                    "problem_id": problem_id,
                    "problem_name": problem_id,
                    # Derive source from the first part of the filename (e.g., 'toi' from 'toi11_place')
                    "source": ''.join(filter(str.isalpha, problem_id.split('_')[0])).upper(),
                    "tags": tag_list,
                    "solution_code": solution_code
                }
            }
            all_documents.append(document_data)
            print(f"✅ Processed '{filename}'")

        except Exception as e:
            print(f"    ❌ Error processing file {filename}: {e}")

    # 5. Write all collected data to the JSON file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_documents, f, indent=4, ensure_ascii=False)
        print(f"\n📄 Successfully generated JSON file at '{output_path}'.")
    except Exception as e:
        print(f"\n    ❌ Error writing to JSON file: {e}")

    print("\n✨ Document processing complete.")