# search_smith/documents_handler.py

import os
import sys
from dotenv import load_dotenv
# CHANGED: Swapped Markdown loader for the generic Text loader
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Constants ---
# Assumes the script is run from the project's root directory
PROMPT_FILE_PATH = "prompts/tagger.txt"
DOCUMENTS_DIR = "databases/solutions"
# NEW: Define the Gemini model to use
GEMINI_MODEL_NAME = "gemini-2.0-flash"

if __name__ == "__main__":
    # Load environment variables from .env file (for GOOGLE_API_KEY)
    load_dotenv()

    print("🚀 Starting document setup with Gemini API...")

    # 2. Set up the Language Model (Gemini API)
    llm = None  # Initialize to None to handle potential errors
    try:
        # Check if the API key is available
        if not os.getenv("GOOGLE_API_KEY"):
            print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
            print("   Please create a .env file and add your GOOGLE_API_KEY.")
            sys.exit(1)  # Exit the script if the key is missing

        print(f"✅ Initializing Gemini model: {GEMINI_MODEL_NAME}")

        # Instantiate the Gemini model with specific parameters
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=0.2,
            max_output_tokens=1000,  # Note: equivalent to max_new_tokens
        )

        print("✅ Gemini model initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini API: {e}")
        sys.exit(1)

    # 3. Load the Prompt Template from file
    prompt_template = None # Initialize to None
    try:
        with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
            prompt_template_string = f.read()
        prompt_template = ChatPromptTemplate.from_template(prompt_template_string)
        print(f"✅ Prompt template loaded from '{PROMPT_FILE_PATH}'.")
    except FileNotFoundError:
        print(f"❌ Error: Prompt file not found at '{PROMPT_FILE_PATH}'.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred while loading the prompt: {e}")
        sys.exit(1)


    # 4. Create the LangChain Chain (LCEL)
    chain = prompt_template | llm | StrOutputParser()

    # 5. Process Each Text Document
    print(f"\n🔎 Processing text documents in '{DOCUMENTS_DIR}'...")
    if not os.path.exists(DOCUMENTS_DIR):
        print(f"⚠️ Warning: Directory not found: '{DOCUMENTS_DIR}'.")
        sys.exit(1)

    for filename in os.listdir(DOCUMENTS_DIR)[50:60]:
        # Now looks for .txt files instead of .md
        if filename.endswith(".txt"):
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            try:
                # Using TextLoader for .txt files with UTF-8 encoding.
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()

                if not docs:
                    print(f"    ⚠️ Could not extract content from '{filename}'. Skipping.")
                    continue

                content = docs[0].page_content
                # FIXED: Changed the key to 'question_markdown' to match the
                # variable expected by your prompt template.
                tags = chain.invoke({"question_markdown": content})

                print(f"🏷️  {filename}: {tags.strip()}")

            except Exception as e:
                print(f"    ❌ Error processing file {filename}: {e}")

    print("\n✅ Document tagging process complete.")
