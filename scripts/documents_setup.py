import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
# CHANGED: Import for Gemini model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Constants ---
# Assumes the script is run from the project's root directory
PROMPT_FILE_PATH = "prompts/tagger.txt"
DOCUMENTS_DIR = "databases/texts"
# NEW: Define the Gemini model to use
GEMINI_MODEL_NAME = "gemini-2.0-flash" 
# REMOVED: MODEL_PATH is no longer needed for API calls

if __name__ == "__main__" :

    # search_smith/documents_handler.py
    """
    Initializes and runs the document tagging process using the Gemini API.

    This function sets up the Gemini API via LangChain, loads a prompt
    template, and then iterates through .md documents in a specified
    directory. For each document, it invokes the model to generate
    descriptive tags based on the content.
    """
    # Load environment variables from .env file (for GOOGLE_API_KEY)
    load_dotenv()
    
    print("🚀 Starting document setup with Gemini API...")

    # 2. Set up the Language Model (Gemini API)
    try:
        # Check if the API key is available
        if not os.getenv("GOOGLE_API_KEY"):
            print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
            print("   Please create a .env file and add your GOOGLE_API_KEY.")

        print(f"✅ Initializing Gemini model: {GEMINI_MODEL_NAME}")
        
        # Instantiate the Gemini model with specific parameters
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=0.2,
            max_output_tokens=1000 # Note: equivalent to max_new_tokens
        )
        
        print("✅ Gemini model initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini API: {e}")

    # 3. Load the Prompt Template from file (No change here)
    try:
        with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
            prompt_template_string = f.read()
        prompt_template = ChatPromptTemplate.from_template(prompt_template_string)
        print(f"✅ Prompt template loaded from '{PROMPT_FILE_PATH}'.")
    except FileNotFoundError:
        print(f"❌ Error: Prompt file not found at '{PROMPT_FILE_PATH}'.")

    # 4. Create the LangChain Chain (LCEL) (No change here)
    chain = prompt_template | llm | StrOutputParser()

    # 5. Process Each Markdown Document
    print(f"\n🔎 Processing Markdown documents in '{DOCUMENTS_DIR}'...")
    if not os.path.exists(DOCUMENTS_DIR):
        print(f"⚠️ Warning: Directory not found: '{DOCUMENTS_DIR}'.")

    # NOTE: I have removed the `[:1]` slice to process ALL files. 
    # Add it back if you only want to test with one file.
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".md"):
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            try:
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()

                if not docs:
                    print(f"    ⚠️ Could not extract content from '{filename}'. Skipping.")
                    continue
                
                content = docs[0].page_content
                tags = chain.invoke({"question_markdown": content})
                
                print(f"🏷️  {filename}: {tags.strip()}")

            except Exception as e:
                print(f"    ❌ Error processing file {filename}: {e}")

    print("\n✅ Document tagging process complete.")