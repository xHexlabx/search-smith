# scripts/documents_setup.py
import sys
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

# This is a common pattern to make the search_smith package importable
# when running scripts from the 'scripts' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import from your new, modularized package
from search_smith import config, get_gemini_llm ,get_huggingface_llm , load_prompt_template, tag_documents , create_langchain_json  # noqa: E402

def main():
    """
    Main function to run the document tagging process.
    """
    # Load environment variables from .env file (for GOOGLE_API_KEY)
    load_dotenv()
    
    print("🚀 Starting document setup...")

    # 1. Initialize the Language Model from the handler
    # llm = get_gemini_llm(
    #     model_name=config.GEMINI_MODEL_NAME
    # )
    llm = get_huggingface_llm(
        model_name=config.HF_MODEL_NAME
    )
    # 2. Load the prompt template from the handler
    prompt_template = load_prompt_template(
        prompt_file_path=config.PROMPT_FILE_PATH
    )

    # 3. Create the LangChain Chain (LCEL)
    chain = prompt_template | llm | StrOutputParser()

    # 4. Process documents using the dedicated processor function
    # tag_documents(
    #     directory=config.SOLUTIONS_DIR,
    #     chain=chain,
    #     file_limit=None  # Example: limit processing to 10 files
    # )

    create_langchain_json(
        problems_dir=config.PROBLEMS_DIR,
        solutions_dir=config.SOLUTIONS_DIR,
        output_path=config.JSON_OUTPUT_DIR,
        chain=chain,
        file_limit=None  # Example: limit processing to 10 files
    )

if __name__ == "__main__":
    main()

