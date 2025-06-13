# search_smith/documents_handler.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Import transformers components for robust local model loading
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- Constants ---
# Assumes the script is run from the project's root directory
PROMPT_FILE_PATH = "prompts/tagger.txt"
DOCUMENTS_DIR = "databases/texts"
# IMPORTANT: This path should point to your local model directory
MODEL_PATH = "Qwen/Qwen3-1.7B" 

def documents_setup():
    """
    Initializes and runs the document tagging process using a local Hugging Face model.

    This function sets up a local Hugging Face model via LangChain, loads a prompt
    template, and then iterates through .md documents in a specified
    directory. For each document, it invokes the model to generate
    descriptive tags based on the content.
    """
    print("🚀 Starting document setup...")

    # Note: API keys are not needed for running a model locally.
    
    # 2. Set up the Language Model (Hugging Face from local path)
    try:
        # This initial log is helpful to confirm the model path
        print(f"✅ Loading LLM from local path: {MODEL_PATH}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="auto"
        )

        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=50,
            temperature=0.5
        )

        llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
        
        print(f"✅ LLM ({os.path.basename(MODEL_PATH)}) loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM from local path: {e}")
        print("   Please ensure the path is correct and the directory contains all necessary model files (e.g., config.json, model weights).")
        return

    # 3. Load the Prompt Template from file
    try:
        with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
            prompt_template_string = f.read()
        prompt_template = ChatPromptTemplate.from_template(prompt_template_string)
        print(f"✅ Prompt template loaded from '{PROMPT_FILE_PATH}'.")
    except FileNotFoundError:
        print(f"❌ Error: Prompt file not found at '{PROMPT_FILE_PATH}'.")
        return

    # 4. Create the LangChain Chain (LCEL)
    chain = prompt_template | llm | StrOutputParser()

    # 5. Process Each Markdown Document
    print(f"\n🔎 Processing Markdown documents in '{DOCUMENTS_DIR}'...")
    if not os.path.exists(DOCUMENTS_DIR):
        print(f"⚠️ Warning: Directory not found: '{DOCUMENTS_DIR}'.")
        return

    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".md"):
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            try:
                # The "Loading..." and "Tagging..." logs are removed from here
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()

                if not docs:
                    print(f"    ⚠️ Could not extract content from '{filename}'. Skipping.")
                    continue
                
                content = docs[0].page_content
                tags = chain.invoke({"question_markdown": content})
                
                # MODIFIED: Cleaner final output line including the filename.
                print(f"🏷️  {filename}: {tags.strip()}")

            except Exception as e:
                print(f"    ❌ Error processing file {filename}: {e}")

    print("\n✅ Document tagging process complete.")