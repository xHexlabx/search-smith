# search_smith/llm_handler.py
import os
import sys
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def get_gemini_llm(model_name: str, temperature: float = 0.2, max_tokens: int = 1000) -> ChatGoogleGenerativeAI:
    """
    Initializes and returns the Gemini Language Model.
    
    Checks for the GOOGLE_API_KEY in environment variables and exits if not found.

    Args:
        model_name (str): The name of the Gemini model to use.
        temperature (float): The temperature for the model's output.
        max_tokens (int): The maximum number of tokens for the output.

    Returns:
        ChatGoogleGenerativeAI: An instance of the Gemini LLM.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
        print("   Please create a .env file and add your GOOGLE_API_KEY.")
        sys.exit(1)

    print(f"✅ Initializing Gemini model: {model_name}")
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        print("✅ Gemini model initialized successfully.")
        return llm
    except Exception as e:
        print(f"❌ Failed to initialize Gemini API: {e}")
        sys.exit(1)

def load_prompt_template(prompt_file_path: Path) -> ChatPromptTemplate:
    """
    Loads a prompt template from a file.

    Args:
        prompt_file_path (Path): The path to the prompt file.

    Returns:
        ChatPromptTemplate: An instance of the prompt template.
    """
    print(f"✅ Loading prompt template from '{prompt_file_path}'...")
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_template_string = f.read()
        
        prompt = ChatPromptTemplate.from_template(prompt_template_string)
        print("✅ Prompt template loaded successfully.")
        return prompt
    except FileNotFoundError:
        print(f"❌ Error: Prompt file not found at '{prompt_file_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred while loading the prompt: {e}")
        sys.exit(1)
