# search_smith/llm_handler.py
import os
import sys
from pathlib import Path
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import login
from langchain_core.prompts import ChatPromptTemplate

def get_huggingface_llm(model_name: str, temperature: float = 0.2, max_tokens: int = 1000) :
    """
    Initializes and returns the Hugging face Language Model.

    Checks for the HF_TOKEN in environment variables and exits if not found.

    Args:
        model_name (str): The name of the Huggingface model to use.
        temperature (float): The temperature for the model's output.
        max_tokens (int): The maximum number of tokens for the output.

    Returns:
        ChatHuggingFace : An instance of the Hugging face LLM.
    """
    if not os.getenv("HF_TOKEN"):
        print("❌ Error: HF_TOKEN not found in environment variables.")
        print("   Please create a .env file and add your HF_TOKEN.")
        sys.exit(1)

    print(f"✅ Initializing Hugging face model: {model_name}")
    try:
        HF_TOKEN = os.getenv("HF_TOKEN")
        login(token=HF_TOKEN)

        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            task="text-generation",
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.03,
            temperature=temperature
        )

        llm = ChatHuggingFace(llm=llm, verbose=True)
        print("✅ Hugging face model initialized successfully.")
        return llm
    except Exception as e:
        print(f"❌ Failed to initialize Huggingface API: {e}")
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
