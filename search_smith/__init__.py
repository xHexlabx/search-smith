# search_smith/__init__.py

"""
search_smith: A package for document processing, embedding, and similarity search.
"""

__version__ = "0.1.0"

# Expose key functions from each module for easy access.
# This allows imports like `from search_smith import load_text_documents`.

from . import config
from .llm_handler import get_gemini_llm ,get_huggingface_llm, load_prompt_template 
from .document_processor import create_langchain_json , tag_documents

# You can define what `from search_smith import *` will import
__all__ = [
    "config",
    "get_gemini_llm",
    "get_huggingface_llm",
    "load_prompt_template",
    "create_langchain_json",
    "tag_documents"
]

