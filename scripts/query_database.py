# scripts/query_database.py
import sys
import os
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from search_smith import get_retriever, recommend_problems  # noqa: E402

def main():
    """
    Main function to handle database querying.
    """
    load_dotenv()
    retriever = get_retriever()

    if not retriever:
        return

    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        recommend_problems(retriever, user_query)
    else:
        print("\nEntering interactive query mode (type 'exit' or 'quit' to end).")
        while True:
            user_query = input("\nEnter your query: ")
            if user_query.lower() in ['exit', 'quit']:
                break
            recommend_problems(retriever, user_query)

if __name__ == "__main__":
    main()