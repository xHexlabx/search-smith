# scripts/create_database.py
import sys
import os
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from search_smith import create_vector_database  # noqa: E402

def main():
    """
    Main function to run the database creation process.
    """
    print("🚀 Starting Database Creation Process...")
    load_dotenv()
    create_vector_database()

if __name__ == "__main__":
    main()