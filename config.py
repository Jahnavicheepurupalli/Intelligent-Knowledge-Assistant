import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Project Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Default Settings
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
DEFAULT_EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
DEFAULT_SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 4))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.35))
