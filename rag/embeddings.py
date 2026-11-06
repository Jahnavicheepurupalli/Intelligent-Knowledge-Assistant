from langchain_huggingface import HuggingFaceEmbeddings
from config import DEFAULT_EMBEDDINGS_MODEL

_embeddings_instance = None

def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Singleton function returning a persistent HuggingFace Embeddings model instance.
    Uses 'sentence-transformers/all-MiniLM-L6-v2' by default.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=DEFAULT_EMBEDDINGS_MODEL,
            model_kwargs={'device': 'cpu'},  # Default to CPU for student local runs
            encode_kwargs={'normalize_embeddings': True}  # Enforce cosine similarity
        )
    return _embeddings_instance
