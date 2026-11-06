import os
import hashlib
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document

def get_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file to check for duplicates.
    """
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def load_single_document(file_path: str) -> List[Document]:
    """
    Load a single document from file_path, extract text and retain metadata.
    Supported extensions: .pdf, .txt, .docx
    """
    ext = os.path.splitext(file_path)[1].lower()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # PyPDFLoader automatically adds source and page numbers (0-indexed).
            # Let's adjust page metadata to be 1-indexed for display friendliness.
            for doc in docs:
                if "page" in doc.metadata:
                    doc.metadata["page"] = int(doc.metadata["page"]) + 1
                else:
                    doc.metadata["page"] = 1
                # Standardize source path representation to base file name
                doc.metadata["source"] = os.path.basename(file_path)
            return docs

        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
                doc.metadata["page"] = 1
            return docs

        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
                doc.metadata["page"] = 1
            return docs

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    except Exception as e:
        raise RuntimeError(f"Error loading document {file_path}: {str(e)}")

def get_document_metadata(file_path: str) -> Dict[str, Any]:
    """
    Retrieve basic system metadata for a document without loading the full text.
    """
    if not os.path.exists(file_path):
        return {}

    stat = os.stat(file_path)
    file_size_kb = round(stat.st_size / 1024, 2)
    file_hash = get_file_hash(file_path)

    return {
        "filename": os.path.basename(file_path),
        "file_size_kb": file_size_kb,
        "file_hash": file_hash,
        "file_path": file_path
    }
