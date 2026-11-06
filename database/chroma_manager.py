import os
import shutil
import json
from typing import List, Dict, Any, Tuple
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config import CHROMA_DIR
from rag.embeddings import get_embeddings_model

# Metadata file path to track loaded file hashes and statistics
METADATA_FILE = os.path.join(CHROMA_DIR, "db_metadata.json")

class ChromaManager:
    """
    Handles initialization, indexing, deduplication, updates, and lifecycle of the Chroma Vector DB index.
    """
    def __init__(self, persist_directory: str = CHROMA_DIR):
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings_model()
        self._vector_store = None
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load internal DB metadata tracking file hashes to prevent duplicates.
        """
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"ingested_files": {}, "chunk_count": 0}

    def _save_metadata(self):
        """
        Save DB metadata.
        """
        os.makedirs(self.persist_directory, exist_ok=True)
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def get_vector_store(self) -> Chroma:
        """
        Retrieve or initialize the Chroma Vector Store instance.
        """
        if self._vector_store is None:
            self._vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="rag_knowledge_base",
                collection_metadata={"hnsw:space": "cosine"}
            )
        return self._vector_store

    def is_file_ingested(self, file_hash: str) -> bool:
        """
        Check if a file with the given SHA-256 hash has already been ingested.
        """
        return file_hash in self.metadata.get("ingested_files", {})

    def add_documents(self, documents: List[Document], file_name: str, file_hash: str) -> bool:
        """
        Add a list of split chunks (Documents) to Chroma DB. Prevents duplicate file ingestion.
        """
        if self.is_file_ingested(file_hash):
            return False  # Already ingested

        vector_store = self.get_vector_store()

        # Inject standard metadata for identification
        for doc in documents:
            doc.metadata["file_hash"] = file_hash
            if "source" not in doc.metadata:
                doc.metadata["source"] = file_name

        # Add to Chroma vector database
        vector_store.add_documents(documents)

        # Save record of ingestion
        self.metadata["ingested_files"][file_hash] = {
            "filename": file_name,
            "chunk_count": len(documents)
        }
        self.metadata["chunk_count"] = self.metadata.get("chunk_count", 0) + len(documents)
        self._save_metadata()
        return True

    def get_ingested_files(self) -> Dict[str, Any]:
        """
        Get the dictionary of currently indexed files.
        """
        return self.metadata.get("ingested_files", {})

    def get_total_chunks(self) -> int:
        """
        Get the total number of chunks currently indexed.
        """
        return self.metadata.get("chunk_count", 0)

    def delete_document(self, filename: str) -> bool:
        """
        Delete a document's chunks from ChromaDB and metadata registry.
        """
        vector_store = self.get_vector_store()
        try:
            # Query documents matching filename
            # ChromaDB supports deleting via filter on metadata matching source
            vector_store.delete(where={"source": filename})

            # Find hash to delete from metadata
            target_hash = None
            for h, info in self.metadata.get("ingested_files", {}).items():
                if info["filename"] == filename:
                    target_hash = h
                    break

            if target_hash:
                chunks_removed = self.metadata["ingested_files"][target_hash]["chunk_count"]
                del self.metadata["ingested_files"][target_hash]
                self.metadata["chunk_count"] = max(0, self.metadata.get("chunk_count", 0) - chunks_removed)
                self._save_metadata()
            return True
        except Exception as e:
            print(f"Error deleting document {filename} from Chroma: {str(e)}")
            return False

    def reset_database(self) -> bool:
        """
        Fully delete and clear the Chroma vector database directory.
        """
        self._vector_store = None
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                # Re-create empty directory
                os.makedirs(self.persist_directory, exist_ok=True)
                self.metadata = {"ingested_files": {}, "chunk_count": 0}
                self._save_metadata()
                return True
            except Exception as e:
                print(f"Error resetting database: {str(e)}")
                return False
        return True
