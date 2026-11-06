import os
import json
from typing import Dict, Any

class DocumentManager:
    """
    Manages document metadata status mapping, types, size tracking, and checksum registries.
    """
    def __init__(self, storage_dir: str):
        self.registry_path = os.path.join(storage_dir, "document_registry.json")
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_registry(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=4, ensure_ascii=False)

    def register_document(self, filename: str, file_hash: str, file_size_kb: float, chunk_count: int, file_type: str):
        self.registry[filename] = {
            "file_hash": file_hash,
            "file_size_kb": file_size_kb,
            "chunk_count": chunk_count,
            "file_type": file_type,
            "status": "Indexed"
        }
        self._save_registry()

    def remove_document(self, filename: str):
        if filename in self.registry:
            del self.registry[filename]
            self._save_registry()

    def get_all_documents(self) -> Dict[str, Any]:
        return self.registry
