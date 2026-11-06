import os
import json
import time
from typing import List, Dict, Any

class ChatManager:
    """
    Manages chat history sessions locally. Saves messages and sessions in a JSON database.
    """
    def __init__(self, storage_dir: str):
        self.storage_path = os.path.join(storage_dir, "chat_history.json")
        self.sessions = self._load_sessions()

    def _load_sessions(self) -> Dict[str, Any]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_sessions(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.sessions, f, indent=4, ensure_ascii=False)

    def get_all_sessions(self) -> Dict[str, Any]:
        return self.sessions

    def create_session(self, title: str = "New Chat") -> str:
        session_id = str(int(time.time() * 1000))
        self.sessions[session_id] = {
            "title": title,
            "created_at": time.time(),
            "messages": []
        }
        self._save_sessions()
        return session_id

    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()

    def rename_session(self, session_id: str, new_title: str):
        if session_id in self.sessions:
            self.sessions[session_id]["title"] = new_title
            self._save_sessions()

    def add_message(self, session_id: str, role: str, content: str, sources: List[Dict[str, Any]] = None):
        if session_id not in self.sessions:
            self.create_session()
            session_id = list(self.sessions.keys())[-1]

        self.sessions[session_id]["messages"].append({
            "role": role,
            "content": content,
            "sources": sources or []
        })

        # Auto rename from first user question if title is default
        if len(self.sessions[session_id]["messages"]) == 1 and role == "user":
            first_msg = content[:30] + ("..." if len(content) > 30 else "")
            self.sessions[session_id]["title"] = first_msg

        self._save_sessions()

    def clear_all_sessions(self):
        self.sessions = {}
        self._save_sessions()
