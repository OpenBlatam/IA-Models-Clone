"""
Conversation manager for tracking student interactions.
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history and context for the AI Tutor.
    """
    
    def __init__(self, storage_path: str = "conversations", max_history: int = 50):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict]] = {}
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """Add a message to conversation history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversations[conversation_id].append(message)
        
        if len(self.conversations[conversation_id]) > self.max_history:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history:]
    
    def get_conversation(self, conversation_id: str) -> List[Dict]:
        """Get conversation history."""
        return self.conversations.get(conversation_id, [])
    
    def get_context(self, conversation_id: str, last_n: int = 5) -> List[Dict]:
        """Get recent conversation context."""
        conversation = self.get_conversation(conversation_id)
        return conversation[-last_n:] if conversation else []
    
    def save_conversation(self, conversation_id: str):
        """Save conversation to disk."""
        file_path = self.storage_path / f"{conversation_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.conversations.get(conversation_id, []), f, indent=2, ensure_ascii=False)
    
    def load_conversation(self, conversation_id: str):
        """Load conversation from disk."""
        file_path = self.storage_path / f"{conversation_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                self.conversations[conversation_id] = json.load(f)
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
        file_path = self.storage_path / f"{conversation_id}.json"
        if file_path.exists():
            file_path.unlink()






