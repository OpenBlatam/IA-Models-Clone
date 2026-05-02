"""
Knowledge Base for Continual Learning
Implements persistent storage for agent knowledge accumulation
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque
import asyncio

logger = logging.getLogger(__name__)


class KnowledgeEntry:
    """Single knowledge entry"""
    
    def __init__(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.content = content
        self.context = context or {}
        self.timestamp = timestamp or datetime.utcnow()
        self.access_count = 0
        self.last_accessed = self.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        """Create from dictionary"""
        entry = cls(
            content=data["content"],
            context=data.get("context", {}),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        entry.access_count = data.get("access_count", 0)
        entry.last_accessed = datetime.fromisoformat(data.get("last_accessed", data["timestamp"]))
        return entry


class KnowledgeBase:
    """
    Persistent knowledge base for continual learning
    Implements concepts from papers on continual learning and long-term memory
    """
    
    def __init__(
        self,
        storage_path: str = "./data/unified_ai/knowledge",
        max_entries: int = 10000,
        retention_days: int = 30
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.retention_days = retention_days
        self._entries: deque = deque(maxlen=max_entries)
        self._lock = asyncio.Lock()
        self._load_from_disk()
    
    def _get_storage_file(self) -> Path:
        """Get path to storage file"""
        return self.storage_path / "knowledge_base.json"
    
    def _load_from_disk(self) -> None:
        """Load knowledge base from disk"""
        storage_file = self._get_storage_file()
        if not storage_file.exists():
            logger.info("Knowledge base file not found, starting fresh")
            return
        
        try:
            with open(storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                entries = data.get("entries", [])
                
                cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
                for entry_data in entries:
                    entry = KnowledgeEntry.from_dict(entry_data)
                    if entry.timestamp >= cutoff_date:
                        self._entries.append(entry)
                
                logger.info(f"Loaded {len(self._entries)} knowledge entries from disk")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            self._entries = deque(maxlen=self.max_entries)
    
    async def _save_to_disk(self) -> None:
        """Save knowledge base to disk"""
        storage_file = self._get_storage_file()
        try:
            async with self._lock:
                entries_data = [entry.to_dict() for entry in self._entries]
                data = {
                    "entries": entries_data,
                    "last_updated": datetime.utcnow().isoformat(),
                    "total_entries": len(self._entries)
                }
                
                # Write atomically
                temp_file = storage_file.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                temp_file.replace(storage_file)
                logger.debug(f"Saved {len(self._entries)} knowledge entries to disk")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
    
    async def add_knowledge(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add new knowledge entry"""
        entry = KnowledgeEntry(content=content, context=context)
        async with self._lock:
            self._entries.append(entry)
        
        # Save periodically (async, non-blocking)
        if len(self._entries) % 10 == 0:
            await self._save_to_disk()
    
    async def search_knowledge(
        self,
        query: str,
        limit: int = 10
    ) -> List[KnowledgeEntry]:
        """Search knowledge base (simple keyword matching)"""
        query_lower = query.lower()
        results = []
        
        async with self._lock:
            for entry in self._entries:
                if query_lower in entry.content.lower():
                    entry.access_count += 1
                    entry.last_accessed = datetime.utcnow()
                    results.append(entry)
                    if len(results) >= limit:
                        break
        
        # Save after search (async, non-blocking)
        await self._save_to_disk()
        return results
    
    async def get_recent_knowledge(self, limit: int = 10) -> List[KnowledgeEntry]:
        """Get most recent knowledge entries"""
        async with self._lock:
            return list(self._entries)[-limit:]
    
    async def get_all_knowledge(self) -> List[KnowledgeEntry]:
        """Get all knowledge entries"""
        async with self._lock:
            return list(self._entries)
    
    async def cleanup_old_entries(self) -> int:
        """Remove entries older than retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        removed = 0
        
        async with self._lock:
            original_count = len(self._entries)
            self._entries = deque(
                [e for e in self._entries if e.timestamp >= cutoff_date],
                maxlen=self.max_entries
            )
            removed = original_count - len(self._entries)
        
        if removed > 0:
            await self._save_to_disk()
            logger.info(f"Cleaned up {removed} old knowledge entries")
        
        return removed
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        async with self._lock:
            if not self._entries:
                return {
                    "total_entries": 0,
                    "oldest_entry": None,
                    "newest_entry": None,
                    "total_access_count": 0
                }
            
            entries = list(self._entries)
            return {
                "total_entries": len(entries),
                "oldest_entry": min(e.timestamp for e in entries).isoformat(),
                "newest_entry": max(e.timestamp for e in entries).isoformat(),
                "total_access_count": sum(e.access_count for e in entries)
            }
