"""
History Manager for Color Grading AI
====================================

Manages processing history and versioning.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ProcessingHistory:
    """Processing history entry."""
    id: str
    input_path: str
    output_path: str
    operation: str
    color_params: Dict[str, Any]
    template_used: Optional[str] = None
    reference_used: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    duration: Optional[float] = None
    file_size: Optional[int] = None
    output_size: Optional[int] = None
    success: bool = True
    error: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data


class HistoryManager:
    """
    Manages processing history.
    
    Features:
    - Store processing history
    - Search and filter
    - Versioning
    - Tags
    """
    
    def __init__(self, history_dir: str = "history"):
        """
        Initialize history manager.
        
        Args:
            history_dir: Directory for history storage
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._history: Dict[str, ProcessingHistory] = {}
        self._load_history()
    
    def _load_history(self):
        """Load history from disk."""
        history_file = self.history_dir / "history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                
                for entry_data in data.get("entries", []):
                    entry = ProcessingHistory(
                        id=entry_data["id"],
                        input_path=entry_data["input_path"],
                        output_path=entry_data["output_path"],
                        operation=entry_data["operation"],
                        color_params=entry_data["color_params"],
                        template_used=entry_data.get("template_used"),
                        reference_used=entry_data.get("reference_used"),
                        description=entry_data.get("description"),
                        created_at=datetime.fromisoformat(entry_data["created_at"]),
                        duration=entry_data.get("duration"),
                        file_size=entry_data.get("file_size"),
                        output_size=entry_data.get("output_size"),
                        success=entry_data.get("success", True),
                        error=entry_data.get("error"),
                        tags=entry_data.get("tags", [])
                    )
                    self._history[entry.id] = entry
                
                logger.info(f"Loaded {len(self._history)} history entries")
            except Exception as e:
                logger.error(f"Error loading history: {e}")
    
    def _save_history(self):
        """Save history to disk."""
        history_file = self.history_dir / "history.json"
        data = {
            "entries": [entry.to_dict() for entry in self._history.values()]
        }
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def add(
        self,
        input_path: str,
        output_path: str,
        operation: str,
        color_params: Dict[str, Any],
        template_used: Optional[str] = None,
        reference_used: Optional[str] = None,
        description: Optional[str] = None,
        duration: Optional[float] = None,
        file_size: Optional[int] = None,
        output_size: Optional[int] = None,
        success: bool = True,
        error: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Add history entry.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            operation: Operation type
            color_params: Color parameters used
            template_used: Template used
            reference_used: Reference used
            description: Description
            duration: Processing duration
            file_size: Input file size
            output_size: Output file size
            success: Success status
            error: Error message
            tags: Tags
            
        Returns:
            History entry ID
        """
        import uuid
        entry_id = str(uuid.uuid4())
        
        entry = ProcessingHistory(
            id=entry_id,
            input_path=input_path,
            output_path=output_path,
            operation=operation,
            color_params=color_params,
            template_used=template_used,
            reference_used=reference_used,
            description=description,
            duration=duration,
            file_size=file_size,
            output_size=output_size,
            success=success,
            error=error,
            tags=tags or []
        )
        
        self._history[entry_id] = entry
        self._save_history()
        
        return entry_id
    
    def get(self, entry_id: str) -> Optional[ProcessingHistory]:
        """Get history entry by ID."""
        return self._history.get(entry_id)
    
    def search(
        self,
        operation: Optional[str] = None,
        template: Optional[str] = None,
        tags: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ProcessingHistory]:
        """
        Search history entries.
        
        Args:
            operation: Filter by operation
            template: Filter by template
            tags: Filter by tags
            date_from: Filter from date
            date_to: Filter to date
            limit: Maximum results
            
        Returns:
            List of matching entries
        """
        results = []
        
        for entry in self._history.values():
            # Filter by operation
            if operation and entry.operation != operation:
                continue
            
            # Filter by template
            if template and entry.template_used != template:
                continue
            
            # Filter by tags
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            # Filter by date
            if date_from and entry.created_at < date_from:
                continue
            if date_to and entry.created_at > date_to:
                continue
            
            results.append(entry)
        
        # Sort by date (newest first)
        results.sort(key=lambda x: x.created_at, reverse=True)
        
        return results[:limit]
    
    def get_recent(self, limit: int = 10) -> List[ProcessingHistory]:
        """Get recent history entries."""
        entries = list(self._history.values())
        entries.sort(key=lambda x: x.created_at, reverse=True)
        return entries[:limit]
    
    def get_by_input(self, input_path: str) -> List[ProcessingHistory]:
        """Get history entries for input file."""
        return [
            entry for entry in self._history.values()
            if entry.input_path == input_path
        ]
    
    def add_tags(self, entry_id: str, tags: List[str]):
        """Add tags to history entry."""
        entry = self._history.get(entry_id)
        if entry:
            entry.tags.extend(tags)
            entry.tags = list(set(entry.tags))  # Remove duplicates
            self._save_history()
    
    def delete(self, entry_id: str):
        """Delete history entry."""
        if entry_id in self._history:
            del self._history[entry_id]
            self._save_history()




