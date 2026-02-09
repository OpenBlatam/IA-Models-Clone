"""Collaboration and change tracking utilities"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
import hashlib
import uuid
import logging

logger = logging.getLogger(__name__)


class CollaborationTracker:
    """Track changes and collaboration on documents"""
    
    def __init__(self, changes_dir: Optional[str] = None):
        """
        Initialize collaboration tracker
        
        Args:
            changes_dir: Directory for storing changes
        """
        if changes_dir is None:
            from config import settings
            changes_dir = settings.temp_dir + "/changes"
        
        self.changes_dir = Path(changes_dir)
        self.changes_dir.mkdir(parents=True, exist_ok=True)
    
    def track_change(
        self,
        document_path: str,
        change_type: str,
        change_data: Dict[str, Any],
        author: Optional[str] = None
    ) -> str:
        """
        Track a change to document
        
        Args:
            document_path: Path to document
            change_type: Type of change (edit, delete, add, etc.)
            change_data: Change data
            author: Author of change
            
        Returns:
            Change ID
        """
        change_id = str(uuid.uuid4())
        
        change = {
            "id": change_id,
            "document_path": document_path,
            "type": change_type,
            "data": change_data,
            "author": author or "anonymous",
            "timestamp": datetime.now().isoformat(),
            "hash": self._calculate_document_hash(document_path)
        }
        
        # Save change
        doc_hash = self._get_document_hash(document_path)
        change_file = self.changes_dir / f"{doc_hash}_{change_id}.json"
        
        with open(change_file, 'w') as f:
            json.dump(change, f, indent=2)
        
        # Update change log
        self._update_change_log(document_path, change)
        
        return change_id
    
    def get_change_history(
        self,
        document_path: str
    ) -> List[Dict[str, Any]]:
        """
        Get change history for document
        
        Args:
            document_path: Path to document
            
        Returns:
            List of changes
        """
        doc_hash = self._get_document_hash(document_path)
        log_file = self.changes_dir / f"{doc_hash}_log.json"
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                log = json.load(f)
                return log.get("changes", [])
        
        return []
    
    def get_collaborators(
        self,
        document_path: str
    ) -> List[str]:
        """
        Get list of collaborators for document
        
        Args:
            document_path: Path to document
            
        Returns:
            List of collaborator names
        """
        changes = self.get_change_history(document_path)
        collaborators = set()
        
        for change in changes:
            author = change.get("author")
            if author:
                collaborators.add(author)
        
        return sorted(list(collaborators))
    
    def get_document_statistics(
        self,
        document_path: str
    ) -> Dict[str, Any]:
        """
        Get collaboration statistics for document
        
        Args:
            document_path: Path to document
            
        Returns:
            Statistics dictionary
        """
        changes = self.get_change_history(document_path)
        
        change_types = {}
        authors = {}
        
        for change in changes:
            change_type = change.get("type", "unknown")
            change_types[change_type] = change_types.get(change_type, 0) + 1
            
            author = change.get("author", "unknown")
            authors[author] = authors.get(author, 0) + 1
        
        return {
            "total_changes": len(changes),
            "change_types": change_types,
            "authors": authors,
            "collaborators": len(authors),
            "first_change": changes[0].get("timestamp") if changes else None,
            "last_change": changes[-1].get("timestamp") if changes else None
        }
    
    def _get_document_hash(self, document_path: str) -> str:
        """Get hash for document path"""
        return hashlib.md5(document_path.encode()).hexdigest()
    
    def _calculate_document_hash(self, document_path: str) -> Optional[str]:
        """Calculate hash of document content"""
        try:
            path = Path(document_path)
            if path.exists():
                sha256 = hashlib.sha256()
                with open(path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256.update(chunk)
                return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating document hash: {e}")
        return None
    
    def _update_change_log(
        self,
        document_path: str,
        change: Dict[str, Any]
    ) -> None:
        """Update change log"""
        doc_hash = self._get_document_hash(document_path)
        log_file = self.changes_dir / f"{doc_hash}_log.json"
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                log = json.load(f)
        else:
            log = {"document_path": document_path, "changes": []}
        
        log["changes"].append(change)
        log["last_updated"] = datetime.now().isoformat()
        
        with open(log_file, 'w') as f:
            json.dump(log, f, indent=2)


# Global collaboration tracker
_collaboration_tracker: Optional[CollaborationTracker] = None


def get_collaboration_tracker() -> CollaborationTracker:
    """Get global collaboration tracker"""
    global _collaboration_tracker
    if _collaboration_tracker is None:
        _collaboration_tracker = CollaborationTracker()
    return _collaboration_tracker

