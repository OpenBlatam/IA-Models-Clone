"""Annotations and comments system for documents"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
import uuid
import logging

logger = logging.getLogger(__name__)


class AnnotationManager:
    """Manage annotations and comments on documents"""
    
    def __init__(self, annotations_dir: Optional[str] = None):
        """
        Initialize annotation manager
        
        Args:
            annotations_dir: Directory for storing annotations
        """
        if annotations_dir is None:
            from config import settings
            annotations_dir = settings.temp_dir + "/annotations"
        
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
    
    def add_annotation(
        self,
        document_path: str,
        annotation_type: str,
        content: str,
        position: Optional[Dict[str, Any]] = None,
        author: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add annotation to document
        
        Args:
            document_path: Path to document
            annotation_type: Type (comment, highlight, note, etc.)
            content: Annotation content
            position: Position information (page, x, y, etc.)
            author: Author name
            metadata: Additional metadata
            
        Returns:
            Annotation ID
        """
        annotation_id = str(uuid.uuid4())
        
        annotation = {
            "id": annotation_id,
            "document_path": document_path,
            "type": annotation_type,
            "content": content,
            "position": position or {},
            "author": author or "anonymous",
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Save annotation
        doc_hash = self._get_document_hash(document_path)
        annotation_file = self.annotations_dir / f"{doc_hash}_{annotation_id}.json"
        
        with open(annotation_file, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        # Update index
        self._update_annotation_index(document_path, annotation_id, annotation)
        
        return annotation_id
    
    def get_annotations(
        self,
        document_path: str
    ) -> List[Dict[str, Any]]:
        """
        Get all annotations for a document
        
        Args:
            document_path: Path to document
            
        Returns:
            List of annotations
        """
        doc_hash = self._get_document_hash(document_path)
        annotations = []
        
        # Load from index
        index_file = self.annotations_dir / f"{doc_hash}_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
                annotations = list(index.values())
        
        # Sort by creation date
        annotations.sort(key=lambda x: x.get("created_at", ""))
        
        return annotations
    
    def update_annotation(
        self,
        annotation_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update annotation
        
        Args:
            annotation_id: Annotation ID
            content: New content
            metadata: New metadata
            
        Returns:
            True if updated, False otherwise
        """
        # Find annotation file
        for annotation_file in self.annotations_dir.glob(f"*_{annotation_id}.json"):
            with open(annotation_file, 'r') as f:
                annotation = json.load(f)
            
            if content is not None:
                annotation["content"] = content
            
            if metadata is not None:
                annotation["metadata"].update(metadata)
            
            annotation["updated_at"] = datetime.now().isoformat()
            
            with open(annotation_file, 'w') as f:
                json.dump(annotation, f, indent=2)
            
            # Update index
            self._update_annotation_index(
                annotation["document_path"],
                annotation_id,
                annotation
            )
            
            return True
        
        return False
    
    def delete_annotation(
        self,
        annotation_id: str
    ) -> bool:
        """
        Delete annotation
        
        Args:
            annotation_id: Annotation ID
            
        Returns:
            True if deleted, False otherwise
        """
        # Find and delete annotation file
        for annotation_file in self.annotations_dir.glob(f"*_{annotation_id}.json"):
            annotation_file.unlink()
            
            # Update index
            # Extract document path from annotation
            with open(annotation_file, 'r') as f:
                annotation = json.load(f)
            
            doc_hash = self._get_document_hash(annotation["document_path"])
            index_file = self.annotations_dir / f"{doc_hash}_index.json"
            
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = json.load(f)
                
                index.pop(annotation_id, None)
                
                with open(index_file, 'w') as f:
                    json.dump(index, f, indent=2)
            
            return True
        
        return False
    
    def _get_document_hash(self, document_path: str) -> str:
        """Get hash for document path"""
        import hashlib
        return hashlib.md5(document_path.encode()).hexdigest()
    
    def _update_annotation_index(
        self,
        document_path: str,
        annotation_id: str,
        annotation: Dict[str, Any]
    ) -> None:
        """Update annotation index"""
        doc_hash = self._get_document_hash(document_path)
        index_file = self.annotations_dir / f"{doc_hash}_index.json"
        
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
            index = {}
        
        index[annotation_id] = annotation
        
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)


# Global annotation manager
_annotation_manager: Optional[AnnotationManager] = None


def get_annotation_manager() -> AnnotationManager:
    """Get global annotation manager"""
    global _annotation_manager
    if _annotation_manager is None:
        _annotation_manager = AnnotationManager()
    return _annotation_manager

