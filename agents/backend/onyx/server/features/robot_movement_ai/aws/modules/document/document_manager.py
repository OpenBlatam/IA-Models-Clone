"""
Document Manager
================

Document management.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentStatus(Enum):
    """Document status."""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class Document:
    """Document definition."""
    id: str
    title: str
    content: str
    status: DocumentStatus
    author: str
    created_at: datetime = None
    updated_at: datetime = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class DocumentManager:
    """Document manager."""
    
    def __init__(self):
        self._documents: Dict[str, Document] = {}
        self._index: Dict[str, List[str]] = {}  # tag -> document_ids
    
    def create_document(
        self,
        document_id: str,
        title: str,
        content: str,
        author: str,
        tags: Optional[List[str]] = None
    ) -> Document:
        """Create document."""
        document = Document(
            id=document_id,
            title=title,
            content=content,
            status=DocumentStatus.DRAFT,
            author=author,
            tags=tags or []
        )
        
        self._documents[document_id] = document
        
        # Index by tags
        for tag in document.tags:
            if tag not in self._index:
                self._index[tag] = []
            self._index[tag].append(document_id)
        
        logger.info(f"Created document: {document_id}")
        return document
    
    def update_document(
        self,
        document_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Update document."""
        if document_id not in self._documents:
            return False
        
        document = self._documents[document_id]
        
        # Remove old tags from index
        for tag in document.tags:
            if tag in self._index and document_id in self._index[tag]:
                self._index[tag].remove(document_id)
        
        if title:
            document.title = title
        if content:
            document.content = content
        if tags:
            document.tags = tags
        
        document.updated_at = datetime.now()
        
        # Add new tags to index
        for tag in document.tags:
            if tag not in self._index:
                self._index[tag] = []
            if document_id not in self._index[tag]:
                self._index[tag].append(document_id)
        
        logger.info(f"Updated document: {document_id}")
        return True
    
    def publish_document(self, document_id: str) -> bool:
        """Publish document."""
        if document_id not in self._documents:
            return False
        
        self._documents[document_id].status = DocumentStatus.PUBLISHED
        logger.info(f"Published document: {document_id}")
        return True
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self._documents.get(document_id)
    
    def search_by_tag(self, tag: str) -> List[Document]:
        """Search documents by tag."""
        document_ids = self._index.get(tag, [])
        return [self._documents[did] for did in document_ids if did in self._documents]
    
    def list_documents(self, status: Optional[DocumentStatus] = None) -> List[Document]:
        """List documents."""
        documents = list(self._documents.values())
        
        if status:
            documents = [d for d in documents if d.status == status]
        
        return documents
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get document statistics."""
        return {
            "total_documents": len(self._documents),
            "by_status": {
                status.value: sum(1 for d in self._documents.values() if d.status == status)
                for status in DocumentStatus
            },
            "total_tags": len(self._index),
            "total_authors": len(set(d.author for d in self._documents.values()))
        }















