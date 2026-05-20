"""
Document Index Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass


@dataclass
class Document:
    """Document model"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class DocumentIndex:
    """Document index definition"""
    
    def __init__(self, name: str, index_type: str = "vector"):
        self.id = str(uuid4())
        self.name = name
        self.index_type = index_type
        self.created_at = datetime.utcnow()
        self.document_count = 0


class IndexBase(ABC):
    """Base interface for document index"""
    
    @abstractmethod
    async def index_document(self, document: Document) -> bool:
        """Index a document"""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10
    ) -> List[Document]:
        """Search documents by embedding"""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from index"""
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        pass

