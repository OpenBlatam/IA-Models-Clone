"""
Document storage abstraction layer.

Provides an interface for document persistence that can be
swapped between in-memory, database, or file-based storage.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from .models import ProfessionalDocument


class DocumentStorage(ABC):
    """Abstract base class for document storage implementations."""
    
    @abstractmethod
    def get(self, document_id: str) -> Optional[ProfessionalDocument]:
        """Get a document by ID."""
        pass
    
    @abstractmethod
    def save(self, document: ProfessionalDocument) -> None:
        """Save a document."""
        pass
    
    @abstractmethod
    def delete(self, document_id: str) -> bool:
        """Delete a document by ID. Returns True if deleted, False if not found."""
        pass
    
    @abstractmethod
    def list_all(self) -> List[ProfessionalDocument]:
        """List all documents."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get total document count."""
        pass


class InMemoryDocumentStorage(DocumentStorage):
    """In-memory document storage implementation."""
    
    def __init__(self):
        self._documents: Dict[str, ProfessionalDocument] = {}
    
    def get(self, document_id: str) -> Optional[ProfessionalDocument]:
        """Get a document by ID."""
        return self._documents.get(document_id)
    
    def save(self, document: ProfessionalDocument) -> None:
        """Save a document."""
        self._documents[document.id] = document
    
    def delete(self, document_id: str) -> bool:
        """Delete a document by ID."""
        if document_id in self._documents:
            del self._documents[document_id]
            return True
        return False
    
    def list_all(self) -> List[ProfessionalDocument]:
        """List all documents."""
        return list(self._documents.values())
    
    def count(self) -> int:
        """Get total document count."""
        return len(self._documents)






