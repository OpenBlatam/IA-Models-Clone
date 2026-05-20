"""
Document Module - Application Layer
Document use cases and business logic
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from .domain import DocumentEntity


# Commands and Queries
@dataclass
class UploadDocumentCommand:
    """Command to upload document"""
    user_id: str
    filename: str
    file_content: bytes
    auto_process: bool = True


@dataclass
class GetDocumentQuery:
    """Query to get document"""
    document_id: str
    user_id: str


@dataclass
class ListDocumentsQuery:
    """Query to list documents"""
    user_id: str
    limit: int = 20
    offset: int = 0
    search: Optional[str] = None


@dataclass
class DeleteDocumentCommand:
    """Command to delete document"""
    document_id: str
    user_id: str


# Use Case Interfaces
class UploadDocumentUseCase(ABC):
    """Upload document use case"""
    
    @abstractmethod
    async def execute(self, command: UploadDocumentCommand) -> DocumentEntity:
        """Execute upload"""
        pass


class GetDocumentUseCase(ABC):
    """Get document use case"""
    
    @abstractmethod
    async def execute(self, query: GetDocumentQuery) -> Optional[DocumentEntity]:
        """Execute get"""
        pass


class ListDocumentsUseCase(ABC):
    """List documents use case"""
    
    @abstractmethod
    async def execute(self, query: ListDocumentsQuery) -> List[DocumentEntity]:
        """Execute list"""
        pass


class DeleteDocumentUseCase(ABC):
    """Delete document use case"""
    
    @abstractmethod
    async def execute(self, command: DeleteDocumentCommand) -> bool:
        """Execute delete"""
        pass






