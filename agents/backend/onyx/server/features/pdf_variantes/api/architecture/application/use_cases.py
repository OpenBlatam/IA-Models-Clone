"""
Application Layer - Use Cases
Business use cases following CQRS pattern
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from ..domain.entities import DocumentEntity, VariantEntity, TopicEntity


# ============================================================================
# Commands (Write Operations)
# ============================================================================

@dataclass
class UploadDocumentCommand:
    """Command to upload a document"""
    user_id: str
    filename: str
    file_content: bytes
    auto_process: bool = True


@dataclass
class GenerateVariantsCommand:
    """Command to generate variants"""
    document_id: str
    user_id: str
    variant_count: int = 10
    variant_type: str = "standard"


@dataclass
class ExtractTopicsCommand:
    """Command to extract topics"""
    document_id: str
    user_id: str
    min_relevance: float = 0.5


# ============================================================================
# Queries (Read Operations)
# ============================================================================

@dataclass
class GetDocumentQuery:
    """Query to get a document"""
    document_id: str
    user_id: str


@dataclass
class ListDocumentsQuery:
    """Query to list documents"""
    user_id: str
    limit: int = 20
    offset: int = 0
    search: Optional[str] = None


# ============================================================================
# Use Case Interfaces
# ============================================================================

class UploadDocumentUseCase(ABC):
    """Upload document use case"""
    
    @abstractmethod
    async def execute(self, command: UploadDocumentCommand) -> DocumentEntity:
        """Execute upload document use case"""
        pass


class GenerateVariantsUseCase(ABC):
    """Generate variants use case"""
    
    @abstractmethod
    async def execute(self, command: GenerateVariantsCommand) -> List[VariantEntity]:
        """Execute generate variants use case"""
        pass


class ExtractTopicsUseCase(ABC):
    """Extract topics use case"""
    
    @abstractmethod
    async def execute(self, command: ExtractTopicsCommand) -> List[TopicEntity]:
        """Execute extract topics use case"""
        pass


class GetDocumentUseCase(ABC):
    """Get document use case"""
    
    @abstractmethod
    async def execute(self, query: GetDocumentQuery) -> Optional[DocumentEntity]:
        """Execute get document use case"""
        pass


class ListDocumentsUseCase(ABC):
    """List documents use case"""
    
    @abstractmethod
    async def execute(self, query: ListDocumentsQuery) -> List[DocumentEntity]:
        """Execute list documents use case"""
        pass






