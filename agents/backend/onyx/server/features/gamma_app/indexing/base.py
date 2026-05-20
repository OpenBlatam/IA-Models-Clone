"""
Indexing Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4


class IndexType(str, Enum):
    """Index types"""
    VECTOR = "vector"
    FULLTEXT = "fulltext"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class Index:
    """Index definition"""
    
    def __init__(
        self,
        name: str,
        index_type: IndexType,
        config: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.name = name
        self.index_type = index_type
        self.config = config or {}
        self.created_at = datetime.utcnow()
        self.document_count = 0


class IndexBase(ABC):
    """Base interface for indexing"""
    
    @abstractmethod
    async def create_index(self, index: Index) -> bool:
        """Create index"""
        pass
    
    @abstractmethod
    async def add_to_index(
        self,
        index_name: str,
        document_id: str,
        content: Any
    ) -> bool:
        """Add document to index"""
        pass
    
    @abstractmethod
    async def search(
        self,
        index_name: str,
        query: Any,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search in index"""
        pass
    
    @abstractmethod
    async def remove_from_index(
        self,
        index_name: str,
        document_id: str
    ) -> bool:
        """Remove document from index"""
        pass

