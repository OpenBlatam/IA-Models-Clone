"""
Search Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class SearchQuery:
    """Search query"""
    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None
    search_type: str = "semantic"  # semantic, keyword, hybrid


@dataclass
class SearchResult:
    """Search result"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: Optional[str] = None


@dataclass
class Context:
    """Context for RAG"""
    query: str
    results: List[SearchResult]
    retrieved_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class SearchBase(ABC):
    """Base interface for search"""
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search"""
        pass
    
    @abstractmethod
    async def retrieve_context(
        self,
        query: str,
        max_results: int = 5
    ) -> Context:
        """Retrieve context for RAG"""
        pass

