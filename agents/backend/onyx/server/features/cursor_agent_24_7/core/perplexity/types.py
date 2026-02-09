"""
Perplexity Types - Data models and enums
========================================

Core data structures for Perplexity query processing.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


class QueryType(Enum):
    """Types of queries supported by Perplexity system."""
    ACADEMIC_RESEARCH = "academic_research"
    RECENT_NEWS = "recent_news"
    WEATHER = "weather"
    PEOPLE = "people"
    CODING = "coding"
    COOKING_RECIPES = "cooking_recipes"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative_writing"
    SCIENCE_MATH = "science_math"
    URL_LOOKUP = "url_lookup"
    GENERAL = "general"


@dataclass
class SearchResult:
    """Represents a search result from external sources."""
    index: int
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    timestamp: Optional[datetime] = None
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'content': self.content,
            'source': self.source,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], index: int) -> 'SearchResult':
        """Create from dictionary format."""
        timestamp = None
        if data.get('timestamp'):
            if isinstance(data['timestamp'], str):
                timestamp = datetime.fromisoformat(data['timestamp'])
            elif isinstance(data['timestamp'], datetime):
                timestamp = data['timestamp']
        
        return cls(
            index=index,
            title=data.get('title', ''),
            url=data.get('url', ''),
            snippet=data.get('snippet', ''),
            content=data.get('content'),
            source=data.get('source'),
            timestamp=timestamp
        )


@dataclass
class ProcessedQuery:
    """Processed query with metadata."""
    original_query: str
    query_type: QueryType
    search_results: List[SearchResult]
    requires_citations: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}




