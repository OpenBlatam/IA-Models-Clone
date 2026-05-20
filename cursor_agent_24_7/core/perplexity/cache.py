"""
Response Cache - Caching for Perplexity queries
================================================

Caches processed queries and formatted responses to improve performance.
"""

import hashlib
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from .types import ProcessedQuery

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for a processed query."""
    query_hash: str
    processed_query: Dict[str, Any]
    formatted_answer: str
    timestamp: datetime
    expires_at: datetime
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query_hash': self.query_hash,
            'processed_query': self.processed_query,
            'formatted_answer': self.formatted_answer,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(
            query_hash=data['query_hash'],
            processed_query=data['processed_query'],
            formatted_answer=data['formatted_answer'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            expires_at=datetime.fromisoformat(data['expires_at'])
        )


class PerplexityCache:
    """Cache for Perplexity query processing."""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time to live for cache entries in seconds
            max_size: Maximum number of cache entries
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
    
    def _generate_hash(self, query: str, search_results: Optional[list] = None) -> str:
        """Generate hash for query and search results."""
        data = {
            'query': query,
            'search_results': search_results or []
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        search_results: Optional[list] = None
    ) -> Optional[str]:
        """
        Get cached answer if available.
        
        Args:
            query: The query string
            search_results: Optional search results
            
        Returns:
            Cached formatted answer or None
        """
        query_hash = self._generate_hash(query, search_results)
        entry = self._cache.get(query_hash)
        
        if entry is None:
            return None
        
        if entry.is_expired():
            del self._cache[query_hash]
            return None
        
        logger.debug(f"Cache hit for query: {query[:50]}...")
        return entry.formatted_answer
    
    def set(
        self,
        query: str,
        search_results: Optional[list],
        processed_query: ProcessedQuery,
        formatted_answer: str
    ) -> None:
        """
        Cache a processed query and answer.
        
        Args:
            query: The query string
            search_results: Search results used
            processed_query: Processed query object
            formatted_answer: Formatted answer
        """
        # Check cache size
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        query_hash = self._generate_hash(query, search_results)
        expires_at = datetime.now() + timedelta(seconds=self.ttl_seconds)
        
        entry = CacheEntry(
            query_hash=query_hash,
            processed_query={
                'original_query': processed_query.original_query,
                'query_type': processed_query.query_type.value,
                'requires_citations': processed_query.requires_citations,
                'metadata': processed_query.metadata
            },
            formatted_answer=formatted_answer,
            timestamp=datetime.now(),
            expires_at=expires_at
        )
        
        self._cache[query_hash] = entry
        logger.debug(f"Cached answer for query: {query[:50]}...")
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self._cache:
            return
        
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].timestamp
        )
        del self._cache[oldest_key]
        logger.debug("Evicted oldest cache entry")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        expired_count = sum(1 for entry in self._cache.values() if entry.is_expired())
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'expired_entries': expired_count,
            'active_entries': len(self._cache) - expired_count
        }




