"""
Search Engine for Flux2 Clothing Changer
========================================

Advanced search and indexing system.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result."""
    item_id: str
    score: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SearchEngine:
    """Advanced search engine system."""
    
    def __init__(self):
        """Initialize search engine."""
        self.index: Dict[str, Dict[str, Any]] = {}
        self.inverted_index: Dict[str, List[str]] = defaultdict(list)
        self.search_history: List[Dict[str, Any]] = []
    
    def index_item(
        self,
        item_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Index item for search.
        
        Args:
            item_id: Item identifier
            content: Content to index
            metadata: Optional metadata
        """
        # Simple tokenization (can be enhanced)
        tokens = content.lower().split()
        
        self.index[item_id] = {
            "content": content,
            "metadata": metadata or {},
            "tokens": set(tokens),
            "indexed_at": time.time(),
        }
        
        # Update inverted index
        for token in tokens:
            if item_id not in self.inverted_index[token]:
                self.inverted_index[token].append(item_id)
        
        logger.debug(f"Indexed item: {item_id}")
    
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search items.
        
        Args:
            query: Search query
            limit: Maximum results
            filters: Optional filters
            
        Returns:
            List of search results
        """
        query_tokens = query.lower().split()
        
        # Calculate scores
        scores: Dict[str, float] = defaultdict(float)
        
        for token in query_tokens:
            if token in self.inverted_index:
                for item_id in self.inverted_index[token]:
                    scores[item_id] += 1.0
        
        # Normalize scores
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}
        
        # Apply filters
        if filters:
            for item_id in list(scores.keys()):
                item = self.index.get(item_id)
                if item:
                    for key, value in filters.items():
                        if key in item["metadata"] and item["metadata"][key] != value:
                            del scores[item_id]
                            break
        
        # Create results
        results = [
            SearchResult(
                item_id=item_id,
                score=score,
                metadata=self.index[item_id]["metadata"] if item_id in self.index else {},
            )
            for item_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Record search
        self.search_history.append({
            "query": query,
            "results_count": len(results),
            "timestamp": time.time(),
        })
        
        return results[:limit]
    
    def delete_item(self, item_id: str) -> bool:
        """
        Delete item from index.
        
        Args:
            item_id: Item identifier
            
        Returns:
            True if deleted
        """
        if item_id not in self.index:
            return False
        
        item = self.index[item_id]
        
        # Remove from inverted index
        for token in item["tokens"]:
            if item_id in self.inverted_index[token]:
                self.inverted_index[token].remove(item_id)
        
        del self.index[item_id]
        logger.debug(f"Deleted item from index: {item_id}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return {
            "total_indexed_items": len(self.index),
            "total_tokens": len(self.inverted_index),
            "total_searches": len(self.search_history),
        }


