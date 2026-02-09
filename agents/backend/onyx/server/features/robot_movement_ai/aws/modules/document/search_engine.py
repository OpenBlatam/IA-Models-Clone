"""
Search Engine
=============

Document search engine.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result."""
    document_id: str
    score: float
    highlights: List[str] = None
    
    def __post_init__(self):
        if self.highlights is None:
            self.highlights = []


class SearchEngine:
    """Document search engine."""
    
    def __init__(self):
        self._index: Dict[str, Dict[str, int]] = {}  # term -> document_id -> count
    
    def index_document(self, document_id: str, content: str):
        """Index document."""
        # Simple tokenization
        words = re.findall(r'\w+', content.lower())
        
        for word in words:
            if word not in self._index:
                self._index[word] = {}
            
            if document_id not in self._index[word]:
                self._index[word][document_id] = 0
            
            self._index[word][document_id] += 1
        
        logger.debug(f"Indexed document: {document_id}")
    
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search documents."""
        query_terms = re.findall(r'\w+', query.lower())
        
        scores: Dict[str, float] = {}
        
        for term in query_terms:
            if term in self._index:
                for document_id, count in self._index[term].items():
                    if document_id not in scores:
                        scores[document_id] = 0.0
                    scores[document_id] += count
        
        # Sort by score
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        results = [
            SearchResult(
                document_id=doc_id,
                score=score
            )
            for doc_id, score in sorted_results[:limit]
        ]
        
        return results
    
    def remove_document(self, document_id: str):
        """Remove document from index."""
        for term in list(self._index.keys()):
            if document_id in self._index[term]:
                del self._index[term][document_id]
                if not self._index[term]:
                    del self._index[term]
        
        logger.debug(f"Removed document from index: {document_id}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_terms": len(self._index),
            "total_documents": len(set(
                doc_id
                for term_index in self._index.values()
                for doc_id in term_index.keys()
            ))
        }















