"""Search and indexing utilities"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SearchIndex:
    """Index documents for search"""
    
    def __init__(self, index_dir: Optional[str] = None):
        """
        Initialize search index
        
        Args:
            index_dir: Directory for index files
        """
        if index_dir is None:
            from config import settings
            index_dir = settings.temp_dir + "/search_index"
        
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / "index.json"
        self._index: Dict[str, Any] = {}
        self._load_index()
    
    def _load_index(self) -> None:
        """Load index from file"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self._index = {}
        else:
            self._index = {}
    
    def _save_index(self) -> None:
        """Save index to file"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def index_document(
        self,
        document_path: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Index a document
        
        Args:
            document_path: Path to document
            content: Document content
            metadata: Optional metadata
        """
        # Extract keywords
        keywords = self._extract_keywords(content)
        
        # Create index entry
        index_entry = {
            "path": document_path,
            "keywords": keywords,
            "content_preview": content[:500],  # First 500 chars
            "word_count": len(content.split()),
            "indexed_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to index
        doc_id = self._get_document_id(document_path)
        self._index[doc_id] = index_entry
        
        # Save index
        self._save_index()
    
    def search(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search documents
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching documents
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        
        for doc_id, entry in self._index.items():
            score = 0
            
            # Check keywords
            keywords = entry.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in query_words:
                    score += 2
            
            # Check content preview
            preview = entry.get("content_preview", "").lower()
            for word in query_words:
                if word in preview:
                    score += 1
            
            if score > 0:
                results.append({
                    "document_path": entry["path"],
                    "score": score,
                    "preview": entry.get("content_preview", ""),
                    "metadata": entry.get("metadata", {})
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:limit]
    
    def remove_document(
        self,
        document_path: str
    ) -> None:
        """
        Remove document from index
        
        Args:
            document_path: Path to document
        """
        doc_id = self._get_document_id(document_path)
        self._index.pop(doc_id, None)
        self._save_index()
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_documents": len(self._index),
            "index_size": len(json.dumps(self._index)),
            "last_updated": max(
                (entry.get("indexed_at", "") for entry in self._index.values()),
                default=None
            )
        }
    
    def _extract_keywords(self, content: str, max_keywords: int = 20) -> List[str]:
        """Extract keywords from content"""
        # Simple keyword extraction (could be improved with NLP)
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Filter common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Count word frequency
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def _get_document_id(self, document_path: str) -> str:
        """Get document ID from path"""
        import hashlib
        return hashlib.md5(document_path.encode()).hexdigest()


# Global search index
_search_index: Optional[SearchIndex] = None


def get_search_index() -> SearchIndex:
    """Get global search index"""
    global _search_index
    if _search_index is None:
        _search_index = SearchIndex()
    return _search_index

