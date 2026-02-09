"""
Search Service
Advanced search and filtering
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime, timedelta
import logging
import re

logger = logging.getLogger(__name__)


class SearchService:
    """Advanced search and filtering service"""
    
    def __init__(self):
        # In production, use Elasticsearch or similar
        self.videos_index: Dict[UUID, Dict[str, Any]] = {}
    
    def index_video(
        self,
        video_id: UUID,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Index video for search"""
        self.videos_index[video_id] = {
            "video_id": str(video_id),
            "title": title or "",
            "description": description or "",
            "tags": tags or [],
            "metadata": metadata or {},
            "indexed_at": datetime.utcnow().isoformat(),
        }
        logger.debug(f"Indexed video: {video_id}")
    
    def search_videos(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search videos
        
        Args:
            query: Search query
            filters: Additional filters (status, date_range, etc.)
            limit: Maximum results
            
        Returns:
            List of matching videos
        """
        query_lower = query.lower()
        results = []
        
        for video_id, video_data in self.videos_index.items():
            score = 0
            
            # Title match
            if query_lower in video_data.get("title", "").lower():
                score += 10
            
            # Description match
            if query_lower in video_data.get("description", "").lower():
                score += 5
            
            # Tag match
            for tag in video_data.get("tags", []):
                if query_lower in tag.lower():
                    score += 3
            
            # Apply filters
            if filters:
                if not self._matches_filters(video_data, filters):
                    continue
            
            if score > 0:
                results.append({
                    "video_id": video_id,
                    "score": score,
                    **video_data
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def _matches_filters(self, video_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if video matches filters"""
        # Status filter
        if "status" in filters:
            if video_data.get("metadata", {}).get("status") != filters["status"]:
                return False
        
        # Date range filter
        if "date_from" in filters or "date_to" in filters:
            indexed_at = datetime.fromisoformat(video_data.get("indexed_at", ""))
            
            if "date_from" in filters:
                date_from = datetime.fromisoformat(filters["date_from"])
                if indexed_at < date_from:
                    return False
            
            if "date_to" in filters:
                date_to = datetime.fromisoformat(filters["date_to"])
                if indexed_at > date_to:
                    return False
        
        # Tag filter
        if "tags" in filters:
            video_tags = set(video_data.get("tags", []))
            filter_tags = set(filters["tags"])
            if not video_tags.intersection(filter_tags):
                return False
        
        return True
    
    def get_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """Get search suggestions"""
        query_lower = query.lower()
        suggestions = set()
        
        for video_data in self.videos_index.values():
            # Title suggestions
            title = video_data.get("title", "")
            if query_lower in title.lower():
                words = title.split()
                for word in words:
                    if word.lower().startswith(query_lower):
                        suggestions.add(word)
            
            # Tag suggestions
            for tag in video_data.get("tags", []):
                if query_lower in tag.lower():
                    suggestions.add(tag)
        
        return sorted(list(suggestions))[:limit]


_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """Get search service instance (singleton)"""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service

