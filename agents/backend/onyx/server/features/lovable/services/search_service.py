"""
Search Service for advanced chat search with relevance scoring.
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging

from ..repositories.chat_repository import ChatRepository
from ..utils.service_base import BaseService
from ..constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, TRENDING_PERIODS
from ..exceptions import ValidationError
from ..utils.decorators import log_execution_time, handle_errors
from ..utils.pagination import calculate_pagination_metadata

logger = logging.getLogger(__name__)


class SearchService(BaseService):
    """Service for advanced search operations."""
    
    def __init__(self, db: Session):
        """Initialize search service."""
        super().__init__(db)
        self.chat_repo = ChatRepository(db)
    
    @log_execution_time
    @handle_errors
    def search(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        sort_by: str = "relevance",
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE
    ) -> Dict[str, Any]:
        """
        Search chats with advanced relevance scoring.
        
        Args:
            query: Search query string
            tags: List of tags to filter by
            category: Category filter
            sort_by: Sort strategy ('relevance', 'score', 'created_at', 'trending')
            page: Page number (1-indexed)
            page_size: Items per page
            
        Returns:
            Dictionary with search results and metadata
        """
        # Validate pagination
        page, page_size = self.validate_pagination_params(page, page_size, max_page_size=MAX_PAGE_SIZE)
        
        # Use repository for search
        chats, total = self.chat_repo.search_chats(
            query=query,
            tags=tags,
            category=category,
            sort_by=sort_by,
            page=page,
            page_size=page_size
        )
        
        # Calculate pagination metadata
        pagination = calculate_pagination_metadata(page, page_size, total)
        
        # Calculate relevance scores for results
        results = []
        for chat in chats:
            chat_dict = self.serialize_list([chat])[0] if chat else {}
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(chat, query, tags)
            chat_dict["relevance_score"] = relevance_score
            
            results.append(chat_dict)
        
        # Re-sort by relevance if needed
        if sort_by == "relevance" and query:
            results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return {
            "results": results,
            "query": query,
            "tags": tags,
            "category": category,
            "sort_by": sort_by,
            "total": total,
            **pagination
        }
    
    def _calculate_relevance_score(
        self,
        chat: Any,  # PublishedChat model
        query: Optional[str],
        tags: Optional[List[str]]
    ) -> float:
        """
        Calculate relevance score for a chat.
        
        Args:
            chat: Chat object
            query: Search query
            tags: Search tags
            
        Returns:
            Relevance score (0-100)
        """
        score = 0.0
        
        # Base score from chat ranking
        score += min(chat.score or 0, 50)  # Cap at 50
        
        # Query match bonus
        if query:
            query_lower = query.lower()
            title_lower = (chat.title or "").lower()
            content_lower = (chat.chat_content or "").lower()
            desc_lower = (chat.description or "").lower()
            
            # Title match (highest weight)
            if query_lower in title_lower:
                score += 30
            
            # Description match (medium weight)
            if query_lower in desc_lower:
                score += 15
            
            # Content match (lower weight)
            if query_lower in content_lower:
                score += 10
        
        # Tag match bonus
        if tags and chat.tags:
            chat_tags = [t.strip().lower() for t in chat.tags.split(",") if t.strip()]
            search_tags = [t.strip().lower() for t in tags if t.strip()]
            
            matching_tags = set(chat_tags) & set(search_tags)
            if matching_tags:
                score += len(matching_tags) * 5
        
        # Recency bonus
        if chat.created_at:
            hours_old = (datetime.now() - chat.created_at).total_seconds() / 3600
            if hours_old < 24:
                score += 10
            elif hours_old < 168:  # 7 days
                score += 5
        
        # Engagement bonus
        engagement_rate = 0
        if chat.view_count and chat.view_count > 0:
            engagement_rate = ((chat.vote_count or 0) + (chat.remix_count or 0) * 2) / chat.view_count
            score += min(engagement_rate * 10, 20)  # Cap at 20
        
        return min(round(score, 2), 100.0)




