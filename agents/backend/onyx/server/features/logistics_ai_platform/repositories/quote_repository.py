"""
Quote repository for data access

This module provides data access layer for quote operations,
abstracting storage implementation details.
"""

import logging
from typing import Optional, List
from datetime import datetime

from models.schemas import QuoteResponse
from repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class QuoteRepository(BaseRepository[QuoteResponse]):
    """
    Repository for quote data access
    
    Provides methods for quote CRUD operations and queries.
    Currently uses in-memory storage, but can be extended to use
    a database or other persistent storage.
    """
    
    def __init__(self):
        """Initialize repository"""
        super().__init__(entity_name="Quote")
    
    async def find_by_request_id(
        self,
        request_id: str
    ) -> List[QuoteResponse]:
        """Find quotes by request ID"""
        return await self.find_all_by_field("request_id", request_id)
    
    async def exists(self, quote_id: str) -> bool:
        """Check if quote exists"""
        return await super().exists(quote_id)
    
    async def find_expired(self) -> List[QuoteResponse]:
        """Find expired quotes"""
        now = datetime.now()
        return await self.find_all(
            filter_func=lambda q: q.valid_until is not None and q.valid_until < now
        )

