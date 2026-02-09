"""
Queries - Read operations in CQRS pattern
Queries represent intent to read data
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Optional, List


class Query(ABC):
    """Base class for queries"""
    pass


@dataclass
class GetAnalysisQuery(Query):
    """Query to get analysis by ID"""
    analysis_id: str
    user_id: Optional[str] = None


@dataclass
class GetAnalysisHistoryQuery(Query):
    """Query to get analysis history"""
    user_id: str
    limit: int = 10
    offset: int = 0


@dataclass
class GetUserQuery(Query):
    """Query to get user by ID"""
    user_id: str


@dataclass
class GetRecommendationsQuery(Query):
    """Query to get recommendations"""
    analysis_id: str
    user_id: Optional[str] = None


@dataclass
class SearchProductsQuery(Query):
    """Query to search products"""
    query: str
    category: Optional[str] = None
    limit: int = 10
    offset: int = 0















