"""
Repositories module for Lovable Community

Implements Repository Pattern for data access abstraction.
Following Clean Architecture principles.
"""

from .base import BaseRepository
from .chat_repository import ChatRepository
from .remix_repository import RemixRepository
from .vote_repository import VoteRepository
from .view_repository import ViewRepository
from .optimizations import QueryOptimizer

__all__ = [
    "BaseRepository",
    "ChatRepository",
    "RemixRepository",
    "VoteRepository",
    "ViewRepository",
    "QueryOptimizer",
]

