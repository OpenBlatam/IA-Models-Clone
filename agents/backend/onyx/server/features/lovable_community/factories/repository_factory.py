"""
Repository Factory

Factory for creating repository instances with dependency injection.
"""

from typing import Optional
from sqlalchemy.orm import Session

from ..repositories import (
    ChatRepository,
    RemixRepository,
    VoteRepository,
    ViewRepository,
)


class RepositoryFactory:
    """
    Factory for creating repository instances.
    
    Implements Factory Pattern for repository creation.
    Uses lazy initialization with singleton pattern per factory instance.
    """
    
    def __init__(self, db: Session):
        """
        Initialize factory with database session.
        
        Args:
            db: Database session
            
        Raises:
            ValueError: If db is None
        """
        if db is None:
            raise ValueError("Database session cannot be None")
        self.db = db
        self._chat_repository: Optional[ChatRepository] = None
        self._remix_repository: Optional[RemixRepository] = None
        self._vote_repository: Optional[VoteRepository] = None
        self._view_repository: Optional[ViewRepository] = None
    
    def _get_or_create_repository(
        self,
        repository_attr: str,
        repository_class: type
    ):
        """
        Generic helper to get or create a repository (singleton pattern).
        
        Args:
            repository_attr: Attribute name for the cached repository
            repository_class: Repository class to instantiate
            
        Returns:
            Repository instance
            
        Raises:
            ValueError: If repository_class is None or invalid
            TypeError: If repository_class cannot be instantiated with db session
        """
        if repository_class is None:
            raise ValueError("Repository class cannot be None")
        
        cached = getattr(self, repository_attr, None)
        if cached is None:
            try:
                cached = repository_class(self.db)
                setattr(self, repository_attr, cached)
            except TypeError as e:
                raise TypeError(
                    f"Repository class {repository_class.__name__} cannot be instantiated with database session: {e}"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create repository {repository_class.__name__}: {e}"
                ) from e
        return cached
    
    def get_chat_repository(self) -> ChatRepository:
        """
        Get or create chat repository (singleton per factory instance).
        
        Returns:
            ChatRepository instance
        """
        return self._get_or_create_repository("_chat_repository", ChatRepository)
    
    def get_remix_repository(self) -> RemixRepository:
        """
        Get or create remix repository (singleton per factory instance).
        
        Returns:
            RemixRepository instance
        """
        return self._get_or_create_repository("_remix_repository", RemixRepository)
    
    def get_vote_repository(self) -> VoteRepository:
        """
        Get or create vote repository (singleton per factory instance).
        
        Returns:
            VoteRepository instance
        """
        return self._get_or_create_repository("_vote_repository", VoteRepository)
    
    def get_view_repository(self) -> ViewRepository:
        """
        Get or create view repository (singleton per factory instance).
        
        Returns:
            ViewRepository instance
        """
        return self._get_or_create_repository("_view_repository", ViewRepository)











