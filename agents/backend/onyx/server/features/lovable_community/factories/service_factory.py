"""
Service Factory

Factory for creating service instances with proper dependency injection.
"""

from typing import Optional
from sqlalchemy.orm import Session

from ..services import ChatService, RankingService, IdentityService
from .repository_factory import RepositoryFactory


class ServiceFactory:
    """
    Factory for creating service instances.
    
    Implements Factory Pattern with Dependency Injection.
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
        self._repository_factory: Optional[RepositoryFactory] = None
        self._chat_service: Optional[ChatService] = None
        self._ranking_service: Optional[RankingService] = None
        self._identity_service: Optional[IdentityService] = None
    
    @property
    def repository_factory(self) -> RepositoryFactory:
        """
        Get or create repository factory.
        
        Returns:
            RepositoryFactory instance
        """
        if self._repository_factory is None:
            self._repository_factory = RepositoryFactory(self.db)
        return self._repository_factory
    
    def get_chat_service(self) -> ChatService:
        """
        Get or create chat service with dependencies.
        
        Uses modular service with full dependency injection.
        Follows Dependency Inversion Principle.
        
        Returns:
            ChatService instance
            
        Raises:
            RuntimeError: If service creation fails
        """
        if self._chat_service is None:
            try:
                from ..services.chat import (
                    ChatService,
                    ChatValidator,
                    ChatAIProcessor,
                    ScoreManager
                )
                
                chat_repo = self.repository_factory.get_chat_repository()
                remix_repo = self.repository_factory.get_remix_repository()
                vote_repo = self.repository_factory.get_vote_repository()
                view_repo = self.repository_factory.get_view_repository()
                ranking_service = self.get_ranking_service()
                
                validator = ChatValidator()
                ai_processor = ChatAIProcessor(chat_repo)
                score_manager = ScoreManager(ranking_service)
                
                self._chat_service = ChatService(
                    chat_repository=chat_repo,
                    remix_repository=remix_repo,
                    vote_repository=vote_repo,
                    view_repository=view_repo,
                    ranking_service=ranking_service,
                    validator=validator,
                    ai_processor=ai_processor,
                    score_manager=score_manager
                )
            except ImportError as e:
                raise RuntimeError(
                    f"Failed to import chat service components: {e}"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create ChatService: {e}"
                ) from e
        return self._chat_service
    
    def get_ranking_service(self) -> RankingService:
        """
        Get or create ranking service.
        
        Returns:
            RankingService instance
        """
        if self._ranking_service is None:
            self._ranking_service = RankingService()
        return self._ranking_service
    
    def get_identity_service(self) -> IdentityService:
        """
        Get or create identity service.
        
        Returns:
            IdentityService instance
        """
        if self._identity_service is None:
            self._identity_service = IdentityService()
        return self._identity_service

