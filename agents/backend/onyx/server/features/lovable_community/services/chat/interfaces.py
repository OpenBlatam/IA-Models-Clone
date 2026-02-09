"""
Chat Service Interfaces

Interfaces for chat service dependencies to enable better testing and loose coupling.
"""

from typing import Protocol, Optional
from ...core.interfaces import (
    IChatRepository,
    IVoteRepository,
    IViewRepository,
    IRemixRepository,
    IRankingService,
    IValidator,
    IAIProcessor,
    IScoreManager
)


class IChatServiceDependencies(Protocol):
    """Protocol defining all dependencies for ChatService."""
    
    chat_repository: IChatRepository
    remix_repository: IRemixRepository
    vote_repository: IVoteRepository
    view_repository: IViewRepository
    ranking_service: Optional[IRankingService]
    validator: IValidator
    ai_processor: IAIProcessor
    score_manager: IScoreManager






