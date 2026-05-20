"""
Domain Layer

Business entities, value objects, domain services, and domain exceptions.
This layer has no dependencies on other layers.

The domain layer represents the core business logic and rules of the application.
It is independent of infrastructure, frameworks, and external systems.
"""

from .entities import (
    ChatAIMetadata,
    ChatEmbedding,
    ChatRemix,
    ChatVote,
    ChatView,
    PublishedChat,
)

from .exceptions import (
    ChatNotFoundError,
    DatabaseError,
    DuplicateVoteError,
    InvalidChatError,
    RemixError,
)

from .interfaces import (
    IAIProcessor,
    IChatRepository,
    IRemixRepository,
    IRankingService,
    IScoreManager,
    IValidator,
    IVoteRepository,
    IViewRepository,
)

__all__ = [
    # Entities
    "ChatAIMetadata",
    "ChatEmbedding",
    "ChatRemix",
    "ChatVote",
    "ChatView",
    "PublishedChat",
    # Exceptions
    "ChatNotFoundError",
    "DatabaseError",
    "DuplicateVoteError",
    "InvalidChatError",
    "RemixError",
    # Interfaces
    "IAIProcessor",
    "IChatRepository",
    "IRemixRepository",
    "IRankingService",
    "IScoreManager",
    "IValidator",
    "IVoteRepository",
    "IViewRepository",
]
