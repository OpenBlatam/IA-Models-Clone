"""
Domain Interfaces

Protocols and interfaces for domain contracts.
These define the contracts that implementations must follow, enabling
dependency inversion and testability.

Interfaces in the domain layer define the contracts that infrastructure
implementations must satisfy. This allows the domain to remain independent
of specific technical implementations.
"""

from ....core.interfaces import (
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
    "IAIProcessor",
    "IChatRepository",
    "IRemixRepository",
    "IRankingService",
    "IScoreManager",
    "IValidator",
    "IVoteRepository",
    "IViewRepository",
]
