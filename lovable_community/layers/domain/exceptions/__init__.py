"""
Domain Exceptions

Business exceptions representing domain errors.
These exceptions are raised when business rules are violated or domain operations fail.

Domain exceptions are part of the domain layer and represent business-level errors
that can occur during domain operations. They are independent of infrastructure concerns.
"""

from ....exceptions import (
    ChatNotFoundError,
    DatabaseError,
    DuplicateVoteError,
    InvalidChatError,
    RemixError,
)

__all__ = [
    "ChatNotFoundError",
    "DatabaseError",
    "DuplicateVoteError",
    "InvalidChatError",
    "RemixError",
]
