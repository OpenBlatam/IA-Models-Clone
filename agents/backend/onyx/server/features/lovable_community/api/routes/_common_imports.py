"""
Common imports for route files

This module provides commonly used imports to reduce duplication
across route files and ensure consistency.
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query, status

from ...dependencies import (
    get_chat_service,
    get_optional_user_id,
    get_user_id
)
from ...services import ChatService
from ..cache import cache_response
from ..decorators import handle_errors

__all__ = [
    # FastAPI
    "APIRouter",
    "Depends",
    "Query",
    "status",
    # Typing
    "Optional",
    # Dependencies
    "get_chat_service",
    "get_optional_user_id",
    "get_user_id",
    # Services
    "ChatService",
    # Decorators
    "cache_response",
    "handle_errors",
]



