"""Routes module for Lovable Community SAM3."""

from .chats import router as chats_router
from .bookmarks import router as bookmarks_router
from .shares import router as shares_router
from .tags import router as tags_router
from .export import router as export_router
from ..ai_controller import router as ai_router

__all__ = [
    "chats_router",
    "bookmarks_router",
    "shares_router",
    "tags_router",
    "export_router",
    "ai_router",
]








