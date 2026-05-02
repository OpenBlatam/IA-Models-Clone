"""
API module for Unified AI Model.
Contains FastAPI routes and schemas.
"""

from .routes import router
from .schemas import (
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    StreamRequest,
    ConversationResponse,
    ModelsResponse,
    StatsResponse
)

__all__ = [
    "router",
    "ChatRequest",
    "ChatResponse",
    "GenerateRequest",
    "GenerateResponse",
    "StreamRequest",
    "ConversationResponse",
    "ModelsResponse",
    "StatsResponse"
]



