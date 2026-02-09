"""
WebSocket Module

Provides:
- WebSocket utilities
- Real-time communication
- WebSocket handlers
"""

from .websocket_handler import (
    WebSocketHandler,
    create_websocket_handler,
    send_message,
    broadcast_message
)

__all__ = [
    "WebSocketHandler",
    "create_websocket_handler",
    "send_message",
    "broadcast_message"
]



