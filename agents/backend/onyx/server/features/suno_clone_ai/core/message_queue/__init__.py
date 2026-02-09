"""
Message Queue Module

Provides:
- Message queue utilities
- Producer/consumer patterns
- Queue management
"""

from .message_queue import (
    MessageQueue,
    create_message_queue,
    publish_message,
    consume_messages
)

__all__ = [
    "MessageQueue",
    "create_message_queue",
    "publish_message",
    "consume_messages"
]



