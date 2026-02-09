"""
Message Queue
=============

Advanced message queue modules.
"""

from aws.modules.message_queue.queue_manager import QueueManager, Message, MessagePriority
from aws.modules.message_queue.message_router import MessageRouter, RoutingStrategy, Route
from aws.modules.message_queue.dead_letter_queue import DeadLetterQueue, DeadLetterMessage

__all__ = [
    "QueueManager",
    "Message",
    "MessagePriority",
    "MessageRouter",
    "RoutingStrategy",
    "Route",
    "DeadLetterQueue",
    "DeadLetterMessage",
]

