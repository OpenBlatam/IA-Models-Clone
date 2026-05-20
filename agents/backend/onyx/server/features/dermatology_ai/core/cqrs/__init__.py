"""
CQRS (Command Query Responsibility Segregation) Pattern
Separates read and write operations for better scalability
"""

from .commands import *
from .queries import *
from .handlers import *

__all__ = [
    "Command",
    "Query",
    "CommandHandler",
    "QueryHandler",
    "CommandBus",
    "QueryBus",
]















