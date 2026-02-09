"""
CQRS (Command Query Responsibility Segregation)
==============================================

CQRS pattern implementation.
"""

from aws.modules.cqrs.command_bus import CommandBus
from aws.modules.cqrs.query_bus import QueryBus
from aws.modules.cqrs.event_store_cqrs import EventStoreCQRS

__all__ = [
    "CommandBus",
    "QueryBus",
    "EventStoreCQRS",
]















