"""
CQRS Pattern - Command Query Responsibility Segregation
======================================================

Implementación del patrón CQRS:
- Command handlers
- Query handlers
- Separate read/write models
- Event sourcing integration
"""

import logging
from typing import Optional, Dict, Any, List, Callable, TypeVar, Generic
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Command(ABC):
    """Base class para commands"""
    pass


class Query(ABC):
    """Base class para queries"""
    pass


class CommandHandler(ABC, Generic[T, R]):
    """Handler para commands"""
    
    @abstractmethod
    async def handle(self, command: T) -> R:
        """Maneja un command"""
        pass


class QueryHandler(ABC, Generic[T, R]):
    """Handler para queries"""
    
    @abstractmethod
    async def handle(self, query: T) -> R:
        """Maneja un query"""
        pass


class CQRSBus:
    """
    Bus CQRS para routing de commands y queries.
    """
    
    def __init__(self) -> None:
        self.command_handlers: Dict[type, CommandHandler] = {}
        self.query_handlers: Dict[type, QueryHandler] = {}
    
    def register_command_handler(
        self,
        command_type: type,
        handler: CommandHandler
    ) -> None:
        """Registra handler de command"""
        self.command_handlers[command_type] = handler
        logger.info(f"Command handler registered for {command_type.__name__}")
    
    def register_query_handler(
        self,
        query_type: type,
        handler: QueryHandler
    ) -> None:
        """Registra handler de query"""
        self.query_handlers[query_type] = handler
        logger.info(f"Query handler registered for {query_type.__name__}")
    
    async def execute_command(self, command: Command) -> Any:
        """Ejecuta un command"""
        handler = self.command_handlers.get(type(command))
        if not handler:
            raise ValueError(f"No handler registered for {type(command).__name__}")
        
        return await handler.handle(command)
    
    async def execute_query(self, query: Query) -> Any:
        """Ejecuta un query"""
        handler = self.query_handlers.get(type(query))
        if not handler:
            raise ValueError(f"No handler registered for {type(query).__name__}")
        
        return await handler.handle(query)


class CreateProjectCommand(Command):
    """Command para crear proyecto"""
    
    def __init__(
        self,
        description: str,
        author: str,
        **kwargs: Any
    ) -> None:
        self.description = description
        self.author = author
        self.kwargs = kwargs


class GetProjectQuery(Query):
    """Query para obtener proyecto"""
    
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id


def get_cqrs_bus() -> CQRSBus:
    """Obtiene bus CQRS"""
    return CQRSBus()















