"""
Command and Query Handlers
Handlers process commands and queries
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Dict, Callable
import logging

from .commands import Command
from .queries import Query

logger = logging.getLogger(__name__)

TCommand = TypeVar('TCommand', bound=Command)
TQuery = TypeVar('TQuery', bound=Query)
TResult = TypeVar('TResult')


class CommandHandler(ABC, Generic[TCommand, TResult]):
    """Base class for command handlers"""
    
    @abstractmethod
    async def handle(self, command: TCommand) -> TResult:
        """Handle command"""
        pass


class QueryHandler(ABC, Generic[TQuery, TResult]):
    """Base class for query handlers"""
    
    @abstractmethod
    async def handle(self, query: TQuery) -> TResult:
        """Handle query"""
        pass


class CommandBus:
    """
    Command bus for dispatching commands to handlers.
    Implements command pattern with dependency injection.
    """
    
    def __init__(self):
        self.handlers: Dict[Type[Command], CommandHandler] = {}
    
    def register_handler(
        self,
        command_type: Type[Command],
        handler: CommandHandler
    ):
        """Register command handler"""
        self.handlers[command_type] = handler
        logger.debug(f"Registered handler for {command_type.__name__}")
    
    async def dispatch(self, command: Command) -> Any:
        """
        Dispatch command to appropriate handler
        
        Args:
            command: Command to dispatch
            
        Returns:
            Handler result
            
        Raises:
            ValueError: If no handler registered for command type
        """
        command_type = type(command)
        handler = self.handlers.get(command_type)
        
        if not handler:
            raise ValueError(f"No handler registered for {command_type.__name__}")
        
        logger.debug(f"Dispatching {command_type.__name__}: {command.command_id if hasattr(command, 'command_id') else 'N/A'}")
        
        try:
            result = await handler.handle(command)
            logger.debug(f"Command {command_type.__name__} handled successfully")
            return result
        except Exception as e:
            logger.error(f"Error handling command {command_type.__name__}: {e}", exc_info=True)
            raise


class QueryBus:
    """
    Query bus for dispatching queries to handlers.
    Implements query pattern with dependency injection.
    """
    
    def __init__(self):
        self.handlers: Dict[Type[Query], QueryHandler] = {}
    
    def register_handler(
        self,
        query_type: Type[Query],
        handler: QueryHandler
    ):
        """Register query handler"""
        self.handlers[query_type] = handler
        logger.debug(f"Registered handler for {query_type.__name__}")
    
    async def dispatch(self, query: Query) -> Any:
        """
        Dispatch query to appropriate handler
        
        Args:
            query: Query to dispatch
            
        Returns:
            Handler result
            
        Raises:
            ValueError: If no handler registered for query type
        """
        query_type = type(query)
        handler = self.handlers.get(query_type)
        
        if not handler:
            raise ValueError(f"No handler registered for {query_type.__name__}")
        
        logger.debug(f"Dispatching {query_type.__name__}")
        
        try:
            result = await handler.handle(query)
            logger.debug(f"Query {query_type.__name__} handled successfully")
            return result
        except Exception as e:
            logger.error(f"Error handling query {query_type.__name__}: {e}", exc_info=True)
            raise


# Global buses
_command_bus: CommandBus = None
_query_bus: QueryBus = None


def get_command_bus() -> CommandBus:
    """Get or create global command bus"""
    global _command_bus
    if _command_bus is None:
        _command_bus = CommandBus()
    return _command_bus


def get_query_bus() -> QueryBus:
    """Get or create global query bus"""
    global _query_bus
    if _query_bus is None:
        _query_bus = QueryBus()
    return _query_bus















