"""
MCP CQRS - Command Query Responsibility Segregation
=====================================================
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Protocol
else:
    Protocol = object
from pydantic import BaseModel, Field
from datetime import datetime

from .exceptions import MCPError

logger = logging.getLogger(__name__)


class Command(BaseModel):
    """Comando CQRS"""
    command_id: str = Field(..., description="ID único del comando")
    command_type: str = Field(..., description="Tipo de comando")
    payload: Dict[str, Any] = Field(..., description="Payload del comando")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Query(BaseModel):
    """Query CQRS"""
    query_id: str = Field(..., description="ID único de la query")
    query_type: str = Field(..., description="Tipo de query")
    parameters: Dict[str, Any] = Field(..., description="Parámetros de la query")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CommandHandler(ABC):
    """Handler abstracto para comandos"""
    
    @abstractmethod
    async def handle(self, command: Command) -> Any:
        """Maneja un comando"""
        pass


class QueryHandler(ABC):
    """Handler abstracto para queries"""
    
    @abstractmethod
    async def handle(self, query: Query) -> Any:
        """Maneja una query"""
        pass


class CQRSBus:
    """
    Bus CQRS
    
    Separa comandos (write) de queries (read) para mejor escalabilidad.
    """
    
    def __init__(self):
        self._command_handlers: Dict[str, CommandHandler] = {}
        self._query_handlers: Dict[str, QueryHandler] = {}
    
    def register_command_handler(self, command_type: str, handler: CommandHandler):
        """
        Registra handler de comando
        
        Args:
            command_type: Tipo de comando
            handler: Handler del comando
        """
        self._command_handlers[command_type] = handler
        logger.info(f"Registered command handler: {command_type}")
    
    def register_query_handler(self, query_type: str, handler: QueryHandler):
        """
        Registra handler de query
        
        Args:
            query_type: Tipo de query
            handler: Handler de la query
        """
        self._query_handlers[query_type] = handler
        logger.info(f"Registered query handler: {query_type}")
    
    async def execute_command(self, command: Command) -> Any:
        """
        Ejecuta un comando
        
        Args:
            command: Comando a ejecutar
            
        Returns:
            Resultado del comando
        """
        handler = self._command_handlers.get(command.command_type)
        if not handler:
            raise MCPError(f"No handler registered for command type: {command.command_type}")
        
        return await handler.handle(command)
    
    async def execute_query(self, query: Query) -> Any:
        """
        Ejecuta una query
        
        Args:
            query: Query a ejecutar
            
        Returns:
            Resultado de la query
        """
        handler = self._query_handlers.get(query.query_type)
        if not handler:
            raise MCPError(f"No handler registered for query type: {query.query_type}")
        
        return await handler.handle(query)

