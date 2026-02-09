"""
CQRS Pattern System
===================

Sistema de Command Query Responsibility Segregation (CQRS).
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class Command:
    """Comando."""
    command_id: str
    command_type: str
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    """Query."""
    query_id: str
    query_type: str
    parameters: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandResult:
    """Resultado de comando."""
    command_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class QueryResult:
    """Resultado de query."""
    query_id: str
    data: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CommandHandler:
    """Manejador de comandos."""
    
    def __init__(self, handler_func: Callable):
        """
        Inicializar manejador.
        
        Args:
            handler_func: Función que maneja el comando
        """
        self.handler_func = handler_func
    
    async def handle(self, command: Command) -> CommandResult:
        """
        Manejar comando.
        
        Args:
            command: Comando a manejar
            
        Returns:
            Resultado del comando
        """
        try:
            result = await self.handler_func(command.payload)
            return CommandResult(
                command_id=command.command_id,
                success=True,
                result=result
            )
        except Exception as e:
            logger.error(f"Error handling command {command.command_type}: {e}", exc_info=True)
            return CommandResult(
                command_id=command.command_id,
                success=False,
                error=str(e)
            )


class QueryHandler:
    """Manejador de queries."""
    
    def __init__(self, handler_func: Callable):
        """
        Inicializar manejador.
        
        Args:
            handler_func: Función que maneja la query
        """
        self.handler_func = handler_func
    
    async def handle(self, query: Query) -> QueryResult:
        """
        Manejar query.
        
        Args:
            query: Query a manejar
            
        Returns:
            Resultado de la query
        """
        try:
            data = await self.handler_func(query.parameters)
            return QueryResult(
                query_id=query.query_id,
                data=data
            )
        except Exception as e:
            logger.error(f"Error handling query {query.query_type}: {e}", exc_info=True)
            raise


class CQRSSystem:
    """
    Sistema CQRS.
    
    Gestiona comandos y queries separadamente.
    """
    
    def __init__(self):
        """Inicializar sistema CQRS."""
        self.command_handlers: Dict[str, CommandHandler] = {}
        self.query_handlers: Dict[str, QueryHandler] = {}
        self.command_history: List[Command] = []
        self.query_history: List[Query] = []
        self.max_history = 10000
    
    def register_command_handler(
        self,
        command_type: str,
        handler_func: Callable
    ) -> None:
        """
        Registrar manejador de comando.
        
        Args:
            command_type: Tipo de comando
            handler_func: Función que maneja el comando
        """
        self.command_handlers[command_type] = CommandHandler(handler_func)
        logger.info(f"Registered command handler: {command_type}")
    
    def register_query_handler(
        self,
        query_type: str,
        handler_func: Callable
    ) -> None:
        """
        Registrar manejador de query.
        
        Args:
            query_type: Tipo de query
            handler_func: Función que maneja la query
        """
        self.query_handlers[query_type] = QueryHandler(handler_func)
        logger.info(f"Registered query handler: {query_type}")
    
    async def execute_command(
        self,
        command_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> CommandResult:
        """
        Ejecutar comando.
        
        Args:
            command_type: Tipo de comando
            payload: Datos del comando
            metadata: Metadata adicional
            
        Returns:
            Resultado del comando
        """
        import uuid
        command_id = str(uuid.uuid4())
        
        command = Command(
            command_id=command_id,
            command_type=command_type,
            payload=payload,
            metadata=metadata or {}
        )
        
        self.command_history.append(command)
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]
        
        if command_type not in self.command_handlers:
            return CommandResult(
                command_id=command_id,
                success=False,
                error=f"Command handler not found: {command_type}"
            )
        
        handler = self.command_handlers[command_type]
        result = await handler.handle(command)
        
        return result
    
    async def execute_query(
        self,
        query_type: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """
        Ejecutar query.
        
        Args:
            query_type: Tipo de query
            parameters: Parámetros de la query
            metadata: Metadata adicional
            
        Returns:
            Resultado de la query
        """
        import uuid
        query_id = str(uuid.uuid4())
        
        query = Query(
            query_id=query_id,
            query_type=query_type,
            parameters=parameters,
            metadata=metadata or {}
        )
        
        self.query_history.append(query)
        if len(self.query_history) > self.max_history:
            self.query_history = self.query_history[-self.max_history:]
        
        if query_type not in self.query_handlers:
            raise ValueError(f"Query handler not found: {query_type}")
        
        handler = self.query_handlers[query_type]
        result = await handler.handle(query)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema CQRS."""
        command_types = {}
        query_types = {}
        
        for cmd in self.command_history:
            command_types[cmd.command_type] = command_types.get(cmd.command_type, 0) + 1
        
        for qry in self.query_history:
            query_types[qry.query_type] = query_types.get(qry.query_type, 0) + 1
        
        return {
            "total_commands": len(self.command_history),
            "total_queries": len(self.query_history),
            "registered_command_handlers": len(self.command_handlers),
            "registered_query_handlers": len(self.query_handlers),
            "commands_by_type": command_types,
            "queries_by_type": query_types
        }


# Instancia global
_cqrs_system: Optional[CQRSSystem] = None


def get_cqrs_system() -> CQRSSystem:
    """Obtener instancia global del sistema CQRS."""
    global _cqrs_system
    if _cqrs_system is None:
        _cqrs_system = CQRSSystem()
    return _cqrs_system


