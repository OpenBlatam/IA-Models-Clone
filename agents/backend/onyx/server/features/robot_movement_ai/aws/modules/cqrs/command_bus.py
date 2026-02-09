"""
Command Bus
===========

CQRS command bus.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Command:
    """Command definition."""
    type: str
    payload: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class CommandBus:
    """Command bus for CQRS."""
    
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._middleware: list[Callable] = []
        self._command_history: list[Command] = []
    
    def register_handler(self, command_type: str, handler: Callable):
        """Register command handler."""
        self._handlers[command_type] = handler
        logger.info(f"Registered command handler: {command_type}")
    
    def add_middleware(self, middleware: Callable):
        """Add command middleware."""
        self._middleware.append(middleware)
        logger.info("Added command middleware")
    
    async def dispatch(self, command: Command) -> Any:
        """Dispatch command."""
        if command.type not in self._handlers:
            raise ValueError(f"No handler registered for command: {command.type}")
        
        handler = self._handlers[command.type]
        
        # Apply middleware
        for middleware in self._middleware:
            handler = middleware(handler)
        
        # Execute handler
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(command.payload)
            else:
                result = await asyncio.to_thread(handler, command.payload)
            
            self._command_history.append(command)
            return result
        
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise
    
    def get_command_history(self, limit: int = 100) -> list[Command]:
        """Get command history."""
        return self._command_history[-limit:]















