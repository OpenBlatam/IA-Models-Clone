"""
Command Pattern Utilities for Piel Mejorador AI SAM3
===================================================

Unified command pattern implementation utilities.
"""

import asyncio
import logging
from typing import TypeVar, Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Command(ABC):
    """Base command interface."""
    
    @abstractmethod
    async def execute(self) -> Any:
        """Execute command."""
        pass
    
    @abstractmethod
    async def undo(self) -> Any:
        """Undo command (if supported)."""
        pass
    
    def can_undo(self) -> bool:
        """Check if command can be undone."""
        return False


@dataclass
class CommandResult:
    """Command execution result."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
        }


class SimpleCommand(Command):
    """Simple command implementation."""
    
    def __init__(
        self,
        execute_func: Callable[[], Any],
        undo_func: Optional[Callable[[], Any]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize simple command.
        
        Args:
            execute_func: Function to execute
            undo_func: Optional undo function
            name: Optional command name
        """
        self._execute_func = execute_func
        self._undo_func = undo_func
        self.name = name or execute_func.__name__
        self._supports_undo = undo_func is not None
    
    async def execute(self) -> Any:
        """Execute command."""
        if asyncio.iscoroutinefunction(self._execute_func):
            return await self._execute_func()
        return self._execute_func()
    
    async def undo(self) -> Any:
        """Undo command."""
        if not self._undo_func:
            raise NotImplementedError("Undo not supported for this command")
        
        if asyncio.iscoroutinefunction(self._undo_func):
            return await self._undo_func()
        return self._undo_func()
    
    def can_undo(self) -> bool:
        """Check if can undo."""
        return self._supports_undo


class CommandInvoker:
    """Command invoker with history and undo support."""
    
    def __init__(self):
        """Initialize invoker."""
        self._history: List[Command] = []
        self._max_history: int = 100
    
    async def execute(self, command: Command) -> CommandResult:
        """
        Execute command.
        
        Args:
            command: Command to execute
            
        Returns:
            CommandResult
        """
        start_time = datetime.now()
        
        try:
            result = await command.execute()
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Add to history if undoable
            if command.can_undo():
                self._history.append(command)
                if len(self._history) > self._max_history:
                    self._history.pop(0)
            
            return CommandResult(
                success=True,
                result=result,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Command execution failed: {e}")
            return CommandResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def undo_last(self) -> CommandResult:
        """
        Undo last command.
        
        Returns:
            CommandResult
        """
        if not self._history:
            return CommandResult(
                success=False,
                error="No commands to undo"
            )
        
        command = self._history.pop()
        start_time = datetime.now()
        
        try:
            result = await command.undo()
            execution_time = (datetime.now() - start_time).total_seconds()
            return CommandResult(
                success=True,
                result=result,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Command undo failed: {e}")
            return CommandResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def clear_history(self):
        """Clear command history."""
        self._history.clear()
    
    def get_history(self) -> List[Command]:
        """Get command history."""
        return self._history.copy()


class CommandUtils:
    """Unified command pattern utilities."""
    
    @staticmethod
    def create_command(
        execute_func: Callable[[], Any],
        undo_func: Optional[Callable[[], Any]] = None,
        name: Optional[str] = None
    ) -> SimpleCommand:
        """
        Create simple command.
        
        Args:
            execute_func: Function to execute
            undo_func: Optional undo function
            name: Optional command name
            
        Returns:
            SimpleCommand
        """
        return SimpleCommand(execute_func, undo_func, name)
    
    @staticmethod
    def create_invoker() -> CommandInvoker:
        """
        Create command invoker.
        
        Returns:
            CommandInvoker
        """
        return CommandInvoker()


# Convenience functions
def create_command(execute_func: Callable[[], Any], **kwargs) -> SimpleCommand:
    """Create command."""
    return CommandUtils.create_command(execute_func, **kwargs)


def create_invoker() -> CommandInvoker:
    """Create invoker."""
    return CommandUtils.create_invoker()




