"""
Command Pattern - Encapsulate operations
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ICommand(ABC):
    """
    Interface for commands
    """
    
    @abstractmethod
    def execute(self) -> Any:
        """Execute command"""
        pass
    
    @abstractmethod
    def undo(self) -> Any:
        """Undo command"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Command name"""
        pass


@dataclass
class CommandResult:
    """Command execution result"""
    success: bool
    data: Any
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class CommandInvoker:
    """
    Invokes commands and manages execution
    """
    
    def __init__(self):
        self.history: List[ICommand] = []
        self.max_history: int = 100
    
    def execute(self, command: ICommand) -> CommandResult:
        """
        Execute a command
        
        Args:
            command: Command to execute
        
        Returns:
            Command result
        """
        try:
            logger.info(f"Executing command: {command.name}")
            result = command.execute()
            
            # Add to history
            self.history.append(command)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            return CommandResult(success=True, data=result)
        
        except Exception as e:
            logger.error(f"Command execution failed: {str(e)}")
            return CommandResult(success=False, data=None, error=str(e))
    
    def undo_last(self) -> CommandResult:
        """Undo last command"""
        if not self.history:
            return CommandResult(success=False, data=None, error="No commands to undo")
        
        command = self.history.pop()
        try:
            result = command.undo()
            logger.info(f"Undid command: {command.name}")
            return CommandResult(success=True, data=result)
        except Exception as e:
            logger.error(f"Undo failed: {str(e)}")
            return CommandResult(success=False, data=None, error=str(e))
    
    def clear_history(self):
        """Clear command history"""
        self.history.clear()


class CommandHistory:
    """
    Manages command history
    """
    
    def __init__(self):
        self.commands: List[ICommand] = []
        self.results: List[CommandResult] = []
    
    def add(self, command: ICommand, result: CommandResult):
        """Add command and result to history"""
        self.commands.append(command)
        self.results.append(result)
    
    def get_history(self) -> List[tuple]:
        """Get full history"""
        return list(zip(self.commands, self.results))
    
    def clear(self):
        """Clear history"""
        self.commands.clear()
        self.results.clear()








