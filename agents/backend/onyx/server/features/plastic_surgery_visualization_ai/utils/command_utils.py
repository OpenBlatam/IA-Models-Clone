"""Command pattern utilities."""

from typing import Any, Callable, Optional, Dict, List
from abc import ABC, abstractmethod
from datetime import datetime
import uuid


class Command(ABC):
    """Base command interface."""
    
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        self.executed = False
    
    @abstractmethod
    def execute(self) -> Any:
        """Execute command."""
        pass
    
    @abstractmethod
    def undo(self) -> Any:
        """Undo command."""
        pass
    
    def can_execute(self) -> bool:
        """Check if command can be executed."""
        return not self.executed


class SimpleCommand(Command):
    """Simple command with execute and undo functions."""
    
    def __init__(self, execute_func: Callable, undo_func: Optional[Callable] = None, *args, **kwargs):
        super().__init__()
        self.execute_func = execute_func
        self.undo_func = undo_func
        self.args = args
        self.kwargs = kwargs
        self.result = None
    
    def execute(self) -> Any:
        """Execute command."""
        if not self.can_execute():
            raise ValueError("Command already executed")
        
        self.result = self.execute_func(*self.args, **self.kwargs)
        self.executed = True
        return self.result
    
    def undo(self) -> Any:
        """Undo command."""
        if not self.executed:
            raise ValueError("Command not executed")
        
        if self.undo_func:
            result = self.undo_func(*self.args, **self.kwargs)
            self.executed = False
            return result
        raise NotImplementedError("Undo not implemented")


class CommandInvoker:
    """Invoker for commands with history."""
    
    def __init__(self):
        self.history: List[Command] = []
        self.undo_stack: List[Command] = []
    
    def execute(self, command: Command) -> Any:
        """Execute command and add to history."""
        result = command.execute()
        self.history.append(command)
        self.undo_stack.clear()
        return result
    
    def undo(self) -> Optional[Any]:
        """Undo last command."""
        if not self.history:
            return None
        
        command = self.history.pop()
        result = command.undo()
        self.undo_stack.append(command)
        return result
    
    def redo(self) -> Optional[Any]:
        """Redo last undone command."""
        if not self.undo_stack:
            return None
        
        command = self.undo_stack.pop()
        result = command.execute()
        self.history.append(command)
        return result
    
    def clear_history(self):
        """Clear command history."""
        self.history.clear()
        self.undo_stack.clear()
    
    def get_history(self) -> List[Command]:
        """Get command history."""
        return self.history.copy()


class MacroCommand(Command):
    """Command that executes multiple commands."""
    
    def __init__(self, commands: List[Command]):
        super().__init__()
        self.commands = commands
        self.executed_commands: List[Command] = []
    
    def execute(self) -> List[Any]:
        """Execute all commands."""
        results = []
        for command in self.commands:
            result = command.execute()
            results.append(result)
            self.executed_commands.append(command)
        self.executed = True
        return results
    
    def undo(self) -> List[Any]:
        """Undo all commands in reverse order."""
        results = []
        for command in reversed(self.executed_commands):
            result = command.undo()
            results.append(result)
        self.executed_commands.clear()
        self.executed = False
        return results


class AsyncCommand(Command):
    """Async command interface."""
    
    @abstractmethod
    async def execute_async(self) -> Any:
        """Execute command asynchronously."""
        pass
    
    @abstractmethod
    async def undo_async(self) -> Any:
        """Undo command asynchronously."""
        pass
    
    def execute(self) -> Any:
        """Sync execute (raises error)."""
        raise NotImplementedError("Use execute_async() for async commands")
    
    def undo(self) -> Any:
        """Sync undo (raises error)."""
        raise NotImplementedError("Use undo_async() for async commands")


class CommandQueue:
    """Queue for command execution."""
    
    def __init__(self):
        self.queue: List[Command] = []
        self.running = False
    
    def enqueue(self, command: Command):
        """Add command to queue."""
        self.queue.append(command)
    
    def dequeue(self) -> Optional[Command]:
        """Remove and return next command."""
        if self.queue:
            return self.queue.pop(0)
        return None
    
    def execute_all(self) -> List[Any]:
        """Execute all commands in queue."""
        results = []
        while self.queue:
            command = self.dequeue()
            if command:
                results.append(command.execute())
        return results
    
    def clear(self):
        """Clear queue."""
        self.queue.clear()


def command(func: Callable) -> Callable:
    """Decorator to create command from function."""
    class FunctionCommand(Command):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args = args
            self.kwargs = kwargs
            self.result = None
        
        def execute(self) -> Any:
            self.result = func(*self.args, **self.kwargs)
            self.executed = True
            return self.result
        
        def undo(self) -> Any:
            raise NotImplementedError("Undo not supported for function commands")
    
    return FunctionCommand


