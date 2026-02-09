"""
Command Handler
===============

IoT command handling.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CommandStatus(Enum):
    """Command status."""
    PENDING = "pending"
    SENT = "sent"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class Command:
    """IoT command."""
    id: str
    device_id: str
    command: str
    parameters: Dict[str, Any]
    status: CommandStatus = CommandStatus.PENDING
    created_at: datetime = None
    executed_at: Optional[datetime] = None
    result: Any = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class CommandHandler:
    """IoT command handler."""
    
    def __init__(self):
        self._commands: Dict[str, Command] = {}
        self._handlers: Dict[str, Callable] = {}
    
    def register_handler(self, command_type: str, handler: Callable):
        """Register command handler."""
        self._handlers[command_type] = handler
        logger.info(f"Registered handler for command: {command_type}")
    
    def create_command(
        self,
        command_id: str,
        device_id: str,
        command: str,
        parameters: Dict[str, Any]
    ) -> Command:
        """Create command."""
        cmd = Command(
            id=command_id,
            device_id=device_id,
            command=command,
            parameters=parameters
        )
        
        self._commands[command_id] = cmd
        logger.info(f"Created command: {command_id} for device {device_id}")
        return cmd
    
    async def execute_command(self, command_id: str) -> Any:
        """Execute command."""
        if command_id not in self._commands:
            raise ValueError(f"Command {command_id} not found")
        
        cmd = self._commands[command_id]
        
        if cmd.command not in self._handlers:
            cmd.status = CommandStatus.FAILED
            raise ValueError(f"No handler for command: {cmd.command}")
        
        handler = self._handlers[cmd.command]
        cmd.status = CommandStatus.SENT
        
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(cmd.device_id, cmd.parameters)
            else:
                result = handler(cmd.device_id, cmd.parameters)
            
            cmd.status = CommandStatus.EXECUTED
            cmd.executed_at = datetime.now()
            cmd.result = result
            
            return result
        
        except Exception as e:
            cmd.status = CommandStatus.FAILED
            cmd.executed_at = datetime.now()
            logger.error(f"Command {command_id} failed: {e}")
            raise
    
    def get_command(self, command_id: str) -> Optional[Command]:
        """Get command by ID."""
        return self._commands.get(command_id)
    
    def get_command_stats(self) -> Dict[str, Any]:
        """Get command statistics."""
        return {
            "total_commands": len(self._commands),
            "by_status": {
                status.value: sum(1 for c in self._commands.values() if c.status == status)
                for status in CommandStatus
            }
        }


# Import asyncio
import asyncio















