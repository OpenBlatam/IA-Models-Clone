"""
Status Management Utilities for Piel Mejorador AI SAM3
======================================================

Unified status management and state tracking utilities.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class Status(Enum):
    """Generic status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"


@dataclass
class StatusInfo:
    """Status information."""
    status: Status
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "error": self.error,
        }


class StatusManager:
    """Manager for status tracking."""
    
    def __init__(self, initial_status: Status = Status.PENDING):
        """
        Initialize status manager.
        
        Args:
            initial_status: Initial status
        """
        self._status = initial_status
        self._history: List[StatusInfo] = []
        self._callbacks: Dict[Status, List[Callable[[StatusInfo], None]]] = {}
        self._add_status(StatusInfo(initial_status))
    
    @property
    def status(self) -> Status:
        """Get current status."""
        return self._status
    
    @property
    def current_info(self) -> StatusInfo:
        """Get current status info."""
        return self._history[-1] if self._history else StatusInfo(Status.PENDING)
    
    @property
    def history(self) -> List[StatusInfo]:
        """Get status history."""
        return self._history.copy()
    
    def set_status(
        self,
        status: Status,
        message: Optional[str] = None,
        error: Optional[str] = None,
        **metadata
    ):
        """
        Set status.
        
        Args:
            status: New status
            message: Optional status message
            error: Optional error message
            **metadata: Additional metadata
        """
        old_status = self._status
        self._status = status
        
        info = StatusInfo(
            status=status,
            message=message,
            error=error,
            metadata=metadata
        )
        
        self._add_status(info)
        
        # Trigger callbacks
        if status in self._callbacks:
            for callback in self._callbacks[status]:
                try:
                    callback(info)
                except Exception as e:
                    logger.error(f"Error in status callback: {e}")
        
        logger.debug(f"Status changed: {old_status.value} -> {status.value}")
    
    def _add_status(self, info: StatusInfo):
        """Add status to history."""
        self._history.append(info)
        # Keep only last 100 status changes
        if len(self._history) > 100:
            self._history = self._history[-100:]
    
    def on_status(
        self,
        status: Status,
        callback: Callable[[StatusInfo], None]
    ):
        """
        Register callback for status change.
        
        Args:
            status: Status to listen for
            callback: Callback function
        """
        if status not in self._callbacks:
            self._callbacks[status] = []
        self._callbacks[status].append(callback)
    
    def is_completed(self) -> bool:
        """Check if status is completed."""
        return self._status == Status.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if status is failed."""
        return self._status == Status.FAILED
    
    def is_in_progress(self) -> bool:
        """Check if status is in progress."""
        return self._status == Status.IN_PROGRESS
    
    def is_pending(self) -> bool:
        """Check if status is pending."""
        return self._status == Status.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_status": self._status.value,
            "current_info": self.current_info.to_dict(),
            "history_count": len(self._history),
            "recent_history": [info.to_dict() for info in self._history[-10:]],
        }


class StatusUtils:
    """Unified status utilities."""
    
    @staticmethod
    def create_manager(initial_status: Status = Status.PENDING) -> StatusManager:
        """
        Create status manager.
        
        Args:
            initial_status: Initial status
            
        Returns:
            StatusManager
        """
        return StatusManager(initial_status)
    
    @staticmethod
    def is_terminal_status(status: Status) -> bool:
        """
        Check if status is terminal (completed, failed, cancelled).
        
        Args:
            status: Status to check
            
        Returns:
            True if terminal
        """
        return status in (Status.COMPLETED, Status.FAILED, Status.CANCELLED)
    
    @staticmethod
    def is_active_status(status: Status) -> bool:
        """
        Check if status is active (in_progress, retrying).
        
        Args:
            status: Status to check
            
        Returns:
            True if active
        """
        return status in (Status.IN_PROGRESS, Status.RETRYING)
    
    @staticmethod
    def can_transition(from_status: Status, to_status: Status) -> bool:
        """
        Check if status transition is valid.
        
        Args:
            from_status: Current status
            to_status: Target status
            
        Returns:
            True if transition is valid
        """
        # Terminal states can't transition
        if StatusUtils.is_terminal_status(from_status):
            return False
        
        # Any non-terminal can transition to any other status
        return True


# Convenience functions
def create_status_manager(initial_status: Status = Status.PENDING) -> StatusManager:
    """Create status manager."""
    return StatusUtils.create_manager(initial_status)


def is_terminal_status(status: Status) -> bool:
    """Check if terminal status."""
    return StatusUtils.is_terminal_status(status)


def is_active_status(status: Status) -> bool:
    """Check if active status."""
    return StatusUtils.is_active_status(status)




