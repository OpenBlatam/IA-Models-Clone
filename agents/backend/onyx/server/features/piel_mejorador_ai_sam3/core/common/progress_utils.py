"""
Progress Tracking Utilities for Piel Mejorador AI SAM3
======================================================

Unified progress tracking and reporting utilities.
"""

import logging
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ProgressStatus(Enum):
    """Progress status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressTracker:
    """Progress tracker for operations."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: ProgressStatus = ProgressStatus.PENDING
    current_item: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start(self):
        """Start tracking."""
        self.start_time = datetime.now()
        self.status = ProgressStatus.IN_PROGRESS
    
    def complete(self):
        """Mark as completed."""
        self.end_time = datetime.now()
        self.status = ProgressStatus.COMPLETED
    
    def fail(self):
        """Mark as failed."""
        self.end_time = datetime.now()
        self.status = ProgressStatus.FAILED
    
    def cancel(self):
        """Mark as cancelled."""
        self.end_time = datetime.now()
        self.status = ProgressStatus.CANCELLED
    
    def increment(self, count: int = 1):
        """Increment completed count."""
        self.completed += count
    
    def increment_failed(self, count: int = 1):
        """Increment failed count."""
        self.failed += count
    
    def increment_skipped(self, count: int = 1):
        """Increment skipped count."""
        self.skipped += count
    
    @property
    def percentage(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100.0
    
    @property
    def remaining(self) -> int:
        """Get remaining count."""
        return max(0, self.total - self.completed - self.failed - self.skipped)
    
    @property
    def elapsed_seconds(self) -> Optional[float]:
        """Get elapsed time in seconds."""
        if not self.start_time:
            return None
        
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def estimated_remaining_seconds(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        if self.completed == 0 or not self.start_time:
            return None
        
        elapsed = self.elapsed_seconds
        if elapsed is None or elapsed == 0:
            return None
        
        rate = self.completed / elapsed
        if rate == 0:
            return None
        
        return self.remaining / rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "remaining": self.remaining,
            "percentage": self.percentage,
            "status": self.status.value,
            "current_item": self.current_item,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
        }


class ProgressUtils:
    """Unified progress tracking utilities."""
    
    @staticmethod
    def create_tracker(total: int, **kwargs) -> ProgressTracker:
        """
        Create progress tracker.
        
        Args:
            total: Total number of items
            **kwargs: Additional tracker parameters
            
        Returns:
            ProgressTracker
        """
        tracker = ProgressTracker(total=total, **kwargs)
        tracker.start()
        return tracker
    
    @staticmethod
    def format_progress(tracker: ProgressTracker) -> str:
        """
        Format progress as string.
        
        Args:
            tracker: Progress tracker
            
        Returns:
            Formatted progress string
        """
        percentage = tracker.percentage
        elapsed = tracker.elapsed_seconds or 0
        remaining = tracker.estimated_remaining_seconds
        
        status_str = f"{tracker.completed}/{tracker.total} ({percentage:.1f}%)"
        
        if elapsed > 0:
            status_str += f" - {elapsed:.1f}s elapsed"
        
        if remaining is not None:
            status_str += f" - ~{remaining:.1f}s remaining"
        
        if tracker.failed > 0:
            status_str += f" - {tracker.failed} failed"
        
        if tracker.skipped > 0:
            status_str += f" - {tracker.skipped} skipped"
        
        return status_str
    
    @staticmethod
    def create_progress_callback(
        tracker: ProgressTracker,
        callback: Optional[Callable[[ProgressTracker], None]] = None
    ) -> Callable[[bool, Optional[str]], None]:
        """
        Create progress callback function.
        
        Args:
            tracker: Progress tracker
            callback: Optional callback function
            
        Returns:
            Callback function (success: bool, item: Optional[str]) -> None
        """
        def progress_callback(success: bool, item: Optional[str] = None):
            tracker.current_item = item
            
            if success:
                tracker.increment()
            else:
                tracker.increment_failed()
            
            if callback:
                callback(tracker)
            else:
                logger.info(ProgressUtils.format_progress(tracker))
        
        return progress_callback


# Convenience functions
def create_tracker(total: int, **kwargs) -> ProgressTracker:
    """Create progress tracker."""
    return ProgressUtils.create_tracker(total, **kwargs)


def format_progress(tracker: ProgressTracker) -> str:
    """Format progress."""
    return ProgressUtils.format_progress(tracker)




