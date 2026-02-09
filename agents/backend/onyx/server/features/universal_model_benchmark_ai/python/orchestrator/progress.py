"""
Progress Tracking - Utilities for tracking benchmark execution progress.

Provides:
- Progress bar management
- Task tracking
- Progress callbacks
- Rich console integration
"""

import time
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class TaskProgress:
    """Progress information for a single task."""
    task_id: str
    description: str
    total: int = 0
    completed: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None
    
    def start(self):
        """Mark task as started."""
        self.status = "running"
        self.start_time = time.time()
    
    def complete(self):
        """Mark task as completed."""
        self.status = "completed"
        self.end_time = time.time()
        self.completed = self.total
    
    def fail(self, error: str):
        """Mark task as failed."""
        self.status = "failed"
        self.end_time = time.time()
        self.error = error
    
    def update(self, completed: int):
        """Update progress."""
        self.completed = min(completed, self.total)
    
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100.0


class ProgressTracker:
    """Track progress of multiple tasks."""
    
    def __init__(self, show_progress: bool = True):
        """
        Initialize progress tracker.
        
        Args:
            show_progress: Whether to show progress bars
        """
        self.show_progress = show_progress
        self.tasks: Dict[str, TaskProgress] = {}
        self.progress: Optional[Progress] = None
        self.task_ids: Dict[str, int] = {}  # Map task_id to progress task ID
    
    def add_task(
        self,
        task_id: str,
        description: str,
        total: int = 0,
    ) -> str:
        """
        Add a new task to track.
        
        Args:
            task_id: Unique task identifier
            description: Task description
            total: Total number of items (0 for indeterminate)
        
        Returns:
            Task ID
        """
        task = TaskProgress(
            task_id=task_id,
            description=description,
            total=total,
        )
        self.tasks[task_id] = task
        
        if self.show_progress and self.progress is None:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            )
            self.progress.start()
        
        if self.progress is not None:
            progress_task_id = self.progress.add_task(
                description,
                total=total if total > 0 else None,
            )
            self.task_ids[task_id] = progress_task_id
        
        return task_id
    
    def start_task(self, task_id: str):
        """Mark a task as started."""
        if task_id in self.tasks:
            self.tasks[task_id].start()
            
            if self.progress is not None and task_id in self.task_ids:
                self.progress.update(
                    self.task_ids[task_id],
                    description=f"[cyan]{self.tasks[task_id].description}",
                )
    
    def update_task(self, task_id: str, completed: int):
        """Update task progress."""
        if task_id in self.tasks:
            self.tasks[task_id].update(completed)
            
            if self.progress is not None and task_id in self.task_ids:
                self.progress.update(
                    self.task_ids[task_id],
                    completed=completed,
                )
    
    def complete_task(self, task_id: str):
        """Mark a task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id].complete()
            
            if self.progress is not None and task_id in self.task_ids:
                self.progress.update(
                    self.task_ids[task_id],
                    completed=self.tasks[task_id].total,
                    description=f"[green]✓ {self.tasks[task_id].description}",
                )
    
    def fail_task(self, task_id: str, error: str):
        """Mark a task as failed."""
        if task_id in self.tasks:
            self.tasks[task_id].fail(error)
            
            if self.progress is not None and task_id in self.task_ids:
                self.progress.update(
                    self.task_ids[task_id],
                    description=f"[red]✗ {self.tasks[task_id].description}",
                )
    
    def get_task(self, task_id: str) -> Optional[TaskProgress]:
        """Get task progress information."""
        return self.tasks.get(task_id)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tasks."""
        total_tasks = len(self.tasks)
        completed = sum(1 for t in self.tasks.values() if t.status == "completed")
        failed = sum(1 for t in self.tasks.values() if t.status == "failed")
        running = sum(1 for t in self.tasks.values() if t.status == "running")
        pending = sum(1 for t in self.tasks.values() if t.status == "pending")
        
        total_time = sum(t.elapsed_time() for t in self.tasks.values())
        
        return {
            "total_tasks": total_tasks,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "total_time": total_time,
            "average_time": total_time / total_tasks if total_tasks > 0 else 0.0,
        }
    
    def display_summary(self):
        """Display progress summary table."""
        summary = self.get_summary()
        
        table = Table(title="Progress Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Tasks", str(summary["total_tasks"]))
        table.add_row("Completed", str(summary["completed"]))
        table.add_row("Failed", str(summary["failed"]))
        table.add_row("Running", str(summary["running"]))
        table.add_row("Pending", str(summary["pending"]))
        table.add_row("Total Time", f"{summary['total_time']:.2f}s")
        table.add_row("Average Time", f"{summary['average_time']:.2f}s")
        
        console.print(table)
    
    def stop(self):
        """Stop progress tracking."""
        if self.progress is not None:
            self.progress.stop()
            self.progress = None


class ProgressCallback:
    """Callback wrapper for progress updates."""
    
    def __init__(
        self,
        callback: Optional[Callable[[int, int, Any], None]] = None,
        tracker: Optional[ProgressTracker] = None,
        task_id: Optional[str] = None,
    ):
        """
        Initialize progress callback.
        
        Args:
            callback: Optional callback function(current, total, result)
            tracker: Optional progress tracker
            task_id: Optional task ID for tracker
        """
        self.callback = callback
        self.tracker = tracker
        self.task_id = task_id
        self.current = 0
        self.total = 0
    
    def set_total(self, total: int):
        """Set total number of items."""
        self.total = total
    
    def update(self, increment: int = 1, result: Any = None):
        """Update progress."""
        self.current += increment
        
        if self.callback:
            self.callback(self.current, self.total, result)
        
        if self.tracker and self.task_id:
            self.tracker.update_task(self.task_id, self.current)
    
    def complete(self, result: Any = None):
        """Mark as completed."""
        self.current = self.total
        self.update(0, result)
        
        if self.tracker and self.task_id:
            self.tracker.complete_task(self.task_id)


__all__ = [
    "TaskProgress",
    "ProgressTracker",
    "ProgressCallback",
]












