"""
Type Definitions
================

Shared type definitions and type aliases.
"""

from typing import Dict, Any, Optional, List, Union, Callable, Awaitable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Type aliases
FilePath = Union[str, Path]
ConfigDict = Dict[str, Any]
ResultDict = Dict[str, Any]
OptionsDict = Dict[str, Any]
MetadataDict = Dict[str, Any]

# Callback types
TaskCallback = Callable[[Any], Awaitable[None]]
ErrorCallback = Callable[[Exception, Dict[str, Any]], Awaitable[None]]
ProgressCallback = Callable[[float, Dict[str, Any]], Awaitable[None]]


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 15


@dataclass
class FileInfo:
    """File information structure."""
    path: FilePath
    size_bytes: int
    mime_type: str
    extension: str
    exists: bool = True
    
    @property
    def size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    @property
    def is_image(self) -> bool:
        """Check if file is an image."""
        return self.mime_type.startswith("image/")
    
    @property
    def is_video(self) -> bool:
        """Check if file is a video."""
        return self.mime_type.startswith("video/")


@dataclass
class ProcessingOptions:
    """Options for processing operations."""
    preserve_quality: bool = True
    output_format: Optional[str] = None
    custom_instructions: Optional[str] = None
    metadata: MetadataDict = field(default_factory=dict)


@dataclass
class TaskContext:
    """Context information for a task."""
    task_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: MetadataDict = field(default_factory=dict)
    created_at: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    success: bool
    output_path: Optional[FilePath] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: MetadataDict = field(default_factory=dict)
    
    def to_dict(self) -> ResultDict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "error": self.error,
            "processing_time": self.processing_time,
            "metadata": self.metadata
        }

