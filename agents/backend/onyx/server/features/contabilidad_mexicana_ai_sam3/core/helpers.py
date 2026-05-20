"""
Common Helper Functions for Contabilidad Mexicana AI SAM3
==========================================================

Refactored with:
- MessageBuilder pattern for structured message creation
- FileHandler protocol for file operations
- DirectoryManager for path management
- ContentBuilder for different content types
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union, Protocol
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of message content."""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"


@dataclass
class Content:
    """
    Represents message content.
    
    Provides type-safe content creation.
    """
    content_type: ContentType
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        if self.content_type == ContentType.TEXT:
            return {"type": "text", "text": self.value}
        elif self.content_type == ContentType.IMAGE:
            return {"type": "image", "url": self.value, **self.metadata}
        elif self.content_type == ContentType.FILE:
            return {"type": "file", "path": str(self.value), **self.metadata}
        return {"type": str(self.content_type.value), "value": self.value}
    
    @classmethod
    def text(cls, text: str) -> "Content":
        """Create text content."""
        return cls(ContentType.TEXT, text)
    
    @classmethod
    def image(cls, url: str, **metadata) -> "Content":
        """Create image content."""
        return cls(ContentType.IMAGE, url, metadata)


@dataclass
class Message:
    """
    Represents a chat message.
    
    Provides structured message creation.
    """
    role: str
    content: Union[str, List[Content]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format."""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        
        return {
            "role": self.role,
            "content": [c.to_dict() for c in self.content]
        }


class MessageBuilder:
    """
    Builder for creating messages.
    
    Provides fluent API for message construction.
    """
    
    def __init__(self, role: str):
        self._role = role
        self._content: List[Content] = []
    
    def text(self, text: str) -> "MessageBuilder":
        """Add text content."""
        self._content.append(Content.text(text))
        return self
    
    def image(self, url: str, **metadata) -> "MessageBuilder":
        """Add image content."""
        self._content.append(Content.image(url, **metadata))
        return self
    
    def build(self) -> Message:
        """Build the message."""
        if len(self._content) == 1 and self._content[0].content_type == ContentType.TEXT:
            return Message(self._role, self._content[0].value)
        return Message(self._role, self._content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Build and convert to dictionary."""
        return self.build().to_dict()
    
    # === Class Factory Methods ===
    
    @classmethod
    def system(cls) -> "MessageBuilder":
        """Create system message builder."""
        return cls("system")
    
    @classmethod
    def user(cls) -> "MessageBuilder":
        """Create user message builder."""
        return cls("user")
    
    @classmethod
    def assistant(cls) -> "MessageBuilder":
        """Create assistant message builder."""
        return cls("assistant")


class FileHandler(Protocol):
    """Protocol for file handlers."""
    
    def read(self, path: str) -> Any:
        """Read from file."""
        ...
    
    def write(self, path: str, data: Any) -> None:
        """Write to file."""
        ...


class JSONFileHandler:
    """Handler for JSON files."""
    
    def __init__(self, indent: int = 4, ensure_ascii: bool = False):
        self.indent = indent
        self.ensure_ascii = ensure_ascii
    
    def read(self, path: str) -> Dict[str, Any]:
        """Read JSON from file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"JSON file not found: {path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {path}: {e}")
            raise
    
    def write(self, path: str, data: Dict[str, Any]) -> None:
        """Write JSON to file."""
        try:
            # Ensure directory exists
            DirectoryManager.ensure_exists(Path(path).parent)
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=self.indent, ensure_ascii=self.ensure_ascii)
        except OSError as e:
            logger.error(f"Error saving JSON file {path}: {e}")
            raise


class DirectoryManager:
    """
    Manages directories and paths.
    
    Provides centralized path management.
    """
    
    @staticmethod
    def ensure_exists(directory: Union[str, Path]) -> Path:
        """Ensure a directory exists, creating it if necessary."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @staticmethod
    def create_structure(
        base_dir: Union[str, Path], 
        subdirs: List[str]
    ) -> Dict[str, Path]:
        """
        Create a directory structure.
        
        Args:
            base_dir: Base directory path
            subdirs: List of subdirectory names
            
        Returns:
            Dictionary mapping names to paths
        """
        base_path = DirectoryManager.ensure_exists(base_dir)
        created_dirs = {}
        
        for subdir in subdirs:
            subdir_path = base_path / subdir
            DirectoryManager.ensure_exists(subdir_path)
            created_dirs[subdir] = subdir_path
        
        return created_dirs
    
    @staticmethod
    def get_project_root() -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent
    
    @staticmethod
    def get_output_dir(name: str = "output") -> Path:
        """Get or create output directory."""
        output_dir = DirectoryManager.get_project_root() / name
        return DirectoryManager.ensure_exists(output_dir)


class FileRegistry:
    """Registry for file handlers."""
    
    _handlers: Dict[str, FileHandler] = {
        ".json": JSONFileHandler(),
    }
    
    @classmethod
    def register(cls, extension: str, handler: FileHandler):
        """Register a handler for an extension."""
        cls._handlers[extension] = handler
    
    @classmethod
    def get_handler(cls, path: str) -> Optional[FileHandler]:
        """Get handler for a file path."""
        ext = Path(path).suffix.lower()
        return cls._handlers.get(ext)
    
    @classmethod
    def read(cls, path: str) -> Any:
        """Read file using appropriate handler."""
        handler = cls.get_handler(path)
        if not handler:
            raise ValueError(f"No handler for file type: {path}")
        return handler.read(path)
    
    @classmethod
    def write(cls, path: str, data: Any) -> None:
        """Write file using appropriate handler."""
        handler = cls.get_handler(path)
        if not handler:
            raise ValueError(f"No handler for file type: {path}")
        handler.write(path, data)


# === Convenience Functions (Backward Compatible) ===

def create_message(role: str, content: Any) -> Dict[str, Any]:
    """Create a message dictionary for OpenRouter API."""
    return {"role": role, "content": content}


def create_text_content(text: str) -> Dict[str, str]:
    """Create a text content item for messages."""
    return Content.text(text).to_dict()


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    return JSONFileHandler().read(file_path)


def save_json_file(data: Dict[str, Any], file_path: str, indent: int = 4) -> None:
    """Save data to JSON file."""
    JSONFileHandler(indent=indent).write(file_path, data)


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    return DirectoryManager.ensure_exists(directory)


def create_output_directories(base_dir: Union[str, Path], subdirs: List[str]) -> Dict[str, Path]:
    """Create a base directory and multiple subdirectories."""
    return DirectoryManager.create_structure(base_dir, subdirs)
