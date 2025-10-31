"""
Analyze Content Command
======================

Single responsibility: Command for analyzing content.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import uuid

from ...domain.value_objects.content_metrics import ContentMetrics


@dataclass
class AnalyzeContentCommand:
    """
    Command to analyze content and create history entry.
    
    Single Responsibility: Represent the intent to analyze content.
    """
    command_id: str
    content: str
    model_version: str
    metadata: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate command after initialization."""
        if not self.content or not self.content.strip():
            raise ValueError("Content cannot be empty")
        
        if not self.model_version or not self.model_version.strip():
            raise ValueError("Model version cannot be empty")
    
    @classmethod
    def create(
        cls,
        content: str,
        model_version: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> 'AnalyzeContentCommand':
        """
        Factory method to create analyze content command.
        
        Args:
            content: Content to analyze
            model_version: AI model version
            metadata: Optional metadata
            user_id: Optional user ID
            request_id: Optional request ID
            
        Returns:
            AnalyzeContentCommand instance
        """
        return cls(
            command_id=str(uuid.uuid4()),
            content=content,
            model_version=model_version,
            metadata=metadata,
            user_id=user_id,
            request_id=request_id
        )
    
    def get_content_length(self) -> int:
        """Get content length."""
        return len(self.content)
    
    def get_word_count(self) -> int:
        """Get word count."""
        return len(self.content.split())
    
    def has_metadata(self, key: str) -> bool:
        """Check if metadata contains key."""
        return self.metadata is not None and key in self.metadata
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        if self.metadata is None:
            return default
        return self.metadata.get(key, default)
    
    def is_valid(self) -> bool:
        """Check if command is valid."""
        try:
            self.__post_init__()
            return True
        except ValueError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "command_id": self.command_id,
            "content": self.content,
            "model_version": self.model_version,
            "metadata": self.metadata,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "content_length": self.get_content_length(),
            "word_count": self.get_word_count()
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"AnalyzeContentCommand(id={self.command_id}, model={self.model_version}, length={self.get_content_length()})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"AnalyzeContentCommand(command_id='{self.command_id}', model_version='{self.model_version}', content_length={self.get_content_length()})"




