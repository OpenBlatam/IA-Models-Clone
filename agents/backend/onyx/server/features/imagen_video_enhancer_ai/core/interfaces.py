"""
Interfaces and Abstract Base Classes
====================================

Abstract interfaces for core components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path

from .types import FilePath, ConfigDict, ResultDict, ProcessingResult, TaskContext


class IRepository(ABC):
    """Abstract interface for data repositories."""
    
    @abstractmethod
    async def save(self, key: str, data: Dict[str, Any]) -> bool:
        """Save data with key."""
        pass
    
    @abstractmethod
    async def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load data by key."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data by key."""
        pass
    
    @abstractmethod
    async def list_keys(self) -> List[str]:
        """List all keys."""
        pass


class IProcessor(ABC):
    """Abstract interface for processors."""
    
    @abstractmethod
    async def process(self, input_path: FilePath, options: Dict[str, Any]) -> ProcessingResult:
        """Process input and return result."""
        pass
    
    @abstractmethod
    def validate_input(self, input_path: FilePath) -> bool:
        """Validate input file."""
        pass


class IEnhancementService(ABC):
    """Abstract interface for enhancement services."""
    
    @abstractmethod
    async def enhance(
        self,
        file_path: FilePath,
        options: Dict[str, Any],
        context: Optional[TaskContext] = None
    ) -> ProcessingResult:
        """Enhance a file."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get supported file formats."""
        pass
    
    @abstractmethod
    def estimate_processing_time(self, file_size_mb: float) -> float:
        """Estimate processing time in seconds."""
        pass


class ICache(ABC):
    """Abstract interface for cache implementations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass


class INotifier(ABC):
    """Abstract interface for notification systems."""
    
    @abstractmethod
    async def send(
        self,
        message: str,
        recipient: str,
        subject: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send notification."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if notifier is available."""
        pass


class IValidator(ABC):
    """Abstract interface for validators."""
    
    @abstractmethod
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate a value."""
        pass
    
    @abstractmethod
    def get_error_message(self) -> Optional[str]:
        """Get last validation error message."""
        pass




