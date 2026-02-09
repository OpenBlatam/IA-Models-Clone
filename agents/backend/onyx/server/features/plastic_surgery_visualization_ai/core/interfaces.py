"""Protocols and interfaces for dependency injection."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
from PIL import Image

from api.schemas.visualization import SurgeryType


class IImageProcessor(ABC):
    """Interface for image processing operations."""
    
    @abstractmethod
    async def load_from_bytes(self, image_data: bytes) -> Image.Image:
        """Load image from bytes."""
        pass
    
    @abstractmethod
    async def load_from_url(self, image_url: str) -> Image.Image:
        """Load image from URL."""
        pass
    
    @abstractmethod
    async def save_image(self, image: Image.Image, output_path: Path) -> None:
        """Save image to file."""
        pass
    
    @abstractmethod
    def validate_image(self, image: Image.Image) -> bool:
        """Validate image format and properties."""
        pass


class IAProcessor(ABC):
    """Interface for AI processing operations."""
    
    @abstractmethod
    async def process_surgery_visualization(
        self,
        image: Image.Image,
        surgery_type: SurgeryType,
        intensity: float,
        target_areas: Optional[List[str]] = None
    ) -> Image.Image:
        """Process image to show surgery visualization."""
        pass


class IStorageRepository(ABC):
    """Interface for storage operations."""
    
    @abstractmethod
    async def save_visualization(
        self,
        visualization_id: str,
        image: Image.Image,
        format: str
    ) -> Path:
        """Save visualization image."""
        pass
    
    @abstractmethod
    async def get_visualization(
        self,
        visualization_id: str
    ) -> Optional[Path]:
        """Get visualization image path."""
        pass
    
    @abstractmethod
    async def delete_visualization(
        self,
        visualization_id: str
    ) -> bool:
        """Delete visualization image."""
        pass


class ICacheRepository(ABC):
    """Interface for cache operations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[dict]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: dict, ttl_hours: Optional[float] = None) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache."""
        pass


class IMetricsCollector(ABC):
    """Interface for metrics collection."""
    
    @abstractmethod
    def increment(self, metric_name: str, value: float = 1.0) -> None:
        """Increment a counter metric."""
        pass
    
    @abstractmethod
    def record_timing(self, metric_name: str, duration: float) -> None:
        """Record a timing metric."""
        pass
    
    @abstractmethod
    def set_gauge(self, metric_name: str, value: float) -> None:
        """Set a gauge metric."""
        pass


class IEventPublisher(ABC):
    """Interface for event publishing."""
    
    @abstractmethod
    async def publish(self, event) -> None:
        """Publish a domain event."""
        pass


class IEventSubscriber(ABC):
    """Interface for event subscription."""
    
    @abstractmethod
    async def subscribe(self, event_type: str, handler) -> None:
        """Subscribe to an event type."""
        pass
