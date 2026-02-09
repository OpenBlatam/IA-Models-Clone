"""
Camera Entity

Represents a camera device and its status.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime


class CameraStatus(str, Enum):
    """Camera status."""
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    STREAMING = "streaming"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class Camera:
    """
    Camera entity representing a camera device.
    
    Attributes:
        index: Camera index/ID
        status: Current camera status
        resolution: Camera resolution (width, height)
        fps: Frames per second
        initialized_at: Timestamp when camera was initialized
        last_error: Last error message if any
    """
    
    index: int
    status: CameraStatus = CameraStatus.UNINITIALIZED
    resolution: Optional[Tuple[int, int]] = None
    fps: Optional[int] = None
    initialized_at: Optional[datetime] = None
    last_error: Optional[str] = None
    
    def initialize(self, resolution: Tuple[int, int], fps: int):
        """Initialize the camera."""
        self.status = CameraStatus.INITIALIZED
        self.resolution = resolution
        self.fps = fps
        self.initialized_at = datetime.utcnow()
        self.last_error = None
    
    def start_streaming(self):
        """Start camera streaming."""
        if self.status != CameraStatus.INITIALIZED:
            raise ValueError(f"Cannot start streaming. Camera status: {self.status}")
        self.status = CameraStatus.STREAMING
        self.last_error = None
    
    def stop_streaming(self):
        """Stop camera streaming."""
        if self.status == CameraStatus.STREAMING:
            self.status = CameraStatus.INITIALIZED
    
    def set_error(self, error_message: str):
        """Set camera error status."""
        self.status = CameraStatus.ERROR
        self.last_error = error_message
    
    def disconnect(self):
        """Disconnect the camera."""
        self.status = CameraStatus.DISCONNECTED
        self.resolution = None
        self.fps = None
        self.initialized_at = None
    
    @property
    def is_available(self) -> bool:
        """Check if camera is available for use."""
        return self.status in [
            CameraStatus.INITIALIZED,
            CameraStatus.STREAMING
        ]
    
    def to_dict(self) -> dict:
        """
        Convert camera to dictionary.
        
        Returns:
            Dictionary representation of the camera
        """
        return {
            "index": self.index,
            "status": self.status.value,
            "resolution": self.resolution,
            "fps": self.fps,
            "is_available": self.is_available,
            "initialized_at": self.initialized_at.isoformat() if self.initialized_at else None,
            "last_error": self.last_error,
        }



