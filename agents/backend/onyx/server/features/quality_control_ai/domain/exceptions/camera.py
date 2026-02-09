"""
Camera-related exceptions.
"""

from .base import QualityControlException


class CameraException(QualityControlException):
    """Exception raised for camera-related errors."""
    pass


class CameraNotInitializedException(CameraException):
    """Exception raised when camera is not initialized."""
    
    def __init__(self, camera_index: int = None):
        message = "Camera is not initialized"
        if camera_index is not None:
            message += f" (camera index: {camera_index})"
        super().__init__(
            message=message,
            error_code="CAMERA_NOT_INITIALIZED",
            details={"camera_index": camera_index}
        )


class CameraConnectionException(CameraException):
    """Exception raised when camera connection fails."""
    
    def __init__(self, camera_index: int, reason: str):
        super().__init__(
            message=f"Failed to connect to camera {camera_index}: {reason}",
            error_code="CAMERA_CONNECTION_FAILED",
            details={"camera_index": camera_index, "reason": reason}
        )


class CameraStreamException(CameraException):
    """Exception raised when camera streaming fails."""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"Camera streaming failed: {reason}",
            error_code="CAMERA_STREAM_FAILED",
            details={"reason": reason}
        )



