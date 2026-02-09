"""
Inspection-related exceptions.
"""

from .base import QualityControlException


class InspectionException(QualityControlException):
    """Exception raised for inspection-related errors."""
    pass


class InspectionFailedException(InspectionException):
    """Exception raised when an inspection fails."""
    
    def __init__(self, reason: str, details: dict = None):
        super().__init__(
            message=f"Inspection failed: {reason}",
            error_code="INSPECTION_FAILED",
            details=details or {}
        )


class InvalidImageException(InspectionException):
    """Exception raised when an invalid image is provided."""
    
    def __init__(self, reason: str = "Invalid image format or corrupted image"):
        super().__init__(
            message=f"Invalid image: {reason}",
            error_code="INVALID_IMAGE",
            details={"reason": reason}
        )


class InspectionTimeoutException(InspectionException):
    """Exception raised when an inspection times out."""
    
    def __init__(self, timeout_seconds: float):
        super().__init__(
            message=f"Inspection timed out after {timeout_seconds} seconds",
            error_code="INSPECTION_TIMEOUT",
            details={"timeout_seconds": timeout_seconds}
        )



