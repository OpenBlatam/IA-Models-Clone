"""
Base Exception for Quality Control AI

All domain exceptions inherit from this base class.
"""

from typing import Optional, Dict, Any


class QualityControlException(Exception):
    """
    Base exception for all Quality Control AI errors.
    
    Attributes:
        message: Error message
        error_code: Optional error code for programmatic handling
        details: Optional dictionary with additional error details
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize QualityControlException.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code (e.g., 'INSPECTION_FAILED')
            details: Optional dictionary with additional context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for API responses.
        
        Returns:
            Dictionary representation of the exception
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }
    
    def __str__(self) -> str:
        """String representation of the exception."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message



