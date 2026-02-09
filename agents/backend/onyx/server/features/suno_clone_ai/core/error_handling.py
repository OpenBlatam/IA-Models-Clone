"""
Unified error handling system for Suno Clone AI.

Provides consistent error handling across the application with proper
categorization, logging, and HTTP response mapping.
"""

import logging
from typing import Optional, Dict, Any, Type, Callable
from enum import Enum
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    GENERATION = "generation"
    AUDIO_PROCESSING = "audio_processing"
    DATABASE = "database"
    CACHE = "cache"
    NETWORK = "network"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM = "system"


class UnifiedErrorHandler:
    """Unified error handler with consistent error mapping."""
    
    # Error pattern mappings
    ERROR_PATTERNS: Dict[str, tuple] = {
        "cuda": (status.HTTP_507_INSUFFICIENT_STORAGE, ErrorCategory.SYSTEM, "GPU memory insufficient"),
        "out of memory": (status.HTTP_507_INSUFFICIENT_STORAGE, ErrorCategory.SYSTEM, "Memory insufficient"),
        "model not found": (status.HTTP_503_SERVICE_UNAVAILABLE, ErrorCategory.GENERATION, "Model not available"),
        "file not found": (status.HTTP_404_NOT_FOUND, ErrorCategory.NOT_FOUND, "File not found"),
        "format": (status.HTTP_400_BAD_REQUEST, ErrorCategory.AUDIO_PROCESSING, "Unsupported format"),
        "codec": (status.HTTP_400_BAD_REQUEST, ErrorCategory.AUDIO_PROCESSING, "Unsupported codec"),
        "permission": (status.HTTP_403_FORBIDDEN, ErrorCategory.AUTHORIZATION, "Permission denied"),
        "unauthorized": (status.HTTP_401_UNAUTHORIZED, ErrorCategory.AUTHENTICATION, "Authentication required"),
    }
    
    @classmethod
    def handle_error(
        cls,
        error: Exception,
        operation: str,
        category: Optional[ErrorCategory] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> HTTPException:
        """
        Handle error with consistent mapping to HTTP exceptions.
        
        Args:
            error: Exception to handle
            operation: Operation name
            category: Error category (auto-detected if not provided)
            context: Additional context
        
        Returns:
            HTTPException with appropriate status code
        """
        error_msg = str(error).lower()
        error_type = type(error).__name__
        
        # Log error
        context_str = f" (context: {context})" if context else ""
        logger.error(
            f"Error in {operation}: {error_type} - {error_msg}{context_str}",
            exc_info=True,
            extra={"operation": operation, "error_type": error_type, "context": context}
        )
        
        # Check for specific error patterns
        for pattern, (status_code, error_cat, detail) in cls.ERROR_PATTERNS.items():
            if pattern in error_msg:
                return HTTPException(
                    status_code=status_code,
                    detail=f"{detail} in {operation}"
                )
        
        # Map common exception types
        if isinstance(error, ValueError):
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input for {operation}: {str(error)}"
            )
        
        if isinstance(error, FileNotFoundError):
            return HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resource not found in {operation}: {str(error)}"
            )
        
        if isinstance(error, PermissionError):
            return HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied in {operation}: {str(error)}"
            )
        
        # Default error
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in {operation}: {str(error)}"
        )
    
    @classmethod
    def handle_generation_error(
        cls,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> HTTPException:
        """Handle music generation errors."""
        return cls.handle_error(
            error,
            "music_generation",
            ErrorCategory.GENERATION,
            context
        )
    
    @classmethod
    def handle_audio_processing_error(
        cls,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> HTTPException:
        """Handle audio processing errors."""
        return cls.handle_error(
            error,
            "audio_processing",
            ErrorCategory.AUDIO_PROCESSING,
            context
        )
    
    @classmethod
    def handle_validation_error(
        cls,
        error: ValueError,
        context: Optional[Dict[str, Any]] = None
    ) -> HTTPException:
        """Handle validation errors."""
        return cls.handle_error(
            error,
            "validation",
            ErrorCategory.VALIDATION,
            context
        )
    
    @classmethod
    def handle_cache_error(
        cls,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Handle cache errors silently (non-critical)."""
        logger.warning(f"Cache error (non-critical): {error}", extra=context)
        return None


# Global error handler instance
error_handler = UnifiedErrorHandler()


def handle_error(
    error: Exception,
    operation: str,
    category: Optional[ErrorCategory] = None,
    context: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """Convenience function for error handling."""
    return error_handler.handle_error(error, operation, category, context)

