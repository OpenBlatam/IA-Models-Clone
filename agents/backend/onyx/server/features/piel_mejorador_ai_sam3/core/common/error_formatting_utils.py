"""
Error Formatting Utilities for Piel Mejorador AI SAM3
=====================================================

Unified error formatting and display utilities.
"""

import traceback
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FormattedError:
    """Formatted error information."""
    error_type: str
    message: str
    category: str = "UNKNOWN"
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "category": self.category,
            "stack_trace": self.stack_trace,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_string(self, include_stack: bool = True) -> str:
        """
        Convert to string.
        
        Args:
            include_stack: Whether to include stack trace
            
        Returns:
            Formatted error string
        """
        lines = [
            f"Error Type: {self.error_type}",
            f"Category: {self.category}",
            f"Message: {self.message}",
        ]
        
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            lines.append(f"Context: {ctx_str}")
        
        if include_stack and self.stack_trace:
            lines.append(f"\nStack Trace:\n{self.stack_trace}")
        
        return "\n".join(lines)


class ErrorFormattingUtils:
    """Unified error formatting utilities."""
    
    @staticmethod
    def format_exception(
        error: Exception,
        include_stack: bool = True,
        include_context: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> FormattedError:
        """
        Format exception.
        
        Args:
            error: Exception to format
            include_stack: Whether to include stack trace
            include_context: Whether to include context
            context: Optional additional context
            
        Returns:
            FormattedError
        """
        error_type = type(error).__name__
        message = str(error)
        
        # Determine category
        category = "UNKNOWN"
        if isinstance(error, ValueError):
            category = "VALIDATION"
        elif isinstance(error, (ConnectionError, TimeoutError)):
            category = "NETWORK"
        elif isinstance(error, (IOError, OSError, FileNotFoundError)):
            category = "STORAGE"
        elif isinstance(error, KeyError):
            category = "KEY_ERROR"
        elif isinstance(error, TypeError):
            category = "TYPE_ERROR"
        elif isinstance(error, AttributeError):
            category = "ATTRIBUTE_ERROR"
        
        # Get stack trace
        stack_trace = None
        if include_stack:
            stack_trace = traceback.format_exc()
        
        # Build context
        error_context = {}
        if include_context:
            error_context = {
                "error_type": error_type,
                "error_message": message,
            }
            if context:
                error_context.update(context)
            
            # Add exception attributes if available
            if hasattr(error, "__dict__"):
                for key, value in error.__dict__.items():
                    if key not in error_context:
                        try:
                            error_context[key] = str(value)
                        except Exception:
                            pass
        
        return FormattedError(
            error_type=error_type,
            message=message,
            category=category,
            stack_trace=stack_trace,
            context=error_context
        )
    
    @staticmethod
    def format_for_logging(
        error: Exception,
        operation: str = "operation",
        task_id: Optional[str] = None,
        **metadata
    ) -> str:
        """
        Format error for logging.
        
        Args:
            error: Exception
            operation: Operation name
            task_id: Optional task ID
            **metadata: Additional metadata
            
        Returns:
            Formatted error string for logging
        """
        formatted = ErrorFormattingUtils.format_exception(
            error,
            include_stack=True,
            context={"operation": operation, "task_id": task_id, **metadata}
        )
        return formatted.to_string(include_stack=True)
    
    @staticmethod
    def format_for_user(
        error: Exception,
        include_details: bool = False
    ) -> str:
        """
        Format error for user display.
        
        Args:
            error: Exception
            include_details: Whether to include technical details
            
        Returns:
            User-friendly error message
        """
        error_type = type(error).__name__
        message = str(error)
        
        # User-friendly messages
        user_messages = {
            "ValueError": "Invalid input provided",
            "FileNotFoundError": "File not found",
            "PermissionError": "Permission denied",
            "ConnectionError": "Connection failed",
            "TimeoutError": "Operation timed out",
            "KeyError": "Required information missing",
            "TypeError": "Invalid data type",
        }
        
        user_message = user_messages.get(error_type, "An error occurred")
        
        if include_details:
            user_message += f": {message}"
        
        return user_message
    
    @staticmethod
    def format_for_api(
        error: Exception,
        status_code: int = 500,
        include_stack: bool = False
    ) -> Dict[str, Any]:
        """
        Format error for API response.
        
        Args:
            error: Exception
            status_code: HTTP status code
            include_stack: Whether to include stack trace
            
        Returns:
            API error response dictionary
        """
        formatted = ErrorFormattingUtils.format_exception(
            error,
            include_stack=include_stack
        )
        
        response = {
            "error": {
                "type": formatted.error_type,
                "message": formatted.message,
                "category": formatted.category,
            }
        }
        
        if include_stack and formatted.stack_trace:
            response["error"]["stack_trace"] = formatted.stack_trace
        
        if formatted.context:
            response["error"]["context"] = formatted.context
        
        return response
    
    @staticmethod
    def get_error_summary(
        errors: List[Exception]
    ) -> Dict[str, Any]:
        """
        Get summary of multiple errors.
        
        Args:
            errors: List of exceptions
            
        Returns:
            Error summary dictionary
        """
        if not errors:
            return {"total": 0, "by_type": {}, "by_category": {}}
        
        by_type = {}
        by_category = {}
        
        for error in errors:
            formatted = ErrorFormattingUtils.format_exception(error, include_stack=False)
            
            # Count by type
            by_type[formatted.error_type] = by_type.get(formatted.error_type, 0) + 1
            
            # Count by category
            by_category[formatted.category] = by_category.get(formatted.category, 0) + 1
        
        return {
            "total": len(errors),
            "by_type": by_type,
            "by_category": by_category,
            "most_common_type": max(by_type.items(), key=lambda x: x[1])[0] if by_type else None,
            "most_common_category": max(by_category.items(), key=lambda x: x[1])[0] if by_category else None,
        }


# Convenience functions
def format_exception(error: Exception, **kwargs) -> FormattedError:
    """Format exception."""
    return ErrorFormattingUtils.format_exception(error, **kwargs)


def format_for_logging(error: Exception, **kwargs) -> str:
    """Format error for logging."""
    return ErrorFormattingUtils.format_for_logging(error, **kwargs)


def format_for_user(error: Exception, **kwargs) -> str:
    """Format error for user."""
    return ErrorFormattingUtils.format_for_user(error, **kwargs)


def format_for_api(error: Exception, **kwargs) -> Dict[str, Any]:
    """Format error for API."""
    return ErrorFormattingUtils.format_for_api(error, **kwargs)




