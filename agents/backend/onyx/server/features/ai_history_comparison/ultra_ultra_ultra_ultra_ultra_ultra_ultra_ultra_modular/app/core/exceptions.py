"""
Custom exceptions for the application.
"""

from typing import Any, Dict, Optional


class BaseAppException(Exception):
    """Base exception for the application."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)


class ValidationError(BaseAppException):
    """Validation error exception."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, 422)


class NotFoundError(BaseAppException):
    """Not found error exception."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, 404)


class ConflictError(BaseAppException):
    """Conflict error exception."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, 409)


class UnauthorizedError(BaseAppException):
    """Unauthorized error exception."""
    
    def __init__(self, message: str = "Unauthorized", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, 401)


class ForbiddenError(BaseAppException):
    """Forbidden error exception."""
    
    def __init__(self, message: str = "Forbidden", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, 403)


class PluginError(BaseAppException):
    """Plugin-related error exception."""
    
    def __init__(self, message: str, plugin_name: str, details: Optional[Dict[str, Any]] = None):
        self.plugin_name = plugin_name
        details = details or {}
        details["plugin_name"] = plugin_name
        super().__init__(message, details, 500)


class ExtensionError(BaseAppException):
    """Extension-related error exception."""
    
    def __init__(self, message: str, extension_name: str, details: Optional[Dict[str, Any]] = None):
        self.extension_name = extension_name
        details = details or {}
        details["extension_name"] = extension_name
        super().__init__(message, details, 500)


class MiddlewareError(BaseAppException):
    """Middleware-related error exception."""
    
    def __init__(self, message: str, middleware_name: str, details: Optional[Dict[str, Any]] = None):
        self.middleware_name = middleware_name
        details = details or {}
        details["middleware_name"] = middleware_name
        super().__init__(message, details, 500)


class ComponentError(BaseAppException):
    """Component-related error exception."""
    
    def __init__(self, message: str, component_name: str, details: Optional[Dict[str, Any]] = None):
        self.component_name = component_name
        details = details or {}
        details["component_name"] = component_name
        super().__init__(message, details, 500)


class EventError(BaseAppException):
    """Event-related error exception."""
    
    def __init__(self, message: str, event_type: str, details: Optional[Dict[str, Any]] = None):
        self.event_type = event_type
        details = details or {}
        details["event_type"] = event_type
        super().__init__(message, details, 500)


class WorkflowError(BaseAppException):
    """Workflow-related error exception."""
    
    def __init__(self, message: str, workflow_id: str, details: Optional[Dict[str, Any]] = None):
        self.workflow_id = workflow_id
        details = details or {}
        details["workflow_id"] = workflow_id
        super().__init__(message, details, 500)




