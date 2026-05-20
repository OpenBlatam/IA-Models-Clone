from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from typing import Any, Dict, TypeVar, Generic, Optional
import structlog
from datetime import datetime, timezone
        import uuid
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Base Use Case

Provides common functionality for all use cases.
"""


logger = structlog.get_logger()

# Type variables for request and response
TRequest = TypeVar('TRequest')
TResponse = TypeVar('TResponse')


class UseCase(ABC, Generic[TRequest, TResponse]):
    """
    Base class for all use cases.
    
    Provides common functionality like logging, validation, and error handling.
    """
    
    def __init__(self) -> Any:
        self._logger = structlog.get_logger(self.__class__.__name__)
    
    async def execute(self, request: TRequest) -> TResponse:
        """
        Execute the use case with request validation and error handling.
        
        This is the main entry point for all use cases.
        """
        start_time = datetime.now(timezone.utc)
        request_id = self._generate_request_id()
        
        self._logger.info(
            "Use case started",
            request_id=request_id,
            request_data=self._sanitize_request_for_logging(request)
        )
        
        try:
            # Validate request
            await self._validate_request(request)
            
            # Execute the use case logic
            response = await self._execute_impl(request)
            
            # Log successful completion
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._logger.info(
                "Use case completed successfully",
                request_id=request_id,
                duration_seconds=duration
            )
            
            return response
            
        except Exception as e:
            # Log error
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._logger.error(
                "Use case failed",
                request_id=request_id,
                duration_seconds=duration,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise
    
    @abstractmethod
    async def _execute_impl(self, request: TRequest) -> TResponse:
        """Implement the use case logic in subclasses."""
        pass
    
    async async def _validate_request(self, request: TRequest) -> None:
        """
        Validate the request. Override in subclasses if needed.
        
        Should raise appropriate exceptions for invalid requests.
        """
        if request is None:
            raise ValueError("Request cannot be None")
    
    async def _sanitize_request_for_logging(self, request: TRequest) -> Dict[str, Any]:
        """
        Sanitize request data for logging (remove sensitive information).
        
        Override in subclasses to customize sanitization.
        """
        if hasattr(request, '__dict__'):
            # Remove sensitive fields like passwords, tokens, etc.
            sensitive_fields = {'password', 'token', 'secret', 'key', 'credential'}
            sanitized = {}
            
            for key, value in request.__dict__.items():
                if any(sensitive in key.lower() for sensitive in sensitive_fields):
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = str(value)
            
            return sanitized
        
        return {"request": str(request)}
    
    async def _generate_request_id(self) -> str:
        """Generate a unique request ID for tracking."""
        return str(uuid.uuid4())[:8]


class Command(UseCase[TRequest, None]):
    """
    Base class for command use cases (operations that modify state).
    
    Commands don't return data, they perform actions.
    """
    pass


class Query(UseCase[TRequest, TResponse]):
    """
    Base class for query use cases (operations that read data).
    
    Queries don't modify state, they only return data.
    """
    pass 