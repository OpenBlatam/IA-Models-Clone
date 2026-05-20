"""
Interface Layer - Ultra Modular API and External Interfaces
Handles all external communication and API interactions
"""

from typing import Optional, Dict, Any, List, Callable
import logging
from abc import ABC, abstractmethod

from .interfaces import IAPIHandler

logger = logging.getLogger(__name__)


# ============================================================================
# Request Processor
# ============================================================================

class RequestProcessor:
    """Process and validate incoming requests"""
    
    def __init__(self):
        self.validators: List[Callable] = []
        self.transformers: List[Callable] = []
    
    def add_validator(self, validator: Callable) -> 'RequestProcessor':
        """Add request validator"""
        self.validators.append(validator)
        return self
    
    def add_transformer(self, transformer: Callable) -> 'RequestProcessor':
        """Add request transformer"""
        self.transformers.append(transformer)
        return self
    
    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request through validators and transformers"""
        # Validate
        for validator in self.validators:
            if not validator(request):
                raise ValueError(f"Request validation failed: {validator.__name__}")
        
        # Transform
        processed = request
        for transformer in self.transformers:
            processed = transformer(processed)
        
        return processed


# ============================================================================
# Response Formatter
# ============================================================================

class ResponseFormatter:
    """Format responses for different output types"""
    
    def __init__(self, format_type: str = "json"):
        self.format_type = format_type
    
    def format(self, data: Any, **kwargs) -> Any:
        """Format response data"""
        if self.format_type == "json":
            return self._format_json(data)
        elif self.format_type == "dict":
            return self._format_dict(data)
        else:
            return data
    
    def _format_json(self, data: Any) -> Dict[str, Any]:
        """Format as JSON-compatible dict"""
        if isinstance(data, dict):
            return data
        elif hasattr(data, '__dict__'):
            return self._format_dict(data.__dict__)
        else:
            return {"result": data}
    
    def _format_dict(self, data: Any) -> Dict[str, Any]:
        """Format as dictionary"""
        if isinstance(data, dict):
            return data
        else:
            return {"result": data}


# ============================================================================
# API Handler
# ============================================================================

class APIHandler:
    """Handle API requests with processing and formatting"""
    
    def __init__(
        self,
        service: Any,
        request_processor: Optional[RequestProcessor] = None,
        response_formatter: Optional[ResponseFormatter] = None
    ):
        self.service = service
        self.request_processor = request_processor or RequestProcessor()
        self.response_formatter = response_formatter or ResponseFormatter()
    
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API request"""
        try:
            # Process request
            processed_request = self.request_processor.process(request)
            
            # Execute service
            result = self.service.execute(processed_request)
            
            # Format response
            response = self.response_formatter.format(result)
            
            return {
                "success": True,
                "data": response
            }
        except Exception as e:
            logger.error(f"API handler error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }


# ============================================================================
# Interface Factory
# ============================================================================

class InterfaceFactory:
    """Factory for creating interface components"""
    
    @staticmethod
    def create_handler(
        service: Any,
        validate_requests: bool = True,
        response_format: str = "json"
    ) -> APIHandler:
        """Create API handler"""
        request_processor = RequestProcessor()
        if validate_requests:
            request_processor.add_validator(lambda r: isinstance(r, dict))
        
        response_formatter = ResponseFormatter(format_type=response_format)
        
        return APIHandler(
            service=service,
            request_processor=request_processor,
            response_formatter=response_formatter
        )


# Export main components
__all__ = [
    "RequestProcessor",
    "ResponseFormatter",
    "APIHandler",
    "InterfaceFactory",
]



