"""
Response Handler
================

Response handling utilities for API clients.
"""

import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)


@dataclass
class ResponseHandler:
    """Response handler configuration."""
    success_codes: list[int] = None
    error_handler: Optional[Callable] = None
    transform: Optional[Callable] = None
    
    def __post_init__(self):
        """Initialize default success codes."""
        if self.success_codes is None:
            self.success_codes = [200, 201, 202, 204]


class ResponseProcessor:
    """Response processor for API clients."""
    
    @staticmethod
    def handle_response(
        response: httpx.Response,
        handler: Optional[ResponseHandler] = None
    ) -> Dict[str, Any]:
        """
        Handle HTTP response.
        
        Args:
            response: HTTP response
            handler: Optional response handler
            
        Returns:
            Processed response data
            
        Raises:
            httpx.HTTPStatusError: If response indicates error
        """
        handler = handler or ResponseHandler()
        
        # Check status code
        if response.status_code not in handler.success_codes:
            error_msg = f"Request failed with status {response.status_code}"
            
            if handler.error_handler:
                handler.error_handler(response)
            
            response.raise_for_status()
            raise httpx.HTTPStatusError(error_msg, request=response.request, response=response)
        
        # Parse JSON
        try:
            data = response.json()
        except Exception:
            data = {"content": response.text}
        
        # Transform if needed
        if handler.transform:
            data = handler.transform(data)
        
        return data
    
    @staticmethod
    def extract_error(response: httpx.Response) -> Dict[str, Any]:
        """
        Extract error information from response.
        
        Args:
            response: HTTP response
            
        Returns:
            Error information dictionary
        """
        try:
            error_data = response.json()
            return {
                "status_code": response.status_code,
                "error": error_data.get("error", {}),
                "message": error_data.get("message", response.text),
                "details": error_data
            }
        except Exception:
            return {
                "status_code": response.status_code,
                "message": response.text,
                "error": None
            }
    
    @staticmethod
    def validate_response(response: httpx.Response, required_fields: list[str]) -> bool:
        """
        Validate response contains required fields.
        
        Args:
            response: HTTP response
            required_fields: List of required field names
            
        Returns:
            True if all fields present
        """
        try:
            data = response.json()
            return all(field in data for field in required_fields)
        except Exception:
            return False




