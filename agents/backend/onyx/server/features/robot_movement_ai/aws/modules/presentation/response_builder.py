"""
Response Builder
================

Builder for API responses.
"""

from typing import Dict, Any, Optional, List
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ResponseBuilder:
    """Builder for API responses."""
    
    def __init__(self):
        self._status_code: int = 200
        self._data: Optional[Dict[str, Any]] = None
        self._message: Optional[str] = None
        self._errors: List[str] = []
        self._headers: Dict[str, str] = {}
        self._meta: Dict[str, Any] = {}
    
    def status(self, code: int) -> "ResponseBuilder":
        """Set status code."""
        self._status_code = code
        return self
    
    def data(self, data: Any) -> "ResponseBuilder":
        """Set response data."""
        self._data = data
        return self
    
    def message(self, message: str) -> "ResponseBuilder":
        """Set response message."""
        self._message = message
        return self
    
    def error(self, error: str) -> "ResponseBuilder":
        """Add error message."""
        self._errors.append(error)
        return self
    
    def header(self, key: str, value: str) -> "ResponseBuilder":
        """Add header."""
        self._headers[key] = value
        return self
    
    def meta(self, key: str, value: Any) -> "ResponseBuilder":
        """Add metadata."""
        self._meta[key] = value
        return self
    
    def build(self) -> JSONResponse:
        """Build JSON response."""
        response_body: Dict[str, Any] = {}
        
        if self._data is not None:
            response_body["data"] = self._data
        
        if self._message:
            response_body["message"] = self._message
        
        if self._errors:
            response_body["errors"] = self._errors
        
        if self._meta:
            response_body["meta"] = self._meta
        
        # If no data structure, use data directly
        if not response_body and self._data:
            response_body = self._data if isinstance(self._data, dict) else {"result": self._data}
        
        return JSONResponse(
            content=response_body,
            status_code=self._status_code,
            headers=self._headers
        )
    
    @classmethod
    def success(cls, data: Any = None, message: Optional[str] = None) -> JSONResponse:
        """Create success response."""
        builder = cls()
        if data:
            builder.data(data)
        if message:
            builder.message(message)
        return builder.build()
    
    @classmethod
    def error_response(cls, message: str, status_code: int = 400, errors: Optional[List[str]] = None) -> JSONResponse:
        """Create error response."""
        builder = cls()
        builder.status(status_code)
        builder.message(message)
        if errors:
            for error in errors:
                builder.error(error)
        return builder.build()















