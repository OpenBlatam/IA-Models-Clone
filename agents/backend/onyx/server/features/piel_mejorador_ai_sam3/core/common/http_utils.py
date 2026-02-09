"""
HTTP Utilities for Piel Mejorador AI SAM3
=========================================

Unified HTTP response handling and utilities.
"""

import logging
from typing import Dict, Any, Optional, TypeVar, Callable
import httpx
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')


class HTTPUtils:
    """Unified HTTP utilities."""
    
    @staticmethod
    def parse_json_response(
        response: httpx.Response,
        default: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parse JSON response safely.
        
        Args:
            response: HTTP response
            default: Default value if parsing fails
            
        Returns:
            Parsed JSON or default
        """
        try:
            return response.json()
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            if default is not None:
                return default
            raise ValueError(f"Invalid JSON response: {e}")
    
    @staticmethod
    def parse_text_response(
        response: httpx.Response,
        encoding: str = "utf-8",
        default: Optional[str] = None
    ) -> str:
        """
        Parse text response safely.
        
        Args:
            response: HTTP response
            encoding: Text encoding
            default: Default value if parsing fails
            
        Returns:
            Response text or default
        """
        try:
            return response.text
        except Exception as e:
            logger.warning(f"Failed to parse text response: {e}")
            if default is not None:
                return default
            raise
    
    @staticmethod
    def handle_response(
        response: httpx.Response,
        raise_on_error: bool = True,
        parse_json: bool = True
    ) -> Dict[str, Any]:
        """
        Handle HTTP response with error checking.
        
        Args:
            response: HTTP response
            raise_on_error: Whether to raise on error status
            parse_json: Whether to parse as JSON
            
        Returns:
            Response data
            
        Raises:
            httpx.HTTPStatusError: If raise_on_error and status is error
        """
        if raise_on_error:
            response.raise_for_status()
        
        if parse_json:
            return HTTPUtils.parse_json_response(response)
        else:
            return {"text": HTTPUtils.parse_text_response(response)}
    
    @staticmethod
    def extract_error_message(
        error: httpx.HTTPStatusError,
        default: Optional[str] = None
    ) -> str:
        """
        Extract error message from HTTP error.
        
        Args:
            error: HTTP status error
            default: Default error message
            
        Returns:
            Error message
        """
        try:
            error_data = error.response.json()
            if isinstance(error_data, dict):
                error_detail = error_data.get("error", {})
                if isinstance(error_detail, dict):
                    return error_detail.get("message", default or str(error))
                return error_detail if isinstance(error_detail, str) else (default or str(error))
            return default or str(error)
        except Exception:
            return default or f"HTTP {error.response.status_code}: {error.response.text[:200]}"
    
    @staticmethod
    def build_query_string(params: Dict[str, Any]) -> str:
        """
        Build query string from parameters.
        
        Args:
            params: Query parameters
            
        Returns:
            Query string (without ?)
        """
        from urllib.parse import urlencode
        
        # Filter out None values
        filtered = {k: v for k, v in params.items() if v is not None}
        return urlencode(filtered)
    
    @staticmethod
    def merge_headers(
        *header_dicts: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Merge multiple header dictionaries.
        
        Args:
            *header_dicts: Header dictionaries to merge
            
        Returns:
            Merged headers (later ones override earlier ones)
        """
        merged = {}
        for headers in header_dicts:
            if headers:
                merged.update(headers)
        return merged
    
    @staticmethod
    def normalize_url(url: str, base_url: Optional[str] = None) -> str:
        """
        Normalize URL (join with base if provided).
        
        Args:
            url: URL or endpoint
            base_url: Optional base URL
            
        Returns:
            Normalized URL
        """
        if base_url:
            # Remove leading slash from endpoint
            endpoint = url.lstrip('/')
            # Remove trailing slash from base
            base = base_url.rstrip('/')
            return f"{base}/{endpoint}"
        return url


# Convenience functions
def parse_json_response(response: httpx.Response, **kwargs) -> Dict[str, Any]:
    """Parse JSON response."""
    return HTTPUtils.parse_json_response(response, **kwargs)


def handle_response(response: httpx.Response, **kwargs) -> Dict[str, Any]:
    """Handle HTTP response."""
    return HTTPUtils.handle_response(response, **kwargs)


def extract_error_message(error: httpx.HTTPStatusError, **kwargs) -> str:
    """Extract error message."""
    return HTTPUtils.extract_error_message(error, **kwargs)




