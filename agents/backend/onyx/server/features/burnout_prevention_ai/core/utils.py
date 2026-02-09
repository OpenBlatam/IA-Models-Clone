"""
Utility functions for Burnout Prevention AI
==========================================
"""

import re
from typing import Dict, Any, Optional, List
from .types import JSONDict

try:
    import orjson as json
    _json_loads = json.loads
    _has_orjson = True
except ImportError:
    import json
    _json_loads = json.loads
    _has_orjson = False

# Export _json_loads for use in other modules (optimized JSON parsing)
__all__ = [
    'extract_json_from_text',
    'extract_content_from_response',
    'extract_suggestions',
    'extract_resources',
    'validate_api_response',
    '_json_loads',  # Export for optimized JSON parsing
    '_has_orjson'  # Export to check if orjson is available
]

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Compile regex patterns once for performance
_JSON_PATTERN = re.compile(r'\{.*\}', re.DOTALL)
_BULLET_PATTERN = re.compile(r'- (.+)')
_URL_PATTERN = re.compile(r'(https?://[^\s]+)')


def extract_json_from_text(text: str) -> Optional[JSONDict]:
    """
    Extract JSON object from text response (optimized).
    
    Handles cases where JSON is embedded in text or has extra whitespace.
    Uses fast path for clean JSON strings.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Parsed JSON dictionary or None if not found/invalid
    """
    if not text or len(text) < 2:
        return None
    
    # Fast path: if starts with {, try direct parse
    text_stripped = text.strip()
    if text_stripped.startswith('{'):
        try:
            # Try direct parse first (most common case)
            # orjson needs bytes, standard json needs string
            if _has_orjson:
                return _json_loads(text_stripped.encode('utf-8'))
            else:
                return _json_loads(text_stripped)
        except (ValueError, TypeError, UnicodeDecodeError):
            # Find matching closing brace (optimized)
            brace_count = 0
            end_pos = -1
            for i, char in enumerate(text_stripped):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            if end_pos > 0:
                try:
                    json_str = text_stripped[:end_pos]
                    if _has_orjson:
                        return _json_loads(json_str.encode('utf-8'))
                    else:
                        return _json_loads(json_str)
                except (ValueError, TypeError, UnicodeDecodeError):
                    pass
    
    # Fallback: regex search (only if fast path failed)
    json_match = _JSON_PATTERN.search(text)
    if json_match:
        try:
            json_str = json_match.group()
            if _has_orjson:
                return _json_loads(json_str.encode('utf-8'))
            else:
                return _json_loads(json_str)
        except (ValueError, TypeError, UnicodeDecodeError):
            pass
    
    return None


def extract_content_from_response(response: JSONDict) -> str:
    """
    Extract content from OpenRouter API response (optimized with better error handling).
    
    Args:
        response: OpenRouter API response dictionary
        
    Returns:
        Content string from the first choice, or empty string if not found
    """
    if not isinstance(response, dict):
        return ""
    
    try:
        choices = response.get("choices", [])
        if not choices or not isinstance(choices, list):
            return ""
        
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return ""
        
        message = first_choice.get("message", {})
        if not isinstance(message, dict):
            return ""
        
        content = message.get("content", "")
        return str(content) if content is not None else ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def extract_suggestions(text: str, limit: int = 3) -> Optional[list]:
    """
    Extract bullet point suggestions from text (optimized).
    
    Args:
        text: Text to extract suggestions from
        limit: Maximum number of suggestions to return
        
    Returns:
        List of suggestions or None if none found
    """
    if not text:
        return None
    suggestions = _BULLET_PATTERN.findall(text)
    return suggestions[:limit] if suggestions else None


def extract_resources(text: str, limit: int = 3) -> Optional[list]:
    """
    Extract URLs from text (optimized).
    
    Args:
        text: Text to extract URLs from
        limit: Maximum number of URLs to return
        
    Returns:
        List of URLs or None if none found
    """
    if not text:
        return None
    resources = _URL_PATTERN.findall(text)
    return resources[:limit] if resources else None


def validate_api_response(response: JSONDict, required_keys: Optional[List[str]] = None) -> bool:
    """
    Validate API response structure (optimized).
    
    Args:
        response: API response dictionary to validate
        required_keys: Optional list of required keys in the response
        
    Returns:
        True if response is valid, False otherwise
    """
    if not isinstance(response, dict):
        return False
    
    if required_keys:
        # Use set for O(1) lookup instead of O(n) iteration
        response_keys = set(response.keys())
        required_set = set(required_keys)
        if not required_set.issubset(response_keys):
            return False
    
    return True

