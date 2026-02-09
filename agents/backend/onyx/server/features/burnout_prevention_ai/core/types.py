"""
Type Aliases and Type Definitions
=================================
Type aliases for better type safety and code readability.

Centralizes common type definitions to improve consistency
and make the codebase more maintainable.
"""

from typing import Dict, Any, List, Optional

# Common type aliases
JSONDict = Dict[str, Any]
"""Dictionary representing JSON data."""

MessageDict = Dict[str, str]
"""Dictionary representing a chat message with 'role' and 'content'."""

MessageList = List[MessageDict]
"""List of chat messages."""

CacheEntry = Dict[str, Any]
"""Cache entry structure with 'value' and 'expires_at'."""

APIResponse = Dict[str, Any]
"""Response from external API."""

ModelList = List[Dict[str, Any]]
"""List of available models from API."""

