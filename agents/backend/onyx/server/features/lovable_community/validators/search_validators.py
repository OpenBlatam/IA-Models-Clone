"""
Search validation functions

Functions for validating search queries and parameters.
"""

from typing import Optional


def validate_search_query(query: Optional[str], max_length: int = 200) -> Optional[str]:
    """
    Validates a search query.
    
    Args:
        query: Search query to validate
        max_length: Maximum allowed length
        
    Returns:
        Sanitized query or None
        
    Raises:
        ValueError: If the query is invalid
    """
    if query is None:
        return None
    
    if not isinstance(query, str):
        raise ValueError("Search query must be a string")
    
    query = query.strip()
    
    if not query:
        return None
    
    if len(query) > max_length:
        raise ValueError(f"Search query cannot exceed {max_length} characters")
    
    return query













