"""
Search helper functions

Functions for normalizing and processing search queries.
"""

from typing import List
from .string_normalization import normalize_to_lower, normalize_whitespace, join_strings
from .validation_common import is_empty_string, is_empty_list


def normalize_search_query(query: str) -> str:
    """
    Normalizes a search query.
    
    Args:
        query: Search query
        
    Returns:
        Normalized query
    """
    if is_empty_string(query):
        return ""
    
    # Remove extra spaces and convert to lowercase
    return normalize_whitespace(query)


def extract_search_terms(query: str) -> List[str]:
    """
    Extracts search terms from a query.
    
    Args:
        query: Search query
        
    Returns:
        List of search terms
    """
    if is_empty_string(query):
        return []
    
    from .string_normalization import normalize_list_to_lower
    
    terms = query.split()
    return normalize_list_to_lower(terms)


def build_search_filter(search_terms: List[str], fields: List[str]) -> str:
    """
    Builds a search filter pattern for SQL LIKE.
    
    Args:
        search_terms: List of search terms
        fields: Fields to search in
        
    Returns:
        Search filter pattern
    """
    if is_empty_list(search_terms):
        return ""
    
    patterns = [f"%{term}%" for term in search_terms]
    return join_strings(patterns, " OR ", filter_empty=True)











