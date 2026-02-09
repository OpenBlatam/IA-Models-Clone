"""
Helper Functions - Common helper functions
==========================================

Reusable helper functions used across the Perplexity system.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime


def convert_search_results_to_dict(
    search_results: Optional[List[Any]]
) -> Optional[List[Dict[str, Any]]]:
    """
    Convert search results to dictionary format.
    
    Handles both Pydantic models and dictionaries.
    
    Args:
        search_results: List of search results (Pydantic models or dicts)
        
    Returns:
        List of dictionaries or None
    """
    if not search_results:
        return None
    
    converted = []
    for result in search_results:
        if isinstance(result, dict):
            converted.append(result)
        elif hasattr(result, 'dict'):
            # Pydantic model
            converted.append(result.dict())
        elif hasattr(result, '__dict__'):
            # Regular object
            converted.append({
                'title': getattr(result, 'title', ''),
                'url': getattr(result, 'url', ''),
                'snippet': getattr(result, 'snippet', ''),
                'content': getattr(result, 'content', None),
                'source': getattr(result, 'source', None),
                'timestamp': getattr(result, 'timestamp', None)
            })
        else:
            # Fallback
            converted.append({
                'title': str(result),
                'url': '',
                'snippet': str(result),
                'content': None,
                'source': None,
                'timestamp': None
            })
    
    return converted


def sanitize_query_text(query: str) -> str:
    """
    Sanitize query text for safe processing.
    
    Args:
        query: Query string to sanitize
        
    Returns:
        Sanitized query string
    """
    if not query:
        return ""
    
    # Remove control characters
    query = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query)
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query)
    return query.strip()


def extract_query_metadata(query: str) -> Dict[str, Any]:
    """
    Extract metadata from query.
    
    Args:
        query: Query string
        
    Returns:
        Dictionary with metadata
    """
    return {
        'length': len(query),
        'word_count': len(query.split()),
        'has_url': bool(re.search(r'https?://', query)),
        'has_question_mark': query.endswith('?'),
        'has_math': bool(re.search(r'[+\-*/=]', query))
    }


def format_search_results_for_display(
    search_results: List[Dict[str, Any]],
    max_results: int = 10
) -> str:
    """
    Format search results for display.
    
    Args:
        search_results: List of search result dictionaries
        max_results: Maximum results to format
        
    Returns:
        Formatted string
    """
    if not search_results:
        return "No search results provided."
    
    formatted = []
    for i, result in enumerate(search_results[:max_results], 1):
        title = result.get('title', 'Untitled')
        url = result.get('url', '')
        snippet = result.get('snippet', '')
        
        formatted.append(f"{i}. {title}")
        if url:
            formatted.append(f"   URL: {url}")
        if snippet:
            formatted.append(f"   {snippet[:100]}...")
        formatted.append("")
    
    return "\n".join(formatted)


def validate_search_result(result: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate a search result dictionary.
    
    Args:
        result: Search result dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ['title', 'url', 'snippet']
    
    for field in required_fields:
        if field not in result:
            return False, f"Missing required field: {field}"
        
        if not isinstance(result[field], str):
            return False, f"Field {field} must be a string"
        
        if not result[field].strip():
            return False, f"Field {field} cannot be empty"
    
    return True, None


def merge_search_results(
    results1: List[Dict[str, Any]],
    results2: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge two lists of search results, removing duplicates.
    
    Args:
        results1: First list of search results
        results2: Second list of search results
        
    Returns:
        Merged list without duplicates
    """
    seen_urls = set()
    merged = []
    
    for result in results1 + results2:
        url = result.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            merged.append(result)
        elif not url:
            # Include results without URLs (might be duplicates)
            merged.append(result)
    
    return merged


def calculate_answer_quality_score(
    answer: str,
    search_results_count: int,
    citation_count: int
) -> float:
    """
    Calculate a quality score for an answer.
    
    Args:
        answer: The answer text
        search_results_count: Number of search results used
        citation_count: Number of citations in answer
        
    Returns:
        Quality score between 0.0 and 1.0
    """
    score = 0.0
    
    # Length score (optimal around 500-2000 chars)
    length = len(answer)
    if 500 <= length <= 2000:
        score += 0.3
    elif 200 <= length < 500 or 2000 < length <= 5000:
        score += 0.2
    else:
        score += 0.1
    
    # Citation score
    if search_results_count > 0:
        citation_ratio = citation_count / search_results_count
        if 0.3 <= citation_ratio <= 0.8:
            score += 0.4
        elif 0.1 <= citation_ratio < 0.3 or 0.8 < citation_ratio <= 1.0:
            score += 0.3
        else:
            score += 0.1
    else:
        score += 0.2  # No citations needed if no results
    
    # Structure score (has headers, proper formatting)
    if '##' in answer:
        score += 0.2
    if len(answer.split('\n')) > 5:
        score += 0.1
    
    return min(1.0, score)




