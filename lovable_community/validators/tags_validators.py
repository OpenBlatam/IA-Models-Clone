"""
Tag validation functions

Functions for validating tag lists and individual tags.
"""

import re
from typing import Optional, List


def validate_tags(tags: Optional[List[str]], max_count: int = 10, max_length: int = 50) -> Optional[List[str]]:
    """
    Validates a list of tags.
    
    Args:
        tags: List of tags to validate
        max_count: Maximum number of tags allowed
        max_length: Maximum length of each tag
        
    Returns:
        List of sanitized tags or None
        
    Raises:
        ValueError: If the tags are invalid
    """
    if tags is None:
        return None
    
    if not isinstance(tags, list):
        raise ValueError("Tags must be a list")
    
    if len(tags) > max_count:
        raise ValueError(f"Maximum {max_count} tags allowed")
    
    sanitized = []
    seen = set()
    
    for tag in tags:
        if not tag or not isinstance(tag, str):
            continue
        
        tag_clean = tag.strip().lower()
        
        if not tag_clean:
            continue
        
        if len(tag_clean) > max_length:
            raise ValueError(f"Tag '{tag_clean}' exceeds maximum length of {max_length} characters")
        
        # Validate that tag only contains alphanumeric, hyphens, and underscores
        if not re.match(r'^[a-z0-9_-]+$', tag_clean):
            raise ValueError(f"Tag '{tag_clean}' contains invalid characters. Only alphanumeric, hyphens, and underscores are allowed")
        
        if tag_clean not in seen:
            sanitized.append(tag_clean)
            seen.add(tag_clean)
    
    return sanitized if sanitized else None













