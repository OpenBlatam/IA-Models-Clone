"""Pattern matching utilities."""

import re
from typing import List, Optional, Callable, Any


def match_pattern(text: str, pattern: str, flags: int = 0) -> Optional[re.Match]:
    """
    Match pattern in text.
    
    Args:
        text: Text to search
        pattern: Regex pattern
        flags: Regex flags
        
    Returns:
        Match object or None
    """
    return re.search(pattern, text, flags)


def find_all_matches(text: str, pattern: str, flags: int = 0) -> List[str]:
    """
    Find all matches of pattern in text.
    
    Args:
        text: Text to search
        pattern: Regex pattern
        flags: Regex flags
        
    Returns:
        List of matches
    """
    return re.findall(pattern, text, flags)


def replace_pattern(text: str, pattern: str, replacement: str, count: int = 0) -> str:
    """
    Replace pattern in text.
    
    Args:
        text: Text to process
        pattern: Regex pattern
        replacement: Replacement string
        count: Maximum replacements (0 = all)
        
    Returns:
        Text with replacements
    """
    return re.sub(pattern, replacement, text, count=count)


def split_pattern(text: str, pattern: str, maxsplit: int = 0) -> List[str]:
    """
    Split text by pattern.
    
    Args:
        text: Text to split
        pattern: Regex pattern
        maxsplit: Maximum splits
        
    Returns:
        List of parts
    """
    return re.split(pattern, text, maxsplit=maxsplit)


def extract_groups(text: str, pattern: str) -> Optional[tuple]:
    """
    Extract groups from pattern match.
    
    Args:
        text: Text to search
        pattern: Regex pattern with groups
        
    Returns:
        Tuple of groups or None
    """
    match = re.search(pattern, text)
    if match:
        return match.groups()
    return None


def validate_pattern(text: str, pattern: str) -> bool:
    """
    Validate text matches pattern exactly.
    
    Args:
        text: Text to validate
        pattern: Regex pattern
        
    Returns:
        True if text matches pattern
    """
    return bool(re.fullmatch(pattern, text))


def escape_regex(text: str) -> str:
    """
    Escape special regex characters.
    
    Args:
        text: Text to escape
        
    Returns:
        Escaped text
    """
    return re.escape(text)

