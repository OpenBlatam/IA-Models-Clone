"""
Value extraction with fallback helper functions.

Refactored to consolidate functions into ValueExtractor class.
"""

from typing import Dict, Any, List, Optional


class ValueExtractor:
    """
    Value extraction with fallback utilities.
    
    Responsibilities:
    - Get values from multiple dictionaries with fallback
    - Get values with alternative sources
    
    Single Responsibility: Handle all value extraction with fallback operations.
    """
    
    def get_value_with_fallback(
        self,
        sources: List[Dict[str, Any]],
        keys: List[str],
        default: Any = None
    ) -> Any:
        """
        Get value from multiple dictionaries using multiple possible keys.
        
        Tries each source dictionary with each key until a value is found.
        
        Args:
            sources: List of dictionaries to search
            keys: List of keys to try (in order)
            default: Default value if none found
        
        Returns:
            First found value or default
        """
        for source in sources:
            for key in keys:
                value = source.get(key)
                if value is not None and value != "":
                    return value
        return default
    
    def get_value_or_alternative(
        self,
        primary_dict: Dict[str, Any],
        primary_key: str,
        fallback_dict: Dict[str, Any],
        fallback_key: str,
        default: Any = None
    ) -> Any:
        """
        Get value from primary dict, fallback to alternative dict.
        
        Args:
            primary_dict: Primary dictionary
            primary_key: Primary key
            fallback_dict: Fallback dictionary
            fallback_key: Fallback key
            default: Default value if both fail
        
        Returns:
            Value from primary, fallback, or default
        """
        value = primary_dict.get(primary_key)
        if value is not None and value != "":
            return value
        
        value = fallback_dict.get(fallback_key)
        if value is not None and value != "":
            return value
        
        return default


# Backward compatibility functions
def get_value_with_fallback(
    sources: List[Dict[str, Any]],
    keys: List[str],
    default: Any = None
) -> Any:
    """
    Get value with fallback (backward compatibility).
    
    Args:
        sources: List of dictionaries to search
        keys: List of keys to try (in order)
        default: Default value if none found
    
    Returns:
        First found value or default
    """
    extractor = ValueExtractor()
    return extractor.get_value_with_fallback(sources, keys, default)


def get_value_or_alternative(
    primary_dict: Dict[str, Any],
    primary_key: str,
    fallback_dict: Dict[str, Any],
    fallback_key: str,
    default: Any = None
) -> Any:
    """
    Get value or alternative (backward compatibility).
    
    Args:
        primary_dict: Primary dictionary
        primary_key: Primary key
        fallback_dict: Fallback dictionary
        fallback_key: Fallback key
        default: Default value if both fail
    
    Returns:
        Value from primary, fallback, or default
    """
    extractor = ValueExtractor()
    return extractor.get_value_or_alternative(
        primary_dict, primary_key, fallback_dict, fallback_key, default
    )
