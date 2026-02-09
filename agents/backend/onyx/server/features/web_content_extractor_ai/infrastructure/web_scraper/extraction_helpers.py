"""
Safe extraction wrapper functions for web scraping.

Refactored to consolidate functions into SafeExtractor class.
"""

import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)


class SafeExtractor:
    """
    Safe extraction wrapper utilities.
    
    Responsibilities:
    - Safely execute extraction functions with error handling
    
    Single Responsibility: Handle all safe extraction operations.
    """
    
    def extract(
        self,
        extractor_func: Callable,
        tool_name: str = "Extractor",
        default: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Safely execute an extraction function with error handling.
        
        Args:
            extractor_func: Function to execute
            tool_name: Name of tool for logging
            default: Default return value on error
        
        Returns:
            Extraction result or default
        """
        if default is None:
            default = {}
        
        try:
            result = extractor_func()
            return result if result else default
        except Exception as e:
            logger.debug(f"Error con {tool_name}: {e}")
            return default


# Backward compatibility function
def safe_extract(
    extractor_func: Callable,
    tool_name: str = "Extractor",
    default: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Safely execute extraction (backward compatibility).
    
    Args:
        extractor_func: Function to execute
        tool_name: Name of tool for logging
        default: Default return value on error
    
    Returns:
        Extraction result or default
    """
    extractor = SafeExtractor()
    return extractor.extract(extractor_func, tool_name, default)
