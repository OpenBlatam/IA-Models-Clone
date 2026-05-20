"""
Metadata Utilities for API
===========================

Utilities for parsing and handling metadata in API requests.
"""

import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def parse_metadata(metadata_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parse JSON metadata string.
    
    Args:
        metadata_str: JSON string or None
        
    Returns:
        Parsed metadata dict or None
        
    Raises:
        ValueError: If JSON parsing fails
    """
    if not metadata_str:
        return None
    
    try:
        return json.loads(metadata_str)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing metadata JSON: {e}")
        raise ValueError(f"Invalid JSON metadata: {e}")

