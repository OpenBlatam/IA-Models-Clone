"""
Batch Validator
===============
Validates batch items and parameters
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Constants
OPERATION_TYPE_CLOTHING_CHANGE = "clothing_change"
OPERATION_TYPE_FACE_SWAP = "face_swap"
MAX_BATCH_ITEMS = 1000
MIN_MAX_CONCURRENT = 1
MAX_MAX_CONCURRENT = 50


class BatchValidator:
    """
    Validates batch items and parameters.
    """
    
    @staticmethod
    def validate_batch_items(items: List[Dict[str, Any]], operation_type: str) -> None:
        """
        Validate batch items.
        
        Args:
            items: List of items to validate
            operation_type: Type of operation ("clothing_change" or "face_swap")
            
        Raises:
            ValueError: If items are invalid
        """
        if not items:
            raise ValueError("Items list cannot be empty")
        
        if not isinstance(items, list):
            raise ValueError("Items must be a list")
        
        if len(items) > MAX_BATCH_ITEMS:
            raise ValueError(
                f"Batch size exceeds maximum of {MAX_BATCH_ITEMS} items. "
                f"Got {len(items)} items."
            )
        
        # Validate operation type
        if operation_type not in (OPERATION_TYPE_CLOTHING_CHANGE, OPERATION_TYPE_FACE_SWAP):
            raise ValueError(
                f"Invalid operation_type: {operation_type}. "
                f"Must be '{OPERATION_TYPE_CLOTHING_CHANGE}' or '{OPERATION_TYPE_FACE_SWAP}'"
            )
        
        # Validate each item
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} must be a dictionary")
            
            # Validate image_url (required for all operations)
            image_url = item.get("image_url")
            if not image_url or not isinstance(image_url, str) or not image_url.strip():
                raise ValueError(f"Item {i} missing or invalid required field: image_url")
            
            # Validate face_url for face swap operations
            if operation_type == OPERATION_TYPE_FACE_SWAP:
                face_url = item.get("face_url")
                if not face_url or not isinstance(face_url, str) or not face_url.strip():
                    raise ValueError(f"Item {i} missing or invalid required field: face_url")
    
    @staticmethod
    def validate_max_concurrent(max_concurrent: Optional[int]) -> int:
        """
        Validate and clamp max_concurrent value.
        
        Args:
            max_concurrent: Maximum concurrent operations to validate
            
        Returns:
            Validated and clamped max_concurrent value
        """
        if max_concurrent is None:
            return 5  # Default
        
        if not isinstance(max_concurrent, int):
            raise ValueError("max_concurrent must be an integer")
        
        if max_concurrent < MIN_MAX_CONCURRENT:
            logger.warning(
                f"max_concurrent ({max_concurrent}) is below minimum ({MIN_MAX_CONCURRENT}), "
                f"clamping to {MIN_MAX_CONCURRENT}"
            )
            return MIN_MAX_CONCURRENT
        
        if max_concurrent > MAX_MAX_CONCURRENT:
            logger.warning(
                f"max_concurrent ({max_concurrent}) exceeds maximum ({MAX_MAX_CONCURRENT}), "
                f"clamping to {MAX_MAX_CONCURRENT}"
            )
            return MAX_MAX_CONCURRENT
        
        return max_concurrent
    
    @staticmethod
    def validate_item(item: Dict[str, Any], index: int, operation_type: str) -> None:
        """
        Validate a single batch item.
        
        Args:
            item: Item dictionary to validate
            index: Item index for error messages
            operation_type: Type of operation
            
        Raises:
            ValueError: If item is invalid
        """
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} must be a dictionary")
        
        # Validate image_url
        image_url = item.get("image_url")
        if not image_url or not isinstance(image_url, str) or not image_url.strip():
            raise ValueError(f"Item {index} missing or invalid required field: image_url")
        
        # Validate face_url for face swap
        if operation_type == OPERATION_TYPE_FACE_SWAP:
            face_url = item.get("face_url")
            if not face_url or not isinstance(face_url, str) or not face_url.strip():
                raise ValueError(f"Item {index} missing or invalid required field: face_url")

