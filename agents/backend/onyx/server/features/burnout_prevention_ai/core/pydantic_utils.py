"""
Pydantic Utilities
==================
Helper functions for working with Pydantic models.

Provides compatibility helpers for Pydantic v1 and v2,
ensuring the code works with both versions seamlessly.
"""

from typing import Any, Dict


def model_to_dict(model: Any) -> Dict[str, Any]:
    """
    Convert Pydantic model to dictionary (compatible with v1 and v2).
    
    Args:
        model: Pydantic model instance
        
    Returns:
        Dictionary representation of the model
    """
    # Try Pydantic v2 first
    try:
        return model.model_dump(mode='json')
    except (AttributeError, TypeError):
        # Fallback to Pydantic v1
        try:
            return model.dict()
        except AttributeError:
            # Last resort: return as-is (shouldn't happen with valid Pydantic models)
            return model

