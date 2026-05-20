"""
Data transformers for support endpoints
Pure functions for transforming request/response data
"""

from typing import Dict, Any

try:
    from schemas.support import CoachingRequest, MotivationRequest
    from utils.transformers import pick_fields
except ImportError:
    from ...schemas.support import CoachingRequest, MotivationRequest
    from ...utils.transformers import pick_fields


def transform_coaching_request_to_dict(
    request: CoachingRequest
) -> Dict[str, Any]:
    """
    Transform CoachingRequest to dictionary (RORO pattern)
    
    Args:
        request: CoachingRequest object
    
    Returns:
        Dictionary with coaching data
    """
    if not request:
        raise ValueError("request cannot be None")
    
    return request.model_dump()


def transform_motivation_request_to_dict(
    request: MotivationRequest
) -> Dict[str, Any]:
    """
    Transform MotivationRequest to dictionary (RORO pattern)
    
    Args:
        request: MotivationRequest object
    
    Returns:
        Dictionary with motivation data
    """
    if not request:
        raise ValueError("request cannot be None")
    
    return request.model_dump()

