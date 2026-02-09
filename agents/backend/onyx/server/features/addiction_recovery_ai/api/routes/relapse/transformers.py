"""
Data transformers for relapse endpoints
Pure functions for transforming request/response data
"""

from typing import Dict, Any

try:
    from schemas.relapse import RelapseRiskRequest, RelapseRiskResponse
    from utils.transformers import pick_fields
except ImportError:
    from ...schemas.relapse import RelapseRiskRequest, RelapseRiskResponse
    from ...utils.transformers import pick_fields


def transform_relapse_request_to_dict(
    request: RelapseRiskRequest
) -> Dict[str, Any]:
    """
    Transform RelapseRiskRequest to dictionary (RORO pattern)
    
    Args:
        request: RelapseRiskRequest object
    
    Returns:
        Dictionary with relapse risk data
    """
    if not request:
        raise ValueError("request cannot be None")
    
    return request.model_dump()


def extract_relapse_summary(
    assessment: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract summary from relapse assessment (RORO pattern)
    
    Args:
        assessment: Full assessment dictionary
    
    Returns:
        Summary dictionary
    """
    if not assessment:
        raise ValueError("assessment cannot be empty")
    
    summary_fields = [
        "user_id",
        "risk_score",
        "risk_level",
        "assessed_at"
    ]
    
    return pick_fields(assessment, summary_fields)

