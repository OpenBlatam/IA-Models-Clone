"""
Data transformers for assessment endpoints
Pure functions for transforming request/response data
"""

from typing import Dict, Any
from schemas.assessment import AssessmentRequest, AssessmentResponse
from utils.transformers import transform_dict, pick_fields


def transform_assessment_request_to_dict(
    request: AssessmentRequest
) -> Dict[str, Any]:
    """
    Transform AssessmentRequest to dictionary (RORO pattern)
    
    Args:
        request: AssessmentRequest object
    
    Returns:
        Dictionary with assessment data
    """
    # Guard clause
    if not request:
        raise ValueError("request cannot be None")
    
    # Return object (RORO pattern)
    return request.model_dump()


def transform_analysis_to_response(
    analysis: Dict[str, Any],
    request: AssessmentRequest
) -> AssessmentResponse:
    """
    Transform analysis result to AssessmentResponse (RORO pattern)
    
    Args:
        analysis: Analysis result dictionary
        request: Original request
    
    Returns:
        AssessmentResponse object
    """
    # Guard clause
    if not analysis:
        raise ValueError("analysis cannot be empty")
    
    # Return object (RORO pattern)
    return AssessmentResponse(
        assessment_id=analysis.get("assessment_id", ""),
        addiction_type=request.addiction_type,
        severity_score=analysis.get("severity_score", 0.0),
        risk_level=analysis.get("risk_level", "unknown"),
        recommendations=analysis.get("recommendations", []),
        next_steps=analysis.get("next_steps", [])
    )


def extract_assessment_summary(
    analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract summary from analysis (RORO pattern)
    
    Args:
        analysis: Full analysis dictionary
    
    Returns:
        Summary dictionary
    """
    # Guard clause
    if not analysis:
        raise ValueError("analysis cannot be empty")
    
    # Pick only summary fields
    summary_fields = [
        "assessment_id",
        "addiction_type",
        "severity_score",
        "risk_level"
    ]
    
    # Return object (RORO pattern)
    return pick_fields(analysis, summary_fields)

