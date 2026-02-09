"""
Pure functions for addiction analysis
Refactored from class-based to functional approach
"""

from typing import Dict, Any, List
from datetime import datetime

try:
    from utils.cache import cache_result
    from utils.guards import guard_not_empty, guard_in_list
    from services.functions.assessment_functions import (
        calculate_severity_score,
        determine_risk_level,
        generate_recommendations,
        generate_next_steps
    )
except ImportError:
    from ...utils.cache import cache_result
    from ...utils.guards import guard_not_empty, guard_in_list
    from ...services.functions.assessment_functions import (
        calculate_severity_score,
        determine_risk_level,
        generate_recommendations,
        generate_next_steps
    )


def assess_addiction(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess addiction based on provided data (RORO pattern)
    
    Args:
        assessment_data: Dictionary with assessment data
    
    Returns:
        Dictionary with complete analysis
    """
    # Guard clauses using utility functions
    guard_not_empty(assessment_data, "assessment_data")
    
    if "addiction_type" not in assessment_data:
        raise ValueError("addiction_type is required")
    
    # Validate severity
    if "severity" in assessment_data:
        guard_in_list(
            assessment_data["severity"],
            ["low", "moderate", "high", "severe"],
            "severity"
        )
    
    # Extract data
    addiction_type = assessment_data.get("addiction_type", "")
    severity = assessment_data.get("severity", "moderate")
    frequency = assessment_data.get("frequency", "occasional")
    duration_years = assessment_data.get("duration_years", 0.0)
    daily_cost = assessment_data.get("daily_cost", 0.0)
    triggers = assessment_data.get("triggers", [])
    motivations = assessment_data.get("motivations", [])
    previous_attempts = assessment_data.get("previous_attempts", 0)
    support_system = assessment_data.get("support_system", False)
    medical_conditions = assessment_data.get("medical_conditions", [])
    
    # Calculate severity score using pure function
    severity_score = calculate_severity_score(
        severity=severity,
        frequency=frequency,
        duration_years=duration_years,
        daily_cost=daily_cost,
        previous_attempts=previous_attempts,
        support_system=support_system,
        medical_conditions=medical_conditions
    )
    
    # Determine risk level using pure function
    risk_level = determine_risk_level(severity_score)
    
    # Generate recommendations using pure function
    recommendations = generate_recommendations(
        addiction_type=addiction_type,
        severity_score=severity_score,
        risk_level=risk_level,
        triggers=triggers,
        motivations=motivations
    )
    
    # Generate next steps using pure function
    next_steps = generate_next_steps(
        risk_level=risk_level,
        previous_attempts=previous_attempts,
        support_system=support_system
    )
    
    # Return object (RORO pattern)
    return {
        "assessment_id": f"assessment_{datetime.now().timestamp()}",
        "addiction_type": addiction_type,
        "severity_score": severity_score,
        "risk_level": risk_level,
        "recommendations": recommendations,
        "next_steps": next_steps,
        "assessed_at": datetime.now().isoformat()
    }


@cache_result(ttl=300, key_prefix="addiction_assessment")
def assess_addiction_cached(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cached version of assess_addiction
    
    Args:
        assessment_data: Dictionary with assessment data
    
    Returns:
        Dictionary with complete analysis
    """
    return assess_addiction(assessment_data)


def validate_assessment_data(assessment_data: Dict[str, Any]) -> None:
    """
    Validate assessment data (guard clause helper)
    
    Args:
        assessment_data: Dictionary with assessment data
    
    Raises:
        ValueError if validation fails
    """
    if not assessment_data:
        raise ValueError("assessment_data cannot be empty")
    
    required_fields = ["addiction_type", "severity", "frequency"]
    for field in required_fields:
        if field not in assessment_data:
            raise ValueError(f"{field} is required")
    
    # Validate severity
    valid_severities = ["low", "moderate", "high", "severe"]
    if assessment_data.get("severity") not in valid_severities:
        raise ValueError(f"severity must be one of: {valid_severities}")
    
    # Validate frequency
    valid_frequencies = ["daily", "weekly", "monthly", "occasional"]
    if assessment_data.get("frequency") not in valid_frequencies:
        raise ValueError(f"frequency must be one of: {valid_frequencies}")

