"""
Pure functions for relapse prevention
Refactored from class-based to functional approach
"""

from typing import Dict, Any, List
from datetime import datetime
from services.functions.relapse_functions import (
    calculate_relapse_risk,
    identify_risk_factors,
    identify_protective_factors
)


def assess_relapse_risk(
    user_id: str,
    days_sober: int,
    stress_level: int,
    support_level: int,
    current_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assess relapse risk (RORO pattern)
    
    Args:
        user_id: User identifier
        days_sober: Days since last use
        stress_level: Current stress level (0-10)
        support_level: Current support level (0-10)
        current_state: Current state information
    
    Returns:
        Dictionary with relapse risk assessment
    """
    # Guard clauses
    if not user_id:
        raise ValueError("user_id is required")
    
    if days_sober < 0:
        raise ValueError("days_sober cannot be negative")
    
    if not (0 <= stress_level <= 10):
        raise ValueError("stress_level must be between 0 and 10")
    
    if not (0 <= support_level <= 10):
        raise ValueError("support_level must be between 0 and 10")
    
    # Identify risk factors
    risk_factors = identify_risk_factors(
        stress_level=stress_level,
        support_level=support_level,
        isolation=current_state.get("isolation", False),
        negative_thinking=current_state.get("negative_thinking", False),
        romanticizing=current_state.get("romanticizing", False),
        skipping_support=current_state.get("skipping_support", False)
    )
    
    # Identify protective factors
    protective_factors = identify_protective_factors(
        support_level=support_level,
        coping_skills=current_state.get("coping_skills", 5),
        motivation=current_state.get("motivation", 5),
        has_plan=current_state.get("has_plan", False)
    )
    
    # Calculate risk using pure function
    risk_assessment = calculate_relapse_risk(
        days_sober=days_sober,
        stress_level=stress_level,
        support_level=support_level,
        risk_factors=risk_factors
    )
    
    # Return object (RORO pattern)
    return {
        "user_id": user_id,
        "assessment_id": f"relapse_risk_{user_id}_{datetime.now().timestamp()}",
        "days_sober": days_sober,
        "risk_score": risk_assessment["risk_score"],
        "risk_level": risk_assessment["risk_level"],
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "recommendations": risk_assessment["recommendations"],
        "assessed_at": datetime.now().isoformat()
    }


def generate_prevention_strategy(
    risk_level: str,
    risk_factors: List[str],
    protective_factors: List[str]
) -> Dict[str, Any]:
    """
    Generate prevention strategy (RORO pattern)
    
    Args:
        risk_level: Risk level (low, moderate, high, critical)
        risk_factors: List of identified risk factors
        protective_factors: List of identified protective factors
    
    Returns:
        Dictionary with prevention strategy
    """
    # Guard clause
    if not risk_level:
        raise ValueError("risk_level is required")
    
    # Determine strategy based on risk level
    strategies = {
        "critical": {
            "immediate_actions": [
                "Contact crisis line immediately",
                "Reach out to support system",
                "Use emergency coping techniques"
            ],
            "daily_actions": [
                "Check in with support system 3x daily",
                "Practice mindfulness techniques",
                "Avoid high-risk situations"
            ]
        },
        "high": {
            "immediate_actions": [
                "Contact support system",
                "Review recovery plan",
                "Use coping strategies"
            ],
            "daily_actions": [
                "Check in with support system 2x daily",
                "Practice stress management",
                "Attend support group"
            ]
        },
        "moderate": {
            "immediate_actions": [
                "Review recovery plan",
                "Practice coping skills"
            ],
            "daily_actions": [
                "Daily check-in",
                "Practice mindfulness",
                "Stay connected"
            ]
        },
        "low": {
            "immediate_actions": [
                "Maintain current routine"
            ],
            "daily_actions": [
                "Continue recovery practices",
                "Stay vigilant"
            ]
        }
    }
    
    strategy = strategies.get(risk_level, strategies["moderate"])
    
    # Return object (RORO pattern)
    return {
        "risk_level": risk_level,
        "strategy": strategy,
        "risk_factors_to_address": risk_factors,
        "protective_factors_to_maintain": protective_factors,
        "generated_at": datetime.now().isoformat()
    }

