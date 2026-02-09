"""
Pure functions for recovery planning
Refactored from class-based to functional approach
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
from utils.date_helpers import add_days, get_current_utc


def create_recovery_plan(
    user_id: str,
    addiction_type: str,
    severity_score: float,
    risk_level: str,
    preferences: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create recovery plan (RORO pattern)
    
    Args:
        user_id: User identifier
        addiction_type: Type of addiction
        severity_score: Calculated severity score
        risk_level: Risk level (low, moderate, high, severe)
        preferences: User preferences
    
    Returns:
        Dictionary with recovery plan
    """
    # Guard clauses
    if not user_id:
        raise ValueError("user_id is required")
    
    if not addiction_type:
        raise ValueError("addiction_type is required")
    
    if severity_score < 0 or severity_score > 100:
        raise ValueError("severity_score must be between 0 and 100")
    
    # Determine plan duration based on severity
    plan_duration_days = _calculate_plan_duration(severity_score, risk_level)
    
    # Generate phases
    phases = _generate_phases(plan_duration_days, risk_level)
    
    # Generate milestones
    milestones = _generate_milestones(plan_duration_days)
    
    # Generate activities
    activities = _generate_activities(
        addiction_type=addiction_type,
        risk_level=risk_level,
        preferences=preferences
    )
    
    # Return object (RORO pattern)
    return {
        "plan_id": f"plan_{user_id}_{datetime.now().timestamp()}",
        "user_id": user_id,
        "addiction_type": addiction_type,
        "severity_score": severity_score,
        "risk_level": risk_level,
        "duration_days": plan_duration_days,
        "start_date": get_current_utc().isoformat(),
        "end_date": add_days(get_current_utc(), plan_duration_days).isoformat(),
        "phases": phases,
        "milestones": milestones,
        "activities": activities,
        "created_at": get_current_utc().isoformat()
    }


def _calculate_plan_duration(severity_score: float, risk_level: str) -> int:
    """Calculate plan duration in days"""
    if risk_level == "severe":
        return 365
    if risk_level == "high":
        return 180
    if risk_level == "moderate":
        return 90
    return 60


def _generate_phases(duration_days: int, risk_level: str) -> List[Dict[str, Any]]:
    """Generate recovery phases"""
    phase_duration = duration_days // 4
    
    phases = [
        {
            "phase": 1,
            "name": "Detoxification",
            "duration_days": phase_duration,
            "focus": "Physical withdrawal and stabilization"
        },
        {
            "phase": 2,
            "name": "Early Recovery",
            "duration_days": phase_duration,
            "focus": "Building new habits and coping strategies"
        },
        {
            "phase": 3,
            "name": "Maintenance",
            "duration_days": phase_duration,
            "focus": "Sustaining recovery and preventing relapse"
        },
        {
            "phase": 4,
            "name": "Long-term Recovery",
            "duration_days": duration_days - (phase_duration * 3),
            "focus": "Building a new life in recovery"
        }
    ]
    
    return phases


def _generate_milestones(duration_days: int) -> List[Dict[str, Any]]:
    """Generate recovery milestones"""
    milestone_days = [7, 30, 60, 90, 180, 365]
    
    milestones = []
    for days in milestone_days:
        if days <= duration_days:
            milestones.append({
                "days": days,
                "name": f"{days} Days Sober",
                "description": f"Celebrate {days} days of sobriety"
            })
    
    return milestones


def _generate_activities(
    addiction_type: str,
    risk_level: str,
    preferences: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate recovery activities"""
    activities = [
        {
            "type": "daily_checkin",
            "frequency": "daily",
            "description": "Daily mood and craving check-in"
        },
        {
            "type": "support_group",
            "frequency": "weekly",
            "description": "Attend support group meeting"
        }
    ]
    
    if risk_level in ["high", "severe"]:
        activities.append({
            "type": "therapy",
            "frequency": "weekly",
            "description": "Individual therapy session"
        })
    
    return activities

