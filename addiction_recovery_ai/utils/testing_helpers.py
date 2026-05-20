"""
Testing helper utilities
Pure functions for testing support
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from utils.date_helpers import get_current_utc


def create_mock_assessment_data(
    addiction_type: str = "smoking",
    severity: str = "moderate",
    frequency: str = "daily"
) -> Dict[str, Any]:
    """
    Create mock assessment data for testing
    
    Args:
        addiction_type: Type of addiction
        severity: Severity level
        frequency: Frequency of use
    
    Returns:
        Mock assessment data dictionary
    """
    return {
        "addiction_type": addiction_type,
        "severity": severity,
        "frequency": frequency,
        "duration_years": 5.0,
        "daily_cost": 10.0,
        "triggers": ["stress", "social"],
        "motivations": ["health", "family"],
        "previous_attempts": 2,
        "support_system": True,
        "medical_conditions": [],
        "additional_info": "Test data"
    }


def create_mock_log_entry(
    user_id: str = "test_user",
    date: Optional[str] = None,
    consumed: bool = False
) -> Dict[str, Any]:
    """
    Create mock log entry for testing
    
    Args:
        user_id: User identifier
        date: Optional date (ISO format)
        consumed: Whether user consumed
    
    Returns:
        Mock log entry dictionary
    """
    if not date:
        date = get_current_utc().isoformat()
    
    return {
        "user_id": user_id,
        "date": date,
        "mood": "good",
        "cravings_level": 3,
        "triggers_encountered": ["stress"],
        "consumed": consumed,
        "notes": "Test entry"
    }


def create_mock_progress_data(
    user_id: str = "test_user",
    days_sober: int = 30
) -> Dict[str, Any]:
    """
    Create mock progress data for testing
    
    Args:
        user_id: User identifier
        days_sober: Days sober
    
    Returns:
        Mock progress data dictionary
    """
    return {
        "user_id": user_id,
        "days_sober": days_sober,
        "total_entries": 30,
        "streak_days": 30,
        "longest_streak": 30,
        "progress_percentage": 8.22,
        "recent_entries": []
    }


def assert_response_structure(
    response: Dict[str, Any],
    required_fields: List[str]
) -> bool:
    """
    Assert response has required fields
    
    Args:
        response: Response dictionary
        required_fields: List of required field names
    
    Returns:
        True if all fields present, raises AssertionError otherwise
    """
    missing_fields = [field for field in required_fields if field not in response]
    
    if missing_fields:
        raise AssertionError(f"Missing required fields: {missing_fields}")
    
    return True

