"""
Helper utilities for Robot Maintenance AI.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json


def format_maintenance_response(
    answer: str,
    recommendations: List[str],
    safety_warnings: List[str],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format maintenance response in a structured way.
    
    Args:
        answer: Main answer text
        recommendations: List of recommendations
        safety_warnings: List of safety warnings
        metadata: Additional metadata
    
    Returns:
        Formatted response dictionary
    """
    return {
        "answer": answer,
        "recommendations": recommendations,
        "safety_warnings": safety_warnings,
        "metadata": metadata or {},
        "timestamp": get_iso_timestamp()
    }


def validate_sensor_data(sensor_data: Dict[str, Any]) -> bool:
    """
    Validate sensor data structure (basic validation).
    Uses the same validation logic as validators module for consistency.
    
    Args:
        sensor_data: Sensor data dictionary
    
    Returns:
        True if valid, False otherwise
    """
    from .validators import validate_sensor_data_strict
    
    is_valid, _ = validate_sensor_data_strict(sensor_data)
    return is_valid


def extract_steps_from_text(text: str) -> List[str]:
    """
    Extract step-by-step instructions from text.
    
    Args:
        text: Text containing instructions
    
    Returns:
        List of steps
    """
    steps = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if any(marker in line.lower() for marker in ['paso', 'step', '1.', '2.', '3.', '-']):
            if len(line) > 10:
                steps.append(line)
    
    return steps


def calculate_maintenance_priority(
    health_score: float,
    anomalies: List[str],
    usage_hours: int
) -> str:
    """
    Calculate maintenance priority based on various factors.
    
    Args:
        health_score: Health score from ML analysis
        anomalies: List of detected anomalies
        usage_hours: Hours of operation
    
    Returns:
        Priority level: "baja", "normal", "media", "alta", "critica"
    """
    if health_score < -0.5 or len(anomalies) > 3:
        return "critica"
    elif health_score < -0.2 or len(anomalies) > 1:
        return "alta"
    elif health_score < 0 or anomalies:
        return "media"
    elif usage_hours > 10000:
        return "normal"
    else:
        return "baja"
