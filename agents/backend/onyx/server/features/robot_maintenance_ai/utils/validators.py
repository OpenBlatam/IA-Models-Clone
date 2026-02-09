"""
Input validators for Robot Maintenance AI.
"""

from typing import Dict, Any, List, Optional, Tuple
import re


def validate_in_list(value: str, allowed_values: List[str], value_name: str = "value") -> Tuple[bool, Optional[str]]:
    """
    Generic validator to check if a value is in an allowed list.
    
    Args:
        value: Value to validate
        allowed_values: List of allowed values
        value_name: Name of the value for error messages
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, str):
        return False, f"{value_name} must be a string"
    
    if value not in allowed_values:
        return False, f"Invalid {value_name}: {value}. Allowed values: {', '.join(allowed_values)}"
    
    return True, None


def validate_robot_type(robot_type: str, allowed_types: List[str]) -> bool:
    """
    Validate robot type.
    
    Args:
        robot_type: Robot type to validate
        allowed_types: List of allowed robot types
    
    Returns:
        True if valid, False otherwise
    """
    is_valid, _ = validate_in_list(robot_type, allowed_types, "robot_type")
    return is_valid


def validate_maintenance_type(maintenance_type: str, allowed_types: List[str]) -> bool:
    """
    Validate maintenance type.
    
    Args:
        maintenance_type: Maintenance type to validate
        allowed_types: List of allowed maintenance types
    
    Returns:
        True if valid, False otherwise
    """
    is_valid, _ = validate_in_list(maintenance_type, allowed_types, "maintenance_type")
    return is_valid


def validate_difficulty_level(difficulty: str, allowed_levels: List[str]) -> bool:
    """
    Validate difficulty level.
    
    Args:
        difficulty: Difficulty level to validate
        allowed_levels: List of allowed difficulty levels
    
    Returns:
        True if valid, False otherwise
    """
    is_valid, _ = validate_in_list(difficulty, allowed_levels, "difficulty_level")
    return is_valid


def validate_question(question: str, min_length: int = 10, max_length: int = 2000) -> Tuple[bool, Optional[str]]:
    """
    Validate question input.
    
    Args:
        question: Question to validate
        min_length: Minimum question length
        max_length: Maximum question length
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(question, str):
        return False, "Question must be a string"
    
    question = question.strip()
    
    if len(question) < min_length:
        return False, f"Question must be at least {min_length} characters long"
    
    if len(question) > max_length:
        return False, f"Question must be at most {max_length} characters long"
    
    if not question:
        return False, "Question cannot be empty"
    
    return True, None


# Valid sensor data keys
VALID_SENSOR_KEYS = [
    "temperature", "pressure", "vibration", "current", "voltage",
    "rpm", "torque", "humidity", "battery_level"
]

SENSOR_VALUE_RANGE = (0, 10000)


def validate_sensor_data_strict(sensor_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Strict validation of sensor data.
    
    Args:
        sensor_data: Sensor data to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(sensor_data, dict):
        return False, "Sensor data must be a dictionary"
    
    if not sensor_data:
        return False, "Sensor data cannot be empty"
    
    for key, value in sensor_data.items():
        if key not in VALID_SENSOR_KEYS:
            return False, f"Invalid sensor key: {key}. Allowed keys: {', '.join(VALID_SENSOR_KEYS)}"
        
        if not isinstance(value, (int, float)):
            return False, f"Sensor value for {key} must be a number"
        
        if isinstance(value, (int, float)) and (value < SENSOR_VALUE_RANGE[0] or value > SENSOR_VALUE_RANGE[1]):
            return False, f"Sensor value for {key} is out of reasonable range ({SENSOR_VALUE_RANGE[0]}-{SENSOR_VALUE_RANGE[1]})"
    
    return True, None


def sanitize_input(text: str) -> str:
    """
    Sanitize user input text.
    
    Args:
        text: Text to sanitize
    
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    return text[:2000]

