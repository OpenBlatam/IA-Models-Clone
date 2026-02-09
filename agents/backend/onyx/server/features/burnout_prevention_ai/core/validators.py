"""
Validators for Burnout Prevention AI
====================================
"""

from typing import Any, Dict, List
from .security import sanitize_string, sanitize_list
from .exceptions import ValidationError
from .constants import (
    MIN_WORK_HOURS,
    MAX_WORK_HOURS,
    MIN_STRESS_LEVEL,
    MAX_STRESS_LEVEL,
    MIN_SLEEP_HOURS,
    MAX_SLEEP_HOURS,
    MIN_SATISFACTION,
    MAX_SATISFACTION,
    MIN_INTERVAL_SECONDS,
    MAX_MESSAGE_LENGTH
)


def validate_assessment_data(data: Dict[str, Any]) -> None:
    """
    Validate assessment data structure (optimized).
    
    Args:
        data: Assessment data dictionary
        
    Raises:
        ValidationError if data is invalid
    """
    if not isinstance(data, dict):
        raise ValidationError("Assessment data must be a dictionary")
    
    # Fast path: check required fields first (using set for O(1) lookup)
    required = {"work_hours_per_week", "stress_level", "sleep_hours_per_night", "work_satisfaction"}
    data_keys = set(data.keys())
    missing = required - data_keys
    if missing:
        raise ValidationError(f"Missing required fields: {', '.join(sorted(missing))}")
    
    # Validate types and ranges (optimized: direct access after checking existence)
    # Helper function to validate numeric range
    def validate_numeric_range(value: Any, field_name: str, min_val: float, max_val: float) -> None:
        """Helper to validate numeric field with range."""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{field_name} must be a number")
        if not (min_val <= value <= max_val):
            raise ValidationError(f"{field_name} must be between {min_val} and {max_val}")
    
    validate_numeric_range(
        data["work_hours_per_week"], 
        "work_hours_per_week", 
        MIN_WORK_HOURS, 
        MAX_WORK_HOURS
    )
    validate_numeric_range(
        data["stress_level"], 
        "stress_level", 
        MIN_STRESS_LEVEL, 
        MAX_STRESS_LEVEL
    )
    validate_numeric_range(
        data["sleep_hours_per_night"], 
        "sleep_hours_per_night", 
        MIN_SLEEP_HOURS, 
        MAX_SLEEP_HOURS
    )
    validate_numeric_range(
        data["work_satisfaction"], 
        "work_satisfaction", 
        MIN_SATISFACTION, 
        MAX_SATISFACTION
    )


def validate_conversation_history(history: List[Dict[str, str]]) -> None:
    """
    Validate conversation history format (optimized with sanitization).
    
    Args:
        history: List of message dictionaries
        
    Raises:
        ValidationError if history is invalid
    """
    valid_roles = {"system", "user", "assistant"}  # Set for O(1) lookup
    max_history_length = 50  # Limit conversation history length
    
    if len(history) > max_history_length:
        raise ValidationError(f"Conversation history too long. Maximum {max_history_length} messages.")
    
    for i, msg in enumerate(history):
        if not isinstance(msg, dict):
            raise ValidationError(f"Message {i} must be a dictionary")
        role = msg.get("role")
        content = msg.get("content")
        if not role or not content:
            raise ValidationError(f"Message {i} must have 'role' and 'content' fields")
        if role not in valid_roles:
            raise ValidationError(f"Message {i} has invalid role: {role}")
        # Validate content length
        content_str = str(content) if content else ""
        if len(content_str) > MAX_MESSAGE_LENGTH:
            raise ValidationError(f"Message {i} content too long. Maximum {MAX_MESSAGE_LENGTH} characters.")
        
        # Sanitize content
        sanitized_content = sanitize_string(content_str, max_length=MAX_MESSAGE_LENGTH)
        if sanitized_content != content_str:
            # Update in place if sanitized
            msg["content"] = sanitized_content


def validate_api_parameters(
    max_tokens: int,
    temperature: float = 0.7,
    min_tokens: int = 1,
    max_tokens_limit: int = 100000,
    min_temperature: float = 0.0,
    max_temperature: float = 2.0
) -> None:
    """
    Validate API generation parameters.
    
    Args:
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        min_tokens: Minimum allowed tokens
        max_tokens_limit: Maximum allowed tokens
        min_temperature: Minimum allowed temperature
        max_temperature: Maximum allowed temperature
        
    Raises:
        ValidationError if parameters are invalid
    """
    if not isinstance(max_tokens, int) or max_tokens < min_tokens:
        raise ValidationError(f"max_tokens must be an integer >= {min_tokens}, got {max_tokens}")
    
    if max_tokens > max_tokens_limit:
        raise ValidationError(f"max_tokens exceeds maximum limit of {max_tokens_limit}, got {max_tokens}")
    
    if not isinstance(temperature, (int, float)) or not (min_temperature <= temperature <= max_temperature):
        raise ValidationError(f"temperature must be between {min_temperature} and {max_temperature}, got {temperature}")


def validate_non_empty_list(items: List[Any], field_name: str = "list") -> None:
    """
    Validate that a list is not empty.
    
    Args:
        items: List to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError if list is empty
    """
    if not items:
        raise ValidationError(f"{field_name} cannot be empty")


def validate_non_empty_dict(data: Dict[str, Any], field_name: str = "dictionary") -> None:
    """
    Validate that a dictionary is not empty.
    
    Args:
        data: Dictionary to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError if dictionary is empty or not a dict
    """
    if not isinstance(data, dict):
        raise ValidationError(f"{field_name} must be a dictionary")
    if not data:
        raise ValidationError(f"{field_name} cannot be empty")


def validate_positive_number(value: float, field_name: str = "number") -> None:
    """
    Validate that a number is positive.
    
    Args:
        value: Number to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError if number is not positive
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be a number")
    if value <= 0:
        raise ValidationError(f"{field_name} must be positive")


def sanitize_assessment_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize assessment data (optimized).
    
    Args:
        data: Assessment data dictionary
        
    Returns:
        Sanitized assessment data (new dictionary, original is not modified)
    """
    if not isinstance(data, dict):
        return {}
    
    sanitized = data.copy()
    
    # Sanitize string fields (optimized: check existence before processing)
    work_env = sanitized.get("work_environment")
    if work_env:
        sanitized["work_environment"] = sanitize_string(str(work_env), max_length=500)
    
    add_context = sanitized.get("additional_context")
    if add_context:
        sanitized["additional_context"] = sanitize_string(str(add_context), max_length=1000)
    
    # Sanitize list fields (optimized: use get with default)
    physical_symptoms = sanitized.get("physical_symptoms", [])
    if isinstance(physical_symptoms, list):
        sanitized["physical_symptoms"] = sanitize_list(physical_symptoms, max_items=20)
    else:
        sanitized["physical_symptoms"] = []
    
    emotional_symptoms = sanitized.get("emotional_symptoms", [])
    if isinstance(emotional_symptoms, list):
        sanitized["emotional_symptoms"] = sanitize_list(emotional_symptoms, max_items=20)
    else:
        sanitized["emotional_symptoms"] = []
    
    return sanitized

