"""
Validation Module - Input validation and sanitization.

Provides:
- Configuration validation
- Input sanitization
- Type checking
- Range validation
"""

import logging
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_path(path: Union[str, Path], must_exist: bool = False) -> Path:
    """
    Validate file or directory path.
    
    Args:
        path: Path to validate
        must_exist: Whether path must exist
    
    Returns:
        Path object
    
    Raises:
        ValidationError: If path is invalid
    """
    try:
        path_obj = Path(path)
        
        if must_exist and not path_obj.exists():
            raise ValidationError(f"Path does not exist: {path}")
        
        return path_obj
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Invalid path: {path}") from e


def validate_range(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    name: str = "value"
) -> Union[int, float]:
    """
    Validate value is within range.
    
    Args:
        value: Value to validate
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        name: Name of value for error messages
    
    Returns:
        Validated value
    
    Raises:
        ValidationError: If value is out of range
    """
    if min_val is not None and value < min_val:
        raise ValidationError(
            f"{name} ({value}) is less than minimum ({min_val})"
        )
    
    if max_val is not None and value > max_val:
        raise ValidationError(
            f"{name} ({value}) is greater than maximum ({max_val})"
        )
    
    return value


def validate_positive(value: Union[int, float], name: str = "value") -> Union[int, float]:
    """
    Validate value is positive.
    
    Args:
        value: Value to validate
        name: Name of value for error messages
    
    Returns:
        Validated value
    
    Raises:
        ValidationError: If value is not positive
    """
    return validate_range(value, min_val=0.0, name=name)


def validate_in_list(
    value: Any,
    valid_values: List[Any],
    name: str = "value"
) -> Any:
    """
    Validate value is in list of valid values.
    
    Args:
        value: Value to validate
        valid_values: List of valid values
        name: Name of value for error messages
    
    Returns:
        Validated value
    
    Raises:
        ValidationError: If value is not in valid values
    """
    if value not in valid_values:
        raise ValidationError(
            f"{name} ({value}) is not in valid values: {valid_values}"
        )
    
    return value


def validate_type(
    value: Any,
    expected_type: type,
    name: str = "value"
) -> Any:
    """
    Validate value is of expected type.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        name: Name of value for error messages
    
    Returns:
        Validated value
    
    Raises:
        ValidationError: If value is not of expected type
    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"{name} ({type(value).__name__}) is not of type {expected_type.__name__}"
        )
    
    return value


def validate_inference_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate inference configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Validated configuration
    
    Raises:
        ValidationError: If configuration is invalid
    """
    validated = {}
    
    # Validate max_tokens
    if "max_tokens" in config:
        validated["max_tokens"] = validate_range(
            validate_type(config["max_tokens"], int, "max_tokens"),
            min_val=1,
            max_val=32768,
            name="max_tokens"
        )
    
    # Validate temperature
    if "temperature" in config:
        validated["temperature"] = validate_range(
            validate_type(config["temperature"], (int, float), "temperature"),
            min_val=0.0,
            max_val=2.0,
            name="temperature"
        )
    
    # Validate top_p
    if "top_p" in config:
        validated["top_p"] = validate_range(
            validate_type(config["top_p"], (int, float), "top_p"),
            min_val=0.0,
            max_val=1.0,
            name="top_p"
        )
    
    # Validate top_k
    if "top_k" in config:
        validated["top_k"] = validate_positive(
            validate_type(config["top_k"], int, "top_k"),
            name="top_k"
        )
    
    # Validate batch_size
    if "batch_size" in config:
        validated["batch_size"] = validate_range(
            validate_type(config["batch_size"], int, "batch_size"),
            min_val=1,
            max_val=128,
            name="batch_size"
        )
    
    return validated


def validate_benchmark_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate benchmark configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Validated configuration
    
    Raises:
        ValidationError: If configuration is invalid
    """
    validated = {}
    
    # Validate model_path
    if "model_path" in config:
        validated["model_path"] = str(validate_path(
            config["model_path"],
            must_exist=False
        ))
    
    # Validate shots
    if "shots" in config:
        validated["shots"] = validate_range(
            validate_type(config["shots"], int, "shots"),
            min_val=0,
            max_val=10,
            name="shots"
        )
    
    # Validate max_samples
    if "max_samples" in config:
        max_samples = config["max_samples"]
        if max_samples is not None:
            validated["max_samples"] = validate_positive(
                validate_type(max_samples, int, "max_samples"),
                name="max_samples"
            )
    
    return validated


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize text input.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length (truncate if longer)
    
    Returns:
        Sanitized text
    """
    # Remove null bytes
    text = text.replace("\x00", "")
    
    # Truncate if needed
    if max_length is not None and len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"Text truncated to {max_length} characters")
    
    return text


def validate_and_sanitize_prompt(
    prompt: str,
    max_length: Optional[int] = None
) -> str:
    """
    Validate and sanitize prompt.
    
    Args:
        prompt: Prompt text
        max_length: Maximum length
    
    Returns:
        Sanitized prompt
    
    Raises:
        ValidationError: If prompt is invalid
    """
    validate_type(prompt, str, "prompt")
    
    if not prompt.strip():
        raise ValidationError("Prompt cannot be empty")
    
    return sanitize_text(prompt, max_length)


__all__ = [
    "ValidationError",
    "validate_path",
    "validate_range",
    "validate_positive",
    "validate_in_list",
    "validate_type",
    "validate_inference_config",
    "validate_benchmark_config",
    "sanitize_text",
    "validate_and_sanitize_prompt",
]












