"""
Core module for Dog Training Coaching AI
"""

from .exceptions import (
    DogTrainingException,
    OpenRouterException,
    ValidationException,
    ProcessingException,
    CacheException,
)
from .error_codes import ErrorCode, get_error_message
from .prompts import (
    COACHING_SYSTEM_PROMPT,
    TRAINING_PLAN_SYSTEM_PROMPT,
    BEHAVIOR_ANALYSIS_SYSTEM_PROMPT,
)

__all__ = [
    "DogTrainingException",
    "OpenRouterException",
    "ValidationException",
    "ProcessingException",
    "CacheException",
    "ErrorCode",
    "get_error_message",
    "COACHING_SYSTEM_PROMPT",
    "TRAINING_PLAN_SYSTEM_PROMPT",
    "BEHAVIOR_ANALYSIS_SYSTEM_PROMPT",
]