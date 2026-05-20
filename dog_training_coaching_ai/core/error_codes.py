"""
Error Codes
===========
Códigos de error estandarizados para la aplicación.
"""

from enum import Enum


class ErrorCode(str, Enum):
    """Códigos de error estandarizados."""
    
    # Validation Errors (1000-1999)
    VALIDATION_ERROR = "ERR_1000"
    INVALID_DOG_BREED = "ERR_1001"
    INVALID_DOG_AGE = "ERR_1002"
    INVALID_DOG_SIZE = "ERR_1003"
    INVALID_TRAINING_GOAL = "ERR_1004"
    INVALID_EXPERIENCE_LEVEL = "ERR_1005"
    MISSING_REQUIRED_FIELD = "ERR_1006"
    INVALID_INPUT_FORMAT = "ERR_1007"
    
    # Service Errors (2000-2999)
    SERVICE_ERROR = "ERR_2000"
    OPENROUTER_ERROR = "ERR_2001"
    CACHE_ERROR = "ERR_2002"
    RATE_LIMIT_EXCEEDED = "ERR_2003"
    
    # External API Errors (3000-3999)
    EXTERNAL_API_ERROR = "ERR_3000"
    OPENROUTER_API_ERROR = "ERR_3001"
    OPENROUTER_TIMEOUT = "ERR_3002"
    OPENROUTER_QUOTA_EXCEEDED = "ERR_3003"
    
    # System Errors (5000-5999)
    INTERNAL_SERVER_ERROR = "ERR_5000"
    CONFIGURATION_ERROR = "ERR_5001"
    DATABASE_ERROR = "ERR_5002"
    UNEXPECTED_ERROR = "ERR_5003"


def get_error_message(error_code: ErrorCode) -> str:
    """Obtener mensaje de error legible."""
    messages = {
        ErrorCode.VALIDATION_ERROR: "Validation error occurred",
        ErrorCode.INVALID_DOG_BREED: "Invalid dog breed provided",
        ErrorCode.INVALID_DOG_AGE: "Invalid dog age provided",
        ErrorCode.INVALID_DOG_SIZE: "Invalid dog size provided",
        ErrorCode.INVALID_TRAINING_GOAL: "Invalid training goal provided",
        ErrorCode.INVALID_EXPERIENCE_LEVEL: "Invalid experience level provided",
        ErrorCode.MISSING_REQUIRED_FIELD: "Required field is missing",
        ErrorCode.INVALID_INPUT_FORMAT: "Invalid input format",
        ErrorCode.SERVICE_ERROR: "Service error occurred",
        ErrorCode.OPENROUTER_ERROR: "OpenRouter API error",
        ErrorCode.CACHE_ERROR: "Cache operation failed",
        ErrorCode.RATE_LIMIT_EXCEEDED: "Rate limit exceeded",
        ErrorCode.EXTERNAL_API_ERROR: "External API error",
        ErrorCode.OPENROUTER_API_ERROR: "OpenRouter API request failed",
        ErrorCode.OPENROUTER_TIMEOUT: "OpenRouter API request timed out",
        ErrorCode.OPENROUTER_QUOTA_EXCEEDED: "OpenRouter API quota exceeded",
        ErrorCode.INTERNAL_SERVER_ERROR: "Internal server error",
        ErrorCode.CONFIGURATION_ERROR: "Configuration error",
        ErrorCode.DATABASE_ERROR: "Database error",
        ErrorCode.UNEXPECTED_ERROR: "Unexpected error occurred"
    }
    return messages.get(error_code, "Unknown error")

