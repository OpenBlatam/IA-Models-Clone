from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import logging
import traceback
import uuid
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union
from functools import wraps
from datetime import datetime, timezone
from enum import Enum
import inspect
from pydantic import BaseModel, Field, ValidationError, field_validator
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
        import re
from typing import Any, List, Dict, Optional
import asyncio
omprehensive Error Handling and Validation System

This module provides:
- Custom exception classes with proper HTTP status codes
- Validation decorators with detailed error messages
- Error response models with structured data
- Input validation utilities with Pydantic v2
- Error logging and monitoring capabilities
"



logger = logging.getLogger(__name__)

T = TypeVar('T')

# ============================================================================
# ERROR CODES AND TYPES
# ============================================================================

class ErrorCode(str, Enum):
 Standardized error codes for the application.""
    # Validation Errors (40  VALIDATION_ERROR =VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT  MISSING_REQUIRED_FIELD =MISSING_REQUIRED_FIELD    INVALID_FORMAT =INVALID_FORMAT    OUT_OF_RANGE = "OUT_OF_RANGE"
    
    # Authentication Errors (401/403)
    UNAUTHORIZED = UNAUTHORIZED"
    FORBIDDEN = FORBIDDENINVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    
    # Resource Errors (404409
    NOT_FOUND = "NOT_FOUND    RESOURCE_CONFLICT = RESOURCE_CONFLICT"
    DUPLICATE_ENTRY = DUPLICATE_ENTRY"
    
    # Rate Limiting (429
    RATE_LIMIT_EXCEEDED = RATE_LIMIT_EXCEEDED"
    TOO_MANY_REQUESTS = TOO_MANY_REQUESTS"
    
    # Server Errors (500)
    INTERNAL_ERROR =INTERNAL_ERROR"
    DATABASE_ERROR =DATABASE_ERROR"
    EXTERNAL_SERVICE_ERROR =EXTERNAL_SERVICE_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    
    # AI/Processing Errors (422
    AI_PROCESSING_ERROR = "AI_PROCESSING_ERROR"
    CONTENT_GENERATION_FAILED = "CONTENT_GENERATION_FAILED   INVALID_CONTENT_TYPE = "INVALID_CONTENT_TYPE"

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class InstagramCaptionsException(Exception):
   Base exception for Instagram Captions API."   
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500   ):
        
    """__init__ function."""
self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        self.timestamp = datetime.now(timezone.utc)
        self.request_id = str(uuid.uuid4        super().__init__(self.message)

class ValidationException(InstagramCaptionsException):
Raised when input validation fails."   
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=message,
            details=details,
            status_code=400ass AuthenticationException(InstagramCaptionsException):
hen authentication fails."   
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            error_code=ErrorCode.UNAUTHORIZED,
            message=message,
            details=details,
            status_code=401 )

class ResourceNotFoundException(InstagramCaptionsException):
  d when a requested resource is not found."   
    def __init__(self, resource_type: str, resource_id: str):
        
    """__init__ function."""
super().__init__(
            error_code=ErrorCode.NOT_FOUND,
            message=f"{resource_type} with id '{resource_id}' not found",
            details={"resource_type": resource_type, "resource_id": resource_id},
            status_code=404     )

class RateLimitException(InstagramCaptionsException):
d when rate limits are exceeded."   
    def __init__(self, retry_after: int = 60        super().__init__(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message="Rate limit exceeded. Please try again later.",
            details={retry_after": retry_after},
            status_code=429       )

class AIProcessingException(InstagramCaptionsException):
   when AI processing fails."   
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            error_code=ErrorCode.AI_PROCESSING_ERROR,
            message=message,
            details=details,
            status_code=422=====================
# ERROR RESPONSE MODELS
# ============================================================================

class ErrorResponse(BaseModel):
 Standardized error response model.   
    error_code: ErrorCode = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details)   request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Error timestamp")
    path: Optionalstr] = Field(None, description="Request path")
    method: Optionalstr] = Field(None, description="HTTP method")

class ValidationErrorResponse(ErrorResponse):ion error response with field-specific details. 
    field_errors: List[Dict[str, Any]] = Field(default_factory=list, description="Field-specific validation errors")

# ============================================================================
# ERROR HANDLING UTILITIES
# ============================================================================

def create_error_response(
    *,
    error_code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized error response (RORO)."""
    return {
   error_code": error_code,
        message": message,
        details: details or {},
   request_id": request_id or str(uuid.uuid4()),
  timestamp": datetime.now(timezone.utc).isoformat(),
    path: path,
       method": method
    }

def log_error(
    *,
    error: Exception,
    request_id: Optional[str] = None,
    context: Optional[Dictstr, Any]] = None,
    module: Optional[str] = None,
    function: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Log error with structured context (module, function, parameters, etc)."""
    # Use inspect to get caller info if not provided
    if module is None or function is None:
        frame = inspect.currentframe()
        outer_frames = inspect.getouterframes(frame)
        # 0: log_error, 1: caller
        if len(outer_frames) > 1:
            caller_frame = outer_frames[1]
            if module is None:
                module = caller_frame.frame.f_globals.get("__name__", "<unknown>")
            if function is None:
                function = caller_frame.function
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "request_id": request_id or str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": module,
        "function": function,
        "parameters": parameters or {},
        "context": context or {},
        "traceback": traceback.format_exc()
    }
    logger.error(f"Error occurred: {error_data}")
    return error_data

# ============================================================================
# VALIDATION DECORATORS
# ============================================================================

def validate_input(*, model_class: type):
 
    """validate_input function."""
orator to validate input using Pydantic model."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                # Extract the first argument as input data
                if args and isinstance(args[0], dict):
                    input_data = args[0]
                    validated_data = model_class(**input_data)
                    # Replace the first argument with validated data
                    new_args = (validated_data, *args[1:])
                    return await func(*new_args, **kwargs)
                else:
                    return await func(*args, **kwargs)
            except ValidationError as e:
                field_errors =                for error in e.errors():
                    field_errors.append({
                      field:error[loc"][0 error["loc"] else "unknown",
                        message": error["msg"],
                     type": error["type"]
                    })
                
                raise ValidationException(
                    message="Input validation failed",
                    details={"field_errors": field_errors}
                )
        return wrapper
    return decorator

async def handle_api_errors(func: Callable[..., T]) -> Callable[..., T]:
   for comprehensive API error handling (RORO)."
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        request_id = str(uuid.uuid4())
        
        try:
            return await func(*args, **kwargs)
        except InstagramCaptionsException as e:
            # Log custom exceptions
            log_error(error=e, request_id=request_id)
            raise HTTPException(
                status_code=e.status_code,
                detail=create_error_response(
                    error_code=e.error_code,
                    message=e.message,
                    details=e.details,
                    request_id=request_id
                )
            )
        except ValidationError as e:
            # Handle Pydantic validation errors
            field_errors =          for error in e.errors():
                field_errors.append({
                   field":..join(str(loc) for loc in error["loc"]),
                    message": error["msg"],
                 type": error["type]               })
            
            log_error(
                error=e,
                request_id=request_id,
                context={"field_errors": field_errors}
            )
            
            raise HTTPException(
                status_code=400            detail=create_error_response(
                    error_code=ErrorCode.VALIDATION_ERROR,
                    message="Input validation failed",
                    details={"field_errors: field_errors},                   request_id=request_id
                )
            )
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            # Handle unexpected errors
            log_error(error=e, request_id=request_id)
            raise HTTPException(
                status_code=500            detail=create_error_response(
                    error_code=ErrorCode.INTERNAL_ERROR,
                    message="An unexpected error occurred",
                    request_id=request_id
                )
            )
    return wrapper

# ============================================================================
# INPUT VALIDATION UTILITIES
# ============================================================================

class StringValidationConfig(BaseModel):
  Configuration for string validation.value: str = Field(..., description="String to validate)   field_name: str = Field(..., description="Field name for error messages)
    min_length: Optionalint] = Field(None, description=Minimum length)
    max_length: Optionalint] = Field(None, description=Maximum length")
    pattern: Optionalstr] = Field(None, description="Regex pattern)
    allow_empty: bool = Field(default=False, description="Allow empty strings")

def validate_string(*, config: StringValidationConfig) -> Dict[str, Any]:
    lidate string with comprehensive rules (RORO)."""
    value = config.value
    
    # Check if empty
    if not config.allow_empty and (not value or not value.strip()):
        return[object Object]        is_valid": False,
            error: f[object Object]config.field_name} cannot be empty",
       error_code": ErrorCode.MISSING_REQUIRED_FIELD
        }
    
    # Check minimum length
    if config.min_length and len(value) < config.min_length:
        return[object Object]        is_valid": False,
            error: f[object Object]config.field_name} must be at least [object Object]config.min_length} characters",
       error_code": ErrorCode.OUT_OF_RANGE
        }
    
    # Check maximum length
    if config.max_length and len(value) > config.max_length:
        return[object Object]        is_valid": False,
            error: f[object Object]config.field_name} must be at most [object Object]config.max_length} characters",
       error_code": ErrorCode.OUT_OF_RANGE
        }
    
    # Check pattern
    if config.pattern:
        if not re.match(config.pattern, value):
            return[object Object]
               is_valid": False,
                error: f[object Object]config.field_name} format is invalid,
           error_code": ErrorCode.INVALID_FORMAT
            }
    
    return {"is_valid": True, "value": value.strip()}

class NumericValidationConfig(BaseModel):
  Configuration for numeric validation.""  value: Union[int, float] = Field(..., description="Numeric value to validate)   field_name: str = Field(..., description="Field name for error messages")
    min_value: Optional[Union[int, float]] = Field(None, description="Minimum value")
    max_value: Optional[Union[int, float]] = Field(None, description="Maximum value)  allow_zero: bool = Field(default=True, description=Allow zero values")
    allow_negative: bool = Field(default=True, description="Allow negative values)

def validate_numeric(*, config: NumericValidationConfig) -> Dict[str, Any]:
    "date numeric value with comprehensive rules (RORO)."""
    value = config.value
    
    # Check zero
    if not config.allow_zero and value ==0
        return[object Object]        is_valid": False,
            error: f[object Object]config.field_name} cannot be zero",
       error_code": ErrorCode.INVALID_INPUT
        }
    
    # Check negative
    if not config.allow_negative and value <0
        return[object Object]        is_valid": False,
            error: f[object Object]config.field_name} cannot be negative",
       error_code": ErrorCode.INVALID_INPUT
        }
    
    # Check minimum value
    if config.min_value is not None and value < config.min_value:
        return[object Object]        is_valid": False,
            error: f[object Object]config.field_name} must be at least {config.min_value}",
       error_code": ErrorCode.OUT_OF_RANGE
        }
    
    # Check maximum value
    if config.max_value is not None and value > config.max_value:
        return[object Object]        is_valid": False,
            error: f[object Object]config.field_name} must be at most {config.max_value}",
       error_code": ErrorCode.OUT_OF_RANGE
        }
    
    return {"is_valid": True, value": value}

# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    " exception handler for FastAPI.  request_id = str(uuid.uuid4())
    
    if isinstance(exc, InstagramCaptionsException):
        error_response = create_error_response(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            request_id=request_id,
            path=str(request.url.path),
            method=request.method
        )
        status_code = exc.status_code
    elif isinstance(exc, ValidationError):
        field_errors = []
        for error in exc.errors():
            field_errors.append({
               field":..join(str(loc) for loc in error["loc"]),
                message": error["msg"],
             type": error["type]      })
        
        error_response = create_error_response(
            error_code=ErrorCode.VALIDATION_ERROR,
            message="Input validation failed",
            details={"field_errors: field_errors},
            request_id=request_id,
            path=str(request.url.path),
            method=request.method
        )
        status_code = 400
    else:
        # Log unexpected errors
        log_error(error=exc, request_id=request_id)
        
        error_response = create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="An unexpected error occurred",
            request_id=request_id,
            path=str(request.url.path),
            method=request.method
        )
        status_code = 50  
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Error Codes
    ErrorCode",
    
    # Exceptions
    InstagramCaptionsException,
    alidationException",
 AuthenticationException",
    "ResourceNotFoundException",
    "RateLimitException",AIProcessingException",
    
    # Models
   ErrorResponse,ValidationErrorResponse",
    
    # Utilitiescreate_error_response",
    log_error",
    
    # Decorators
    validate_input,
    handle_api_errors",
    
    # Validation
   StringValidationConfig",
    NumericValidationConfig",
  validate_string,validate_numeric,    
    # FastAPI Integration
   global_exception_handler"
] 

class TimeoutError(InstagramCaptionsException):
    """Raised when an operation times out."""
    def __init__(self, message: str = "Operation timed out", details: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            error_code=ErrorCode.TIMEOUT_ERROR,
            message=message,
            details=details,
            status_code=504
        )

class InvalidTargetError(InstagramCaptionsException):
    """Raised when a target address or resource is invalid or malformed."""
    def __init__(self, message: str = "Invalid target address", details: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(
            error_code=ErrorCode.INVALID_INPUT,
            message=message,
            details=details,
            status_code=400
        ) 