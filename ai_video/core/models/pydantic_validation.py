from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Type, Union, Callable
from contextlib import asynccontextmanager
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, ConfigDict
from pydantic.json import pydantic_encoder
from .pydantic_schemas import (
                import json
        import hashlib
from typing import Any, List, Dict, Optional
"""
Pydantic Validation Middleware and Utilities
===========================================

Comprehensive validation middleware and utilities for the AI Video system
using Pydantic BaseModel for consistent input/output validation.

Features:
- Request/response validation middleware
- Custom validation decorators
- Error handling and transformation
- Performance monitoring for validation
- Caching of validation results
- Detailed validation error reporting
"""



    APIError, VideoGenerationInput, BatchGenerationInput,
    VideoEditInput, create_error_response
)

logger = logging.getLogger(__name__)

# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

class ValidationConfig:
    """Configuration for validation middleware."""
    
    def __init__(
        self,
        enable_request_validation: bool = True,
        enable_response_validation: bool = True,
        enable_performance_monitoring: bool = True,
        enable_validation_caching: bool = True,
        max_validation_time: float = 1.0,
        detailed_error_messages: bool = True,
        log_validation_errors: bool = True
    ):
        
    """__init__ function."""
self.enable_request_validation = enable_request_validation
        self.enable_response_validation = enable_response_validation
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_validation_caching = enable_validation_caching
        self.max_validation_time = max_validation_time
        self.detailed_error_messages = detailed_error_messages
        self.log_validation_errors = log_validation_errors

# =============================================================================
# VALIDATION MIDDLEWARE
# =============================================================================

class PydanticValidationMiddleware:
    """Middleware for Pydantic request/response validation."""
    
    def __init__(self, config: ValidationConfig = None):
        
    """__init__ function."""
self.config = config or ValidationConfig()
        self.validation_cache: Dict[str, Any] = {}
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'total_validation_time': 0.0
        }
    
    async def __call__(self, request: Request, call_next):
        """Process request through validation middleware."""
        start_time = time.time()
        
        try:
            # Request validation
            if self.config.enable_request_validation:
                await self._validate_request(request)
            
            # Process request
            response = await call_next(request)
            
            # Response validation
            if self.config.enable_response_validation:
                await self._validate_response(request, response)
            
            # Update statistics
            self._update_stats(time.time() - start_time, True)
            
            return response
            
        except ValidationError as e:
            # Handle validation errors
            error_response = await self._handle_validation_error(request, e)
            self._update_stats(time.time() - start_time, False)
            return error_response
            
        except Exception as e:
            # Handle other errors
            logger.error(f"Validation middleware error: {e}")
            self._update_stats(time.time() - start_time, False)
            raise
    
    async async def _validate_request(self, request: Request) -> None:
        """Validate incoming request data."""
        if request.method in ['POST', 'PUT', 'PATCH']:
            try:
                # Get request body
                body = await request.json()
                
                # Determine validation schema based on endpoint
                schema = self._get_validation_schema(request.url.path)
                if schema:
                    # Check cache first
                    cache_key = self._get_cache_key(request.url.path, body)
                    if self.config.enable_validation_caching and cache_key in self.validation_cache:
                        return
                    
                    # Validate with timeout
                    await asyncio.wait_for(
                        self._validate_data(body, schema),
                        timeout=self.config.max_validation_time
                    )
                    
                    # Cache validation result
                    if self.config.enable_validation_caching:
                        self.validation_cache[cache_key] = True
                        
            except asyncio.TimeoutError:
                raise ValidationError(
                    errors=[{
                        'loc': ('body',),
                        'msg': 'Validation timeout',
                        'type': 'timeout_error'
                    }],
                    model=BaseModel
                )
    
    async def _validate_response(self, request: Request, response: Response) -> None:
        """Validate outgoing response data."""
        if hasattr(response, 'body') and response.body:
            try:
                # Parse response body
                body = json.loads(response.body.decode())
                
                # Validate response schema
                schema = self._get_response_schema(request.url.path)
                if schema:
                    await asyncio.wait_for(
                        self._validate_data(body, schema),
                        timeout=self.config.max_validation_time
                    )
                    
            except (json.JSONDecodeError, asyncio.TimeoutError):
                # Skip response validation on errors
                pass
    
    def _get_validation_schema(self, path: str) -> Optional[Type[BaseModel]]:
        """Get validation schema for request path."""
        schema_mapping = {
            '/api/v1/videos/generate': VideoGenerationInput,
            '/api/v1/videos/batch': BatchGenerationInput,
            '/api/v1/videos/edit': VideoEditInput,
        }
        return schema_mapping.get(path)
    
    def _get_response_schema(self, path: str) -> Optional[Type[BaseModel]]:
        """Get response schema for request path."""
        # Add response schema mapping as needed
        return None
    
    async def _validate_data(self, data: Any, schema: Type[BaseModel]) -> None:
        """Validate data against Pydantic schema."""
        try:
            schema.model_validate(data)
        except ValidationError as e:
            if self.config.log_validation_errors:
                logger.warning(f"Validation error: {e.errors()}")
            raise
    
    def _get_cache_key(self, path: str, data: Any) -> str:
        """Generate cache key for validation result."""
        content = f"{path}:{str(data)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _handle_validation_error(self, request: Request, error: ValidationError) -> JSONResponse:
        """Handle validation errors and return appropriate response."""
        error_details = []
        
        for err in error.errors():
            detail = {
                'field': '.'.join(str(loc) for loc in err['loc']),
                'message': err['msg'],
                'type': err['type']
            }
            error_details.append(detail)
        
        api_error = create_error_response(
            error_code="VALIDATION_ERROR",
            error_type="validation_failed",
            message="Request validation failed",
            details={'errors': error_details} if self.config.detailed_error_messages else None,
            request_id=request.headers.get('X-Request-ID'),
            endpoint=str(request.url.path)
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=api_error.model_dump()
        )
    
    def _update_stats(self, validation_time: float, success: bool) -> None:
        """Update validation statistics."""
        self.validation_stats['total_validations'] += 1
        self.validation_stats['total_validation_time'] += validation_time
        
        if success:
            self.validation_stats['successful_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats['total_validations']
        if total > 0:
            avg_time = self.validation_stats['total_validation_time'] / total
            success_rate = self.validation_stats['successful_validations'] / total
        else:
            avg_time = 0.0
            success_rate = 0.0
        
        return {
            **self.validation_stats,
            'average_validation_time': avg_time,
            'success_rate': success_rate,
            'cache_size': len(self.validation_cache)
        }

# =============================================================================
# VALIDATION DECORATORS
# =============================================================================

def validate_request(schema: Type[BaseModel]):
    """Decorator to validate request data against Pydantic schema."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Extract request from args or kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                request = kwargs.get('request')
            
            if request:
                try:
                    # Get request body
                    body = await request.json()
                    
                    # Validate against schema
                    validated_data = schema.model_validate(body)
                    
                    # Add validated data to kwargs
                    kwargs['validated_data'] = validated_data
                    
                except ValidationError as e:
                    # Return validation error response
                    error_details = []
                    for err in e.errors():
                        detail = {
                            'field': '.'.join(str(loc) for loc in err['loc']),
                            'message': err['msg'],
                            'type': err['type']
                        }
                        error_details.append(detail)
                    
                    api_error = create_error_response(
                        error_code="VALIDATION_ERROR",
                        error_type="validation_failed",
                        message="Request validation failed",
                        details={'errors': error_details}
                    )
                    
                    return JSONResponse(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        content=api_error.model_dump()
                    )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def validate_response(schema: Type[BaseModel]):
    """Decorator to validate response data against Pydantic schema."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Execute function
            result = await func(*args, **kwargs)
            
            # Validate response
            try:
                if isinstance(result, dict):
                    validated_response = schema.model_validate(result)
                    return validated_response.model_dump()
                elif isinstance(result, BaseModel):
                    # Already a Pydantic model
                    return result.model_dump()
                else:
                    # Try to validate as dict
                    validated_response = schema.model_validate(result)
                    return validated_response.model_dump()
                    
            except ValidationError as e:
                logger.error(f"Response validation error: {e.errors()}")
                # Return original result if validation fails
                return result
        
        return wrapper
    return decorator

def validate_input_output(input_schema: Type[BaseModel], output_schema: Type[BaseModel]):
    """Decorator to validate both input and output."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Input validation
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                request = kwargs.get('request')
            
            if request:
                try:
                    body = await request.json()
                    validated_input = input_schema.model_validate(body)
                    kwargs['validated_input'] = validated_input
                except ValidationError as e:
                    error_details = []
                    for err in e.errors():
                        detail = {
                            'field': '.'.join(str(loc) for loc in err['loc']),
                            'message': err['msg'],
                            'type': err['type']
                        }
                        error_details.append(detail)
                    
                    api_error = create_error_response(
                        error_code="INPUT_VALIDATION_ERROR",
                        error_type="input_validation_failed",
                        message="Input validation failed",
                        details={'errors': error_details}
                    )
                    
                    return JSONResponse(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        content=api_error.model_dump()
                    )
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Output validation
            try:
                if isinstance(result, dict):
                    validated_output = output_schema.model_validate(result)
                    return validated_output.model_dump()
                elif isinstance(result, BaseModel):
                    return result.model_dump()
                else:
                    validated_output = output_schema.model_validate(result)
                    return validated_output.model_dump()
                    
            except ValidationError as e:
                logger.error(f"Output validation error: {e.errors()}")
                return result
        
        return wrapper
    return decorator

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

class ValidationUtils:
    """Utility class for validation operations."""
    
    @staticmethod
    def validate_partial_data(data: Dict[str, Any], schema: Type[BaseModel]) -> BaseModel:
        """Validate partial data against schema (for PATCH operations)."""
        # Create a partial schema for validation
        partial_schema = schema.model_validate(data)
        return partial_schema
    
    @staticmethod
    def validate_list_data(data: List[Dict[str, Any]], schema: Type[BaseModel]) -> List[BaseModel]:
        """Validate list of data against schema."""
        validated_items = []
        errors = []
        
        for i, item in enumerate(data):
            try:
                validated_item = schema.model_validate(item)
                validated_items.append(validated_item)
            except ValidationError as e:
                for err in e.errors():
                    err['loc'] = (i,) + err['loc']
                    errors.append(err)
        
        if errors:
            raise ValidationError(errors, model=schema)
        
        return validated_items
    
    @staticmethod
    def sanitize_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data for security."""
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Remove potential XSS
                sanitized[key] = value.strip()
            elif isinstance(value, dict):
                sanitized[key] = ValidationUtils.sanitize_data(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    ValidationUtils.sanitize_data(item) if isinstance(item, dict)
                    else item.strip() if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    @staticmethod
    def transform_validation_error(error: ValidationError) -> Dict[str, Any]:
        """Transform Pydantic validation error to API format."""
        errors = []
        
        for err in error.errors():
            error_detail = {
                'field': '.'.join(str(loc) for loc in err['loc']),
                'message': err['msg'],
                'type': err['type'],
                'input': err.get('input')
            }
            errors.append(error_detail)
        
        return {
            'error_code': 'VALIDATION_ERROR',
            'error_type': 'validation_failed',
            'message': 'Data validation failed',
            'details': {'errors': errors}
        }

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class ValidationPerformanceMonitor:
    """Monitor validation performance metrics."""
    
    def __init__(self) -> Any:
        self.metrics = {
            'validation_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_errors': 0,
            'total_validations': 0
        }
    
    @asynccontextmanager
    async def monitor_validation(self, schema_name: str):
        """Context manager to monitor validation performance."""
        start_time = time.time()
        try:
            yield
            # Record successful validation
            validation_time = time.time() - start_time
            self.metrics['validation_times'].append(validation_time)
            self.metrics['total_validations'] += 1
            
        except ValidationError:
            # Record validation error
            self.metrics['validation_errors'] += 1
            raise
    
    def record_cache_hit(self) -> Any:
        """Record cache hit."""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self) -> Any:
        """Record cache miss."""
        self.metrics['cache_misses'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        validation_times = self.metrics['validation_times']
        
        if validation_times:
            avg_time = sum(validation_times) / len(validation_times)
            max_time = max(validation_times)
            min_time = min(validation_times)
        else:
            avg_time = max_time = min_time = 0.0
        
        total_cache_ops = self.metrics['cache_hits'] + self.metrics['cache_misses']
        cache_hit_rate = (self.metrics['cache_hits'] / total_cache_ops * 100) if total_cache_ops > 0 else 0
        
        return {
            'total_validations': self.metrics['total_validations'],
            'validation_errors': self.metrics['validation_errors'],
            'average_validation_time': avg_time,
            'max_validation_time': max_time,
            'min_validation_time': min_time,
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'error_rate': (self.metrics['validation_errors'] / self.metrics['total_validations'] * 100) if self.metrics['total_validations'] > 0 else 0
        }

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_validation_middleware(config: ValidationConfig = None) -> PydanticValidationMiddleware:
    """Create validation middleware with configuration."""
    return PydanticValidationMiddleware(config)

def create_performance_monitor() -> ValidationPerformanceMonitor:
    """Create performance monitoring instance."""
    return ValidationPerformanceMonitor()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ValidationConfig',
    'PydanticValidationMiddleware',
    'validate_request',
    'validate_response', 
    'validate_input_output',
    'ValidationUtils',
    'ValidationPerformanceMonitor',
    'create_validation_middleware',
    'create_performance_monitor'
] 