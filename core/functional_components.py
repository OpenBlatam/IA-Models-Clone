from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from __future__ import annotations
from typing import (
from datetime import datetime, timedelta
import uuid
import time
import asyncio
from functools import lru_cache, wraps, partial
from dataclasses import dataclass, field
from enum import Enum
import logging
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from fastapi import HTTPException, status
import structlog
from ..utils.optimized_base_model import OptimizedBaseModel
from ..utils.error_system import error_factory, ErrorContext, ValidationError
    import hashlib
from typing import Any, List, Dict, Optional
"""
Functional Components with Pydantic Models
=========================================

A comprehensive system for creating functional components with Pydantic models
for input validation and response schemas in FastAPI applications.

Features:
- Pure functional components (no classes)
- Pydantic v2 models for validation
- Type-safe input/output schemas
- Performance monitoring
- Caching support
- Error handling
- Async support
- Dependency injection ready
"""

    Any, Dict, List, Optional, Type, TypeVar, Union, Callable, 
    Generic, Awaitable, Protocol, runtime_checkable
)



logger = structlog.get_logger(__name__)

# Type variables for generic components
InputT = TypeVar('InputT', bound=BaseModel)
OutputT = TypeVar('OutputT', bound=BaseModel)
ContextT = TypeVar('ContextT')

# Performance metrics
@dataclass
class ComponentMetrics:
    """Performance metrics for functional components."""
    name: str
    execution_count: int = 0
    total_execution_time: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time."""
        return self.total_execution_time / self.execution_count if self.execution_count > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return self.error_count / self.execution_count if self.execution_count > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_cache_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0

# Global metrics registry
_component_metrics: Dict[str, ComponentMetrics] = {}

def get_component_metrics(component_name: str) -> ComponentMetrics:
    """Get or create metrics for a component."""
    if component_name not in _component_metrics:
        _component_metrics[component_name] = ComponentMetrics(name=component_name)
    return _component_metrics[component_name]

def get_all_metrics() -> Dict[str, ComponentMetrics]:
    """Get all component metrics."""
    return _component_metrics.copy()

def reset_metrics() -> None:
    """Reset all component metrics."""
    global _component_metrics
    _component_metrics.clear()

# Base schemas
class BaseInputModel(OptimizedBaseModel):
    """Base input model for functional components."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    # Common fields for all inputs
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BaseOutputModel(OptimizedBaseModel):
    """Base output model for functional components."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )
    
    # Common fields for all outputs
    success: bool = Field(..., description="Whether the operation was successful")
    data: Optional[Any] = Field(None, description="Result data")
    error: Optional[str] = Field(None, description="Error message if any")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")

class ErrorOutputModel(BaseOutputModel):
    """Standardized error output model."""
    
    success: bool = Field(default=False)
    error_code: str = Field(..., description="Error code for programmatic handling")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")
    suggestions: Optional[List[str]] = Field(None, description="Suggested solutions")

# Component protocols
@runtime_checkable
class ComponentProtocol(Protocol[InputT, OutputT]):
    """Protocol for functional components."""
    
    def __call__(self, input_data: InputT, **kwargs) -> OutputT:
        """Execute the component."""
        ...

@runtime_checkable
class AsyncComponentProtocol(Protocol[InputT, OutputT]):
    """Protocol for async functional components."""
    
    async def __call__(self, input_data: InputT, **kwargs) -> Awaitable[OutputT]:
        """Execute the component asynchronously."""
        ...

# Component decorators
def component(
    name: Optional[str] = None,
    cache_result: bool = False,
    cache_ttl: int = 300,
    validate_input: bool = True,
    validate_output: bool = True,
    log_execution: bool = True
):
    """
    Decorator for functional components with performance monitoring and caching.
    
    Args:
        name: Component name for metrics and logging
        cache_result: Whether to cache results
        cache_ttl: Cache TTL in seconds
        validate_input: Whether to validate input
        validate_output: Whether to validate output
        log_execution: Whether to log execution details
    """
    def decorator(func: Callable) -> Callable:
        component_name = name or func.__name__
        metrics = get_component_metrics(component_name)
        
        # Cache for results
        result_cache = {}
        
        @wraps(func)
        def wrapper(input_data: BaseInputModel, **kwargs) -> BaseOutputModel:
            start_time = time.perf_counter()
            
            try:
                # Input validation
                if validate_input and isinstance(input_data, BaseModel):
                    input_data.model_validate(input_data.model_dump())
                
                # Check cache
                if cache_result:
                    cache_key = _generate_cache_key(input_data, kwargs)
                    if cache_key in result_cache:
                        cached_result, cached_time = result_cache[cache_key]
                        if time.time() - cached_time < cache_ttl:
                            metrics.cache_hits += 1
                            return cached_result
                    metrics.cache_misses += 1
                
                # Execute component
                result = func(input_data, **kwargs)
                
                # Output validation
                if validate_output and isinstance(result, BaseModel):
                    result.model_validate(result.model_dump())
                
                # Cache result
                if cache_result:
                    cache_key = _generate_cache_key(input_data, kwargs)
                    result_cache[cache_key] = (result, time.time())
                
                # Update metrics
                execution_time = time.perf_counter() - start_time
                metrics.execution_count += 1
                metrics.total_execution_time += execution_time
                
                # Log execution
                if log_execution:
                    logger.info(
                        "Component executed successfully",
                        component=component_name,
                        execution_time_ms=execution_time * 1000,
                        input_type=type(input_data).__name__,
                        output_type=type(result).__name__
                    )
                
                # Add execution time to result if it's a BaseOutputModel
                if isinstance(result, BaseOutputModel):
                    result.execution_time_ms = execution_time * 1000
                    result.request_id = getattr(input_data, 'request_id', None)
                
                return result
                
            except Exception as e:
                # Update error metrics
                metrics.error_count += 1
                metrics.execution_count += 1
                
                execution_time = time.perf_counter() - start_time
                metrics.total_execution_time += execution_time
                
                # Log error
                logger.error(
                    "Component execution failed",
                    component=component_name,
                    error=str(e),
                    execution_time_ms=execution_time * 1000,
                    input_type=type(input_data).__name__
                )
                
                # Return error output
                return ErrorOutputModel(
                    success=False,
                    error_code="COMPONENT_ERROR",
                    error=str(e),
                    error_details={"component": component_name},
                    execution_time_ms=execution_time * 1000,
                    request_id=getattr(input_data, 'request_id', None)
                )
        
        return wrapper
    
    return decorator

def async_component(
    name: Optional[str] = None,
    cache_result: bool = False,
    cache_ttl: int = 300,
    validate_input: bool = True,
    validate_output: bool = True,
    log_execution: bool = True
):
    """
    Decorator for async functional components.
    """
    def decorator(func: Callable) -> Callable:
        component_name = name or func.__name__
        metrics = get_component_metrics(component_name)
        
        # Cache for results
        result_cache = {}
        
        @wraps(func)
        async def wrapper(input_data: BaseInputModel, **kwargs) -> BaseOutputModel:
            start_time = time.perf_counter()
            
            try:
                # Input validation
                if validate_input and isinstance(input_data, BaseModel):
                    input_data.model_validate(input_data.model_dump())
                
                # Check cache
                if cache_result:
                    cache_key = _generate_cache_key(input_data, kwargs)
                    if cache_key in result_cache:
                        cached_result, cached_time = result_cache[cache_key]
                        if time.time() - cached_time < cache_ttl:
                            metrics.cache_hits += 1
                            return cached_result
                    metrics.cache_misses += 1
                
                # Execute component
                result = await func(input_data, **kwargs)
                
                # Output validation
                if validate_output and isinstance(result, BaseModel):
                    result.model_validate(result.model_dump())
                
                # Cache result
                if cache_result:
                    cache_key = _generate_cache_key(input_data, kwargs)
                    result_cache[cache_key] = (result, time.time())
                
                # Update metrics
                execution_time = time.perf_counter() - start_time
                metrics.execution_count += 1
                metrics.total_execution_time += execution_time
                
                # Log execution
                if log_execution:
                    logger.info(
                        "Async component executed successfully",
                        component=component_name,
                        execution_time_ms=execution_time * 1000,
                        input_type=type(input_data).__name__,
                        output_type=type(result).__name__
                    )
                
                # Add execution time to result if it's a BaseOutputModel
                if isinstance(result, BaseOutputModel):
                    result.execution_time_ms = execution_time * 1000
                    result.request_id = getattr(input_data, 'request_id', None)
                
                return result
                
            except Exception as e:
                # Update error metrics
                metrics.error_count += 1
                metrics.execution_count += 1
                
                execution_time = time.perf_counter() - start_time
                metrics.total_execution_time += execution_time
                
                # Log error
                logger.error(
                    "Async component execution failed",
                    component=component_name,
                    error=str(e),
                    execution_time_ms=execution_time * 1000,
                    input_type=type(input_data).__name__
                )
                
                # Return error output
                return ErrorOutputModel(
                    success=False,
                    error_code="ASYNC_COMPONENT_ERROR",
                    error=str(e),
                    error_details={"component": component_name},
                    execution_time_ms=execution_time * 1000,
                    request_id=getattr(input_data, 'request_id', None)
                )
        
        return wrapper
    
    return decorator

def _generate_cache_key(input_data: BaseInputModel, kwargs: Dict[str, Any]) -> str:
    """Generate cache key from input data and kwargs."""
    
    # Create a hashable representation
    data_dict = input_data.model_dump() if isinstance(input_data, BaseModel) else input_data
    key_data = {
        "input": data_dict,
        "kwargs": kwargs
    }
    
    # Generate hash
    key_str = str(key_data)
    return hashlib.md5(key_str.encode()).hexdigest()

# Component composition
def compose_components(*components: Callable) -> Callable:
    """
    Compose multiple components into a pipeline.
    
    Args:
        *components: Components to compose
        
    Returns:
        Composed component function
    """
    def composed_component(input_data: BaseInputModel, **kwargs) -> BaseOutputModel:
        current_result = input_data
        
        for component_func in components:
            try:
                current_result = component_func(current_result, **kwargs)
                
                # Check if component returned an error
                if isinstance(current_result, ErrorOutputModel) and not current_result.success:
                    return current_result
                    
            except Exception as e:
                return ErrorOutputModel(
                    success=False,
                    error_code="COMPOSITION_ERROR",
                    error=str(e),
                    error_details={"component": component_func.__name__}
                )
        
        return current_result
    
    return composed_component

def compose_async_components(*components: Callable) -> Callable:
    """
    Compose multiple async components into a pipeline.
    """
    async def composed_component(input_data: BaseInputModel, **kwargs) -> BaseOutputModel:
        current_result = input_data
        
        for component_func in components:
            try:
                current_result = await component_func(current_result, **kwargs)
                
                # Check if component returned an error
                if isinstance(current_result, ErrorOutputModel) and not current_result.success:
                    return current_result
                    
            except Exception as e:
                return ErrorOutputModel(
                    success=False,
                    error_code="ASYNC_COMPOSITION_ERROR",
                    error=str(e),
                    error_details={"component": component_func.__name__}
                )
        
        return current_result
    
    return composed_component

# Parallel execution
async def execute_parallel(
    components: List[Callable],
    input_data: BaseInputModel,
    **kwargs
) -> List[BaseOutputModel]:
    """
    Execute multiple components in parallel.
    
    Args:
        components: List of component functions
        input_data: Input data for all components
        **kwargs: Additional arguments
        
    Returns:
        List of results from all components
    """
    async def execute_component(component_func: Callable) -> BaseOutputModel:
        try:
            if asyncio.iscoroutinefunction(component_func):
                return await component_func(input_data, **kwargs)
            else:
                return component_func(input_data, **kwargs)
        except Exception as e:
            return ErrorOutputModel(
                success=False,
                error_code="PARALLEL_EXECUTION_ERROR",
                error=str(e),
                error_details={"component": component_func.__name__}
            )
    
    # Execute all components in parallel
    tasks = [execute_component(comp) for comp in components]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append(ErrorOutputModel(
                success=False,
                error_code="PARALLEL_EXECUTION_ERROR",
                error=str(result),
                error_details={"component": components[i].__name__}
            ))
        else:
            processed_results.append(result)
    
    return processed_results

# Conditional execution
def conditional_component(
    condition: Callable[[BaseInputModel], bool],
    true_component: Callable,
    false_component: Optional[Callable] = None
) -> Callable:
    """
    Create a conditional component that executes different components based on a condition.
    
    Args:
        condition: Function that returns True/False based on input
        true_component: Component to execute when condition is True
        false_component: Component to execute when condition is False (optional)
        
    Returns:
        Conditional component function
    """
    def conditional_wrapper(input_data: BaseInputModel, **kwargs) -> BaseOutputModel:
        try:
            if condition(input_data):
                return true_component(input_data, **kwargs)
            elif false_component:
                return false_component(input_data, **kwargs)
            else:
                # Return input as output if no false component
                return input_data
        except Exception as e:
            return ErrorOutputModel(
                success=False,
                error_code="CONDITIONAL_EXECUTION_ERROR",
                error=str(e),
                error_details={"condition": condition.__name__}
            )
    
    return conditional_wrapper

# Retry logic
def retry_component(
    component_func: Callable,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_on_errors: Optional[List[Type[Exception]]] = None
) -> Callable:
    """
    Add retry logic to a component.
    
    Args:
        component_func: Component function to retry
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries
        backoff_factor: Multiplier for delay on each retry
        retry_on_errors: List of exception types to retry on
        
    Returns:
        Component function with retry logic
    """
    def retry_wrapper(input_data: BaseInputModel, **kwargs) -> BaseOutputModel:
        last_exception = None
        current_delay = retry_delay
        
        for attempt in range(max_retries + 1):
            try:
                result = component_func(input_data, **kwargs)
                
                # Check if result indicates an error
                if isinstance(result, ErrorOutputModel) and not result.success:
                    if retry_on_errors is None or any(
                        error_type.__name__ in result.error 
                        for error_type in retry_on_errors
                    ):
                        if attempt < max_retries:
                            time.sleep(current_delay)
                            current_delay *= backoff_factor
                            continue
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if retry_on_errors is None or any(
                    isinstance(e, error_type) for error_type in retry_on_errors
                ):
                    if attempt < max_retries:
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                        continue
                
                # Don't retry or max retries reached
                break
        
        # All retries failed
        return ErrorOutputModel(
            success=False,
            error_code="RETRY_EXHAUSTED",
            error=f"Component failed after {max_retries + 1} attempts",
            error_details={
                "component": component_func.__name__,
                "last_error": str(last_exception) if last_exception else "Unknown"
            }
        )
    
    return retry_wrapper

# Example input/output models
class UserInputModel(BaseInputModel):
    """Example input model for user operations."""
    
    user_id: str = Field(..., min_length=1, description="User ID")
    name: str = Field(..., min_length=1, max_length=100, description="User name")
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$", description="User email")
    age: Optional[int] = Field(None, ge=0, le=150, description="User age")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and clean name."""
        return v.strip().title()
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate and normalize email."""
        return v.strip().lower()

class UserOutputModel(BaseOutputModel):
    """Example output model for user operations."""
    
    success: bool = Field(default=True)
    user_id: str = Field(..., description="User ID")
    name: str = Field(..., description="User name")
    email: str = Field(..., description="User email")
    age: Optional[int] = Field(None, description="User age")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    @computed_field
    @property
    def display_name(self) -> str:
        """Computed display name."""
        return f"{self.name} ({self.email})"
    
    @computed_field
    @property
    def is_adult(self) -> bool:
        """Computed adult status."""
        return self.age is not None and self.age >= 18

# Example functional components
@component(name="validate_user_input", cache_result=False)
def validate_user_input(input_data: UserInputModel) -> UserOutputModel:
    """Validate user input data."""
    try:
        # Additional validation logic here
        if input_data.age and input_data.age < 13:
            raise ValueError("User must be at least 13 years old")
        
        return UserOutputModel(
            success=True,
            user_id=input_data.user_id,
            name=input_data.name,
            email=input_data.email,
            age=input_data.age,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    except Exception as e:
        return ErrorOutputModel(
            success=False,
            error_code="VALIDATION_ERROR",
            error=str(e),
            error_details={"field": "validation"}
        )

@component(name="enrich_user_data", cache_result=True, cache_ttl=600)
def enrich_user_data(input_data: UserOutputModel) -> UserOutputModel:
    """Enrich user data with additional information."""
    try:
        # Simulate data enrichment
        enriched_data = input_data.model_copy()
        enriched_data.metadata = {
            "enriched": True,
            "enrichment_timestamp": datetime.utcnow().isoformat(),
            "source": "functional_component"
        }
        
        return enriched_data
    except Exception as e:
        return ErrorOutputModel(
            success=False,
            error_code="ENRICHMENT_ERROR",
            error=str(e),
            error_details={"operation": "enrichment"}
        )

@async_component(name="async_user_processing", cache_result=False)
async def async_user_processing(input_data: UserOutputModel) -> UserOutputModel:
    """Async user processing component."""
    try:
        # Simulate async processing
        await asyncio.sleep(0.1)
        
        processed_data = input_data.model_copy()
        processed_data.metadata = {
            "processed": True,
            "processing_timestamp": datetime.utcnow().isoformat(),
            "async": True
        }
        
        return processed_data
    except Exception as e:
        return ErrorOutputModel(
            success=False,
            error_code="ASYNC_PROCESSING_ERROR",
            error=str(e),
            error_details={"operation": "async_processing"}
        )

# Example composition
user_processing_pipeline = compose_components(
    validate_user_input,
    enrich_user_data
)

async_user_processing_pipeline = compose_async_components(
    validate_user_input,
    async_user_processing
)

# Example conditional component
def is_premium_user(input_data: UserInputModel) -> bool:
    """Check if user is premium."""
    return input_data.metadata.get("premium", False)

premium_user_processing = conditional_component(
    condition=is_premium_user,
    true_component=enrich_user_data,
    false_component=lambda x: x  # No enrichment for non-premium users
)

# Example retry component
robust_user_processing = retry_component(
    component_func=async_user_processing,
    max_retries=3,
    retry_delay=1.0,
    backoff_factor=2.0,
    retry_on_errors=[ConnectionError, TimeoutError]
)

# Utility functions
def create_input_model(
    name: str,
    fields: Dict[str, Any],
    base_class: Type[BaseInputModel] = BaseInputModel
) -> Type[BaseInputModel]:
    """Dynamically create an input model."""
    return create_optimized_model(name, fields, base_class)

def create_output_model(
    name: str,
    fields: Dict[str, Any],
    base_class: Type[BaseOutputModel] = BaseOutputModel
) -> Type[BaseOutputModel]:
    """Dynamically create an output model."""
    return create_optimized_model(name, fields, base_class)

def log_component_metrics() -> None:
    """Log all component metrics."""
    logger.info("Component metrics summary", metrics=get_all_metrics())
    
    for name, metrics in _component_metrics.items():
        logger.info(
            "Component performance",
            component=name,
            execution_count=metrics.execution_count,
            average_time_ms=metrics.average_execution_time * 1000,
            error_rate=metrics.error_rate,
            cache_hit_rate=metrics.cache_hit_rate
        )

# Export main components
__all__ = [
    # Base models
    "BaseInputModel",
    "BaseOutputModel", 
    "ErrorOutputModel",
    
    # Decorators
    "component",
    "async_component",
    
    # Composition
    "compose_components",
    "compose_async_components",
    "execute_parallel",
    "conditional_component",
    "retry_component",
    
    # Metrics
    "get_component_metrics",
    "get_all_metrics",
    "reset_metrics",
    "log_component_metrics",
    
    # Example models and components
    "UserInputModel",
    "UserOutputModel",
    "validate_user_input",
    "enrich_user_data",
    "async_user_processing",
    
    # Example pipelines
    "user_processing_pipeline",
    "async_user_processing_pipeline",
    "premium_user_processing",
    "robust_user_processing",
    
    # Utilities
    "create_input_model",
    "create_output_model"
] 