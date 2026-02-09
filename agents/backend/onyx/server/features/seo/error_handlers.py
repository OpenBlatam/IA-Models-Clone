from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import traceback
import sys
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import functools
import logging
from contextlib import contextmanager, asynccontextmanager
import signal
import os
import structlog
import orjson
import msgspec
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx
import aiohttp
from pybreaker import CircuitBreaker, CircuitBreakerError
import redis.asyncio as redis
from selectolax.parser import HTMLParser
import trafilatura
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import gradio as gr
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Ultra-Optimized Error Handling Module v14 - MAXIMUM RELIABILITY
Comprehensive try-except blocks for error-prone operations in data processing
Production-ready error handling with advanced logging and recovery mechanisms
"""


# Ultra-fast imports with error handling

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Type variables for generic error handling
T = TypeVar('T')
R = TypeVar('R')

# Use custom error types or error factories for consistent error handling
class SEOError(Exception):
    """Base exception for SEO-related errors"""
    def __init__(self, message: str, operation: str = "", component: str = "", **kwargs):
        
    """__init__ function."""
super().__init__(message)
        self.operation = operation
        self.component = component
        self.metadata = kwargs

class NetworkError(SEOError):
    """Network-related errors"""
    pass

class DataProcessingError(SEOError):
    """Data processing errors"""
    pass

class ModelInferenceError(SEOError):
    """Model inference errors"""
    pass

class ValidationError(SEOError):
    """Validation errors"""
    pass

class CacheError(SEOError):
    """Cache-related errors"""
    pass

class ConfigurationError(SEOError):
    """Configuration errors"""
    pass

def create_error(error_type: str, message: str, operation: str = "", component: str = "", **kwargs) -> SEOError:
    """Error factory for creating consistent error instances"""
    error_map = {
        "network": NetworkError,
        "data_processing": DataProcessingError,
        "model_inference": ModelInferenceError,
        "validation": ValidationError,
        "cache": CacheError,
        "configuration": ConfigurationError
    }
    
    error_class = error_map.get(error_type, SEOError)
    return error_class(message, operation, component, **kwargs)

class ErrorSeverity(Enum):
    """Error severity levels for classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"
    DATA_PROCESSING = "data_processing"
    MODEL_INFERENCE = "model_inference"
    CACHE = "cache"
    VALIDATION = "validation"
    SYSTEM = "system"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for error handling"""
    operation: str
    component: str
    severity: ErrorSeverity
    category: ErrorCategory
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0
    fallback_value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorResult(Generic[T]):
    """Result wrapper with error information"""
    success: bool
    data: Optional[T] = None
    error: Optional[Exception] = None
    error_context: Optional[ErrorContext] = None
    processing_time: float = 0.0
    retry_count: int = 0

class UltraErrorHandler:
    """Ultra-optimized error handler with comprehensive try-except blocks"""
    
    def __init__(self) -> Any:
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {},
            "recovery_success_rate": 0.0
        }
        self.circuit_breakers = {}
        self.fallback_strategies = {}
        self._setup_circuit_breakers()
    
    def _setup_circuit_breakers(self) -> Any:
        """Setup circuit breakers for different operations"""
        self.circuit_breakers = {
            "network": CircuitBreaker(fail_max=5, reset_timeout=60),
            "data_processing": CircuitBreaker(fail_max=3, reset_timeout=30),
            "model_inference": CircuitBreaker(fail_max=2, reset_timeout=120),
            "cache": CircuitBreaker(fail_max=10, reset_timeout=15)
        }
    
    def safe_execute(self, func: Callable[..., R], *args, 
                    error_context: ErrorContext, **kwargs) -> ErrorResult[R]:
        """Execute function with comprehensive error handling"""
        # Handle errors and edge cases at the beginning of functions
        if not func:
            return ErrorResult[R](
                success=False, 
                error=ValueError("Function cannot be None"),
                error_context=error_context
            )
        
        if not error_context:
            return ErrorResult[R](
                success=False, 
                error=ValueError("Error context cannot be None"),
                error_context=ErrorContext(
                    operation="unknown",
                    component="unknown", 
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.UNKNOWN
                )
            )
        
        start_time = time.time()
        result = ErrorResult[R](success=False, error_context=error_context)
        
        try:
            # Check circuit breaker
            circuit_breaker = self.circuit_breakers.get(error_context.category.value)
            if circuit_breaker and circuit_breaker.current_state == "open":
                raise CircuitBreakerError("Circuit breaker is open")
            
            # Execute function
            result.data = func(*args, **kwargs)
            result.success = True
            
        except Exception as e:
            result.error = e
            result.retry_count = error_context.retry_count
            self._handle_error(e, error_context)
            
            # Apply fallback strategy
            if error_context.fallback_value is not None:
                result.data = error_context.fallback_value
                result.success = True
                logger.info("Applied fallback value", 
                          operation=error_context.operation,
                          fallback_type=type(error_context.fallback_value).__name__)
        
        finally:
            result.processing_time = time.time() - start_time
            self._update_stats(result, error_context)
        
        return result
    
    async def safe_execute_async(self, func: Callable[..., R], *args,
                               error_context: ErrorContext, **kwargs) -> ErrorResult[R]:
        """Execute async function with comprehensive error handling"""
        # Use early returns for error conditions to avoid deeply nested if statements
        if not func:
            return ErrorResult[R](
                success=False, 
                error=ValueError("Function cannot be None"),
                error_context=error_context
            )
        
        if not error_context:
            return ErrorResult[R](
                success=False, 
                error=ValueError("Error context cannot be None"),
                error_context=ErrorContext(
                    operation="unknown",
                    component="unknown", 
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.UNKNOWN
                )
            )
        
        start_time = time.time()
        result = ErrorResult[R](success=False, error_context=error_context)
        
        try:
            # Check circuit breaker
            circuit_breaker = self.circuit_breakers.get(error_context.category.value)
            if circuit_breaker and circuit_breaker.current_state == "open":
                raise CircuitBreakerError("Circuit breaker is open")
            
            # Execute async function
            if asyncio.iscoroutinefunction(func):
                result.data = await func(*args, **kwargs)
            else:
                result.data = func(*args, **kwargs)
            result.success = True
            
        except Exception as e:
            result.error = e
            result.retry_count = error_context.retry_count
            await self._handle_error_async(e, error_context)
            
            # Apply fallback strategy
            if error_context.fallback_value is not None:
                result.data = error_context.fallback_value
                result.success = True
                logger.info("Applied fallback value", 
                          operation=error_context.operation,
                          fallback_type=type(error_context.fallback_value).__name__)
        
        finally:
            result.processing_time = time.time() - start_time
            self._update_stats(result, error_context)
        
        return result
    
    def _handle_error(self, error: Exception, context: ErrorContext):
        """Handle error with appropriate logging and recovery"""
        # Use guard clauses to handle preconditions and invalid states early
        if not error:
            logger.warning("Received None error in error handler")
            return
        
        if not context:
            logger.error("Missing error context for error handling")
            return
        
        error_category = context.category.value
        error_severity = context.severity.value
        
        # Update error statistics
        self.error_stats["total_errors"] += 1
        self.error_stats["errors_by_category"][error_category] = \
            self.error_stats["errors_by_category"].get(error_category, 0) + 1
        self.error_stats["errors_by_severity"][error_severity] = \
            self.error_stats["errors_by_severity"].get(error_severity, 0) + 1
        
        # Implement proper error logging and user-friendly error messages
        user_friendly_message = self._get_user_friendly_message(error, context)
        
        # Log error with context
        logger.error("Operation failed",
                    operation=context.operation,
                    component=context.component,
                    category=error_category,
                    severity=error_severity,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    user_friendly_message=user_friendly_message,
                    retry_count=context.retry_count,
                    metadata=context.metadata,
                    traceback=traceback.format_exc())
        
        # Apply circuit breaker
        circuit_breaker = self.circuit_breakers.get(error_category)
        if circuit_breaker:
            circuit_breaker.call(lambda: None)
    
    def _get_user_friendly_message(self, error: Exception, context: ErrorContext) -> str:
        """Generate user-friendly error messages"""
        error_type = type(error).__name__
        
        # Network errors
        if error_type in ['ConnectionError', 'TimeoutError', 'httpx.RequestError']:
            return f"Unable to connect to {context.metadata.get('url', 'the service')}. Please check your internet connection and try again."
        
        # Validation errors
        if error_type in ['ValueError', 'TypeError']:
            return f"Invalid data provided for {context.operation}. Please check your input and try again."
        
        # Model errors
        if error_type in ['RuntimeError', 'OSError'] and 'model' in context.operation:
            return f"Model operation failed. Please try again or contact support if the problem persists."
        
        # Cache errors
        if error_type in ['RedisError', 'ConnectionError'] and 'cache' in context.operation:
            return f"Temporary storage issue. Your request will be processed without caching."
        
        # Default user-friendly message
        return f"An unexpected error occurred during {context.operation}. Please try again or contact support."
    
    async def _handle_error_async(self, error: Exception, context: ErrorContext):
        """Handle async error with appropriate logging and recovery"""
        await asyncio.to_thread(self._handle_error, error, context)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    def _update_stats(self, result: ErrorResult, context: ErrorContext):
        """Update error handling statistics"""
        if result.success and result.retry_count > 0:
            self.error_stats["recovery_success_rate"] = \
                (self.error_stats.get("recovery_success", 0) + 1) / \
                max(self.error_stats["total_errors"], 1) * 100

class DataProcessingErrorHandler:
    """Specialized error handler for data processing operations"""
    
    def __init__(self, error_handler: UltraErrorHandler):
        
    """__init__ function."""
self.error_handler = error_handler
        self.html_parser = HTMLParser()
    
    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(multiplier=1, min=4, max=10),
           retry=retry_if_exception_type((ValueError, AttributeError, TypeError)))
    def safe_html_parsing(self, html_content: str, url: str) -> ErrorResult[Dict[str, Any]]:
        """Safely parse HTML content with multiple fallback strategies"""
        context = ErrorContext(
            operation="html_parsing",
            component="data_processing",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_PROCESSING,
            metadata={"url": url, "content_length": len(html_content)}
        )
        
        def parse_with_trafilatura():
            
    """parse_with_trafilatura function."""
try:
                extracted = trafilatura.extract(html_content, include_formatting=True, 
                                              include_links=True, include_images=True)
                if not extracted:
                    raise ValueError("No content extracted by Trafilatura")
                return {"method": "trafilatura", "content": extracted}
            except Exception as e:
                logger.warning("Trafilatura parsing failed", error=str(e))
                raise
        
        def parse_with_selectolax():
            
    """parse_with_selectolax function."""
try:
                parser = HTMLParser()
                parser.feed(html_content)
                return {"method": "selectolax", "parser": parser}
            except Exception as e:
                logger.warning("Selectolax parsing failed", error=str(e))
                raise
        
        def parse_with_regex():
            
    """parse_with_regex function."""
try:
                # Basic regex extraction as final fallback
                title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else ""
                
                desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']', 
                                     html_content, re.IGNORECASE)
                description = desc_match.group(1) if desc_match else ""
                
                return {
                    "method": "regex",
                    "title": title,
                    "description": description,
                    "raw_content": html_content[:1000]  # Limit content size
                }
            except Exception as e:
                logger.warning("Regex parsing failed", error=str(e))
                raise
        
        # Try parsing methods in order of preference
        for parse_method in [parse_with_trafilatura, parse_with_selectolax, parse_with_regex]:
            try:
                result = self.error_handler.safe_execute(parse_method, context=context)
                if result.success:
                    return result
            except Exception:
                continue
        
        # Final fallback
        context.fallback_value = {"method": "fallback", "error": "All parsing methods failed"}
        return self.error_handler.safe_execute(lambda: context.fallback_value, context=context)
    
    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10),
           retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)))
    async async def safe_http_request(self, url: str, timeout: float = 30.0) -> ErrorResult[httpx.Response]:
        """Safely make HTTP requests with comprehensive error handling"""
        context = ErrorContext(
            operation="http_request",
            component="network",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.NETWORK,
            timeout=timeout,
            metadata={"url": url}
        )
        
        async def make_request():
            
    """make_request function."""
async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response
        
        return await self.error_handler.safe_execute_async(make_request, context=context)
    
    def safe_data_validation(self, data: Any, validation_rules: Dict[str, Any]) -> ErrorResult[bool]:
        """Safely validate data according to rules"""
        context = ErrorContext(
            operation="data_validation",
            component="validation",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            metadata={"data_type": type(data).__name__, "rules": validation_rules}
        )
        
        def validate():
            
    """validate function."""
# Place the happy path last in the function for improved readability
            if not validation_rules:
                return True
            
            for field, rule in validation_rules.items():
                # Early validation checks
                if field not in data:
                    if rule.get("required", False):
                        raise create_error(
                            "validation", 
                            f"Required field '{field}' missing",
                            "data_validation",
                            "validation",
                            field=field,
                            rule=rule
                        )
                    continue
                
                value = data[field]
                
                # Type validation
                expected_type = rule.get("type")
                if expected_type and not isinstance(value, expected_type):
                    raise create_error(
                        "validation",
                        f"Field '{field}' must be of type {expected_type}",
                        "data_validation",
                        "validation",
                        field=field,
                        expected_type=expected_type,
                        actual_type=type(value).__name__
                    )
                
                # Length validation
                if "min_length" in rule and len(str(value)) < rule["min_length"]:
                    raise create_error(
                        "validation",
                        f"Field '{field}' too short (min: {rule['min_length']})",
                        "data_validation",
                        "validation",
                        field=field,
                        min_length=rule["min_length"],
                        actual_length=len(str(value))
                    )
                if "max_length" in rule and len(str(value)) > rule["max_length"]:
                    raise create_error(
                        "validation",
                        f"Field '{field}' too long (max: {rule['max_length']})",
                        "data_validation",
                        "validation",
                        field=field,
                        max_length=rule["max_length"],
                        actual_length=len(str(value))
                    )
                
                # Range validation
                if "min" in rule and value < rule["min"]:
                    raise create_error(
                        "validation",
                        f"Field '{field}' below minimum (min: {rule['min']})",
                        "data_validation",
                        "validation",
                        field=field,
                        min_value=rule["min"],
                        actual_value=value
                    )
                if "max" in rule and value > rule["max"]:
                    raise create_error(
                        "validation",
                        f"Field '{field}' above maximum (max: {rule['max']})",
                        "data_validation",
                        "validation",
                        field=field,
                        max_value=rule["max"],
                        actual_value=value
                    )
            
            # Happy path - all validations passed
            return True
        
        return self.error_handler.safe_execute(validate, context=context)

class ModelInferenceErrorHandler:
    """Specialized error handler for model inference operations"""
    
    def __init__(self, error_handler: UltraErrorHandler):
        
    """__init__ function."""
self.error_handler = error_handler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
    
    def safe_model_loading(self, model_name: str, model_type: str = "transformer") -> ErrorResult[Any]:
        """Safely load models with error handling"""
        context = ErrorContext(
            operation="model_loading",
            component="model_inference",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MODEL_INFERENCE,
            metadata={"model_name": model_name, "model_type": model_type}
        )
        
        def load_model():
            
    """load_model function."""
# Avoid unnecessary else statements; use the if-return pattern instead
            if model_type == "transformer":
                if model_name not in self.models:
                    self.models[model_name] = AutoModel.from_pretrained(model_name)
                    self.models[model_name].to(self.device)
                if model_name not in self.tokenizers:
                    self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
                return {"model": self.models[model_name], "tokenizer": self.tokenizers[model_name]}
            
            if model_type == "diffusion":
                if model_name not in self.models:
                    self.models[model_name] = StableDiffusionPipeline.from_pretrained(model_name)
                    self.models[model_name].to(self.device)
                return {"pipeline": self.models[model_name]}
            
            # Default case - unsupported model type
            raise create_error(
                "configuration",
                f"Unsupported model type: {model_type}",
                "model_loading",
                "model_inference",
                model_type=model_type,
                supported_types=["transformer", "diffusion"]
            )
        
        return self.error_handler.safe_execute(load_model, context=context)
    
    def safe_text_tokenization(self, text: str, tokenizer_name: str) -> ErrorResult[Dict[str, Any]]:
        """Safely tokenize text with error handling"""
        context = ErrorContext(
            operation="text_tokenization",
            component="model_inference",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.MODEL_INFERENCE,
            metadata={"text_length": len(text), "tokenizer": tokenizer_name}
        )
        
        def tokenize():
            
    """tokenize function."""
# Avoid unnecessary else statements; use the if-return pattern instead
            if tokenizer_name not in self.tokenizers:
                self.tokenizers[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
            
            tokenizer = self.tokenizers[tokenizer_name]
            
            # Handle long texts
            max_length = tokenizer.model_max_length
            if len(text) > max_length * 4:  # Rough estimate
                text = text[:max_length * 4]
            
            tokens = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
            return {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "token_count": tokens["input_ids"].shape[1]
            }
        
        return self.error_handler.safe_execute(tokenize, context=context)
    
    def safe_model_inference(self, model_name: str, inputs: Any, 
                           inference_type: str = "transformer") -> ErrorResult[Any]:
        """Safely perform model inference with error handling"""
        context = ErrorContext(
            operation="model_inference",
            component="model_inference",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MODEL_INFERENCE,
            metadata={"model_name": model_name, "inference_type": inference_type}
        )
        
        def infer():
            
    """infer function."""
# Use guard clauses to handle preconditions and invalid states early
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not loaded")
            
            if not inputs:
                raise ValueError("Input data cannot be empty")
            
            with torch.no_grad():
                # Avoid unnecessary else statements; use the if-return pattern instead
                if inference_type == "transformer":
                    model = self.models[model_name]
                    model.eval()
                    outputs = model(**inputs)
                    return outputs
                
                if inference_type == "diffusion":
                    pipeline = self.models[model_name]
                    if isinstance(inputs, str):
                        outputs = pipeline(inputs)
                    else:
                        outputs = pipeline(**inputs)
                    return outputs
                
                # Default case - unsupported inference type
                raise ValueError(f"Unsupported inference type: {inference_type}")
        
        return self.error_handler.safe_execute(infer, context=context)

class CacheErrorHandler:
    """Specialized error handler for cache operations"""
    
    def __init__(self, error_handler: UltraErrorHandler):
        
    """__init__ function."""
self.error_handler = error_handler
        self.redis_client: Optional[redis.Redis] = None
    
    async def safe_cache_get(self, key: str, cache_type: str = "memory") -> ErrorResult[Any]:
        """Safely get value from cache with error handling"""
        context = ErrorContext(
            operation="cache_get",
            component="cache",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.CACHE,
            metadata={"key": key, "cache_type": cache_type}
        )
        
        async def get_from_cache():
            
    """get_from_cache function."""
if cache_type == "redis" and self.redis_client:
                try:
                    value = await self.redis_client.get(key)
                    if value:
                        return orjson.loads(value)
                except Exception as e:
                    logger.warning("Redis cache get failed", error=str(e))
                    raise
            
            return None
        
        return await self.error_handler.safe_execute_async(get_from_cache, context=context)
    
    async def safe_cache_set(self, key: str, value: Any, ttl: int = 3600, 
                           cache_type: str = "memory") -> ErrorResult[bool]:
        """Safely set value in cache with error handling"""
        context = ErrorContext(
            operation="cache_set",
            component="cache",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.CACHE,
            metadata={"key": key, "cache_type": cache_type, "ttl": ttl}
        )
        
        async def set_in_cache():
            
    """set_in_cache function."""
if cache_type == "redis" and self.redis_client:
                try:
                    serialized = orjson.dumps(value)
                    await self.redis_client.setex(key, ttl, serialized)
                    return True
                except Exception as e:
                    logger.warning("Redis cache set failed", error=str(e))
                    raise
            
            return True
        
        return await self.error_handler.safe_execute_async(set_in_cache, context=context)

# Global error handler instance
error_handler = UltraErrorHandler()
data_handler = DataProcessingErrorHandler(error_handler)
model_handler = ModelInferenceErrorHandler(error_handler)
cache_handler = CacheErrorHandler(error_handler)

# Decorators for easy error handling
def safe_execution(error_context: ErrorContext):
    """Decorator for safe function execution"""
    def decorator(func: Callable[..., R]) -> Callable[..., ErrorResult[R]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> ErrorResult[R]:
            return error_handler.safe_execute(func, *args, error_context=error_context, **kwargs)
        return wrapper
    return decorator

def safe_async_execution(error_context: ErrorContext):
    """Decorator for safe async function execution"""
    def decorator(func: Callable[..., R]) -> Callable[..., ErrorResult[R]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> ErrorResult[R]:
            return await error_handler.safe_execute_async(func, *args, error_context=error_context, **kwargs)
        return wrapper
    return decorator

# Context managers for error handling
@contextmanager
def error_boundary(operation: str, component: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Context manager for error boundary"""
    context = ErrorContext(
        operation=operation,
        component=component,
        severity=severity,
        category=ErrorCategory.SYSTEM
    )
    
    try:
        yield context
    except Exception as e:
        error_handler._handle_error(e, context)
        raise

@asynccontextmanager
async def async_error_boundary(operation: str, component: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Async context manager for error boundary"""
    context = ErrorContext(
        operation=operation,
        component=component,
        severity=severity,
        category=ErrorCategory.SYSTEM
    )
    
    try:
        yield context
    except Exception as e:
        await error_handler._handle_error_async(e, context)
        raise

# Utility functions
def get_error_stats() -> Dict[str, Any]:
    """Get error handling statistics"""
    return error_handler.error_stats.copy()

def reset_error_stats():
    """Reset error handling statistics"""
    error_handler.error_stats = {
        "total_errors": 0,
        "errors_by_category": {},
        "errors_by_severity": {},
        "recovery_success_rate": 0.0
    }

def get_circuit_breaker_status() -> Dict[str, str]:
    """Get circuit breaker status"""
    return {
        name: breaker.current_state 
        for name, breaker in error_handler.circuit_breakers.items()
    } 