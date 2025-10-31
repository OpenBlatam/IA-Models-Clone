from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass
import structlog
from contextlib import asynccontextmanager
from functools import partial, reduce
import uuid
from .models import (
from .ml.models import create_model, ModelEnsemble
from .ml.data_loader import DataManager
from .ml.evaluation import ModelEvaluator
    import hashlib
from typing import Any, List, Dict, Optional
import logging
"""
Production-ready functional service for Key Messages feature with guard clauses and early validation.
"""

    KeyMessageRequest, KeyMessageResponse, BatchKeyMessageRequest, 
    BatchKeyMessageResponse, GeneratedResponse, MessageAnalysis
)

logger = structlog.get_logger(__name__)

@dataclass
class ServiceConfig:
    """Configuration for the Key Messages service."""
    model_name: str = "gpt2"
    max_concurrent_requests: int = 10
    cache_size: int = 1000
    timeout_seconds: int = 30
    enable_analytics: bool = True
    enable_caching: bool = True
    enable_monitoring: bool = True

@dataclass
class ServiceState:
    """Immutable service state."""
    config: ServiceConfig
    model: Optional[Any] = None
    data_manager: Optional[DataManager] = None
    evaluator: Optional[ModelEvaluator] = None
    cache: Dict[str, Any] = None
    is_healthy: bool = False
    startup_time: Optional[float] = None
    request_semaphore: Optional[asyncio.Semaphore] = None

# Global state (functional approach)
_service_state: Optional[ServiceState] = None

# Modular validation functions
VALIDATION_FUNCTIONS = {
    "config": lambda config: (
        not config.model_name or not config.model_name.strip(),
        "Model name cannot be empty"
    ),
    "max_concurrent": lambda config: (
        config.max_concurrent_requests <= 0,
        "Max concurrent requests must be positive"
    ),
    "cache_size": lambda config: (
        config.cache_size <= 0,
        "Cache size must be positive"
    ),
    "timeout": lambda config: (
        config.timeout_seconds <= 0,
        "Timeout must be positive"
    )
}

def run_validations(validations: Dict[str, Callable], target: Any) -> None:
    """Run multiple validations on a target."""
    # Guard clause: Check if validations dict is empty
    if not validations:
        return
    
    # Guard clause: Check if target is None
    if target is None:
        raise ValueError("Target cannot be None")
    
    for validation_name, validation_func in validations.items():
        # Guard clause: Check if validation function is callable
        if not callable(validation_func):
            raise ValueError(f"Validation function '{validation_name}' is not callable")
        
        is_invalid, error_message = validation_func(target)
        if is_invalid:
            raise ValueError(error_message)

def validate_config(config: ServiceConfig) -> None:
    """Validate service configuration using modular validations."""
    # Guard clause: Check if config is None
    if config is None:
        raise ValueError("Service configuration cannot be None")
    
    # Guard clause: Check if config is the correct type
    if not isinstance(config, ServiceConfig):
        raise TypeError("Config must be an instance of ServiceConfig")
    
    run_validations(VALIDATION_FUNCTIONS, config)

# Modular request validation functions
REQUEST_VALIDATIONS = {
    "empty_message": lambda request: (
        not request.message or not request.message.strip(),
        "Message cannot be empty"
    ),
    "message_length": lambda request: (
        len(request.message) > 10000,
        "Message too long (max 10000 characters)"
    )
}

BATCH_VALIDATIONS = {
    "no_messages": lambda batch_request: (
        not batch_request.messages,
        "No messages provided"
    ),
    "too_many_messages": lambda batch_request: (
        len(batch_request.messages) > 100,
        "Too many messages (max 100)"
    )
}

SERVICE_HEALTH_VALIDATIONS = {
    "service_initialized": lambda: (
        not _service_state or not _service_state.is_healthy,
        "Service is not healthy"
    ),
    "model_loaded": lambda: (
        _service_state.model is None,
        "Model not loaded"
    )
}

async def validate_request(request: KeyMessageRequest) -> None:
    """Validate message request using modular validations."""
    # Guard clause: Check if request is None
    if request is None:
        raise ValueError("Request cannot be None")
    
    # Guard clause: Check if request is the correct type
    if not isinstance(request, KeyMessageRequest):
        raise TypeError("Request must be an instance of KeyMessageRequest")
    
    run_validations(REQUEST_VALIDATIONS, request)

async def validate_batch_request(batch_request: BatchKeyMessageRequest) -> None:
    """Validate batch request using modular validations."""
    # Guard clause: Check if batch_request is None
    if batch_request is None:
        raise ValueError("Batch request cannot be None")
    
    # Guard clause: Check if batch_request is the correct type
    if not isinstance(batch_request, BatchKeyMessageRequest):
        raise TypeError("Batch request must be an instance of BatchKeyMessageRequest")
    
    run_validations(BATCH_VALIDATIONS, batch_request)

def validate_service_health() -> None:
    """Validate service health using modular validations."""
    # Guard clause: Check if service state exists
    if _service_state is None:
        raise RuntimeError("Service not initialized")
    
    run_validations(SERVICE_HEALTH_VALIDATIONS, None)

# Pure functions for state management
def get_service_state() -> ServiceState:
    """Get current service state."""
    global _service_state
    
    # Guard clause: Check if service state is None
    if _service_state is None:
        raise RuntimeError("Service not initialized")
    
    return _service_state

def update_service_state(**kwargs) -> ServiceState:
    """Update service state immutably."""
    global _service_state
    
    # Guard clause: Check if service state is None
    if _service_state is None:
        raise RuntimeError("Service not initialized")
    
    # Guard clause: Check if kwargs is empty
    if not kwargs:
        return _service_state
    
    _service_state = ServiceState(**{**vars(_service_state), **kwargs})
    return _service_state

def initialize_service_state(config: ServiceConfig) -> ServiceState:
    """Initialize service state."""
    global _service_state
    
    # Guard clause: Check if config is None
    if config is None:
        raise ValueError("Configuration cannot be None")
    
    # Guard clause: Check if service is already initialized
    if _service_state is not None:
        raise RuntimeError("Service already initialized")
    
    validate_config(config)
    
    _service_state = ServiceState(
        config=config,
        cache={},
        request_semaphore=asyncio.Semaphore(config.max_concurrent_requests)
    )
    
    logger.info("Service state initialized", 
               model_name=config.model_name,
               max_concurrent_requests=config.max_concurrent_requests,
               cache_size=config.cache_size)
    
    return _service_state

# Modular cache management functions
CACHE_OPERATIONS = {
    "generate_key": lambda request: f"{request.message}_{request.message_type.value}_{request.tone.value}",
    "get_response": lambda cache_key: get_service_state().cache.get(cache_key),
    "store_response": lambda cache_key, response: setattr(get_service_state().cache, cache_key, response),
    "clear_all": lambda: get_service_state().cache.clear(),
    "get_size": lambda: len(get_service_state().cache),
    "get_max_size": lambda: get_service_state().config.cache_size
}

def generate_cache_key(request: KeyMessageRequest) -> str:
    """Generate cache key for request."""
    # Guard clause: Check if request is None
    if request is None:
        raise ValueError("Request cannot be None")
    
    # Guard clause: Check if request has required attributes
    if not hasattr(request, 'message') or not hasattr(request, 'message_type') or not hasattr(request, 'tone'):
        raise ValueError("Request missing required attributes")
    
    key_data = CACHE_OPERATIONS["generate_key"](request)
    return hashlib.md5(key_data.encode()).hexdigest()

def get_cached_response(cache_key: str) -> Optional[GeneratedResponse]:
    """Get cached response."""
    # Guard clause: Check if cache_key is None or empty
    if not cache_key or not cache_key.strip():
        return None
    
    return CACHE_OPERATIONS["get_response"](cache_key)

def cache_response(cache_key: str, response: GeneratedResponse) -> None:
    """Cache response."""
    # Guard clause: Check if cache_key is None or empty
    if not cache_key or not cache_key.strip():
        return
    
    # Guard clause: Check if response is None
    if response is None:
        return
    
    state = get_service_state()
    if state.config.enable_caching:
        CACHE_OPERATIONS["store_response"](cache_key, response)
        cleanup_cache_if_needed()

def cleanup_cache_if_needed() -> None:
    """Cleanup cache if needed."""
    # Guard clause: Check if service state exists
    try:
        state = get_service_state()
    except RuntimeError:
        return
    
    current_size = CACHE_OPERATIONS["get_size"]()
    max_size = CACHE_OPERATIONS["get_max_size"]()
    
    # Guard clause: Check if cleanup is needed
    if current_size <= max_size:
        return
    
    # Remove oldest entries
    sorted_items = sorted(state.cache.items(), key=lambda x: x[1].created_at)
    items_to_remove = current_size - max_size
    for i in range(items_to_remove):
        del state.cache[sorted_items[i][0]]

def calculate_cache_hit_rate() -> float:
    """Calculate cache hit rate."""
    # Guard clause: Check if service state exists
    try:
        state = get_service_state()
    except RuntimeError:
        return 0.0
    
    # Guard clause: Check if cache is empty
    if not state.cache:
        return 0.0
    
    total_accesses = sum(entry.access_count for entry in state.cache.values() if hasattr(entry, 'access_count'))
    cache_hits = sum(entry.access_count for entry in state.cache.values() if hasattr(entry, 'access_count') and entry.access_count > 0)
    
    # Guard clause: Check if no accesses
    if total_accesses == 0:
        return 0.0
    
    return cache_hits / total_accesses

# Modular text processing functions
TEXT_PROCESSORS = {
    "build_prompt": lambda request: " ".join([
        f"Generate a {request.message_type.value} message",
        f"with a {request.tone.value} tone",
        f"for {request.target_audience}" if request.target_audience else "",
        f"Context: {request.context}" if request.context else "",
        f"Include keywords: {', '.join(request.keywords)}" if request.keywords else "",
        f"Original message: {request.message}"
    ]),
    "generate_id": lambda: str(uuid.uuid4()),
    "analyze_content": lambda text: {
        "word_count": len(text.split()),
        "char_count": len(text),
        "sentiment": "neutral",
        "readability_score": 0.7
    }
}

def build_prompt(request: KeyMessageRequest) -> str:
    """Build prompt for text generation."""
    # Guard clause: Check if request is None
    if request is None:
        raise ValueError("Request cannot be None")
    
    return TEXT_PROCESSORS["build_prompt"](request)

def generate_response_id() -> str:
    """Generate unique response ID."""
    return TEXT_PROCESSORS["generate_id"]()

def analyze_text_content(text: str) -> MessageAnalysis:
    """Analyze text content."""
    # Guard clause: Check if text is None
    if text is None:
        raise ValueError("Text cannot be None")
    
    # Guard clause: Check if text is empty
    if not text.strip():
        return MessageAnalysis(
            sentiment="neutral",
            confidence=0.0,
            keywords=[],
            word_count=0
        )
    
    # Simple analysis (in production, use proper NLP libraries)
    words = text.lower().split()
    word_count = len(words)
    
    # Basic sentiment analysis
    positive_words = {"good", "great", "excellent", "amazing", "wonderful", "fantastic"}
    negative_words = {"bad", "terrible", "awful", "horrible", "disgusting"}
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    if positive_count > negative_count:
        sentiment = "positive"
        confidence = min(0.9, positive_count / word_count)
    elif negative_count > positive_count:
        sentiment = "negative"
        confidence = min(0.9, negative_count / word_count)
    else:
        sentiment = "neutral"
        confidence = 0.5
    
    # Extract keywords (simple approach)
    keywords = [word for word in words if len(word) > 3 and word.isalpha()]
    
    return MessageAnalysis(
        sentiment=sentiment,
        confidence=confidence,
        keywords=keywords[:10],  # Limit to 10 keywords
        word_count=word_count
    )

# Modular response creation functions
RESPONSE_CREATORS = {
    "generated": lambda request, generated_text, processing_time: GeneratedResponse(
        id=generate_response_id(),
        original_message=request.message,
        response=generated_text,
        message_type=request.message_type,
        tone=request.tone,
        word_count=len(generated_text.split()),
        character_count=len(generated_text),
        keywords_used=request.keywords,
        processing_time=processing_time
    ),
    "success": lambda data, processing_time: KeyMessageResponse(
        success=True,
        data=data,
        processing_time=processing_time
    ),
    "error": lambda error, processing_time: KeyMessageResponse(
        success=False,
        error=error,
        processing_time=processing_time
    )
}

def create_generated_response(
    request: KeyMessageRequest,
    generated_text: str,
    processing_time: float
) -> GeneratedResponse:
    """Create generated response."""
    # Guard clause: Check if request is None
    if request is None:
        raise ValueError("Request cannot be None")
    
    # Guard clause: Check if generated_text is None
    if generated_text is None:
        raise ValueError("Generated text cannot be None")
    
    # Guard clause: Check if processing_time is negative
    if processing_time < 0:
        processing_time = 0.0
    
    return RESPONSE_CREATORS["generated"](request, generated_text, processing_time)

def create_success_response(
    data: GeneratedResponse,
    processing_time: float
) -> KeyMessageResponse:
    """Create success response."""
    # Guard clause: Check if data is None
    if data is None:
        raise ValueError("Data cannot be None")
    
    # Guard clause: Check if processing_time is negative
    if processing_time < 0:
        processing_time = 0.0
    
    return RESPONSE_CREATORS["success"](data, processing_time)

def create_error_response(
    error: str,
    processing_time: float
) -> KeyMessageResponse:
    """Create error response."""
    # Guard clause: Check if error is None or empty
    if not error or not error.strip():
        error = "Unknown error occurred"
    
    # Guard clause: Check if processing_time is negative
    if processing_time < 0:
        processing_time = 0.0
    
    return RESPONSE_CREATORS["error"](error, processing_time)

# Modular service operations
SERVICE_OPERATIONS = {
    "startup": {
        "initialize_state": lambda config: initialize_service_state(config),
        "load_model": lambda config: create_model(config.model_name),
        "create_components": lambda: (DataManager(), ModelEvaluator()),
        "initialize_components": lambda components: asyncio.gather(
            components[0].initialize(),
            components[1].initialize()
        )
    },
    "shutdown": {
        "update_state": lambda: update_service_state(is_healthy=False),
        "clear_cache": lambda: CACHE_OPERATIONS["clear_all"]() if get_service_state().config.enable_caching else None,
        "cleanup_model": lambda: delattr(get_service_state(), 'model') if get_service_state().model else None
    }
}

# Async functions for service operations
async def startup_service(config: ServiceConfig) -> None:
    """Startup service with model loading."""
    # Guard clause: Check if config is None
    if config is None:
        raise ValueError("Configuration cannot be None")
    
    # Guard clause: Check if service is already started
    if _service_state is not None and _service_state.is_healthy:
        logger.warning("Service already started")
        return
    
    try:
        # Initialize state
        initialize_service_state(config)
        
        # Load model
        model = await create_model(config.model_name)
        update_service_state(model=model, is_healthy=True, startup_time=time.time())
        
        # Initialize data manager and evaluator
        data_manager = DataManager()
        evaluator = ModelEvaluator()
        update_service_state(data_manager=data_manager, evaluator=evaluator)
        
        logger.info("Service started successfully", model_name=config.model_name)
        
    except Exception as e:
        logger.error("Failed to start service", error=str(e))
        update_service_state(is_healthy=False)
        raise

async def shutdown_service() -> None:
    """Shutdown service gracefully."""
    # Guard clause: Check if service is not initialized
    if _service_state is None:
        logger.warning("Service not initialized, nothing to shutdown")
        return
    
    try:
        # Update state
        update_service_state(is_healthy=False)
        
        # Clear cache
        if _service_state.cache:
            _service_state.cache.clear()
        
        # Close semaphore
        if _service_state.request_semaphore:
            _service_state.request_semaphore = None
        
        logger.info("Service shutdown successfully")
        
    except Exception as e:
        logger.error("Error during service shutdown", error=str(e))
        raise

# Modular health check functions
HEALTH_CHECKERS = {
    "service_status": lambda state: (
        not state.is_healthy,
        "Service not initialized"
    ),
    "model_status": lambda state: (
        state.model is None,
        "Model not loaded"
    ),
    "test_functionality": lambda: test_model_functionality()
}

async def check_service_health() -> Dict[str, Any]:
    """Check service health status."""
    # Guard clause: Check if service is not initialized
    if _service_state is None:
        return {
            "is_healthy": False,
            "status": "not_initialized",
            "uptime": 0.0,
            "cache_size": 0,
            "cache_hit_rate": 0.0
        }
    
    try:
        uptime = time.time() - (_service_state.startup_time or time.time())
        cache_size = len(_service_state.cache) if _service_state.cache else 0
        cache_hit_rate = calculate_cache_hit_rate()
        
        return {
            "is_healthy": _service_state.is_healthy,
            "status": "healthy" if _service_state.is_healthy else "unhealthy",
            "uptime": max(0.0, uptime),
            "cache_size": cache_size,
            "cache_hit_rate": cache_hit_rate,
            "model_loaded": _service_state.model is not None
        }
        
    except Exception as e:
        logger.error("Error checking service health", error=str(e))
        return {
            "is_healthy": False,
            "status": "error",
            "uptime": 0.0,
            "cache_size": 0,
            "cache_hit_rate": 0.0,
            "error": str(e)
        }

async def generate_text(request: KeyMessageRequest) -> str:
    """Generate text using the loaded model."""
    # Guard clause: Check if request is None
    if request is None:
        raise ValueError("Request cannot be None")
    
    # Guard clause: Check if service is healthy
    validate_service_health()
    
    # Guard clause: Check if model is loaded
    state = get_service_state()
    if state.model is None:
        raise RuntimeError("Model not loaded")
    
    try:
        prompt = build_prompt(request)
        generated_text = await state.model.generate(prompt)
        
        # Guard clause: Check if generated text is empty
        if not generated_text or not generated_text.strip():
            raise RuntimeError("Model generated empty text")
        
        return generated_text.strip()
        
    except Exception as e:
        logger.error("Error generating text", error=str(e))
        raise

async def test_model_functionality() -> bool:
    """Test model functionality."""
    # Guard clause: Check if service is not initialized
    if _service_state is None:
        return False
    
    # Guard clause: Check if model is not loaded
    if _service_state.model is None:
        return False
    
    try:
        test_request = KeyMessageRequest(
            message="Test message",
            message_type="social_media",
            tone="professional"
        )
        
        test_text = await generate_text(test_request)
        
        # Guard clause: Check if test generation failed
        if not test_text or not test_text.strip():
            return False
        
        return True
        
    except Exception as e:
        logger.error("Model functionality test failed", error=str(e))
        return False

# Modular processing pipeline
PROCESSING_STEPS = {
    "validate": lambda request: validate_request(request),
    "check_cache": lambda request: (generate_cache_key(request), get_cached_response(generate_cache_key(request))),
    "generate_text": lambda request: generate_text(request),
    "create_response": lambda request, text, time: create_generated_response(request, text, time),
    "cache_response": lambda key, response: cache_response(key, response),
    "log_success": lambda request, text, time: logger.info(
        "Response generated successfully",
        message_length=len(request.message),
        response_length=len(text),
        processing_time=time
    )
}

# Main service functions
async def generate_response(request: KeyMessageRequest) -> KeyMessageResponse:
    """Generate a key message response with validation."""
    start_time = time.perf_counter()
    
    # Guard clause: Check if request is None
    if request is None:
        return create_error_response("Request cannot be None", time.perf_counter() - start_time)
    
    try:
        # Guard clauses for early validation
        validate_service_health()
        PROCESSING_STEPS["validate"](request)
        
        state = get_service_state()
        
        async with state.request_semaphore:
            try:
                # Check cache first
                cache_key, cached_response = PROCESSING_STEPS["check_cache"](request)
                
                if state.config.enable_caching and cached_response:
                    processing_time = time.perf_counter() - start_time
                    
                    logger.info("Response served from cache", 
                               cache_key=cache_key,
                               processing_time=processing_time)
                    
                    return create_success_response(cached_response, processing_time)
                
                # Generate response
                generated_text = await PROCESSING_STEPS["generate_text"](request)
                
                # Create response object
                response_data = PROCESSING_STEPS["create_response"](
                    request, 
                    generated_text, 
                    time.perf_counter() - start_time
                )
                
                # Cache response
                PROCESSING_STEPS["cache_response"](cache_key, response_data)
                
                processing_time = time.perf_counter() - start_time
                
                # Log success
                PROCESSING_STEPS["log_success"](request, generated_text, processing_time)
                
                return create_success_response(response_data, processing_time)
                
            except ValueError as e:
                return create_error_response(f"Validation error: {str(e)}", time.perf_counter() - start_time)
            except RuntimeError as e:
                return create_error_response(f"Runtime error: {str(e)}", time.perf_counter() - start_time)
            except Exception as e:
                logger.error("Unexpected error in generate_response", error=str(e))
                return create_error_response(f"Unexpected error: {str(e)}", time.perf_counter() - start_time)

async def analyze_message(request: KeyMessageRequest) -> KeyMessageResponse:
    """Analyze a message with comprehensive metrics."""
    start_time = time.perf_counter()
    
    # Guard clause: Check if request is None
    if request is None:
        return create_error_response("Request cannot be None", time.perf_counter() - start_time)
    
    try:
        # Guard clauses for early validation
        validate_service_health()
        validate_request(request)
        
        state = get_service_state()
        
        async with state.request_semaphore:
            try:
                # Perform analysis
                analysis = analyze_text_content(request.message)
                
                # Create response
                response_data = create_generated_response(
                    request, 
                    request.message,  # Original message for analysis
                    time.perf_counter() - start_time
                )
                
                processing_time = time.perf_counter() - start_time
                
                return create_success_response(response_data, processing_time)
                
            except ValueError as e:
                return create_error_response(f"Validation error: {str(e)}", time.perf_counter() - start_time)
            except Exception as e:
                logger.error("Unexpected error in analyze_message", error=str(e))
                return create_error_response(f"Unexpected error: {str(e)}", time.perf_counter() - start_time)

# Modular batch processing
BATCH_PROCESSORS = {
    "validate": lambda batch_request: validate_batch_request(batch_request),
    "create_tasks": lambda batch_request: [generate_response(msg) for msg in batch_request.messages],
    "process_results": lambda responses: [
        create_error_response(str(response), 0.0) if isinstance(response, Exception) else response
        for response in responses
    ],
    "count_failures": lambda results: sum(1 for r in results if not r.success),
    "log_completion": lambda batch_request, results, failed_count, processing_time: logger.info(
        "Batch generation completed",
        total_messages=len(batch_request.messages),
        successful=len(results) - failed_count,
        failed=failed_count,
        processing_time=processing_time
    )
}

async def generate_batch(batch_request: BatchKeyMessageRequest) -> BatchKeyMessageResponse:
    """Generate responses for multiple messages in batch."""
    start_time = time.perf_counter()
    
    # Guard clause: Check if batch_request is None
    if batch_request is None:
        return BatchKeyMessageResponse(
            success=False,
            responses=[],
            processing_time=time.perf_counter() - start_time,
            error="Batch request cannot be None"
        )
    
    try:
        # Guard clauses for early validation
        validate_service_health()
        BATCH_PROCESSORS["validate"](batch_request)
        
        # Process messages concurrently using functional approach
        tasks = BATCH_PROCESSORS["create_tasks"](batch_request)
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results functionally
        results = BATCH_PROCESSORS["process_results"](responses)
        failed_count = BATCH_PROCESSORS["count_failures"](results)
        
        processing_time = time.perf_counter() - start_time
        
        # Log completion
        BATCH_PROCESSORS["log_completion"](batch_request, results, failed_count, processing_time)
        
        return BatchKeyMessageResponse(
            success=failed_count == 0,
            results=results,
            total_processed=len(batch_request.messages),
            failed_count=failed_count,
            processing_time=processing_time
        )
        
    except ValueError as e:
        return BatchKeyMessageResponse(
            success=False,
            responses=[],
            processing_time=time.perf_counter() - start_time,
            error=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error("Unexpected error in generate_batch", error=str(e))
        return BatchKeyMessageResponse(
            success=False,
            responses=[],
            processing_time=time.perf_counter() - start_time,
            error=f"Unexpected error: {str(e)}"
        )

async def clear_cache() -> None:
    """Clear all cached responses."""
    # Guard clause: Check if service is not initialized
    if _service_state is None:
        return
    
    # Guard clause: Check if cache is empty
    if not _service_state.cache:
        return
    
    try:
        _service_state.cache.clear()
        logger.info("Cache cleared successfully")
    except Exception as e:
        logger.error("Error clearing cache", error=str(e))
        raise

async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    # Guard clause: Check if service is not initialized
    if _service_state is None:
        return {
            "cache_size": 0,
            "cache_hit_rate": 0.0,
            "cache_enabled": False
        }
    
    try:
        cache_size = len(_service_state.cache) if _service_state.cache else 0
        cache_hit_rate = calculate_cache_hit_rate()
        
        return {
            "cache_size": cache_size,
            "cache_hit_rate": cache_hit_rate,
            "cache_enabled": _service_state.config.enable_caching,
            "max_cache_size": _service_state.config.cache_size
        }
        
    except Exception as e:
        logger.error("Error getting cache stats", error=str(e))
        return {
            "cache_size": 0,
            "cache_hit_rate": 0.0,
            "cache_enabled": False,
            "error": str(e)
        }

# Service lifecycle management
@asynccontextmanager
async def service_lifecycle(config: ServiceConfig):
    """Context manager for service lifecycle."""
    # Guard clause: Check if config is None
    if config is None:
        raise ValueError("Configuration cannot be None")
    
    try:
        await startup_service(config)
        yield
    finally:
        await shutdown_service()

# Functional service interface
def create_service_interface(config: ServiceConfig):
    """Create service interface with dependency injection."""
    # Guard clause: Check if config is None
    if config is None:
        raise ValueError("Configuration cannot be None")
    
    return {
        "generate_response": generate_response,
        "analyze_message": analyze_message,
        "generate_batch": generate_batch,
        "check_health": check_service_health,
        "clear_cache": clear_cache,
        "get_cache_stats": get_cache_stats
    }
