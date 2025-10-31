from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import requests
from typing import Dict, Any, List, Callable
from functools import reduce, partial, compose
import logging
from .security_functions import (
from .api_functional import (
from typing import Any, List, Dict, Optional
"""
Functional Programming Demo for Instagram Captions API

Demonstrates pure functions, declarative patterns, and functional composition.
No classes - purely functional approach.
"""


# Import functional components
    hash_password, verify_password, create_access_token, verify_token,
    generate_api_key, validate_api_key, check_rate_limit, enforce_rate_limit,
    sanitize_content, validate_content_description, validate_style,
    log_security_event, generate_request_id, calculate_request_hash
)

    process_caption_request, process_batch_request, get_health_status,
    get_metrics, compose_security_checks, compose_processing_pipeline
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PURE FUNCTIONS - FUNCTIONAL UTILITIES
# =============================================================================

def pipe(*functions: Callable) -> Callable:
    """Functional pipe operator - compose functions left to right."""
    def pipe_inner(value) -> Any:
        return reduce(lambda acc, func: func(acc), functions, value)
    return pipe_inner


def curry(func: Callable, *args, **kwargs) -> Callable:
    """Curry a function with partial arguments."""
    return partial(func, *args, **kwargs)


def map_over_list(func: Callable) -> Callable:
    """Apply function to each item in a list."""
    return lambda items: list(map(func, items))


def filter_by_condition(condition: Callable) -> Callable:
    """Filter list by condition."""
    return lambda items: list(filter(condition, items))


def reduce_with_func(func: Callable, initial=None) -> Callable:
    """Reduce list with function."""
    return lambda items: reduce(func, items, initial)


# =============================================================================
# PURE FUNCTIONS - DATA TRANSFORMATION
# =============================================================================

async def transform_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform request data using functional patterns."""
    transformations = pipe(
        lambda d: {**d, 'content_description': sanitize_content(d.get('content_description', ''))},
        lambda d: {**d, 'style': d.get('style', 'casual').lower()},
        lambda d: {**d, 'hashtag_count': min(max(d.get('hashtag_count', 15), 0), 30)},
        lambda d: {**d, 'request_id': generate_request_id()},
        lambda d: {**d, 'timestamp': time.time()}
    )
    return transformations(data)


async def validate_multiple_requests(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate multiple requests using functional patterns."""
    validation_pipeline = pipe(
        map_over_list(transform_request_data),
        filter_by_condition(lambda req: validate_content_description(req['content_description'])[0]),
        map_over_list(lambda req: {**req, 'validated': True})
    )
    return validation_pipeline(requests)


def calculate_batch_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate batch metrics using functional patterns."""
    metrics_pipeline = pipe(
        lambda items: {
            'total_count': len(items),
            'successful_count': len(filter_by_condition(lambda r: r.get('success', False))(items)),
            'failed_count': len(filter_by_condition(lambda r: not r.get('success', False))(items)),
            'average_processing_time': reduce_with_func(lambda acc, r: acc + r.get('processing_time', 0), 0)(items) / len(items) if items else 0,
            'total_processing_time': reduce_with_func(lambda acc, r: acc + r.get('processing_time', 0), 0)(items)
        }
    )
    return metrics_pipeline(results)


# =============================================================================
# PURE FUNCTIONS - SECURITY DEMONSTRATION
# =============================================================================

def demonstrate_authentication() -> None:
    """Demonstrate functional authentication."""
    logger.info("🔐 Demonstrating Functional Authentication")
    
    # Generate API key
    user_id = "user_123"
    api_key = generate_api_key(user_id)
    logger.info(f"✅ Generated API key: {api_key[:20]}...")
    
    # Validate API key
    is_valid = validate_api_key(api_key)
    logger.info(f"✅ API key validation: {is_valid}")
    
    # Create access token
    token_data = {"user_id": user_id, "role": "user"}
    access_token = create_access_token(token_data)
    logger.info(f"✅ Created access token: {access_token[:20]}...")
    
    # Verify token
    try:
        payload = verify_token(access_token)
        logger.info(f"✅ Token verification successful: {payload}")
    except Exception as e:
        logger.error(f"❌ Token verification failed: {e}")


def demonstrate_rate_limiting() -> None:
    """Demonstrate functional rate limiting."""
    logger.info("⏱️ Demonstrating Functional Rate Limiting")
    
    user_id = "demo_user"
    
    # Check rate limit multiple times
    for i in range(5):
        can_proceed = check_rate_limit(user_id, requests_per_minute=10)
        logger.info(f"Request {i+1}: Rate limit check - {can_proceed}")
        
        if not can_proceed:
            logger.warning("⚠️ Rate limit exceeded!")
            break


def demonstrate_input_validation() -> None:
    """Demonstrate functional input validation."""
    logger.info("✅ Demonstrating Functional Input Validation")
    
    test_cases = [
        {
            "content_description": "Beautiful sunset over mountains",
            "style": "casual",
            "hashtag_count": 15
        },
        {
            "content_description": "<script>alert('xss')</script>",
            "style": "invalid_style",
            "hashtag_count": 50
        },
        {
            "content_description": "Short",
            "style": "casual",
            "hashtag_count": -5
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"\nTest case {i+1}:")
        
        # Transform and validate
        transformed = transform_request_data(test_case)
        is_valid, error_msg = validate_content_description(transformed['content_description'])
        style_valid, style_error = validate_style(transformed['style'])
        
        logger.info(f"  Original: {test_case}")
        logger.info(f"  Transformed: {transformed}")
        logger.info(f"  Content valid: {is_valid} - {error_msg}")
        logger.info(f"  Style valid: {style_valid} - {style_error}")


# =============================================================================
# PURE FUNCTIONS - PROCESSING DEMONSTRATION
# =============================================================================

def demonstrate_single_processing() -> None:
    """Demonstrate functional single request processing."""
    logger.info("🎯 Demonstrating Functional Single Request Processing")
    
    request_data = {
        "content_description": "Beautiful sunset over mountains with golden light reflecting on a calm lake",
        "style": "casual",
        "tone": "friendly",
        "hashtag_count": 15,
        "language": "en",
        "include_emoji": True
    }
    
    user_id = "demo_user"
    
    # Process using functional pipeline
    try:
        result = process_caption_request(request_data, user_id)
        logger.info(f"✅ Processing successful:")
        logger.info(f"  Caption: {result.caption}")
        logger.info(f"  Hashtags: {result.hashtags}")
        logger.info(f"  Processing time: {result.processing_time:.3f}s")
        logger.info(f"  Request ID: {result.request_id}")
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")


def demonstrate_batch_processing() -> None:
    """Demonstrate functional batch processing."""
    logger.info("📦 Demonstrating Functional Batch Processing")
    
    batch_data = {
        "requests": [
            {
                "content_description": "Delicious homemade pizza with melted cheese",
                "style": "casual",
                "tone": "enthusiastic",
                "hashtag_count": 10
            },
            {
                "content_description": "Modern office space with natural lighting",
                "style": "professional",
                "tone": "professional",
                "hashtag_count": 8
            },
            {
                "content_description": "Cozy coffee shop with warm atmosphere",
                "style": "creative",
                "tone": "calm",
                "hashtag_count": 12
            }
        ],
        "batch_id": f"batch_{int(time.time())}"
    }
    
    user_id = "demo_user"
    
    # Process batch using functional approach
    try:
        result = process_batch_request(batch_data, user_id)
        logger.info(f"✅ Batch processing successful:")
        logger.info(f"  Batch ID: {result.batch_id}")
        logger.info(f"  Total requests: {len(result.results)}")
        logger.info(f"  Successful: {result.successful_count}")
        logger.info(f"  Failed: {result.failed_count}")
        logger.info(f"  Total time: {result.total_processing_time:.3f}s")
        
        # Show individual results
        for i, res in enumerate(result.results):
            logger.info(f"  Result {i+1}: {res.caption[:50]}...")
            
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")


# =============================================================================
# PURE FUNCTIONS - COMPOSITION DEMONSTRATION
# =============================================================================

def demonstrate_functional_composition() -> None:
    """Demonstrate functional composition patterns."""
    logger.info("🔗 Demonstrating Functional Composition")
    
    # Example 1: Compose data transformation pipeline
    data_pipeline = pipe(
        lambda data: {**data, 'processed': True},
        lambda data: {**data, 'timestamp': time.time()},
        lambda data: {**data, 'hash': calculate_request_hash(data)}
    )
    
    test_data = {"content": "test content", "style": "casual"}
    result = data_pipeline(test_data)
    logger.info(f"✅ Data pipeline result: {result}")
    
    # Example 2: Compose validation pipeline
    validation_pipeline = pipe(
        lambda req: (req, validate_content_description(req.get('content_description', ''))),
        lambda result: (result[0], result[1][0]),  # Extract validation result
        lambda result: result[0] if result[1] else None
    )
    
    valid_request = {"content_description": "Valid content with sufficient length"}
    invalid_request = {"content_description": "Short"}
    
    valid_result = validation_pipeline(valid_request)
    invalid_result = validation_pipeline(invalid_request)
    
    logger.info(f"✅ Valid request result: {valid_result is not None}")
    logger.info(f"✅ Invalid request result: {invalid_result is not None}")
    
    # Example 3: Compose processing pipeline
    processing_pipeline = pipe(
        transform_request_data,
        lambda data: process_caption_request(data, "demo_user")
    )
    
    try:
        processing_result = processing_pipeline({
            "content_description": "Amazing landscape photography",
            "style": "creative",
            "hashtag_count": 20
        })
        logger.info(f"✅ Processing pipeline result: {processing_result.caption[:50]}...")
    except Exception as e:
        logger.error(f"❌ Processing pipeline failed: {e}")


# =============================================================================
# PURE FUNCTIONS - HIGHER-ORDER FUNCTIONS
# =============================================================================

def demonstrate_higher_order_functions() -> None:
    """Demonstrate higher-order functions."""
    logger.info("🔝 Demonstrating Higher-Order Functions")
    
    # Map over list of requests
    requests = [
        {"content": "First content", "style": "casual"},
        {"content": "Second content", "style": "formal"},
        {"content": "Third content", "style": "creative"}
    ]
    
    # Transform all requests
    transformed_requests = map_over_list(transform_request_data)(requests)
    logger.info(f"✅ Transformed requests: {len(transformed_requests)}")
    
    # Filter valid requests
    valid_requests = filter_by_condition(
        lambda req: len(req.get('content_description', '')) > 10
    )(transformed_requests)
    logger.info(f"✅ Valid requests: {len(valid_requests)}")
    
    # Reduce to summary
    summary = reduce_with_func(
        lambda acc, req: {
            'total_requests': acc.get('total_requests', 0) + 1,
            'total_content_length': acc.get('total_content_length', 0) + len(req.get('content_description', ''))
        }
    )(valid_requests)
    logger.info(f"✅ Summary: {summary}")


# =============================================================================
# PURE FUNCTIONS - CURRYING DEMONSTRATION
# =============================================================================

def demonstrate_currying() -> None:
    """Demonstrate function currying."""
    logger.info("🎯 Demonstrating Function Currying")
    
    # Curry the log_security_event function
    log_caption_event = curry(log_security_event, "caption_generated")
    log_error_event = curry(log_security_event, "error_occurred")
    
    # Use curried functions
    log_caption_event("user_123", {"processing_time": 1.5})
    log_error_event("user_456", {"error": "validation_failed"})
    
    logger.info("✅ Curried functions executed")
    
    # Curry with multiple parameters
    validate_for_user = curry(validate_content_description)
    # This creates a function that takes content_description and validates it
    
    logger.info("✅ Function currying demonstrated")


# =============================================================================
# MAIN DEMO FUNCTION
# =============================================================================

def run_functional_demo() -> None:
    """Run complete functional programming demo."""
    logger.info("🚀 Starting Functional Programming Demo")
    logger.info("=" * 60)
    
    try:
        # Authentication demo
        demonstrate_authentication()
        logger.info("-" * 40)
        
        # Rate limiting demo
        demonstrate_rate_limiting()
        logger.info("-" * 40)
        
        # Input validation demo
        demonstrate_input_validation()
        logger.info("-" * 40)
        
        # Single processing demo
        demonstrate_single_processing()
        logger.info("-" * 40)
        
        # Batch processing demo
        demonstrate_batch_processing()
        logger.info("-" * 40)
        
        # Functional composition demo
        demonstrate_functional_composition()
        logger.info("-" * 40)
        
        # Higher-order functions demo
        demonstrate_higher_order_functions()
        logger.info("-" * 40)
        
        # Currying demo
        demonstrate_currying()
        logger.info("-" * 40)
        
        logger.info("✅ All functional programming demos completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_test_data() -> List[Dict[str, Any]]:
    """Create test data for demonstrations."""
    return [
        {
            "content_description": "Beautiful sunset over mountains with golden light",
            "style": "casual",
            "tone": "friendly",
            "hashtag_count": 15
        },
        {
            "content_description": "Delicious homemade pizza with melted cheese and fresh basil",
            "style": "creative",
            "tone": "enthusiastic",
            "hashtag_count": 12
        },
        {
            "content_description": "Modern office space with natural lighting and ergonomic furniture",
            "style": "professional",
            "tone": "professional",
            "hashtag_count": 8
        }
    ]


def benchmark_functional_approach() -> Dict[str, float]:
    """Benchmark functional approach performance."""
    logger.info("⚡ Benchmarking Functional Approach")
    
    test_data = create_test_data()
    user_id = "benchmark_user"
    
    start_time = time.time()
    
    # Process all test data
    results = []
    for data in test_data:
        try:
            result = process_caption_request(data, user_id)
            results.append(result)
        except Exception:
            pass
    
    total_time = time.time() - start_time
    
    metrics = {
        "total_requests": len(test_data),
        "successful_requests": len(results),
        "total_time": total_time,
        "average_time_per_request": total_time / len(test_data) if test_data else 0,
        "requests_per_second": len(test_data) / total_time if total_time > 0 else 0
    }
    
    logger.info(f"✅ Benchmark results: {metrics}")
    return metrics


if __name__ == "__main__":
    # Run the demo
    run_functional_demo()
    
    # Run benchmark
    benchmark_functional_approach()
    
    logger.info("🎉 Functional programming demo completed!") 