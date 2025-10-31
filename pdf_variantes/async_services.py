"""Enhanced async service layer with functional patterns."""

from typing import Dict, Any, List, Optional, Callable, Awaitable
from functools import wraps, partial
import asyncio
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def async_timer(func: Callable) -> Callable:
    """Decorator to measure async function execution time."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
        return result
    return wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay}s")
                        await asyncio.sleep(delay)
                        delay *= 2  # Exponential backoff
            
            logger.error(f"All {max_retries} attempts failed")
            raise last_exception
        return wrapper
    return decorator


async def process_pdf_pipeline(
    file_content: bytes,
    filename: str,
    processors: List[Callable[[bytes, str], Awaitable[Dict[str, Any]]]],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process PDF through a pipeline of async processors."""
    options = options or {}
    results = {"file_id": None, "processing_steps": []}
    
    current_content = file_content
    current_filename = filename
    
    for i, processor in enumerate(processors):
        try:
            step_result = await processor(current_content, current_filename)
            results["processing_steps"].append({
                "step": i + 1,
                "processor": processor.__name__,
                "result": step_result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Update content for next processor if needed
            if "processed_content" in step_result:
                current_content = step_result["processed_content"]
            
        except Exception as e:
            logger.error(f"Processor {processor.__name__} failed: {e}")
            results["processing_steps"].append({
                "step": i + 1,
                "processor": processor.__name__,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            break
    
    return results


async def batch_process_with_concurrency(
    items: List[Any],
    processor: Callable[[Any], Awaitable[Any]],
    max_concurrent: int = 5,
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """Process items in batches with controlled concurrency."""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_item(item):
            async with semaphore:
                try:
                    result = await processor(item)
                    return {"item": item, "result": result, "status": "success"}
                except Exception as e:
                    return {"item": item, "error": str(e), "status": "failed"}
        
        batch_results = await asyncio.gather(
            *[process_item(item) for item in batch],
            return_exceptions=True
        )
        
        results.extend(batch_results)
    
    return results


def create_async_cache(
    ttl_seconds: int = 3600,
    max_size: int = 1000
) -> Dict[str, Any]:
    """Create an async cache with TTL and size limits."""
    cache = {
        "_data": {},
        "_timestamps": {},
        "_ttl": ttl_seconds,
        "_max_size": max_size
    }
    
    async def get(key: str) -> Optional[Any]:
        if key not in cache["_data"]:
            return None
        
        # Check TTL
        if time.time() - cache["_timestamps"][key] > cache["_ttl"]:
            del cache["_data"][key]
            del cache["_timestamps"][key]
            return None
        
        return cache["_data"][key]
    
    async def set(key: str, value: Any) -> None:
        # Evict oldest entries if cache is full
        if len(cache["_data"]) >= cache["_max_size"]:
            oldest_key = min(cache["_timestamps"], key=cache["_timestamps"].get)
            del cache["_data"][oldest_key]
            del cache["_timestamps"][oldest_key]
        
        cache["_data"][key] = value
        cache["_timestamps"][key] = time.time()
    
    async def clear() -> None:
        cache["_data"].clear()
        cache["_timestamps"].clear()
    
    cache["get"] = get
    cache["set"] = set
    cache["clear"] = clear
    
    return cache


async def parallel_extract_features(
    file_id: str,
    extractors: Dict[str, Callable[[str], Awaitable[Any]]]
) -> Dict[str, Any]:
    """Extract multiple features in parallel."""
    tasks = {
        name: asyncio.create_task(extractor(file_id))
        for name, extractor in extractors.items()
    }
    
    results = {}
    for name, task in tasks.items():
        try:
            results[name] = await task
        except Exception as e:
            logger.error(f"Feature extraction failed for {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


def create_processing_pipeline(
    steps: List[Callable[[Any], Awaitable[Any]]],
    error_handler: Optional[Callable[[Exception], Any]] = None
) -> Callable[[Any], Awaitable[Any]]:
    """Create a processing pipeline from a list of steps."""
    async def pipeline(input_data: Any) -> Any:
        current_data = input_data
        
        for step in steps:
            try:
                current_data = await step(current_data)
            except Exception as e:
                if error_handler:
                    return error_handler(e)
                raise e
        
        return current_data
    
    return pipeline


async def with_timeout(
    coro: Awaitable[Any],
    timeout_seconds: float = 30.0,
    default_value: Any = None
) -> Any:
    """Execute coroutine with timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout_seconds}s")
        return default_value


def create_rate_limiter(
    max_requests: int = 100,
    window_seconds: int = 60
) -> Callable[[str], bool]:
    """Create a rate limiter function."""
    request_counts = {}
    
    def is_allowed(client_id: str) -> bool:
        current_time = time.time()
        
        # Clean old entries
        request_counts[client_id] = [
            req_time for req_time in request_counts.get(client_id, [])
            if current_time - req_time < window_seconds
        ]
        
        # Check if limit exceeded
        if len(request_counts[client_id]) >= max_requests:
            return False
        
        # Add current request
        request_counts[client_id].append(current_time)
        return True
    
    return is_allowed


async def health_check_services(
    services: Dict[str, Callable[[], Awaitable[bool]]]
) -> Dict[str, Any]:
    """Check health of multiple services."""
    health_results = {}
    
    for service_name, health_check in services.items():
        try:
            is_healthy = await with_timeout(health_check(), timeout_seconds=5.0)
            health_results[service_name] = {
                "status": "healthy" if is_healthy else "unhealthy",
                "checked_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            health_results[service_name] = {
                "status": "error",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
    
    return health_results
