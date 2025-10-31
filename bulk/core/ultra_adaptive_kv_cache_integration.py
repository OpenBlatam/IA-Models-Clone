"""
Integration Helpers for Ultra-Adaptive K/V Cache Engine
Provides integration patterns for common frameworks and use cases
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
import logging

try:
    from fastapi import Request, HTTPException
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from ultra_adaptive_kv_cache_engine import (
        UltraAdaptiveKVCacheEngine,
        AdaptiveConfig,
        TruthGPTIntegration
    )
except ImportError:
    UltraAdaptiveKVCacheEngine = None

logger = logging.getLogger(__name__)


class FastAPIIntegration:
    """Integration helpers for FastAPI."""
    
    def __init__(self, engine: UltraAdaptiveKVCacheEngine):
        self.engine = engine
    
    @staticmethod
    def create_middleware(engine: UltraAdaptiveKVCacheEngine):
        """
        Create FastAPI middleware for automatic request processing.
        
        Usage:
            app.add_middleware(FastAPIIntegration.create_middleware(engine))
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is not available")
        
        from starlette.middleware.base import BaseHTTPMiddleware
        
        class EngineMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, engine_instance):
                super().__init__(app)
                self.engine = engine_instance
            
            async def dispatch(self, request: Request, call_next):
                # Add engine to request state
                request.state.engine = self.engine
                
                # Process request
                response = await call_next(request)
                
                return response
        
        return lambda app: EngineMiddleware(app, engine)
    
    @staticmethod
    def create_endpoint_handler(engine: UltraAdaptiveKVCacheEngine):
        """
        Create FastAPI endpoint handler.
        
        Usage:
            @app.post("/process")
            async def process(request: ProcessRequest, engine = Depends(get_engine)):
                return await FastAPIIntegration.create_endpoint_handler(engine)(request)
        """
        async def handler(request_data: Dict[str, Any]) -> Dict[str, Any]:
            try:
                result = await engine.process_request(request_data.dict() if hasattr(request_data, 'dict') else request_data)
                
                if result['success']:
                    return {
                        'success': True,
                        'data': result['response'],
                        'metadata': {
                            'processing_time': result['processing_time'],
                            'session_id': result.get('session_id'),
                            'gpu_used': result.get('gpu_used')
                        }
                    }
                else:
                    raise HTTPException(status_code=500, detail=result.get('error', 'Processing failed'))
            
            except Exception as e:
                logger.error(f"Error in endpoint handler: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return handler


class AsyncTaskQueue:
    """Async task queue for batch processing."""
    
    def __init__(self, engine: UltraAdaptiveKVCacheEngine, max_batch_size: int = 20, 
                 max_wait_time: float = 1.0):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue = asyncio.Queue()
        self.processing = False
    
    async def add_request(self, request: Dict[str, Any]) -> asyncio.Future:
        """
        Add request to queue and return future.
        
        Args:
            request: Request dictionary
            
        Returns:
            Future that will resolve with result
        """
        future = asyncio.Future()
        await self.queue.put((request, future))
        return future
    
    async def start_processing(self):
        """Start background processing loop."""
        self.processing = True
        
        while self.processing:
            try:
                batch = []
                batch_futures = []
                
                # Collect batch
                deadline = asyncio.get_event_loop().time() + self.max_wait_time
                
                while len(batch) < self.max_batch_size:
                    try:
                        timeout = max(0, deadline - asyncio.get_event_loop().time())
                        if timeout <= 0:
                            break
                        
                        request, future = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        batch.append(request)
                        batch_futures.append(future)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    # Process batch
                    results = await self.engine.process_batch(batch)
                    
                    # Resolve futures
                    for i, result in enumerate(results):
                        if i < len(batch_futures):
                            if not batch_futures[i].done():
                                batch_futures[i].set_result(result)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(0.1)
    
    def stop_processing(self):
        """Stop background processing."""
        self.processing = False


class CircuitBreakerIntegration:
    """Circuit breaker pattern for engine operations."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
        self.half_open_calls = 0
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        # Check state
        if self.state == 'open':
            if self.last_failure_time and \
               (asyncio.get_event_loop().time() - self.last_failure_time) > self.recovery_timeout:
                # Try to recover
                self.state = 'half_open'
                self.half_open_calls = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Success - reset on success
            if self.state == 'half_open':
                self.half_open_calls += 1
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = 'closed'
                    self.failure_count = 0
            
            if self.state == 'closed':
                self.failure_count = 0
            
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = asyncio.get_event_loop().time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
            
            raise e


def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0, 
                       max_delay: float = 60.0, exponential_base: float = 2.0):
    """
    Decorator for retrying operations with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Exponential base for backoff
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e
                    
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
            
            return None
        
        return wrapper
    return decorator


class RateLimiter:
    """Simple rate limiter for request processing."""
    
    def __init__(self, max_requests: int, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def acquire(self):
        """Acquire permission to process a request."""
        now = asyncio.get_event_loop().time()
        
        # Remove old requests
        self.requests = [r for r in self.requests if now - r < self.time_window]
        
        # Check if we can proceed
        if len(self.requests) >= self.max_requests:
            # Calculate wait time
            oldest_request = min(self.requests)
            wait_time = self.time_window - (now - oldest_request)
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                now = asyncio.get_event_loop().time()
                self.requests = [r for r in self.requests if now - r < self.time_window]
        
        # Record this request
        self.requests.append(now)


# Example usage patterns

async def example_fastapi_integration():
    """Example FastAPI integration."""
    if not FASTAPI_AVAILABLE or not UltraAdaptiveKVCacheEngine:
        logger.warning("FastAPI or engine not available")
        return
    
    # Create engine
    engine = TruthGPTIntegration.create_engine_for_truthgpt()
    
    # Create integration
    integration = FastAPIIntegration(engine)
    
    # Use in FastAPI app
    # app.add_middleware(integration.create_middleware(engine))
    
    logger.info("FastAPI integration example ready")


async def example_async_queue():
    """Example async queue usage."""
    if not UltraAdaptiveKVCacheEngine:
        logger.warning("Engine not available")
        return
    
    engine = TruthGPTIntegration.create_engine_for_truthgpt()
    queue = AsyncTaskQueue(engine, max_batch_size=10)
    
    # Start processing
    processing_task = asyncio.create_task(queue.start_processing())
    
    # Add requests
    futures = []
    for i in range(20):
        future = await queue.add_request({
            'text': f'Request {i}',
            'max_length': 50,
            'temperature': 0.7,
            'session_id': 'test'
        })
        futures.append(future)
    
    # Wait for results
    results = await asyncio.gather(*futures)
    
    # Stop processing
    queue.stop_processing()
    processing_task.cancel()
    
    logger.info(f"Processed {len(results)} requests via queue")


async def example_circuit_breaker():
    """Example circuit breaker usage."""
    if not UltraAdaptiveKVCacheEngine:
        logger.warning("Engine not available")
        return
    
    engine = TruthGPTIntegration.create_engine_for_truthgpt()
    breaker = CircuitBreakerIntegration(failure_threshold=3, recovery_timeout=30.0)
    
    @retry_with_backoff(max_retries=3)
    async def process_with_protection(request):
        return await breaker.call(engine.process_request, request)
    
    # Use it
    try:
        result = await process_with_protection({
            'text': 'Test',
            'max_length': 50,
            'temperature': 0.7
        })
        logger.info("Processed with circuit breaker protection")
    except Exception as e:
        logger.error(f"Circuit breaker prevented processing: {e}")


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_async_queue())

