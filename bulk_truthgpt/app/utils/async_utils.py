"""
Advanced async utilities for Ultimate Enhanced Supreme Production system
Following Flask best practices with functional programming patterns
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Coroutine
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiohttp
import aiofiles

logger = logging.getLogger(__name__)

class AsyncManager:
    """Advanced async manager with thread and process pools."""
    
    def __init__(self, max_workers: int = 10):
        """Initialize async manager with early returns."""
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.session = None
    
    async def init_session(self) -> None:
        """Initialize aiohttp session with early returns."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self) -> None:
        """Close aiohttp session with early returns."""
        if self.session:
            await self.session.close()
            self.session = None
    
    def run_in_thread(self, func: Callable, *args, **kwargs) -> asyncio.Future:
        """Run function in thread pool with early returns."""
        if not func:
            return None
        
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
    
    def run_in_process(self, func: Callable, *args, **kwargs) -> asyncio.Future:
        """Run function in process pool with early returns."""
        if not func:
            return None
        
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self.process_pool, func, *args, **kwargs)
    
    async def fetch_url(self, url: str, method: str = 'GET', **kwargs) -> Dict[str, Any]:
        """Fetch URL asynchronously with early returns."""
        if not url:
            return {'error': 'URL is required'}
        
        await self.init_session()
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                data = await response.json()
                return {
                    'status': response.status,
                    'data': data,
                    'headers': dict(response.headers)
                }
        except Exception as e:
            logger.error(f"❌ Async fetch error: {e}")
            return {'error': str(e)}
    
    async def fetch_multiple_urls(self, urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Fetch multiple URLs concurrently with early returns."""
        if not urls:
            return []
        
        tasks = [self.fetch_url(url, **kwargs) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_batch(self, items: List[Any], process_func: Callable, 
                           batch_size: int = 10) -> List[Any]:
        """Process items in batches asynchronously with early returns."""
        if not items or not process_func:
            return []
        
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [self.run_in_thread(process_func, item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results

# Global async manager instance
async_manager = AsyncManager()

def init_async_manager(app) -> None:
    """Initialize async manager with app."""
    global async_manager
    async_manager = AsyncManager(max_workers=app.config.get('ASYNC_MAX_WORKERS', 10))
    app.logger.info("⚡ Async manager initialized")

def async_performance_monitor(func: Callable) -> Callable:
    """Decorator for async performance monitoring with early returns."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        request_id = getattr(g, 'request_id', 'unknown')
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            logger.info(f"⚡ {func.__name__} executed in {execution_time:.3f}s [req:{request_id}]")
            
            # Store metrics in request context
            if not hasattr(g, 'performance_metrics'):
                g.performance_metrics = {}
            g.performance_metrics[func.__name__] = execution_time
            
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"❌ {func.__name__} failed after {execution_time:.3f}s [req:{request_id}]: {e}")
            raise
    return wrapper

def async_error_handler(func: Callable) -> Callable:
    """Decorator for async error handling with early returns."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValueError as e:
            logger.error(f"❌ ValueError in {func.__name__}: {e}")
            return {'success': False, 'message': 'Invalid value', 'error': str(e)}
        except PermissionError as e:
            logger.error(f"❌ PermissionError in {func.__name__}: {e}")
            return {'success': False, 'message': 'Permission denied', 'error': str(e)}
        except FileNotFoundError as e:
            logger.error(f"❌ FileNotFoundError in {func.__name__}: {e}")
            return {'success': False, 'message': 'File not found', 'error': str(e)}
        except Exception as e:
            logger.error(f"❌ Unexpected error in {func.__name__}: {e}")
            return {'success': False, 'message': 'Internal server error', 'error': str(e)}
    return wrapper

def async_retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """Decorator for async retrying on failure with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (backoff_factor ** attempt)
                        logger.warning(f"🔄 Retry {attempt + 1}/{max_retries} for {func.__name__} in {wait_time:.2f}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"❌ All retries failed for {func.__name__}: {e}")
                        raise last_exception
            
            return None
        return wrapper
    return decorator

def async_timeout(seconds: float):
    """Decorator for async function timeout with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"❌ {func.__name__} timed out after {seconds} seconds")
                return {'success': False, 'message': 'Operation timed out', 'error': 'timeout'}
        return wrapper
    return decorator

def async_cache_result(ttl: int = 300, key_prefix: str = None):
    """Decorator for async caching function results with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            prefix = key_prefix or func.__name__
            cache_key = f"{prefix}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            if hasattr(current_app, 'cache'):
                cached_result = current_app.cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"📦 Cache hit for {func.__name__}")
                    return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            if hasattr(current_app, 'cache'):
                current_app.cache.set(cache_key, result, timeout=ttl)
                logger.debug(f"💾 Cached result for {func.__name__}")
            
            return result
        return wrapper
    return decorator

def async_rate_limit(requests_per_minute: int = 60, key_func: Callable = None):
    """Decorator for async rate limiting with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate rate limit key
            if key_func:
                rate_key = key_func()
            else:
                client_ip = request.remote_addr
                rate_key = f"rate_limit:{client_ip}:{func.__name__}"
            
            # Check rate limit
            if hasattr(current_app, 'cache'):
                current_requests = current_app.cache.get(rate_key) or 0
                if current_requests >= requests_per_minute:
                    logger.warning(f"🚫 Rate limit exceeded for {rate_key}")
                    return {'success': False, 'message': 'Rate limit exceeded', 'error_type': 'rate_limit_exceeded'}
                
                # Increment counter
                current_app.cache.set(rate_key, current_requests + 1, timeout=60)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

async def run_concurrent_tasks(tasks: List[Coroutine], max_concurrent: int = 10) -> List[Any]:
    """Run concurrent tasks with early returns."""
    if not tasks:
        return []
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_task(task: Coroutine) -> Any:
        async with semaphore:
            return await task
    
    return await asyncio.gather(*[run_task(task) for task in tasks], return_exceptions=True)

async def run_sequential_tasks(tasks: List[Coroutine]) -> List[Any]:
    """Run tasks sequentially with early returns."""
    if not tasks:
        return []
    
    results = []
    for task in tasks:
        try:
            result = await task
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Task failed: {e}")
            results.append({'error': str(e)})
    
    return results

async def run_batch_processing(items: List[Any], process_func: Callable, 
                              batch_size: int = 10, max_concurrent: int = 5) -> List[Any]:
    """Run batch processing with early returns."""
    if not items or not process_func:
        return []
    
    # Create batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    # Process batches concurrently
    batch_tasks = []
    for batch in batches:
        task = asyncio.create_task(process_batch_async(batch, process_func))
        batch_tasks.append(task)
    
    # Limit concurrent batches
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_batch_task(task: asyncio.Task) -> Any:
        async with semaphore:
            return await task
    
    results = await asyncio.gather(*[run_batch_task(task) for task in batch_tasks], return_exceptions=True)
    
    # Flatten results
    flattened_results = []
    for result in results:
        if isinstance(result, list):
            flattened_results.extend(result)
        else:
            flattened_results.append(result)
    
    return flattened_results

async def process_batch_async(batch: List[Any], process_func: Callable) -> List[Any]:
    """Process batch asynchronously with early returns."""
    if not batch or not process_func:
        return []
    
    tasks = [async_manager.run_in_thread(process_func, item) for item in batch]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def fetch_data_async(url: str, **kwargs) -> Dict[str, Any]:
    """Fetch data asynchronously with early returns."""
    return await async_manager.fetch_url(url, **kwargs)

async def fetch_multiple_data_async(urls: List[str], **kwargs) -> List[Dict[str, Any]]:
    """Fetch multiple data sources asynchronously with early returns."""
    return await async_manager.fetch_multiple_urls(urls, **kwargs)

async def process_file_async(file_path: str, process_func: Callable) -> Any:
    """Process file asynchronously with early returns."""
    if not file_path or not process_func:
        return None
    
    try:
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            return await async_manager.run_in_thread(process_func, content)
    except Exception as e:
        logger.error(f"❌ File processing error: {e}")
        return None

async def save_file_async(file_path: str, content: str) -> bool:
    """Save file asynchronously with early returns."""
    if not file_path or not content:
        return False
    
    try:
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(content)
            return True
    except Exception as e:
        logger.error(f"❌ File save error: {e}")
        return False

async def run_periodic_task(task_func: Callable, interval: float, 
                           stop_event: asyncio.Event = None) -> None:
    """Run periodic task asynchronously with early returns."""
    if not task_func or interval <= 0:
        return
    
    while not (stop_event and stop_event.is_set()):
        try:
            await task_func()
            await asyncio.sleep(interval)
        except Exception as e:
            logger.error(f"❌ Periodic task error: {e}")
            await asyncio.sleep(interval)

async def run_health_check_async(health_funcs: List[Callable]) -> Dict[str, Any]:
    """Run health checks asynchronously with early returns."""
    if not health_funcs:
        return {'status': 'healthy', 'checks': []}
    
    tasks = [asyncio.create_task(run_single_health_check(func)) for func in health_funcs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    checks = []
    all_healthy = True
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            checks.append({
                'name': health_funcs[i].__name__,
                'status': 'unhealthy',
                'error': str(result)
            })
            all_healthy = False
        else:
            checks.append({
                'name': health_funcs[i].__name__,
                'status': 'healthy',
                'result': result
            })
    
    return {
        'status': 'healthy' if all_healthy else 'unhealthy',
        'checks': checks,
        'timestamp': time.time()
    }

async def run_single_health_check(health_func: Callable) -> Any:
    """Run single health check asynchronously with early returns."""
    if not health_func:
        return None
    
    try:
        if asyncio.iscoroutinefunction(health_func):
            return await health_func()
        else:
            return await async_manager.run_in_thread(health_func)
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        raise

# Utility functions
def create_async_task(func: Callable, *args, **kwargs) -> asyncio.Task:
    """Create async task with early returns."""
    if not func:
        return None
    
    return asyncio.create_task(func(*args, **kwargs))

def run_sync_in_async(func: Callable, *args, **kwargs) -> asyncio.Future:
    """Run sync function in async context with early returns."""
    if not func:
        return None
    
    return async_manager.run_in_thread(func, *args, **kwargs)

def run_cpu_intensive_in_async(func: Callable, *args, **kwargs) -> asyncio.Future:
    """Run CPU intensive function in process pool with early returns."""
    if not func:
        return None
    
    return async_manager.run_in_process(func, *args, **kwargs)

async def cleanup_async_resources() -> None:
    """Cleanup async resources with early returns."""
    await async_manager.close_session()
    logger.info("🧹 Async resources cleaned up")

# Async context managers
class AsyncContextManager:
    """Async context manager for resource management."""
    
    def __init__(self, resource_func: Callable, cleanup_func: Callable = None):
        """Initialize async context manager with early returns."""
        self.resource_func = resource_func
        self.cleanup_func = cleanup_func
        self.resource = None
    
    async def __aenter__(self):
        """Enter async context with early returns."""
        if self.resource_func:
            self.resource = await self.resource_func()
        return self.resource
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context with early returns."""
        if self.cleanup_func and self.resource:
            await self.cleanup_func(self.resource)

def async_context_manager(resource_func: Callable, cleanup_func: Callable = None):
    """Create async context manager with early returns."""
    return AsyncContextManager(resource_func, cleanup_func)









