"""
Model Serving Optimizations for Production
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
import logging
import time
from functools import wraps
import threading
from queue import Queue

logger = logging.getLogger(__name__)


class ModelServer:
    """Optimized model server for production"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        max_batch_size: int = 32,
        max_queue_size: int = 100,
        num_workers: int = 2
    ):
        """
        Initialize model server
        
        Args:
            model: Model to serve
            device: Device to use
            max_batch_size: Maximum batch size
            max_queue_size: Maximum queue size
            num_workers: Number of worker threads
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers
        self.request_queue = Queue(maxsize=max_queue_size)
        self.response_dict = {}
        self.workers = []
        self.running = False
        
        logger.info(f"ModelServer initialized on {self.device}")
    
    def start(self):
        """Start server workers"""
        if self.running:
            return
        
        self.running = True
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"ModelServer started with {self.num_workers} workers")
    
    def _worker(self):
        """Worker thread for processing requests"""
        while self.running:
            try:
                request_id, batch = self.request_queue.get(timeout=1.0)
                
                # Process batch
                with torch.no_grad():
                    outputs = self.model(batch.to(self.device))
                
                # Store result
                self.response_dict[request_id] = outputs.cpu()
                self.request_queue.task_done()
            except:
                continue
    
    def predict(self, inputs: torch.Tensor, timeout: float = 10.0) -> torch.Tensor:
        """
        Predict with timeout
        
        Args:
            inputs: Input tensor
            timeout: Timeout in seconds
        
        Returns:
            Prediction result
        """
        request_id = f"{time.time()}_{id(inputs)}"
        
        # Add to queue
        self.request_queue.put((request_id, inputs))
        
        # Wait for result
        start_time = time.time()
        while request_id not in self.response_dict:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Prediction timeout after {timeout}s")
            time.sleep(0.001)
        
        result = self.response_dict.pop(request_id)
        return result
    
    def predict_batch(self, inputs_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Predict batch of inputs"""
        # Batch inputs
        batch = torch.cat(inputs_list, dim=0)
        
        # Split into chunks
        results = []
        for i in range(0, len(batch), self.max_batch_size):
            chunk = batch[i:i + self.max_batch_size]
            output = self.predict(chunk)
            results.append(output)
        
        return torch.cat(results, dim=0)
    
    def stop(self):
        """Stop server"""
        self.running = False
        for worker in self.workers:
            worker.join()
        logger.info("ModelServer stopped")


class ModelCache:
    """Intelligent model output caching"""
    
    def __init__(self, max_size: int = 10000, ttl: float = 3600.0):
        """
        Initialize model cache
        
        Args:
            max_size: Maximum cache size
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired"""
        if key not in self.creation_times:
            return True
        
        age = time.time() - self.creation_times[key]
        return age > self.ttl
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times, key=self.access_times.get)
        del self.cache[lru_key]
        del self.access_times[lru_key]
        del self.creation_times[lru_key]
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get cached result"""
        if key not in self.cache or self._is_expired(key):
            return None
        
        self.access_times[key] = time.time()
        return self.cache[key]
    
    def set(self, key: str, value: torch.Tensor):
        """Set cached result"""
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.creation_times[key] = time.time()
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_times.clear()
        self.creation_times.clear()


class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, max_requests: int = 100, window: float = 60.0):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests per window
            window: Time window in seconds
        """
        self.max_requests = max_requests
        self.window = window
        self.requests = []
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        # Remove old requests
        self.requests = [t for t in self.requests if now - t < self.window]
        
        if len(self.requests) >= self.max_requests:
            return False
        
        self.requests.append(now)
        return True
    
    def get_remaining(self) -> int:
        """Get remaining requests"""
        now = time.time()
        self.requests = [t for t in self.requests if now - t < self.window]
        return max(0, self.max_requests - len(self.requests))


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(delay * (attempt + 1))
        return wrapper
    return decorator


class HealthMonitor:
    """Health monitoring for model serving"""
    
    def __init__(self):
        """Initialize health monitor"""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_latency": 0.0,
            "p95_latency": 0.0,
            "p99_latency": 0.0
        }
        self.latencies = []
    
    def record_request(self, success: bool, latency: float):
        """Record request"""
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        self.latencies.append(latency)
        
        # Keep only last 1000
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]
        
        # Update percentiles
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            self.metrics["avg_latency"] = sum(sorted_latencies) / len(sorted_latencies)
            self.metrics["p95_latency"] = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            self.metrics["p99_latency"] = sorted_latencies[int(len(sorted_latencies) * 0.99)]
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status"""
        success_rate = (
            self.metrics["successful_requests"] / self.metrics["total_requests"]
            if self.metrics["total_requests"] > 0 else 0.0
        )
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "is_healthy": success_rate > 0.95 and self.metrics["avg_latency"] < 100.0
        }

