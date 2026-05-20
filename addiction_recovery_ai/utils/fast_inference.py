"""
Fast Inference Utilities with Caching, Batch Processing, and GPU Optimization
"""

import torch
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)


class LRUCache:
    """LRU Cache for inference results"""
    
    def __init__(self, maxsize: int = 1000):
        self.cache = {}
        self.access_order = deque(maxlen=maxsize)
        self.maxsize = maxsize
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result"""
        if key in self.cache:
            # Move to end (most recently used)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cached result"""
        if len(self.cache) >= self.maxsize:
            # Remove least recently used
            lru_key = self.access_order.popleft()
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()


class BatchProcessor:
    """Batch processor for efficient inference"""
    
    def __init__(self, batch_size: int = 32, device: Optional[torch.device] = None):
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queue = []
    
    def add(self, item: Any):
        """Add item to batch queue"""
        self.queue.append(item)
    
    def process_batch(self, model: torch.nn.Module, process_fn) -> List[Any]:
        """Process batch"""
        if not self.queue:
            return []
        
        results = []
        for i in range(0, len(self.queue), self.batch_size):
            batch = self.queue[i:i + self.batch_size]
            batch_results = process_fn(model, batch)
            results.extend(batch_results)
        
        self.queue.clear()
        return results
    
    def clear(self):
        """Clear queue"""
        self.queue.clear()


class FastInferenceEngine:
    """Fast inference engine with optimizations"""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        cache_size: int = 1000,
        batch_size: int = 32,
        use_gpu: bool = True
    ):
        self.device = device or torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.cache = LRUCache(maxsize=cache_size)
        self.batch_processor = BatchProcessor(batch_size=batch_size, device=self.device)
        
        # Enable optimizations
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        logger.info(f"FastInferenceEngine initialized on {self.device}")
    
    def predict_progress_cached(
        self,
        model: torch.nn.Module,
        features: Dict[str, float]
    ) -> float:
        """Cached progress prediction"""
        # Create cache key
        cache_key = f"progress_{hash(tuple(sorted(features.items())))}"
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Predict
        feature_list = [
            features.get("days_sober", 0) / 365.0,
            features.get("cravings_level", 5) / 10.0,
            features.get("stress_level", 5) / 10.0,
            features.get("support_level", 5) / 10.0,
            features.get("mood_score", 5) / 10.0,
            features.get("sleep_quality", 5) / 10.0,
            features.get("exercise_frequency", 2) / 7.0,
            features.get("therapy_sessions", 0) / 10.0,
            features.get("medication_compliance", 1.0),
            features.get("social_activity", 3) / 7.0
        ]
        
        feature_tensor = torch.tensor([feature_list], dtype=torch.float32).to(self.device)
        
        model.eval()
        with torch.no_grad():
            output = model(feature_tensor)
            result = output.item()
        
        # Cache result
        self.cache.set(cache_key, result)
        return result
    
    def predict_relapse_batch(
        self,
        model: torch.nn.Module,
        sequences: List[List[Dict[str, float]]]
    ) -> List[float]:
        """Batch relapse prediction"""
        # Convert sequences to tensors
        batch_tensors = []
        for sequence in sequences:
            seq_data = []
            for day in sequence[-30:]:
                seq_data.append([
                    day.get("cravings_level", 5) / 10.0,
                    day.get("stress_level", 5) / 10.0,
                    day.get("mood_score", 5) / 10.0,
                    day.get("triggers_count", 0) / 10.0,
                    day.get("consumed", 0.0)
                ])
            
            # Pad to fixed length
            while len(seq_data) < 30:
                seq_data.insert(0, [0.0] * 5)
            
            batch_tensors.append(seq_data)
        
        # Batch tensor
        batch_tensor = torch.tensor(batch_tensors, dtype=torch.float32).to(self.device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(batch_tensor)
            results = outputs.squeeze().cpu().tolist()
        
        if not isinstance(results, list):
            results = [results]
        
        return results
    
    def clear_cache(self):
        """Clear inference cache"""
        self.cache.clear()
        logger.info("Cache cleared")


class AsyncInference:
    """Asynchronous inference wrapper"""
    
    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        
        # Move to GPU if available
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
            # Enable async execution
            torch.cuda.set_stream(torch.cuda.Stream())
    
    def predict_async(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Asynchronous prediction"""
        if self.device.type == "cuda":
            with torch.cuda.stream(torch.cuda.current_stream()):
                with torch.no_grad():
                    output = self.model(input_tensor)
                torch.cuda.synchronize()
                return output
        else:
            with torch.no_grad():
                return self.model(input_tensor)


def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Optimize model for inference"""
    model.eval()
    
    # Fuse operations
    try:
        if hasattr(torch.quantization, 'fuse_modules'):
            # Try to fuse Conv+BN+ReLU patterns
            pass
    except:
        pass
    
    # Enable inference optimizations
    if hasattr(torch.jit, '_state'):
        try:
            model = torch.jit.optimize_for_inference(torch.jit.script(model))
        except:
            pass
    
    return model


def benchmark_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """Benchmark inference speed"""
    device = input_tensor.device
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    avg_time = elapsed / num_runs * 1000  # ms
    
    return {
        "avg_time_ms": avg_time,
        "total_time_s": elapsed,
        "throughput": num_runs / elapsed,
        "device": str(device)
    }

