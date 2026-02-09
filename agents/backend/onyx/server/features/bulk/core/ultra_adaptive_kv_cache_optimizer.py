"""
Advanced Optimizations for Ultra-Adaptive K/V Cache Engine
Performance optimizations, memory management, and intelligent caching strategies
"""

import time
import gc
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict, deque
from dataclasses import dataclass
import logging
import threading
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    ADAPTIVE = "adaptive"  # Adaptive based on patterns
    SIZE_BASED = "size_based"  # Evict largest entries first


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    access_count: int = 0
    last_access: float = 0
    size_bytes: int = 0
    created_at: float = 0
    
    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = time.time()
        if self.last_access == 0:
            self.last_access = time.time()


class IntelligentCache:
    """Intelligent cache with multiple eviction policies."""
    
    def __init__(self, max_size: int = 1000, policy: CachePolicy = CachePolicy.ADAPTIVE):
        self.max_size = max_size
        self.policy = policy
        self.cache: Dict[str, CacheEntry] = OrderedDict()
        self.access_history = deque(maxlen=1000)
        self.eviction_stats = {
            'total_evictions': 0,
            'policy_evictions': {}
        }
        
        # Adaptive policy state
        self.adaptive_weights = {
            'recency': 0.5,
            'frequency': 0.3,
            'size': 0.2
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        entry.access_count += 1
        entry.last_access = time.time()
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        
        # Track access
        self.access_history.append({
            'key': key,
            'timestamp': time.time(),
            'policy': self.policy.value
        })
        
        return entry.value
    
    def set(self, key: str, value: Any, size_bytes: int = 0):
        """Set value in cache."""
        # Check if we need to evict
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict()
        
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes,
            created_at=time.time(),
            last_access=time.time()
        )
        
        self.cache[key] = entry
        self.cache.move_to_end(key)
    
    def _evict(self):
        """Evict entry based on policy."""
        if not self.cache:
            return
        
        key_to_evict = None
        
        if self.policy == CachePolicy.LRU:
            # Evict least recently used (first item)
            key_to_evict = next(iter(self.cache))
        
        elif self.policy == CachePolicy.LFU:
            # Evict least frequently used
            key_to_evict = min(
                self.cache.items(),
                key=lambda x: x[1].access_count
            )[0]
        
        elif self.policy == CachePolicy.FIFO:
            # Evict oldest (first item)
            key_to_evict = next(iter(self.cache))
        
        elif self.policy == CachePolicy.SIZE_BASED:
            # Evict largest entry
            key_to_evict = max(
                self.cache.items(),
                key=lambda x: x[1].size_bytes
            )[0]
        
        elif self.policy == CachePolicy.ADAPTIVE:
            # Use weighted score
            key_to_evict = self._adaptive_evict()
        
        if key_to_evict:
            del self.cache[key_to_evict]
            self.eviction_stats['total_evictions'] += 1
            policy_key = self.policy.value
            self.eviction_stats['policy_evictions'][policy_key] = \
                self.eviction_stats['policy_evictions'].get(policy_key, 0) + 1
    
    def _adaptive_evict(self) -> Optional[str]:
        """Adaptive eviction based on multiple factors."""
        if not self.cache:
            return None
        
        current_time = time.time()
        scores = {}
        
        for key, entry in self.cache.items():
            # Recency score (higher = more recent = should keep)
            recency_score = 1.0 - ((current_time - entry.last_access) / 3600.0)
            recency_score = max(0, min(1, recency_score))
            
            # Frequency score (normalized)
            max_access = max(e.access_count for e in self.cache.values())
            frequency_score = entry.access_count / max(max_access, 1)
            
            # Size score (larger = more expensive to keep)
            max_size = max(e.size_bytes for e in self.cache.values())
            size_score = entry.size_bytes / max(max_size, 1)
            
            # Combined score (lower = should evict)
            score = (
                self.adaptive_weights['recency'] * (1 - recency_score) +
                self.adaptive_weights['frequency'] * (1 - frequency_score) +
                self.adaptive_weights['size'] * size_score
            )
            
            scores[key] = score
        
        # Evict highest score (worst candidate to keep)
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.access_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {
                'size': 0,
                'max_size': self.max_size,
                'hit_rate': 0.0,
                'evictions': self.eviction_stats
            }
        
        total_accesses = sum(e.access_count for e in self.cache.values())
        avg_accesses = total_accesses / len(self.cache)
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'policy': self.policy.value,
            'total_accesses': total_accesses,
            'avg_accesses_per_entry': avg_accesses,
            'evictions': self.eviction_stats,
            'memory_estimate': sum(e.size_bytes for e in self.cache.values())
        }


class MemoryOptimizer:
    """Advanced memory optimization utilities."""
    
    @staticmethod
    def clear_cuda_cache():
        """Clear CUDA cache if available."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("CUDA cache cleared")
    
    @staticmethod
    def force_gc():
        """Force garbage collection."""
        collected = gc.collect()
        logger.debug(f"Garbage collection: {collected} objects collected")
        return collected
    
    @staticmethod
    def optimize_memory():
        """Perform comprehensive memory optimization."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            MemoryOptimizer.clear_cuda_cache()
        
        collected = MemoryOptimizer.force_gc()
        
        return {
            'gc_collected': collected,
            'cuda_cache_cleared': TORCH_AVAILABLE and torch.cuda.is_available()
        }
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get detailed memory information."""
        info = {}
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
                max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)  # GB
                
                info[f'gpu_{i}'] = {
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'max_allocated_gb': max_allocated
                }
        
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            
            info['system'] = {
                'rss_gb': mem_info.rss / (1024 ** 3),
                'vms_gb': mem_info.vms / (1024 ** 3),
                'percent': process.memory_percent()
            }
        except ImportError:
            pass
        
        return info


from contextlib import contextmanager


class PerformanceProfiler:
    """Performance profiling and analysis."""
    
    def __init__(self):
        self.profiles: Dict[str, List[float]] = {}
        self.lock = threading.Lock()
    
    def start_profile(self, name: str):
        """Start profiling a section."""
        if name not in self.profiles:
            self.profiles[name] = []
    
    def end_profile(self, name: str, duration: float):
        """End profiling and record duration."""
        with self.lock:
            if name not in self.profiles:
                self.profiles[name] = []
            self.profiles[name].append(duration)
    
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling."""
        start = time.time()
        self.start_profile(name)
        try:
            yield
        finally:
            duration = time.time() - start
            self.end_profile(name, duration)
    
    def get_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a profile."""
        if name not in self.profiles or not self.profiles[name]:
            return {}
        
        durations = self.profiles[name]
        durations.sort()
        
        return {
            'count': len(durations),
            'total': sum(durations),
            'avg': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations),
            'p50': durations[int(len(durations) * 0.50)],
            'p95': durations[int(len(durations) * 0.95)],
            'p99': durations[int(len(durations) * 0.99)]
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all profiles."""
        return {name: self.get_stats(name) for name in self.profiles}


class RequestPredictor:
    """Predict future requests based on patterns."""
    
    def __init__(self, history_window: int = 100):
        self.history_window = history_window
        self.request_history = deque(maxlen=history_window)
        self.pattern_cache = {}
    
    def record_request(self, request: Dict[str, Any]):
        """Record a request for pattern analysis."""
        key = self._extract_key(request)
        self.request_history.append({
            'key': key,
            'timestamp': time.time(),
            'request': request
        })
    
    def _extract_key(self, request: Dict[str, Any]) -> str:
        """Extract key pattern from request."""
        return f"{request.get('session_id', '')}:{len(request.get('text', ''))}"
    
    def predict_next(self, current_request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict likely next requests."""
        if len(self.request_history) < 2:
            return []
        
        # Simple pattern matching
        current_key = self._extract_key(current_request)
        predictions = []
        
        # Find similar patterns in history
        for i, hist_entry in enumerate(self.request_history):
            if hist_entry['key'] == current_key and i < len(self.request_history) - 1:
                next_entry = self.request_history[i + 1]
                predictions.append(next_entry['request'])
        
        # Return most frequent predictions
        from collections import Counter
        if predictions:
            counter = Counter(str(p) for p in predictions)
            most_common = counter.most_common(3)
            return [eval(p) for p, _ in most_common]
        
        return []


class AdaptiveLoadBalancer:
    """Intelligent load balancing across resources."""
    
    def __init__(self):
        self.resource_loads: Dict[str, float] = {}
        self.resource_capacities: Dict[str, float] = {}
        self.allocation_history = deque(maxlen=100)
    
    def register_resource(self, resource_id: str, capacity: float = 1.0):
        """Register a resource with capacity."""
        self.resource_capacities[resource_id] = capacity
        if resource_id not in self.resource_loads:
            self.resource_loads[resource_id] = 0.0
    
    def update_load(self, resource_id: str, load: float):
        """Update load for a resource."""
        if resource_id in self.resource_loads:
            self.resource_loads[resource_id] = load
    
    def select_resource(self, task_complexity: float = 1.0) -> Optional[str]:
        """Select best resource for a task."""
        if not self.resource_loads:
            return None
        
        best_resource = None
        best_score = float('-inf')
        
        for resource_id, current_load in self.resource_loads.items():
            capacity = self.resource_capacities.get(resource_id, 1.0)
            available_capacity = capacity - current_load
            
            # Score based on available capacity and task fit
            score = available_capacity - (task_complexity * 0.1)
            
            if score > best_score and available_capacity > 0:
                best_score = score
                best_resource = resource_id
        
        if best_resource:
            # Record allocation
            self.allocation_history.append({
                'resource': best_resource,
                'complexity': task_complexity,
                'timestamp': time.time()
            })
        
        return best_resource
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get statistics about resource usage."""
        stats = {}
        
        for resource_id in self.resource_loads:
            capacity = self.resource_capacities.get(resource_id, 1.0)
            load = self.resource_loads[resource_id]
            utilization = (load / capacity) * 100 if capacity > 0 else 0
            
            stats[resource_id] = {
                'load': load,
                'capacity': capacity,
                'utilization_percent': utilization,
                'available': capacity - load
            }
        
        return stats

