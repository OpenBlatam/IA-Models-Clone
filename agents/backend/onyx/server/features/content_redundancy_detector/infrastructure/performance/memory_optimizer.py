"""
Memory Optimization System - Object pooling, memory management
Optimized for memory efficiency and garbage collection
"""

import gc
import sys
import weakref
import threading
import time
from typing import Any, Dict, List, Optional, Type, Callable, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import psutil
import os

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    process_memory: int
    python_objects: int
    gc_collections: int

class ObjectPool:
    """Generic object pool for memory optimization"""
    
    def __init__(
        self,
        factory: Callable[[], Any],
        max_size: int = 100,
        min_size: int = 10,
        reset_func: Optional[Callable[[Any], None]] = None
    ):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.reset_func = reset_func or (lambda obj: None)
        
        # Pool storage
        self.available: deque = deque()
        self.in_use: set = set()
        self.total_created = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Pre-populate pool
        self._pre_populate()

    def _pre_populate(self):
        """Pre-populate pool with minimum objects"""
        for _ in range(self.min_size):
            obj = self.factory()
            self.available.append(obj)
            self.total_created += 1

    def acquire(self) -> Any:
        """Acquire object from pool"""
        with self.lock:
            if self.available:
                obj = self.available.popleft()
            else:
                if self.total_created < self.max_size:
                    obj = self.factory()
                    self.total_created += 1
                else:
                    # Wait for object to become available
                    while not self.available:
                        time.sleep(0.001)  # 1ms
                    obj = self.available.popleft()
            
            self.in_use.add(id(obj))
            return obj

    def release(self, obj: Any) -> None:
        """Release object back to pool"""
        with self.lock:
            if id(obj) in self.in_use:
                self.in_use.remove(id(obj))
                
                # Reset object state
                self.reset_func(obj)
                
                # Return to pool if not full
                if len(self.available) < self.max_size:
                    self.available.append(obj)

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self.lock:
            return {
                "total_created": self.total_created,
                "available_count": len(self.available),
                "in_use_count": len(self.in_use),
                "max_size": self.max_size,
                "min_size": self.min_size,
                "utilization": len(self.in_use) / max(self.total_created, 1)
            }

class MemoryOptimizer:
    """Advanced memory optimization and monitoring"""
    
    def __init__(self, auto_optimize: bool = True):
        self.auto_optimize = auto_optimize
        self.object_pools: Dict[str, ObjectPool] = {}
        self.memory_threshold = 80.0  # 80% memory usage threshold
        self.gc_threshold = 1000  # Objects threshold for GC
        
        # Memory monitoring
        self.memory_history: deque = deque(maxlen=100)
        self.gc_stats = defaultdict(int)
        
        # Background optimization
        self.optimization_thread: Optional[threading.Thread] = None
        self.running = False
        
        if self.auto_optimize:
            self.start()

    def start(self):
        """Start memory optimization"""
        if self.running:
            return
        
        self.running = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_worker,
            daemon=True
        )
        self.optimization_thread.start()

    def stop(self):
        """Stop memory optimization"""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5.0)

    def _optimization_worker(self):
        """Background memory optimization worker"""
        while self.running:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                # Check memory usage
                memory_stats = self.get_memory_stats()
                self.memory_history.append(memory_stats)
                
                # Trigger optimization if needed
                if memory_stats.memory_percent > self.memory_threshold:
                    self._optimize_memory()
                
                # Trigger GC if too many objects
                if memory_stats.python_objects > self.gc_threshold:
                    self._force_garbage_collection()
                
            except Exception as e:
                print(f"Memory optimization error: {e}")

    def create_pool(
        self,
        name: str,
        factory: Callable[[], Any],
        max_size: int = 100,
        min_size: int = 10,
        reset_func: Optional[Callable[[Any], None]] = None
    ) -> ObjectPool:
        """Create a new object pool"""
        pool = ObjectPool(factory, max_size, min_size, reset_func)
        self.object_pools[name] = pool
        return pool

    def get_pool(self, name: str) -> Optional[ObjectPool]:
        """Get object pool by name"""
        return self.object_pools.get(name)

    def _optimize_memory(self):
        """Perform memory optimization"""
        # Force garbage collection
        self._force_garbage_collection()
        
        # Optimize object pools
        for pool in self.object_pools.values():
            # Reduce pool size if memory is high
            if len(pool.available) > pool.min_size:
                # Remove excess objects
                excess = len(pool.available) - pool.min_size
                for _ in range(excess):
                    if pool.available:
                        pool.available.popleft()

    def _force_garbage_collection(self):
        """Force garbage collection and collect stats"""
        # Collect before GC
        before_objects = len(gc.get_objects())
        
        # Run GC
        collected = gc.collect()
        
        # Collect after GC
        after_objects = len(gc.get_objects())
        
        # Update stats
        self.gc_stats["collections"] += 1
        self.gc_stats["objects_collected"] += collected
        self.gc_stats["objects_before"] = before_objects
        self.gc_stats["objects_after"] = after_objects

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        # System memory
        memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss
        
        # Python objects
        python_objects = len(gc.get_objects())
        
        # GC collections
        gc_collections = sum(gc.get_stats()[i]["collections"] for i in range(len(gc.get_stats())))
        
        return MemoryStats(
            total_memory=memory.total,
            available_memory=memory.available,
            used_memory=memory.used,
            memory_percent=memory.percent,
            process_memory=process_memory,
            python_objects=python_objects,
            gc_collections=gc_collections
        )

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get detailed optimization report"""
        memory_stats = self.get_memory_stats()
        
        # Pool statistics
        pool_stats = {}
        for name, pool in self.object_pools.items():
            pool_stats[name] = pool.get_stats()
        
        # Memory history analysis
        if self.memory_history:
            memory_percentages = [stats.memory_percent for stats in self.memory_history]
            avg_memory = sum(memory_percentages) / len(memory_percentages)
            max_memory = max(memory_percentages)
            min_memory = min(memory_percentages)
        else:
            avg_memory = max_memory = min_memory = memory_stats.memory_percent
        
        return {
            "current_memory": {
                "total_mb": memory_stats.total_memory // (1024 * 1024),
                "used_mb": memory_stats.used_memory // (1024 * 1024),
                "available_mb": memory_stats.available_memory // (1024 * 1024),
                "percent": memory_stats.memory_percent,
                "process_mb": memory_stats.process_memory // (1024 * 1024)
            },
            "python_objects": {
                "count": memory_stats.python_objects,
                "gc_collections": memory_stats.gc_collections
            },
            "memory_trend": {
                "average_percent": avg_memory,
                "max_percent": max_memory,
                "min_percent": min_memory,
                "samples": len(self.memory_history)
            },
            "object_pools": pool_stats,
            "gc_stats": dict(self.gc_stats),
            "optimization_enabled": self.auto_optimize
        }

    def optimize_for_workload(self, workload_type: str):
        """Optimize memory settings for specific workload types"""
        if workload_type == "cpu_intensive":
            # Reduce object pools, increase GC frequency
            for pool in self.object_pools.values():
                pool.max_size = max(pool.min_size, pool.max_size // 2)
            self.gc_threshold = 500
            
        elif workload_type == "memory_intensive":
            # Increase object pools, reduce GC frequency
            for pool in self.object_pools.values():
                pool.max_size = min(pool.max_size * 2, 1000)
            self.gc_threshold = 2000
            
        elif workload_type == "io_intensive":
            # Moderate settings
            for pool in self.object_pools.values():
                pool.max_size = int(pool.max_size * 1.5)
            self.gc_threshold = 1500

# Specialized object pools for common types

class StringPool(ObjectPool):
    """Optimized pool for string objects"""
    
    def __init__(self, max_size: int = 1000, min_size: int = 100):
        def factory():
            return ""
        
        def reset(obj):
            obj = ""
        
        super().__init__(factory, max_size, min_size, reset)

class ListPool(ObjectPool):
    """Optimized pool for list objects"""
    
    def __init__(self, max_size: int = 500, min_size: int = 50):
        def factory():
            return []
        
        def reset(obj):
            obj.clear()
        
        super().__init__(factory, max_size, min_size, reset)

class DictPool(ObjectPool):
    """Optimized pool for dictionary objects"""
    
    def __init__(self, max_size: int = 500, min_size: int = 50):
        def factory():
            return {}
        
        def reset(obj):
            obj.clear()
        
        super().__init__(factory, max_size, min_size, reset)

class BytesPool(ObjectPool):
    """Optimized pool for bytes objects"""
    
    def __init__(self, max_size: int = 200, min_size: int = 20):
        def factory():
            return b""
        
        def reset(obj):
            obj = b""
        
        super().__init__(factory, max_size, min_size, reset)

# Memory-efficient data structures

class MemoryEfficientList:
    """Memory-efficient list with automatic optimization"""
    
    def __init__(self, initial_capacity: int = 1000):
        self._data = [None] * initial_capacity
        self._size = 0
        self._capacity = initial_capacity
        self._optimizer = MemoryOptimizer(auto_optimize=False)
    
    def append(self, item: Any) -> None:
        """Add item to list"""
        if self._size >= self._capacity:
            self._resize()
        
        self._data[self._size] = item
        self._size += 1
    
    def _resize(self) -> None:
        """Resize internal array"""
        new_capacity = self._capacity * 2
        new_data = [None] * new_capacity
        
        for i in range(self._size):
            new_data[i] = self._data[i]
        
        self._data = new_data
        self._capacity = new_capacity
    
    def __getitem__(self, index: int) -> Any:
        """Get item by index"""
        if 0 <= index < self._size:
            return self._data[index]
        raise IndexError("Index out of range")
    
    def __len__(self) -> int:
        """Get list length"""
        return self._size
    
    def __iter__(self):
        """Iterate over items"""
        for i in range(self._size):
            yield self._data[i]
    
    def compact(self) -> None:
        """Remove None values and compact memory"""
        compacted = [item for item in self._data[:self._size] if item is not None]
        self._data = compacted + [None] * (self._capacity - len(compacted))
        self._size = len(compacted)

class MemoryEfficientDict:
    """Memory-efficient dictionary with automatic optimization"""
    
    def __init__(self, initial_capacity: int = 1000):
        self._keys = [None] * initial_capacity
        self._values = [None] * initial_capacity
        self._size = 0
        self._capacity = initial_capacity
        self._key_to_index = {}
    
    def __setitem__(self, key: Any, value: Any) -> None:
        """Set key-value pair"""
        if key in self._key_to_index:
            # Update existing
            index = self._key_to_index[key]
            self._values[index] = value
        else:
            # Add new
            if self._size >= self._capacity:
                self._resize()
            
            index = self._size
            self._keys[index] = key
            self._values[index] = value
            self._key_to_index[key] = index
            self._size += 1
    
    def __getitem__(self, key: Any) -> Any:
        """Get value by key"""
        if key in self._key_to_index:
            index = self._key_to_index[key]
            return self._values[index]
        raise KeyError(key)
    
    def __contains__(self, key: Any) -> bool:
        """Check if key exists"""
        return key in self._key_to_index
    
    def _resize(self) -> None:
        """Resize internal arrays"""
        new_capacity = self._capacity * 2
        new_keys = [None] * new_capacity
        new_values = [None] * new_capacity
        
        for i in range(self._size):
            new_keys[i] = self._keys[i]
            new_values[i] = self._values[i]
        
        self._keys = new_keys
        self._values = new_values
        self._capacity = new_capacity
    
    def __len__(self) -> int:
        """Get dictionary size"""
        return self._size
    
    def keys(self):
        """Get all keys"""
        for i in range(self._size):
            yield self._keys[i]
    
    def values(self):
        """Get all values"""
        for i in range(self._size):
            yield self._values[i]
    
    def items(self):
        """Get all key-value pairs"""
        for i in range(self._size):
            yield (self._keys[i], self._values[i])





