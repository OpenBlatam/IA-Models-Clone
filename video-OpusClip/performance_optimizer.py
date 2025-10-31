"""
Performance Optimizer for Video-OpusClip

Advanced performance optimization system with:
- Automatic optimization strategies
- Resource management
- Caching optimization
- Parallel processing
- Memory optimization
- GPU optimization
- Load balancing
- Adaptive performance tuning
"""

import asyncio
import time
import psutil
import threading
import multiprocessing
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import structlog
import numpy as np
import torch
import gc
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, lru_cache
import weakref

logger = structlog.get_logger()

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    
    # Resource Management
    max_cpu_usage: float = 90.0
    max_memory_usage: float = 85.0
    max_gpu_usage: float = 95.0
    target_response_time: float = 2.0
    
    # Caching
    enable_smart_caching: bool = True
    cache_size_limit: int = 1000
    cache_ttl: int = 3600  # seconds
    enable_predictive_caching: bool = True
    
    # Parallel Processing
    max_workers: int = None  # Auto-detect
    enable_async_processing: bool = True
    enable_multiprocessing: bool = True
    chunk_size: int = 100
    
    # Memory Optimization
    enable_memory_optimization: bool = True
    memory_cleanup_threshold: float = 80.0
    enable_garbage_collection: bool = True
    tensor_memory_fraction: float = 0.8
    
    # GPU Optimization
    enable_gpu_optimization: bool = True
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    enable_cuda_graphs: bool = True
    
    # Load Balancing
    enable_load_balancing: bool = True
    load_balancing_strategy: str = "round_robin"  # round_robin, least_loaded, adaptive
    
    # Adaptive Tuning
    enable_adaptive_tuning: bool = True
    tuning_interval: int = 300  # seconds
    performance_history_size: int = 1000
    
    # Monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval: float = 1.0  # seconds
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = min(multiprocessing.cpu_count(), 8)

# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

@dataclass
class OptimizationMetrics:
    """Performance optimization metrics."""
    
    # Timing
    processing_time: float = 0.0
    optimization_time: float = 0.0
    throughput: float = 0.0
    
    # Resource Usage
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: Optional[float] = None
    memory_efficiency: float = 0.0
    
    # Cache Performance
    cache_hit_rate: float = 0.0
    cache_efficiency: float = 0.0
    cache_size: int = 0
    
    # Optimization Impact
    speedup_factor: float = 1.0
    memory_savings: float = 0.0
    optimization_score: float = 0.0
    
    # System Health
    system_health: float = 100.0
    bottleneck_identified: Optional[str] = None
    
    timestamp: datetime = field(default_factory=datetime.now)

# =============================================================================
# SMART CACHE SYSTEM
# =============================================================================

class SmartCache:
    """Intelligent caching system with predictive capabilities."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache = {}
        self.access_patterns = defaultdict(int)
        self.access_times = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "predictions": 0
        }
        
        # Predictive caching
        self.prediction_model = self._create_prediction_model()
        
        # Start background optimization
        if config.enable_predictive_caching:
            self._start_background_optimization()
    
    def _create_prediction_model(self) -> Dict[str, Any]:
        """Create a simple prediction model for cache optimization."""
        return {
            "access_frequency": defaultdict(int),
            "temporal_patterns": defaultdict(list),
            "correlation_matrix": defaultdict(dict)
        }
    
    def _start_background_optimization(self):
        """Start background cache optimization."""
        def optimize_cache():
            while True:
                try:
                    self._optimize_cache()
                    time.sleep(60)  # Optimize every minute
                except Exception as e:
                    logger.error("Cache optimization error", error=str(e))
                    time.sleep(300)  # Wait longer on error
        
        thread = threading.Thread(target=optimize_cache, daemon=True)
        thread.start()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with access tracking."""
        if key in self.cache:
            # Cache hit
            self.cache_stats["hits"] += 1
            self.access_patterns[key] += 1
            self.access_times[key] = time.time()
            
            # Update prediction model
            self._update_prediction_model(key, "hit")
            
            return self.cache[key]["value"]
        else:
            # Cache miss
            self.cache_stats["misses"] += 1
            self._update_prediction_model(key, "miss")
            return default
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set item in cache with TTL."""
        if ttl is None:
            ttl = self.config.cache_ttl
        
        # Check cache size limit
        if len(self.cache) >= self.config.cache_size_limit:
            self._evict_least_valuable()
        
        self.cache[key] = {
            "value": value,
            "created": time.time(),
            "ttl": ttl,
            "access_count": 0
        }
        
        self.access_times[key] = time.time()
    
    def _evict_least_valuable(self) -> None:
        """Evict least valuable items from cache."""
        if not self.cache:
            return
        
        # Calculate value score for each item
        current_time = time.time()
        item_scores = {}
        
        for key, item in self.cache.items():
            age = current_time - item["created"]
            access_count = self.access_patterns.get(key, 0)
            time_since_access = current_time - self.access_times.get(key, item["created"])
            
            # Value score based on access frequency, recency, and age
            value_score = (
                access_count * 0.4 +
                (1.0 / (1.0 + time_since_access)) * 0.4 +
                (1.0 / (1.0 + age)) * 0.2
            )
            
            item_scores[key] = value_score
        
        # Remove lowest scoring items
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1])
        items_to_remove = len(self.cache) - self.config.cache_size_limit + 1
        
        for key, _ in sorted_items[:items_to_remove]:
            del self.cache[key]
            self.cache_stats["evictions"] += 1
    
    def _update_prediction_model(self, key: str, access_type: str):
        """Update prediction model with access pattern."""
        current_time = time.time()
        
        # Update access frequency
        self.prediction_model["access_frequency"][key] += 1
        
        # Update temporal patterns
        self.prediction_model["temporal_patterns"][key].append(current_time)
        
        # Keep only recent patterns
        if len(self.prediction_model["temporal_patterns"][key]) > 100:
            self.prediction_model["temporal_patterns"][key] = \
                self.prediction_model["temporal_patterns"][key][-50:]
    
    def _optimize_cache(self):
        """Optimize cache based on prediction model."""
        if not self.config.enable_predictive_caching:
            return
        
        # Predict which items will be accessed soon
        predictions = self._predict_access_patterns()
        
        # Pre-warm cache with predicted items
        for key, probability in predictions.items():
            if probability > 0.7 and key not in self.cache:  # High probability items
                # Try to load the item (this would need to be implemented based on your data source)
                self.cache_stats["predictions"] += 1
    
    def _predict_access_patterns(self) -> Dict[str, float]:
        """Predict access patterns based on historical data."""
        predictions = {}
        current_time = time.time()
        
        for key, frequency in self.prediction_model["access_frequency"].items():
            if key in self.cache:
                continue
            
            # Simple prediction based on access frequency and recency
            temporal_patterns = self.prediction_model["temporal_patterns"][key]
            if temporal_patterns:
                last_access = max(temporal_patterns)
                time_since_access = current_time - last_access
                
                # Probability decreases with time since last access
                probability = frequency / (1.0 + time_since_access / 3600)  # Normalize by hour
                predictions[key] = min(probability / 10.0, 1.0)  # Cap at 1.0
        
        return predictions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "total_requests": total_requests,
            "predictions": self.cache_stats["predictions"],
            "evictions": self.cache_stats["evictions"]
        }

# =============================================================================
# MEMORY OPTIMIZER
# =============================================================================

class MemoryOptimizer:
    """Advanced memory optimization system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_history = deque(maxlen=100)
        self.optimization_history = []
        self.tensor_refs = weakref.WeakSet()
        
        # Start memory monitoring
        if config.enable_memory_optimization:
            self._start_memory_monitoring()
    
    def _start_memory_monitoring(self):
        """Start background memory monitoring."""
        def monitor_memory():
            while True:
                try:
                    memory_usage = psutil.virtual_memory().percent
                    self.memory_history.append(memory_usage)
                    
                    if memory_usage > self.config.memory_cleanup_threshold:
                        self.optimize_memory()
                    
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    logger.error("Memory monitoring error", error=str(e))
                    time.sleep(30)
        
        thread = threading.Thread(target=monitor_memory, daemon=True)
        thread.start()
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        start_time = time.perf_counter()
        initial_memory = psutil.virtual_memory().used
        
        optimizations = {
            "garbage_collection": False,
            "tensor_cleanup": False,
            "cache_cleanup": False,
            "memory_compaction": False
        }
        
        # Garbage collection
        if self.config.enable_garbage_collection:
            collected = gc.collect()
            if collected > 0:
                optimizations["garbage_collection"] = True
        
        # Tensor cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            optimizations["tensor_cleanup"] = True
        
        # Memory compaction (simulated)
        if len(self.memory_history) > 10:
            recent_avg = np.mean(list(self.memory_history)[-10:])
            if recent_avg > 90:
                optimizations["memory_compaction"] = True
        
        end_time = time.perf_counter()
        final_memory = psutil.virtual_memory().used
        memory_saved = initial_memory - final_memory
        
        optimization_result = {
            "optimizations_applied": optimizations,
            "memory_saved_mb": memory_saved / 1024 / 1024,
            "optimization_time": end_time - start_time,
            "timestamp": datetime.now()
        }
        
        self.optimization_history.append(optimization_result)
        logger.info("Memory optimization completed", **optimization_result)
        
        return optimization_result
    
    def register_tensor(self, tensor: torch.Tensor, name: str = None):
        """Register a tensor for memory tracking."""
        self.tensor_refs.add(tensor)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        memory = psutil.virtual_memory()
        
        return {
            "total_memory_gb": memory.total / 1024**3,
            "available_memory_gb": memory.available / 1024**3,
            "used_memory_gb": memory.used / 1024**3,
            "memory_percent": memory.percent,
            "tensor_count": len(self.tensor_refs),
            "optimization_count": len(self.optimization_history)
        }

# =============================================================================
# GPU OPTIMIZER
# =============================================================================

class GPUOptimizer:
    """GPU optimization and management system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.gpu_stats = {}
        self.optimization_history = []
        
        if config.enable_gpu_optimization and torch.cuda.is_available():
            self._initialize_gpu_optimization()
    
    def _initialize_gpu_optimization(self):
        """Initialize GPU optimization settings."""
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(self.config.tensor_memory_fraction)
        
        # Enable mixed precision if available
        if self.config.mixed_precision:
            try:
                from torch.cuda.amp import autocast
                self.autocast_enabled = True
            except ImportError:
                self.autocast_enabled = False
                logger.warning("Mixed precision not available")
        
        # Enable CUDA graphs if available
        if self.config.enable_cuda_graphs:
            self.cuda_graphs_enabled = True
        else:
            self.cuda_graphs_enabled = False
        
        logger.info("GPU optimization initialized", 
                   mixed_precision=self.autocast_enabled,
                   cuda_graphs=self.cuda_graphs_enabled)
    
    def optimize_gpu_memory(self) -> Dict[str, Any]:
        """Optimize GPU memory usage."""
        if not torch.cuda.is_available():
            return {"error": "GPU not available"}
        
        start_time = time.perf_counter()
        initial_memory = torch.cuda.memory_allocated()
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Synchronize if needed
        torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        final_memory = torch.cuda.memory_allocated()
        memory_saved = initial_memory - final_memory
        
        result = {
            "memory_saved_mb": memory_saved / 1024 / 1024,
            "optimization_time": end_time - start_time,
            "timestamp": datetime.now()
        }
        
        self.optimization_history.append(result)
        return result
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics."""
        if not torch.cuda.is_available():
            return {"error": "GPU not available"}
        
        return {
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name()
        }
    
    def create_optimized_context(self):
        """Create optimized GPU context."""
        if not torch.cuda.is_available():
            return None
        
        context = {
            "autocast": None,
            "memory_fraction": self.config.tensor_memory_fraction
        }
        
        if self.autocast_enabled:
            context["autocast"] = torch.cuda.amp.autocast()
        
        return context

# =============================================================================
# LOAD BALANCER
# =============================================================================

class LoadBalancer:
    """Intelligent load balancing system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.workers = []
        self.worker_stats = defaultdict(lambda: {
            "load": 0.0,
            "response_time": 0.0,
            "error_rate": 0.0,
            "last_update": time.time()
        })
        self.current_worker_index = 0
        
        # Initialize workers
        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize worker pool."""
        for i in range(self.config.max_workers):
            worker = {
                "id": i,
                "status": "available",
                "current_load": 0.0,
                "total_processed": 0
            }
            self.workers.append(worker)
    
    def get_next_worker(self) -> int:
        """Get next available worker based on strategy."""
        if self.config.load_balancing_strategy == "round_robin":
            return self._round_robin()
        elif self.config.load_balancing_strategy == "least_loaded":
            return self._least_loaded()
        elif self.config.load_balancing_strategy == "adaptive":
            return self._adaptive_balancing()
        else:
            return self._round_robin()
    
    def _round_robin(self) -> int:
        """Round-robin worker selection."""
        worker_id = self.current_worker_index
        self.current_worker_index = (self.current_worker_index + 1) % len(self.workers)
        return worker_id
    
    def _least_loaded(self) -> int:
        """Select least loaded worker."""
        return min(range(len(self.workers)), 
                  key=lambda i: self.workers[i]["current_load"])
    
    def _adaptive_balancing(self) -> int:
        """Adaptive load balancing based on performance metrics."""
        # Consider load, response time, and error rate
        scores = []
        for i, worker in enumerate(self.workers):
            load_score = 1.0 - worker["current_load"]
            response_score = 1.0 / (1.0 + self.worker_stats[i]["response_time"])
            error_score = 1.0 - self.worker_stats[i]["error_rate"]
            
            total_score = load_score * 0.4 + response_score * 0.4 + error_score * 0.2
            scores.append((i, total_score))
        
        return max(scores, key=lambda x: x[1])[0]
    
    def update_worker_stats(self, worker_id: int, load: float, response_time: float, 
                           error_rate: float = 0.0):
        """Update worker statistics."""
        self.workers[worker_id]["current_load"] = load
        self.worker_stats[worker_id].update({
            "load": load,
            "response_time": response_time,
            "error_rate": error_rate,
            "last_update": time.time()
        })

# =============================================================================
# ADAPTIVE PERFORMANCE TUNER
# =============================================================================

class AdaptivePerformanceTuner:
    """Adaptive performance tuning system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.performance_history = deque(maxlen=config.performance_history_size)
        self.optimization_strategies = []
        self.current_strategy = None
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Start adaptive tuning
        if config.enable_adaptive_tuning:
            self._start_adaptive_tuning()
    
    def _initialize_strategies(self):
        """Initialize optimization strategies."""
        self.optimization_strategies = [
            {
                "name": "aggressive_caching",
                "description": "Increase cache size and TTL",
                "conditions": {"cache_hit_rate": "< 0.7", "memory_usage": "< 70"},
                "actions": {"cache_size_limit": "increase", "cache_ttl": "increase"}
            },
            {
                "name": "memory_optimization",
                "description": "Enable aggressive memory cleanup",
                "conditions": {"memory_usage": "> 80", "memory_efficiency": "< 0.8"},
                "actions": {"memory_cleanup_threshold": "decrease", "enable_garbage_collection": True}
            },
            {
                "name": "parallel_processing",
                "description": "Increase parallel processing",
                "conditions": {"cpu_usage": "< 70", "throughput": "< target"},
                "actions": {"max_workers": "increase", "chunk_size": "decrease"}
            },
            {
                "name": "gpu_optimization",
                "description": "Enable GPU optimizations",
                "conditions": {"gpu_usage": "> 50", "processing_time": "> target"},
                "actions": {"enable_cuda_graphs": True, "mixed_precision": True}
            }
        ]
    
    def _start_adaptive_tuning(self):
        """Start adaptive tuning loop."""
        def tune_performance():
            while True:
                try:
                    self._evaluate_and_tune()
                    time.sleep(self.config.tuning_interval)
                except Exception as e:
                    logger.error("Adaptive tuning error", error=str(e))
                    time.sleep(60)
        
        thread = threading.Thread(target=tune_performance, daemon=True)
        thread.start()
    
    def _evaluate_and_tune(self):
        """Evaluate current performance and apply tuning."""
        if len(self.performance_history) < 10:
            return
        
        # Analyze recent performance
        recent_metrics = list(self.performance_history)[-10:]
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        
        # Find applicable strategies
        applicable_strategies = []
        for strategy in self.optimization_strategies:
            if self._evaluate_strategy_conditions(strategy, avg_throughput, avg_memory, avg_cpu):
                applicable_strategies.append(strategy)
        
        # Apply best strategy
        if applicable_strategies:
            best_strategy = max(applicable_strategies, key=lambda s: self._calculate_strategy_score(s))
            self._apply_strategy(best_strategy)
    
    def _evaluate_strategy_conditions(self, strategy: Dict, throughput: float, 
                                    memory_usage: float, cpu_usage: float) -> bool:
        """Evaluate if strategy conditions are met."""
        conditions = strategy["conditions"]
        
        for metric, condition in conditions.items():
            if metric == "cache_hit_rate" and condition == "< 0.7":
                # This would need actual cache hit rate data
                continue
            elif metric == "memory_usage" and condition == "> 80":
                if memory_usage <= 80:
                    return False
            elif metric == "cpu_usage" and condition == "< 70":
                if cpu_usage >= 70:
                    return False
        
        return True
    
    def _calculate_strategy_score(self, strategy: Dict) -> float:
        """Calculate strategy effectiveness score."""
        # Simple scoring based on strategy type
        scores = {
            "aggressive_caching": 0.8,
            "memory_optimization": 0.9,
            "parallel_processing": 0.7,
            "gpu_optimization": 0.6
        }
        return scores.get(strategy["name"], 0.5)
    
    def _apply_strategy(self, strategy: Dict):
        """Apply optimization strategy."""
        logger.info("Applying optimization strategy", strategy=strategy["name"])
        self.current_strategy = strategy
        
        # Apply strategy actions (this would need to be implemented based on your system)
        actions = strategy["actions"]
        for action, value in actions.items():
            logger.info("Applying action", action=action, value=value)
    
    def record_performance(self, metrics: OptimizationMetrics):
        """Record performance metrics for adaptive tuning."""
        self.performance_history.append(metrics)

# =============================================================================
# MAIN PERFORMANCE OPTIMIZER
# =============================================================================

class PerformanceOptimizer:
    """Main performance optimization system."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        
        # Initialize components
        self.smart_cache = SmartCache(self.config)
        self.memory_optimizer = MemoryOptimizer(self.config)
        self.gpu_optimizer = GPUOptimizer(self.config)
        self.load_balancer = LoadBalancer(self.config)
        self.adaptive_tuner = AdaptivePerformanceTuner(self.config)
        
        # Performance monitoring
        self.metrics_history = deque(maxlen=1000)
        self.optimization_history = []
        
        logger.info("Performance optimizer initialized", config=self.config.__dict__)
    
    def optimize_operation(self, operation_name: str, operation_func: Callable, 
                          *args, **kwargs) -> Tuple[Any, OptimizationMetrics]:
        """Optimize and execute an operation."""
        start_time = time.perf_counter()
        
        # Pre-optimization
        pre_metrics = self._collect_metrics()
        
        # Execute operation with optimizations
        try:
            # Check cache first
            cache_key = f"{operation_name}_{hash(str(args) + str(sorted(kwargs.items())))}"
            result = self.smart_cache.get(cache_key)
            
            if result is None:
                # Execute operation
                result = operation_func(*args, **kwargs)
                
                # Cache result
                self.smart_cache.set(cache_key, result)
            
            # Post-optimization
            post_metrics = self._collect_metrics()
            
            # Calculate optimization metrics
            optimization_metrics = self._calculate_optimization_metrics(
                pre_metrics, post_metrics, time.perf_counter() - start_time
            )
            
            # Record for adaptive tuning
            self.adaptive_tuner.record_performance(optimization_metrics)
            self.metrics_history.append(optimization_metrics)
            
            return result, optimization_metrics
            
        except Exception as e:
            logger.error("Operation optimization failed", operation=operation_name, error=str(e))
            raise
    
    def _collect_metrics(self) -> OptimizationMetrics:
        """Collect current performance metrics."""
        memory_stats = self.memory_optimizer.get_memory_stats()
        gpu_stats = self.gpu_optimizer.get_gpu_stats()
        cache_stats = self.smart_cache.get_stats()
        
        return OptimizationMetrics(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=memory_stats["memory_percent"],
            gpu_usage=gpu_stats.get("memory_allocated_gb", 0) if "error" not in gpu_stats else None,
            cache_hit_rate=cache_stats["hit_rate"],
            cache_size=cache_stats["cache_size"],
            timestamp=datetime.now()
        )
    
    def _calculate_optimization_metrics(self, pre_metrics: OptimizationMetrics, 
                                      post_metrics: OptimizationMetrics, 
                                      processing_time: float) -> OptimizationMetrics:
        """Calculate optimization impact metrics."""
        return OptimizationMetrics(
            processing_time=processing_time,
            cpu_usage=post_metrics.cpu_usage,
            memory_usage=post_metrics.memory_usage,
            gpu_usage=post_metrics.gpu_usage,
            cache_hit_rate=post_metrics.cache_hit_rate,
            cache_size=post_metrics.cache_size,
            memory_efficiency=1.0 - (post_metrics.memory_usage / 100.0),
            speedup_factor=1.0,  # Would need baseline comparison
            memory_savings=pre_metrics.memory_usage - post_metrics.memory_usage,
            optimization_score=self._calculate_optimization_score(pre_metrics, post_metrics),
            timestamp=datetime.now()
        )
    
    def _calculate_optimization_score(self, pre_metrics: OptimizationMetrics, 
                                    post_metrics: OptimizationMetrics) -> float:
        """Calculate overall optimization score."""
        # Simple scoring based on resource usage and performance
        cpu_score = 1.0 - (post_metrics.cpu_usage / 100.0)
        memory_score = 1.0 - (post_metrics.memory_usage / 100.0)
        cache_score = post_metrics.cache_hit_rate
        
        return (cpu_score * 0.3 + memory_score * 0.3 + cache_score * 0.4)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        if not self.metrics_history:
            return {"error": "No performance data available"}
        
        recent_metrics = list(self.metrics_history)[-100:]
        
        return {
            "cache_performance": self.smart_cache.get_stats(),
            "memory_stats": self.memory_optimizer.get_memory_stats(),
            "gpu_stats": self.gpu_optimizer.get_gpu_stats(),
            "load_balancer_stats": {
                "worker_count": len(self.load_balancer.workers),
                "strategy": self.config.load_balancing_strategy
            },
            "performance_summary": {
                "avg_processing_time": np.mean([m.processing_time for m in recent_metrics]),
                "avg_throughput": np.mean([m.throughput for m in recent_metrics]),
                "avg_optimization_score": np.mean([m.optimization_score for m in recent_metrics]),
                "total_optimizations": len(self.optimization_history)
            },
            "current_strategy": self.adaptive_tuner.current_strategy,
            "timestamp": datetime.now()
        }
    
    def force_optimization(self) -> Dict[str, Any]:
        """Force immediate optimization."""
        optimizations = {}
        
        # Memory optimization
        memory_result = self.memory_optimizer.optimize_memory()
        optimizations["memory"] = memory_result
        
        # GPU optimization
        if torch.cuda.is_available():
            gpu_result = self.gpu_optimizer.optimize_gpu_memory()
            optimizations["gpu"] = gpu_result
        
        # Cache optimization
        self.smart_cache._optimize_cache()
        optimizations["cache"] = {"status": "optimized"}
        
        self.optimization_history.append(optimizations)
        return optimizations

# =============================================================================
# DECORATORS AND UTILITIES
# =============================================================================

def optimize_performance(operation_name: str = None):
    """Decorator for automatic performance optimization."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            name = operation_name or func.__name__
            return optimizer.optimize_operation(name, func, *args, **kwargs)
        return wrapper
    return decorator

def cache_result(ttl: int = 3600):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            result = optimizer.smart_cache.get(cache_key)
            if result is None:
                result = func(*args, **kwargs)
                optimizer.smart_cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

_performance_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer

def initialize_performance_optimizer(config: OptimizationConfig):
    """Initialize global performance optimizer with custom config."""
    global _performance_optimizer
    _performance_optimizer = PerformanceOptimizer(config)
    return _performance_optimizer

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Example usage of performance optimizer."""
    
    # Initialize optimizer
    config = OptimizationConfig(
        enable_smart_caching=True,
        enable_memory_optimization=True,
        enable_gpu_optimization=True,
        enable_load_balancing=True,
        enable_adaptive_tuning=True
    )
    
    optimizer = PerformanceOptimizer(config)
    
    # Example operation
    @optimize_performance("video_processing")
    def process_video(video_path: str):
        # Simulate video processing
        time.sleep(0.1)
        return {"processed": True, "path": video_path}
    
    # Execute with optimization
    result, metrics = process_video("example.mp4")
    
    # Get optimization report
    report = optimizer.get_optimization_report()
    print("Optimization report:", report)
    
    return optimizer

if __name__ == "__main__":
    example_usage() 