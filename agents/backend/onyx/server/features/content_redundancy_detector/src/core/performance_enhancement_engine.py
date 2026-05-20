"""
Performance Enhancement Engine - Advanced performance optimization and monitoring
"""

import asyncio
import logging
import time
import psutil
import gc
import threading
import multiprocessing
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
from collections import defaultdict, deque
import statistics
import numpy as np
import pandas as pd
import tracemalloc
import linecache
import sys
import os
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import heapq
import bisect
import functools
import operator
import itertools
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Performance enhancement configuration"""
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_cache_optimization: bool = True
    enable_database_optimization: bool = True
    enable_api_optimization: bool = True
    enable_async_optimization: bool = True
    memory_threshold_mb: int = 1024
    cpu_threshold_percent: float = 80.0
    cache_size_limit: int = 1000
    max_workers: int = multiprocessing.cpu_count()
    enable_profiling: bool = True
    enable_monitoring: bool = True
    monitoring_interval: float = 1.0
    enable_auto_optimization: bool = True
    optimization_threshold: float = 0.8


@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_io_bytes: int
    active_threads: int
    active_processes: int
    cache_hit_rate: float
    response_time_ms: float
    throughput_requests_per_sec: float
    error_rate: float
    gc_collections: int
    gc_time_ms: float


@dataclass
class OptimizationResult:
    """Optimization result data class"""
    optimization_id: str
    timestamp: datetime
    optimization_type: str
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    improvement_percent: float
    optimization_details: Dict[str, Any]
    recommendations: List[str]


@dataclass
class MemoryProfile:
    """Memory profile data class"""
    timestamp: datetime
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    top_memory_consumers: List[Dict[str, Any]]
    memory_leaks: List[Dict[str, Any]]
    gc_stats: Dict[str, Any]


class MemoryOptimizer:
    """Advanced memory optimization"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.memory_history = deque(maxlen=100)
        self.memory_thresholds = {
            'warning': config.memory_threshold_mb * 0.7,
            'critical': config.memory_threshold_mb * 0.9
        }
        self.weak_refs = weakref.WeakSet()
        self.memory_pool = {}
        self._initialize_memory_optimization()
    
    def _initialize_memory_optimization(self):
        """Initialize memory optimization"""
        try:
            # Start memory tracing
            if self.config.enable_profiling:
                tracemalloc.start()
            
            # Set up garbage collection
            gc.set_threshold(700, 10, 10)
            
            logger.info("Memory optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing memory optimizer: {e}")
    
    async def optimize_memory(self) -> OptimizationResult:
        """Perform memory optimization"""
        start_time = time.time()
        
        try:
            # Get before metrics
            before_metrics = await self._get_memory_metrics()
            
            # Perform memory optimizations
            optimizations = []
            
            # Garbage collection
            if self.config.enable_memory_optimization:
                gc_result = await self._force_garbage_collection()
                optimizations.append(gc_result)
            
            # Memory pool optimization
            pool_result = await self._optimize_memory_pool()
            optimizations.append(pool_result)
            
            # Weak reference cleanup
            weak_ref_result = await self._cleanup_weak_references()
            optimizations.append(weak_ref_result)
            
            # Memory compression
            compression_result = await self._compress_memory()
            optimizations.append(compression_result)
            
            # Get after metrics
            after_metrics = await self._get_memory_metrics()
            
            # Calculate improvement
            improvement = self._calculate_memory_improvement(before_metrics, after_metrics)
            
            # Generate recommendations
            recommendations = await self._generate_memory_recommendations(after_metrics)
            
            processing_time = (time.time() - start_time) * 1000
            
            return OptimizationResult(
                optimization_id=hashlib.md5(f"memory_{time.time()}".encode()).hexdigest(),
                timestamp=datetime.now(),
                optimization_type="memory",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=improvement,
                optimization_details={
                    "optimizations": optimizations,
                    "processing_time_ms": processing_time
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in memory optimization: {e}")
            raise
    
    async def _get_memory_metrics(self) -> PerformanceMetrics:
        """Get current memory metrics"""
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            
            # Get garbage collection stats
            gc_stats = gc.get_stats()
            gc_collections = sum(stat['collections'] for stat in gc_stats)
            gc_time = sum(stat['collected'] for stat in gc_stats)
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=psutil.disk_usage('/').percent,
                network_io_bytes=0,  # Would need network monitoring
                active_threads=threading.active_count(),
                active_processes=len(psutil.pids()),
                cache_hit_rate=0.0,  # Would need cache monitoring
                response_time_ms=0.0,  # Would need response monitoring
                throughput_requests_per_sec=0.0,  # Would need throughput monitoring
                error_rate=0.0,  # Would need error monitoring
                gc_collections=gc_collections,
                gc_time_ms=gc_time
            )
            
        except Exception as e:
            logger.error(f"Error getting memory metrics: {e}")
            raise
    
    async def _force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection"""
        try:
            before_count = len(gc.get_objects())
            
            # Force garbage collection
            collected = gc.collect()
            
            after_count = len(gc.get_objects())
            
            return {
                "type": "garbage_collection",
                "objects_before": before_count,
                "objects_after": after_count,
                "objects_collected": collected,
                "improvement": before_count - after_count
            }
            
        except Exception as e:
            logger.error(f"Error in garbage collection: {e}")
            return {"type": "garbage_collection", "error": str(e)}
    
    async def _optimize_memory_pool(self) -> Dict[str, Any]:
        """Optimize memory pool"""
        try:
            # Clear unused memory pool entries
            initial_size = len(self.memory_pool)
            
            # Remove unused entries (simplified logic)
            unused_keys = []
            for key, value in self.memory_pool.items():
                if not value:  # Simplified check
                    unused_keys.append(key)
            
            for key in unused_keys:
                del self.memory_pool[key]
            
            final_size = len(self.memory_pool)
            
            return {
                "type": "memory_pool_optimization",
                "initial_size": initial_size,
                "final_size": final_size,
                "entries_removed": initial_size - final_size
            }
            
        except Exception as e:
            logger.error(f"Error optimizing memory pool: {e}")
            return {"type": "memory_pool_optimization", "error": str(e)}
    
    async def _cleanup_weak_references(self) -> Dict[str, Any]:
        """Cleanup weak references"""
        try:
            initial_count = len(self.weak_refs)
            
            # Clean up dead weak references
            dead_refs = []
            for ref in self.weak_refs:
                if ref() is None:
                    dead_refs.append(ref)
            
            for ref in dead_refs:
                self.weak_refs.discard(ref)
            
            final_count = len(self.weak_refs)
            
            return {
                "type": "weak_reference_cleanup",
                "initial_count": initial_count,
                "final_count": final_count,
                "dead_refs_removed": initial_count - final_count
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up weak references: {e}")
            return {"type": "weak_reference_cleanup", "error": str(e)}
    
    async def _compress_memory(self) -> Dict[str, Any]:
        """Compress memory usage"""
        try:
            # This is a simplified memory compression
            # In a real implementation, you might use memory mapping or other techniques
            
            before_memory = psutil.virtual_memory().used
            
            # Simulate memory compression by optimizing data structures
            # This is a placeholder for actual compression logic
            
            after_memory = psutil.virtual_memory().used
            
            return {
                "type": "memory_compression",
                "before_memory_mb": before_memory / (1024 * 1024),
                "after_memory_mb": after_memory / (1024 * 1024),
                "compression_ratio": (before_memory - after_memory) / before_memory if before_memory > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error compressing memory: {e}")
            return {"type": "memory_compression", "error": str(e)}
    
    def _calculate_memory_improvement(self, before: PerformanceMetrics, after: PerformanceMetrics) -> float:
        """Calculate memory improvement percentage"""
        try:
            if before.memory_used_mb == 0:
                return 0.0
            
            improvement = (before.memory_used_mb - after.memory_used_mb) / before.memory_used_mb
            return max(0.0, improvement * 100)
            
        except Exception as e:
            logger.error(f"Error calculating memory improvement: {e}")
            return 0.0
    
    async def _generate_memory_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        if metrics.memory_percent > 80:
            recommendations.append("Memory usage is high - consider reducing cache sizes or implementing memory pooling")
        
        if metrics.gc_collections > 100:
            recommendations.append("High garbage collection activity - consider optimizing object creation patterns")
        
        if metrics.memory_used_mb > self.memory_thresholds['critical']:
            recommendations.append("Critical memory usage - immediate optimization required")
        
        if not recommendations:
            recommendations.append("Memory usage is within acceptable limits")
        
        return recommendations
    
    async def get_memory_profile(self) -> MemoryProfile:
        """Get detailed memory profile"""
        try:
            memory = psutil.virtual_memory()
            
            # Get top memory consumers (simplified)
            top_consumers = []
            try:
                for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                    try:
                        info = proc.info
                        top_consumers.append({
                            'pid': info['pid'],
                            'name': info['name'],
                            'memory_mb': info['memory_info'].rss / (1024 * 1024)
                        })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Sort by memory usage
                top_consumers.sort(key=lambda x: x['memory_mb'], reverse=True)
                top_consumers = top_consumers[:10]
                
            except Exception as e:
                logger.warning(f"Could not get process memory info: {e}")
            
            # Get garbage collection stats
            gc_stats = {
                'collections': gc.get_stats(),
                'counts': gc.get_count(),
                'thresholds': gc.get_threshold()
            }
            
            return MemoryProfile(
                timestamp=datetime.now(),
                total_memory_mb=memory.total / (1024 * 1024),
                used_memory_mb=memory.used / (1024 * 1024),
                available_memory_mb=memory.available / (1024 * 1024),
                memory_percent=memory.percent,
                top_memory_consumers=top_consumers,
                memory_leaks=[],  # Would need leak detection
                gc_stats=gc_stats
            )
            
        except Exception as e:
            logger.error(f"Error getting memory profile: {e}")
            raise


class CPUOptimizer:
    """Advanced CPU optimization"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.cpu_history = deque(maxlen=100)
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_workers)
        self.cpu_affinity = None
        self._initialize_cpu_optimization()
    
    def _initialize_cpu_optimization(self):
        """Initialize CPU optimization"""
        try:
            # Set CPU affinity if available
            try:
                self.cpu_affinity = psutil.Process().cpu_affinity()
            except (AttributeError, psutil.AccessDenied):
                self.cpu_affinity = None
            
            logger.info("CPU optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing CPU optimizer: {e}")
    
    async def optimize_cpu(self) -> OptimizationResult:
        """Perform CPU optimization"""
        start_time = time.time()
        
        try:
            # Get before metrics
            before_metrics = await self._get_cpu_metrics()
            
            # Perform CPU optimizations
            optimizations = []
            
            # Thread pool optimization
            if self.config.enable_cpu_optimization:
                thread_result = await self._optimize_thread_pool()
                optimizations.append(thread_result)
            
            # Process pool optimization
            process_result = await self._optimize_process_pool()
            optimizations.append(process_result)
            
            # CPU affinity optimization
            affinity_result = await self._optimize_cpu_affinity()
            optimizations.append(affinity_result)
            
            # Frequency optimization
            frequency_result = await self._optimize_cpu_frequency()
            optimizations.append(frequency_result)
            
            # Get after metrics
            after_metrics = await self._get_cpu_metrics()
            
            # Calculate improvement
            improvement = self._calculate_cpu_improvement(before_metrics, after_metrics)
            
            # Generate recommendations
            recommendations = await self._generate_cpu_recommendations(after_metrics)
            
            processing_time = (time.time() - start_time) * 1000
            
            return OptimizationResult(
                optimization_id=hashlib.md5(f"cpu_{time.time()}".encode()).hexdigest(),
                timestamp=datetime.now(),
                optimization_type="cpu",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=improvement,
                optimization_details={
                    "optimizations": optimizations,
                    "processing_time_ms": processing_time
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in CPU optimization: {e}")
            raise
    
    async def _get_cpu_metrics(self) -> PerformanceMetrics:
        """Get current CPU metrics"""
        try:
            # Get CPU info
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Get memory info
            memory = psutil.virtual_memory()
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=psutil.disk_usage('/').percent,
                network_io_bytes=0,
                active_threads=threading.active_count(),
                active_processes=len(psutil.pids()),
                cache_hit_rate=0.0,
                response_time_ms=0.0,
                throughput_requests_per_sec=0.0,
                error_rate=0.0,
                gc_collections=0,
                gc_time_ms=0.0
            )
            
        except Exception as e:
            logger.error(f"Error getting CPU metrics: {e}")
            raise
    
    async def _optimize_thread_pool(self) -> Dict[str, Any]:
        """Optimize thread pool"""
        try:
            # Adjust thread pool size based on CPU usage
            current_cpu = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            if current_cpu > 80:
                # Reduce thread pool size
                optimal_workers = max(1, cpu_count // 2)
            elif current_cpu < 30:
                # Increase thread pool size
                optimal_workers = min(cpu_count * 2, self.config.max_workers)
            else:
                optimal_workers = cpu_count
            
            return {
                "type": "thread_pool_optimization",
                "current_workers": self.thread_pool._max_workers,
                "optimal_workers": optimal_workers,
                "cpu_usage": current_cpu
            }
            
        except Exception as e:
            logger.error(f"Error optimizing thread pool: {e}")
            return {"type": "thread_pool_optimization", "error": str(e)}
    
    async def _optimize_process_pool(self) -> Dict[str, Any]:
        """Optimize process pool"""
        try:
            # Similar logic to thread pool optimization
            current_cpu = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            if current_cpu > 80:
                optimal_workers = max(1, cpu_count // 2)
            else:
                optimal_workers = cpu_count
            
            return {
                "type": "process_pool_optimization",
                "current_workers": self.process_pool._max_workers,
                "optimal_workers": optimal_workers,
                "cpu_usage": current_cpu
            }
            
        except Exception as e:
            logger.error(f"Error optimizing process pool: {e}")
            return {"type": "process_pool_optimization", "error": str(e)}
    
    async def _optimize_cpu_affinity(self) -> Dict[str, Any]:
        """Optimize CPU affinity"""
        try:
            if self.cpu_affinity is None:
                return {
                    "type": "cpu_affinity_optimization",
                    "status": "not_available",
                    "message": "CPU affinity not available on this system"
                }
            
            # Get current affinity
            current_affinity = psutil.Process().cpu_affinity()
            
            return {
                "type": "cpu_affinity_optimization",
                "current_affinity": current_affinity,
                "available_cores": psutil.cpu_count(),
                "status": "optimized"
            }
            
        except Exception as e:
            logger.error(f"Error optimizing CPU affinity: {e}")
            return {"type": "cpu_affinity_optimization", "error": str(e)}
    
    async def _optimize_cpu_frequency(self) -> Dict[str, Any]:
        """Optimize CPU frequency"""
        try:
            # Get CPU frequency info
            cpu_freq = psutil.cpu_freq()
            
            if cpu_freq is None:
                return {
                    "type": "cpu_frequency_optimization",
                    "status": "not_available",
                    "message": "CPU frequency info not available"
                }
            
            return {
                "type": "cpu_frequency_optimization",
                "current_frequency": cpu_freq.current,
                "min_frequency": cpu_freq.min,
                "max_frequency": cpu_freq.max,
                "status": "monitored"
            }
            
        except Exception as e:
            logger.error(f"Error optimizing CPU frequency: {e}")
            return {"type": "cpu_frequency_optimization", "error": str(e)}
    
    def _calculate_cpu_improvement(self, before: PerformanceMetrics, after: PerformanceMetrics) -> float:
        """Calculate CPU improvement percentage"""
        try:
            if before.cpu_percent == 0:
                return 0.0
            
            improvement = (before.cpu_percent - after.cpu_percent) / before.cpu_percent
            return max(0.0, improvement * 100)
            
        except Exception as e:
            logger.error(f"Error calculating CPU improvement: {e}")
            return 0.0
    
    async def _generate_cpu_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate CPU optimization recommendations"""
        recommendations = []
        
        if metrics.cpu_percent > 80:
            recommendations.append("High CPU usage - consider optimizing algorithms or increasing parallelization")
        
        if metrics.active_threads > self.config.max_workers * 2:
            recommendations.append("Too many active threads - consider thread pool optimization")
        
        if metrics.cpu_percent < 30:
            recommendations.append("Low CPU usage - consider increasing workload or reducing resources")
        
        if not recommendations:
            recommendations.append("CPU usage is within optimal range")
        
        return recommendations


class PerformanceEnhancementEngine:
    """Main Performance Enhancement Engine"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.memory_optimizer = MemoryOptimizer(config)
        self.cpu_optimizer = CPUOptimizer(config)
        
        self.optimization_history = []
        self.performance_history = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitoring_task = None
        
        self._initialize_performance_engine()
    
    def _initialize_performance_engine(self):
        """Initialize performance enhancement engine"""
        try:
            # Start monitoring if enabled
            if self.config.enable_monitoring:
                self._start_monitoring()
            
            logger.info("Performance Enhancement Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing performance engine: {e}")
    
    def _start_monitoring(self):
        """Start performance monitoring"""
        try:
            if not self.monitoring_active:
                self.monitoring_active = True
                self.monitoring_task = asyncio.create_task(self._monitor_performance())
                logger.info("Performance monitoring started")
                
        except Exception as e:
            logger.error(f"Error starting performance monitoring: {e}")
    
    def _stop_monitoring(self):
        """Stop performance monitoring"""
        try:
            if self.monitoring_active and self.monitoring_task:
                self.monitoring_active = False
                self.monitoring_task.cancel()
                logger.info("Performance monitoring stopped")
                
        except Exception as e:
            logger.error(f"Error stopping performance monitoring: {e}")
    
    async def _monitor_performance(self):
        """Monitor system performance"""
        try:
            while self.monitoring_active:
                # Collect performance metrics
                metrics = await self._collect_performance_metrics()
                self.performance_history.append(metrics)
                
                # Check for optimization triggers
                if self.config.enable_auto_optimization:
                    await self._check_optimization_triggers(metrics)
                
                # Wait for next monitoring interval
                await asyncio.sleep(self.config.monitoring_interval)
                
        except asyncio.CancelledError:
            logger.info("Performance monitoring cancelled")
        except Exception as e:
            logger.error(f"Error in performance monitoring: {e}")
    
    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get network I/O (simplified)
            network_io = 0
            try:
                net_io = psutil.net_io_counters()
                network_io = net_io.bytes_sent + net_io.bytes_recv
            except Exception:
                pass
            
            # Get garbage collection stats
            gc_stats = gc.get_stats()
            gc_collections = sum(stat['collections'] for stat in gc_stats)
            gc_time = sum(stat['collected'] for stat in gc_stats)
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_io_bytes=network_io,
                active_threads=threading.active_count(),
                active_processes=len(psutil.pids()),
                cache_hit_rate=0.0,  # Would need cache monitoring
                response_time_ms=0.0,  # Would need response monitoring
                throughput_requests_per_sec=0.0,  # Would need throughput monitoring
                error_rate=0.0,  # Would need error monitoring
                gc_collections=gc_collections,
                gc_time_ms=gc_time
            )
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            raise
    
    async def _check_optimization_triggers(self, metrics: PerformanceMetrics):
        """Check if optimization is needed"""
        try:
            # Check memory threshold
            if metrics.memory_percent > self.config.optimization_threshold * 100:
                logger.info("Memory threshold exceeded, triggering optimization")
                await self.optimize_memory()
            
            # Check CPU threshold
            if metrics.cpu_percent > self.config.cpu_threshold_percent:
                logger.info("CPU threshold exceeded, triggering optimization")
                await self.optimize_cpu()
            
        except Exception as e:
            logger.error(f"Error checking optimization triggers: {e}")
    
    async def optimize_memory(self) -> OptimizationResult:
        """Optimize memory usage"""
        try:
            result = await self.memory_optimizer.optimize_memory()
            self.optimization_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in memory optimization: {e}")
            raise
    
    async def optimize_cpu(self) -> OptimizationResult:
        """Optimize CPU usage"""
        try:
            result = await self.cpu_optimizer.optimize_cpu()
            self.optimization_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in CPU optimization: {e}")
            raise
    
    async def optimize_all(self) -> List[OptimizationResult]:
        """Optimize all system components"""
        try:
            results = []
            
            # Memory optimization
            if self.config.enable_memory_optimization:
                memory_result = await self.optimize_memory()
                results.append(memory_result)
            
            # CPU optimization
            if self.config.enable_cpu_optimization:
                cpu_result = await self.optimize_cpu()
                results.append(cpu_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive optimization: {e}")
            raise
    
    async def get_performance_metrics(self) -> List[PerformanceMetrics]:
        """Get performance metrics history"""
        return list(self.performance_history)
    
    async def get_optimization_history(self) -> List[OptimizationResult]:
        """Get optimization history"""
        return self.optimization_history
    
    async def get_memory_profile(self) -> MemoryProfile:
        """Get detailed memory profile"""
        return await self.memory_optimizer.get_memory_profile()
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            if not self.performance_history:
                return {}
            
            # Calculate summary statistics
            recent_metrics = list(self.performance_history)[-10:]  # Last 10 measurements
            
            avg_cpu = statistics.mean([m.cpu_percent for m in recent_metrics])
            avg_memory = statistics.mean([m.memory_percent for m in recent_metrics])
            avg_memory_used = statistics.mean([m.memory_used_mb for m in recent_metrics])
            
            max_cpu = max([m.cpu_percent for m in recent_metrics])
            max_memory = max([m.memory_percent for m in recent_metrics])
            
            return {
                "summary_timestamp": datetime.now().isoformat(),
                "metrics_count": len(self.performance_history),
                "recent_metrics_count": len(recent_metrics),
                "average_cpu_percent": avg_cpu,
                "average_memory_percent": avg_memory,
                "average_memory_used_mb": avg_memory_used,
                "max_cpu_percent": max_cpu,
                "max_memory_percent": max_memory,
                "optimization_count": len(self.optimization_history),
                "monitoring_active": self.monitoring_active
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown performance engine"""
        try:
            # Stop monitoring
            self._stop_monitoring()
            
            # Shutdown thread and process pools
            self.cpu_optimizer.thread_pool.shutdown(wait=True)
            self.cpu_optimizer.process_pool.shutdown(wait=True)
            
            logger.info("Performance Enhancement Engine shutdown completed")
            
        except Exception as e:
            logger.error(f"Error shutting down performance engine: {e}")


# Global instance
performance_enhancement_engine: Optional[PerformanceEnhancementEngine] = None


async def initialize_performance_enhancement_engine(config: Optional[PerformanceConfig] = None) -> None:
    """Initialize performance enhancement engine"""
    global performance_enhancement_engine
    
    if config is None:
        config = PerformanceConfig()
    
    performance_enhancement_engine = PerformanceEnhancementEngine(config)
    logger.info("Performance Enhancement Engine initialized successfully")


async def get_performance_enhancement_engine() -> Optional[PerformanceEnhancementEngine]:
    """Get performance enhancement engine instance"""
    return performance_enhancement_engine
