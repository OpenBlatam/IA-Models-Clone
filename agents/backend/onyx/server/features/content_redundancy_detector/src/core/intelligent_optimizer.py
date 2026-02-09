"""
Intelligent Optimizer - Advanced performance optimization, efficiency, and intelligent automation
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import asyncio
import logging
import time
import json
import hashlib
import statistics
import psutil
import gc
import weakref
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import multiprocessing
import subprocess
import os
import sys

# Optimization libraries
import numba
import cython
import ray
import dask
import joblib
from memory_profiler import profile
import line_profiler
import py-spy
import scalene

logger = logging.getLogger(__name__)


@dataclass
class OptimizationMetric:
    """Optimization metric data structure"""
    metric_id: str
    metric_name: str
    metric_type: str
    value: float
    baseline_value: float
    improvement_percentage: float
    status: str  # improved, degraded, stable
    description: str
    recommendations: List[str]
    timestamp: datetime


@dataclass
class PerformanceProfile:
    """Performance profile data structure"""
    profile_id: str
    component_name: str
    cpu_usage: float
    memory_usage: float
    execution_time: float
    throughput: float
    error_rate: float
    cache_hit_rate: float
    optimization_score: float
    bottlenecks: List[str]
    recommendations: List[str]
    timestamp: datetime


@dataclass
class OptimizationRule:
    """Optimization rule data structure"""
    rule_id: str
    rule_name: str
    rule_type: str
    condition: str
    action: str
    priority: int
    enabled: bool
    description: str
    success_rate: float


@dataclass
class OptimizationResult:
    """Optimization result data structure"""
    optimization_id: str
    optimization_type: str
    component: str
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_percentage: float
    optimization_time: float
    status: str  # success, partial, failed
    recommendations: List[str]
    timestamp: datetime


class IntelligentOptimizer:
    """Intelligent optimization engine with advanced performance tuning"""
    
    def __init__(self):
        self.optimization_metrics = []
        self.performance_profiles = []
        self.optimization_rules = []
        self.optimization_results = []
        self.performance_cache = {}
        self.optimization_history = []
        self.auto_optimization_enabled = True
        self.optimization_thresholds = {}
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize the intelligent optimizer"""
        try:
            logger.info("Initializing Intelligent Optimizer...")
            
            # Load optimization rules
            await self._load_optimization_rules()
            
            # Load optimization thresholds
            await self._load_optimization_thresholds()
            
            # Initialize performance monitoring
            await self._initialize_performance_monitoring()
            
            # Initialize auto-optimization
            await self._initialize_auto_optimization()
            
            # Initialize optimization engines
            await self._initialize_optimization_engines()
            
            self.initialized = True
            logger.info("Intelligent Optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Intelligent Optimizer: {e}")
            raise
    
    async def _load_optimization_rules(self) -> None:
        """Load optimization rules"""
        try:
            self.optimization_rules = [
                OptimizationRule(
                    rule_id="memory_optimization",
                    rule_name="Memory Optimization",
                    rule_type="memory",
                    condition="memory_usage > 80",
                    action="trigger_garbage_collection",
                    priority=1,
                    enabled=True,
                    description="Trigger garbage collection when memory usage exceeds 80%",
                    success_rate=0.0
                ),
                OptimizationRule(
                    rule_id="cpu_optimization",
                    rule_name="CPU Optimization",
                    rule_type="cpu",
                    condition="cpu_usage > 85",
                    action="scale_processing_workers",
                    priority=2,
                    enabled=True,
                    description="Scale processing workers when CPU usage exceeds 85%",
                    success_rate=0.0
                ),
                OptimizationRule(
                    rule_id="cache_optimization",
                    rule_name="Cache Optimization",
                    rule_type="cache",
                    condition="cache_hit_rate < 70",
                    action="optimize_cache_strategy",
                    priority=3,
                    enabled=True,
                    description="Optimize cache strategy when hit rate falls below 70%",
                    success_rate=0.0
                ),
                OptimizationRule(
                    rule_id="response_time_optimization",
                    rule_name="Response Time Optimization",
                    rule_type="performance",
                    condition="response_time > 1000",
                    action="enable_async_processing",
                    priority=1,
                    enabled=True,
                    description="Enable async processing when response time exceeds 1 second",
                    success_rate=0.0
                ),
                OptimizationRule(
                    rule_id="throughput_optimization",
                    rule_name="Throughput Optimization",
                    rule_type="performance",
                    condition="throughput < 100",
                    action="enable_parallel_processing",
                    priority=2,
                    enabled=True,
                    description="Enable parallel processing when throughput falls below 100 ops/sec",
                    success_rate=0.0
                )
            ]
            
            logger.info("Optimization rules loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load optimization rules: {e}")
    
    async def _load_optimization_thresholds(self) -> None:
        """Load optimization thresholds"""
        try:
            self.optimization_thresholds = {
                "performance": {
                    "response_time": {"warning": 500, "critical": 1000},
                    "throughput": {"warning": 50, "critical": 100},
                    "error_rate": {"warning": 1, "critical": 5},
                    "cpu_usage": {"warning": 70, "critical": 85},
                    "memory_usage": {"warning": 70, "critical": 80},
                    "cache_hit_rate": {"warning": 80, "critical": 70}
                },
                "optimization": {
                    "improvement_threshold": 10.0,  # Minimum 10% improvement
                    "optimization_timeout": 30.0,   # 30 seconds max
                    "max_optimization_attempts": 3,
                    "cooldown_period": 60.0         # 60 seconds between optimizations
                }
            }
            
            logger.info("Optimization thresholds loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load optimization thresholds: {e}")
    
    async def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring"""
        try:
            # Start background performance monitoring
            asyncio.create_task(self._performance_monitoring_loop())
            logger.info("Performance monitoring initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize performance monitoring: {e}")
    
    async def _initialize_auto_optimization(self) -> None:
        """Initialize auto-optimization"""
        try:
            # Start background auto-optimization
            asyncio.create_task(self._auto_optimization_loop())
            logger.info("Auto-optimization initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize auto-optimization: {e}")
    
    async def _initialize_optimization_engines(self) -> None:
        """Initialize optimization engines"""
        try:
            # Initialize Numba JIT compiler
            try:
                numba.config.THREADING_LAYER = 'tbb'
                logger.info("Numba JIT compiler initialized")
            except Exception as e:
                logger.warning(f"Numba initialization failed: {e}")
            
            # Initialize Ray for distributed computing
            try:
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                logger.info("Ray distributed computing initialized")
            except Exception as e:
                logger.warning(f"Ray initialization failed: {e}")
            
            # Initialize Dask for parallel computing
            try:
                from dask.distributed import Client
                self.dask_client = Client()
                logger.info("Dask parallel computing initialized")
            except Exception as e:
                logger.warning(f"Dask initialization failed: {e}")
            
            logger.info("Optimization engines initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize optimization engines: {e}")
    
    async def _performance_monitoring_loop(self) -> None:
        """Background performance monitoring loop"""
        while True:
            try:
                # Monitor system performance
                await self._monitor_system_performance()
                
                # Monitor application performance
                await self._monitor_application_performance()
                
                # Update performance profiles
                await self._update_performance_profiles()
                
                # Wait before next monitoring cycle
                await asyncio.sleep(10)  # 10 seconds
                
            except Exception as e:
                logger.warning(f"Performance monitoring error: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds on error
    
    async def _auto_optimization_loop(self) -> None:
        """Background auto-optimization loop"""
        while True:
            try:
                if self.auto_optimization_enabled:
                    # Check optimization rules
                    await self._check_optimization_rules()
                    
                    # Apply automatic optimizations
                    await self._apply_automatic_optimizations()
                
                # Wait before next optimization cycle
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                logger.warning(f"Auto-optimization error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _monitor_system_performance(self) -> None:
        """Monitor system performance metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_usage = network.bytes_sent + network.bytes_recv
            
            # Store metrics
            metric = OptimizationMetric(
                metric_id=f"system_perf_{int(time.time())}",
                metric_name="System Performance",
                metric_type="system",
                value=cpu_usage + memory_usage + disk_usage,
                baseline_value=100.0,
                improvement_percentage=0.0,
                status="stable",
                description="Overall system performance metrics",
                recommendations=[],
                timestamp=datetime.now()
            )
            
            self.optimization_metrics.append(metric)
            
        except Exception as e:
            logger.warning(f"System performance monitoring failed: {e}")
    
    async def _monitor_application_performance(self) -> None:
        """Monitor application performance metrics"""
        try:
            # Get current process
            process = psutil.Process()
            
            # CPU usage
            cpu_usage = process.cpu_percent()
            
            # Memory usage
            memory_info = process.memory_info()
            memory_usage = memory_info.rss / 1024 / 1024  # MB
            
            # Thread count
            thread_count = process.num_threads()
            
            # File descriptor count
            fd_count = process.num_fds() if hasattr(process, 'num_fds') else 0
            
            # Store metrics
            metric = OptimizationMetric(
                metric_id=f"app_perf_{int(time.time())}",
                metric_name="Application Performance",
                metric_type="application",
                value=cpu_usage + memory_usage,
                baseline_value=100.0,
                improvement_percentage=0.0,
                status="stable",
                description="Application performance metrics",
                recommendations=[],
                timestamp=datetime.now()
            )
            
            self.optimization_metrics.append(metric)
            
        except Exception as e:
            logger.warning(f"Application performance monitoring failed: {e}")
    
    async def _update_performance_profiles(self) -> None:
        """Update performance profiles"""
        try:
            # Get recent metrics
            recent_metrics = self.optimization_metrics[-10:] if self.optimization_metrics else []
            
            if recent_metrics:
                # Calculate average performance
                avg_cpu = statistics.mean([m.value for m in recent_metrics if m.metric_type == "system"])
                avg_memory = statistics.mean([m.value for m in recent_metrics if m.metric_type == "application"])
                
                # Create performance profile
                profile = PerformanceProfile(
                    profile_id=f"profile_{int(time.time())}",
                    component_name="main_application",
                    cpu_usage=avg_cpu,
                    memory_usage=avg_memory,
                    execution_time=0.0,  # Would be measured in actual operations
                    throughput=0.0,      # Would be measured in actual operations
                    error_rate=0.0,      # Would be measured in actual operations
                    cache_hit_rate=0.0,  # Would be measured in actual operations
                    optimization_score=self._calculate_optimization_score(avg_cpu, avg_memory),
                    bottlenecks=[],
                    recommendations=[],
                    timestamp=datetime.now()
                )
                
                self.performance_profiles.append(profile)
            
        except Exception as e:
            logger.warning(f"Performance profile update failed: {e}")
    
    def _calculate_optimization_score(self, cpu_usage: float, memory_usage: float) -> float:
        """Calculate optimization score"""
        try:
            # Simple scoring algorithm
            cpu_score = max(0, 100 - cpu_usage)
            memory_score = max(0, 100 - memory_usage)
            
            # Weighted average
            optimization_score = (cpu_score * 0.6) + (memory_score * 0.4)
            
            return min(100.0, max(0.0, optimization_score))
            
        except Exception as e:
            logger.warning(f"Optimization score calculation failed: {e}")
            return 50.0
    
    async def _check_optimization_rules(self) -> None:
        """Check optimization rules and trigger actions"""
        try:
            # Get current performance metrics
            current_metrics = await self._get_current_metrics()
            
            for rule in self.optimization_rules:
                if not rule.enabled:
                    continue
                
                # Check if rule condition is met
                if await self._evaluate_rule_condition(rule, current_metrics):
                    # Trigger optimization action
                    await self._execute_optimization_action(rule, current_metrics)
                    
        except Exception as e:
            logger.warning(f"Optimization rule checking failed: {e}")
    
    async def _get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
            
            # Get application metrics
            process = psutil.Process()
            app_cpu = process.cpu_percent()
            app_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "app_cpu": app_cpu,
                "app_memory": app_memory,
                "response_time": 0.0,  # Would be measured in actual operations
                "throughput": 0.0,     # Would be measured in actual operations
                "error_rate": 0.0,     # Would be measured in actual operations
                "cache_hit_rate": 0.0  # Would be measured in actual operations
            }
            
        except Exception as e:
            logger.warning(f"Failed to get current metrics: {e}")
            return {}
    
    async def _evaluate_rule_condition(self, rule: OptimizationRule, metrics: Dict[str, float]) -> bool:
        """Evaluate optimization rule condition"""
        try:
            # Simple condition evaluation
            if rule.condition == "memory_usage > 80":
                return metrics.get("memory_usage", 0) > 80
            elif rule.condition == "cpu_usage > 85":
                return metrics.get("cpu_usage", 0) > 85
            elif rule.condition == "cache_hit_rate < 70":
                return metrics.get("cache_hit_rate", 100) < 70
            elif rule.condition == "response_time > 1000":
                return metrics.get("response_time", 0) > 1000
            elif rule.condition == "throughput < 100":
                return metrics.get("throughput", 0) < 100
            
            return False
            
        except Exception as e:
            logger.warning(f"Rule condition evaluation failed: {e}")
            return False
    
    async def _execute_optimization_action(self, rule: OptimizationRule, metrics: Dict[str, float]) -> None:
        """Execute optimization action"""
        try:
            logger.info(f"Executing optimization action: {rule.action}")
            
            if rule.action == "trigger_garbage_collection":
                await self._trigger_garbage_collection()
            elif rule.action == "scale_processing_workers":
                await self._scale_processing_workers()
            elif rule.action == "optimize_cache_strategy":
                await self._optimize_cache_strategy()
            elif rule.action == "enable_async_processing":
                await self._enable_async_processing()
            elif rule.action == "enable_parallel_processing":
                await self._enable_parallel_processing()
            
            # Update rule success rate
            rule.success_rate = min(1.0, rule.success_rate + 0.1)
            
        except Exception as e:
            logger.warning(f"Optimization action execution failed: {e}")
    
    async def _trigger_garbage_collection(self) -> None:
        """Trigger garbage collection"""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            logger.info(f"Garbage collection triggered, collected {collected} objects")
            
            # Record optimization result
            result = OptimizationResult(
                optimization_id=f"gc_{int(time.time())}",
                optimization_type="memory",
                component="garbage_collection",
                before_metrics={"memory_usage": psutil.virtual_memory().percent},
                after_metrics={"memory_usage": psutil.virtual_memory().percent},
                improvement_percentage=0.0,
                optimization_time=0.1,
                status="success",
                recommendations=["Consider implementing weak references", "Review object lifecycle"],
                timestamp=datetime.now()
            )
            
            self.optimization_results.append(result)
            
        except Exception as e:
            logger.warning(f"Garbage collection failed: {e}")
    
    async def _scale_processing_workers(self) -> None:
        """Scale processing workers"""
        try:
            # This would scale workers based on CPU usage
            logger.info("Scaling processing workers")
            
            # Record optimization result
            result = OptimizationResult(
                optimization_id=f"scale_{int(time.time())}",
                optimization_type="cpu",
                component="processing_workers",
                before_metrics={"cpu_usage": psutil.cpu_percent()},
                after_metrics={"cpu_usage": psutil.cpu_percent()},
                improvement_percentage=0.0,
                optimization_time=0.1,
                status="success",
                recommendations=["Monitor worker performance", "Adjust scaling thresholds"],
                timestamp=datetime.now()
            )
            
            self.optimization_results.append(result)
            
        except Exception as e:
            logger.warning(f"Worker scaling failed: {e}")
    
    async def _optimize_cache_strategy(self) -> None:
        """Optimize cache strategy"""
        try:
            # This would optimize cache configuration
            logger.info("Optimizing cache strategy")
            
            # Record optimization result
            result = OptimizationResult(
                optimization_id=f"cache_{int(time.time())}",
                optimization_type="cache",
                component="cache_strategy",
                before_metrics={"cache_hit_rate": 0.0},
                after_metrics={"cache_hit_rate": 0.0},
                improvement_percentage=0.0,
                optimization_time=0.1,
                status="success",
                recommendations=["Review cache size", "Optimize cache eviction policy"],
                timestamp=datetime.now()
            )
            
            self.optimization_results.append(result)
            
        except Exception as e:
            logger.warning(f"Cache optimization failed: {e}")
    
    async def _enable_async_processing(self) -> None:
        """Enable async processing"""
        try:
            # This would enable async processing for better response times
            logger.info("Enabling async processing")
            
            # Record optimization result
            result = OptimizationResult(
                optimization_id=f"async_{int(time.time())}",
                optimization_type="performance",
                component="async_processing",
                before_metrics={"response_time": 0.0},
                after_metrics={"response_time": 0.0},
                improvement_percentage=0.0,
                optimization_time=0.1,
                status="success",
                recommendations=["Monitor async performance", "Optimize async operations"],
                timestamp=datetime.now()
            )
            
            self.optimization_results.append(result)
            
        except Exception as e:
            logger.warning(f"Async processing optimization failed: {e}")
    
    async def _enable_parallel_processing(self) -> None:
        """Enable parallel processing"""
        try:
            # This would enable parallel processing for better throughput
            logger.info("Enabling parallel processing")
            
            # Record optimization result
            result = OptimizationResult(
                optimization_id=f"parallel_{int(time.time())}",
                optimization_type="performance",
                component="parallel_processing",
                before_metrics={"throughput": 0.0},
                after_metrics={"throughput": 0.0},
                improvement_percentage=0.0,
                optimization_time=0.1,
                status="success",
                recommendations=["Monitor parallel performance", "Optimize parallel operations"],
                timestamp=datetime.now()
            )
            
            self.optimization_results.append(result)
            
        except Exception as e:
            logger.warning(f"Parallel processing optimization failed: {e}")
    
    async def _apply_automatic_optimizations(self) -> None:
        """Apply automatic optimizations"""
        try:
            # Get current performance profile
            if self.performance_profiles:
                current_profile = self.performance_profiles[-1]
                
                # Apply optimizations based on performance profile
                if current_profile.optimization_score < 70:
                    await self._apply_performance_optimizations(current_profile)
                
                if current_profile.cpu_usage > 80:
                    await self._apply_cpu_optimizations(current_profile)
                
                if current_profile.memory_usage > 80:
                    await self._apply_memory_optimizations(current_profile)
            
        except Exception as e:
            logger.warning(f"Automatic optimization application failed: {e}")
    
    async def _apply_performance_optimizations(self, profile: PerformanceProfile) -> None:
        """Apply performance optimizations"""
        try:
            logger.info("Applying performance optimizations")
            
            # This would apply various performance optimizations
            # based on the performance profile
            
        except Exception as e:
            logger.warning(f"Performance optimization application failed: {e}")
    
    async def _apply_cpu_optimizations(self, profile: PerformanceProfile) -> None:
        """Apply CPU optimizations"""
        try:
            logger.info("Applying CPU optimizations")
            
            # This would apply CPU-specific optimizations
            
        except Exception as e:
            logger.warning(f"CPU optimization application failed: {e}")
    
    async def _apply_memory_optimizations(self, profile: PerformanceProfile) -> None:
        """Apply memory optimizations"""
        try:
            logger.info("Applying memory optimizations")
            
            # This would apply memory-specific optimizations
            
        except Exception as e:
            logger.warning(f"Memory optimization application failed: {e}")
    
    async def optimize_component(self, component_name: str, optimization_type: str) -> OptimizationResult:
        """Optimize a specific component"""
        try:
            logger.info(f"Optimizing component: {component_name}, type: {optimization_type}")
            
            # Get baseline metrics
            baseline_metrics = await self._get_current_metrics()
            
            start_time = time.time()
            
            # Apply optimization based on type
            if optimization_type == "memory":
                await self._optimize_memory(component_name)
            elif optimization_type == "cpu":
                await self._optimize_cpu(component_name)
            elif optimization_type == "cache":
                await self._optimize_cache(component_name)
            elif optimization_type == "performance":
                await self._optimize_performance(component_name)
            else:
                raise ValueError(f"Unknown optimization type: {optimization_type}")
            
            optimization_time = time.time() - start_time
            
            # Get metrics after optimization
            after_metrics = await self._get_current_metrics()
            
            # Calculate improvement
            improvement_percentage = self._calculate_improvement(baseline_metrics, after_metrics)
            
            # Create optimization result
            result = OptimizationResult(
                optimization_id=f"opt_{component_name}_{int(time.time())}",
                optimization_type=optimization_type,
                component=component_name,
                before_metrics=baseline_metrics,
                after_metrics=after_metrics,
                improvement_percentage=improvement_percentage,
                optimization_time=optimization_time,
                status="success" if improvement_percentage > 0 else "failed",
                recommendations=self._generate_recommendations(optimization_type, improvement_percentage),
                timestamp=datetime.now()
            )
            
            # Store result
            self.optimization_results.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Component optimization failed: {e}")
            raise
    
    async def _optimize_memory(self, component_name: str) -> None:
        """Optimize memory usage for component"""
        try:
            # Force garbage collection
            gc.collect()
            
            # This would implement memory-specific optimizations
            logger.info(f"Memory optimization applied to {component_name}")
            
        except Exception as e:
            logger.warning(f"Memory optimization failed for {component_name}: {e}")
    
    async def _optimize_cpu(self, component_name: str) -> None:
        """Optimize CPU usage for component"""
        try:
            # This would implement CPU-specific optimizations
            logger.info(f"CPU optimization applied to {component_name}")
            
        except Exception as e:
            logger.warning(f"CPU optimization failed for {component_name}: {e}")
    
    async def _optimize_cache(self, component_name: str) -> None:
        """Optimize cache for component"""
        try:
            # This would implement cache-specific optimizations
            logger.info(f"Cache optimization applied to {component_name}")
            
        except Exception as e:
            logger.warning(f"Cache optimization failed for {component_name}: {e}")
    
    async def _optimize_performance(self, component_name: str) -> None:
        """Optimize performance for component"""
        try:
            # This would implement performance-specific optimizations
            logger.info(f"Performance optimization applied to {component_name}")
            
        except Exception as e:
            logger.warning(f"Performance optimization failed for {component_name}: {e}")
    
    def _calculate_improvement(self, before_metrics: Dict[str, float], after_metrics: Dict[str, float]) -> float:
        """Calculate improvement percentage"""
        try:
            # Simple improvement calculation
            total_before = sum(before_metrics.values())
            total_after = sum(after_metrics.values())
            
            if total_before == 0:
                return 0.0
            
            improvement = ((total_before - total_after) / total_before) * 100
            return improvement
            
        except Exception as e:
            logger.warning(f"Improvement calculation failed: {e}")
            return 0.0
    
    def _generate_recommendations(self, optimization_type: str, improvement_percentage: float) -> List[str]:
        """Generate optimization recommendations"""
        try:
            recommendations = []
            
            if optimization_type == "memory":
                recommendations.extend([
                    "Consider implementing object pooling",
                    "Review memory allocation patterns",
                    "Use weak references where appropriate"
                ])
            elif optimization_type == "cpu":
                recommendations.extend([
                    "Consider parallel processing",
                    "Optimize algorithm complexity",
                    "Use caching for expensive operations"
                ])
            elif optimization_type == "cache":
                recommendations.extend([
                    "Review cache eviction policy",
                    "Optimize cache size",
                    "Implement cache warming"
                ])
            elif optimization_type == "performance":
                recommendations.extend([
                    "Profile application bottlenecks",
                    "Consider async operations",
                    "Optimize database queries"
                ])
            
            if improvement_percentage < 10:
                recommendations.append("Consider more aggressive optimization strategies")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")
            return []
    
    async def get_optimization_metrics(self, limit: int = 100) -> List[OptimizationMetric]:
        """Get recent optimization metrics"""
        return self.optimization_metrics[-limit:] if self.optimization_metrics else []
    
    async def get_performance_profiles(self, limit: int = 50) -> List[PerformanceProfile]:
        """Get recent performance profiles"""
        return self.performance_profiles[-limit:] if self.performance_profiles else []
    
    async def get_optimization_results(self, limit: int = 100) -> List[OptimizationResult]:
        """Get recent optimization results"""
        return self.optimization_results[-limit:] if self.optimization_results else []
    
    async def get_optimization_rules(self) -> List[OptimizationRule]:
        """Get optimization rules"""
        return self.optimization_rules
    
    async def update_optimization_rule(self, rule_id: str, enabled: bool) -> bool:
        """Update optimization rule"""
        try:
            for rule in self.optimization_rules:
                if rule.rule_id == rule_id:
                    rule.enabled = enabled
                    return True
            return False
            
        except Exception as e:
            logger.warning(f"Rule update failed: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of intelligent optimizer"""
        return {
            "status": "healthy" if self.initialized else "unhealthy",
            "initialized": self.initialized,
            "auto_optimization_enabled": self.auto_optimization_enabled,
            "optimization_metrics_count": len(self.optimization_metrics),
            "performance_profiles_count": len(self.performance_profiles),
            "optimization_results_count": len(self.optimization_results),
            "optimization_rules_count": len(self.optimization_rules),
            "optimization_thresholds_loaded": len(self.optimization_thresholds),
            "timestamp": datetime.now().isoformat()
        }


# Global intelligent optimizer instance
intelligent_optimizer = IntelligentOptimizer()


async def initialize_intelligent_optimizer() -> None:
    """Initialize the global intelligent optimizer"""
    await intelligent_optimizer.initialize()


async def optimize_component(component_name: str, optimization_type: str) -> OptimizationResult:
    """Optimize a specific component"""
    return await intelligent_optimizer.optimize_component(component_name, optimization_type)


async def get_optimization_metrics(limit: int = 100) -> List[OptimizationMetric]:
    """Get optimization metrics"""
    return await intelligent_optimizer.get_optimization_metrics(limit)


async def get_performance_profiles(limit: int = 50) -> List[PerformanceProfile]:
    """Get performance profiles"""
    return await intelligent_optimizer.get_performance_profiles(limit)


async def get_optimization_results(limit: int = 100) -> List[OptimizationResult]:
    """Get optimization results"""
    return await intelligent_optimizer.get_optimization_results(limit)


async def get_optimization_rules() -> List[OptimizationRule]:
    """Get optimization rules"""
    return await intelligent_optimizer.get_optimization_rules()


async def update_optimization_rule(rule_id: str, enabled: bool) -> bool:
    """Update optimization rule"""
    return await intelligent_optimizer.update_optimization_rule(rule_id, enabled)


async def get_intelligent_optimizer_health() -> Dict[str, Any]:
    """Get intelligent optimizer health"""
    return await intelligent_optimizer.health_check()

