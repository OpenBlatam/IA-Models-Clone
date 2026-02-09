"""
Performance Optimizer - Advanced Performance Optimization Engine
Optimizes system performance, memory usage, database queries, and API responses
"""

import asyncio
import gc
import logging
import psutil
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Performance monitoring
from memory_profiler import profile
import tracemalloc
from line_profiler import LineProfiler

# Caching and optimization
from functools import lru_cache, wraps
import cachetools
from cachetools import TTLCache, LRUCache

# Database optimization
from sqlalchemy import text, select
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.pool import QueuePool, StaticPool

# Async optimization
import aiofiles
import aiohttp
from asyncio import Semaphore, BoundedSemaphore

# Memory optimization
import numpy as np
import pandas as pd
from pympler import tracker, muppy, summary

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_bytes: int = 0
    active_connections: int = 0
    response_time_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    db_query_time_ms: float = 0.0
    db_connection_pool_size: int = 0
    gc_collections: int = 0
    gc_time_ms: float = 0.0


@dataclass
class OptimizationRule:
    """Optimization rule configuration"""
    rule_id: str
    name: str
    description: str
    condition: str
    action: str
    threshold: float
    enabled: bool = True
    priority: int = 1
    cooldown_seconds: int = 60
    last_triggered: Optional[datetime] = None


@dataclass
class OptimizationResult:
    """Optimization result data class"""
    rule_id: str
    timestamp: datetime
    success: bool
    improvement_percent: float
    details: Dict[str, Any]
    execution_time_ms: float


class MemoryOptimizer:
    """Advanced memory optimization"""
    
    def __init__(self):
        self.memory_tracker = tracker.SummaryTracker()
        self.weak_refs: Dict[str, weakref.WeakValueDictionary] = {}
        self.memory_threshold_mb = 1000  # 1GB threshold
        self.gc_threshold = 100  # GC threshold
        
    async def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        start_time = time.time()
        
        # Get current memory usage
        memory_info = psutil.virtual_memory()
        current_memory_mb = memory_info.used / 1024 / 1024
        
        optimizations = []
        
        # Force garbage collection
        if current_memory_mb > self.memory_threshold_mb:
            gc_before = len(gc.get_objects())
            collected = gc.collect()
            gc_after = len(gc.get_objects())
            
            optimizations.append({
                "type": "garbage_collection",
                "objects_collected": collected,
                "objects_before": gc_before,
                "objects_after": gc_after,
                "memory_freed_mb": (gc_before - gc_after) * 0.001  # Rough estimate
            })
        
        # Clear weak references
        cleared_refs = 0
        for name, weak_dict in self.weak_refs.items():
            before_count = len(weak_dict)
            weak_dict.clear()
            cleared_refs += before_count - len(weak_dict)
        
        if cleared_refs > 0:
            optimizations.append({
                "type": "weak_reference_cleanup",
                "references_cleared": cleared_refs
            })
        
        # Memory profiling
        memory_summary = self.memory_tracker.diff()
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "timestamp": datetime.now(),
            "current_memory_mb": current_memory_mb,
            "optimizations": optimizations,
            "memory_summary": memory_summary,
            "execution_time_ms": execution_time
        }
    
    def track_object(self, obj: Any, name: str) -> None:
        """Track object for memory optimization"""
        if name not in self.weak_refs:
            self.weak_refs[name] = weakref.WeakValueDictionary()
        self.weak_refs[name][id(obj)] = obj


class DatabaseOptimizer:
    """Advanced database optimization"""
    
    def __init__(self):
        self.query_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes
        self.connection_pool_size = 20
        self.max_overflow = 30
        self.pool_timeout = 30
        self.pool_recycle = 3600  # 1 hour
        
    async def optimize_queries(self, queries: List[str]) -> Dict[str, Any]:
        """Optimize database queries"""
        start_time = time.time()
        
        optimizations = []
        
        for query in queries:
            # Query analysis
            query_analysis = self._analyze_query(query)
            
            # Suggest optimizations
            suggestions = self._suggest_optimizations(query_analysis)
            
            if suggestions:
                optimizations.append({
                    "query": query,
                    "analysis": query_analysis,
                    "suggestions": suggestions
                })
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "timestamp": datetime.now(),
            "queries_analyzed": len(queries),
            "optimizations": optimizations,
            "execution_time_ms": execution_time
        }
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query for optimization opportunities"""
        analysis = {
            "has_joins": "JOIN" in query.upper(),
            "has_subqueries": "SELECT" in query.upper().split("FROM")[0] if "FROM" in query.upper() else False,
            "has_aggregations": any(func in query.upper() for func in ["COUNT", "SUM", "AVG", "MAX", "MIN"]),
            "has_order_by": "ORDER BY" in query.upper(),
            "has_group_by": "GROUP BY" in query.upper(),
            "has_where": "WHERE" in query.upper(),
            "estimated_complexity": "high" if "JOIN" in query.upper() and "GROUP BY" in query.upper() else "medium"
        }
        
        return analysis
    
    def _suggest_optimizations(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest query optimizations"""
        suggestions = []
        
        if analysis["has_joins"] and not analysis["has_where"]:
            suggestions.append("Add WHERE clause to filter joined tables")
        
        if analysis["has_subqueries"]:
            suggestions.append("Consider using JOINs instead of subqueries")
        
        if analysis["has_aggregations"] and not analysis["has_group_by"]:
            suggestions.append("Add GROUP BY clause for aggregations")
        
        if analysis["estimated_complexity"] == "high":
            suggestions.append("Consider adding indexes on join columns")
            suggestions.append("Consider query result caching")
        
        return suggestions
    
    async def optimize_connection_pool(self) -> Dict[str, Any]:
        """Optimize database connection pool"""
        start_time = time.time()
        
        # Get current pool stats
        pool_stats = {
            "size": self.connection_pool_size,
            "max_overflow": self.max_overflow,
            "timeout": self.pool_timeout,
            "recycle": self.pool_recycle
        }
        
        # Suggest pool optimizations
        suggestions = []
        
        if self.connection_pool_size < 10:
            suggestions.append("Consider increasing pool size for better concurrency")
        
        if self.max_overflow < 20:
            suggestions.append("Consider increasing max overflow for peak loads")
        
        if self.pool_recycle > 7200:  # 2 hours
            suggestions.append("Consider reducing pool recycle time")
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "timestamp": datetime.now(),
            "current_pool_stats": pool_stats,
            "suggestions": suggestions,
            "execution_time_ms": execution_time
        }


class APIOptimizer:
    """Advanced API optimization"""
    
    def __init__(self):
        self.response_cache = TTLCache(maxsize=5000, ttl=600)  # 10 minutes
        self.compression_threshold = 1024  # 1KB
        self.batch_size = 100
        self.max_concurrent_requests = 100
        
    async def optimize_response(self, response_data: Any, endpoint: str) -> Dict[str, Any]:
        """Optimize API response"""
        start_time = time.time()
        
        optimizations = []
        
        # Response size analysis
        response_size = len(str(response_data))
        
        # Compression optimization
        if response_size > self.compression_threshold:
            optimizations.append({
                "type": "compression_recommended",
                "current_size_bytes": response_size,
                "threshold_bytes": self.compression_threshold
            })
        
        # Caching optimization
        cache_key = f"{endpoint}:{hash(str(response_data))}"
        if cache_key not in self.response_cache:
            self.response_cache[cache_key] = response_data
            optimizations.append({
                "type": "response_cached",
                "cache_key": cache_key
            })
        
        # Pagination optimization
        if isinstance(response_data, list) and len(response_data) > self.batch_size:
            optimizations.append({
                "type": "pagination_recommended",
                "current_count": len(response_data),
                "recommended_batch_size": self.batch_size
            })
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "timestamp": datetime.now(),
            "endpoint": endpoint,
            "response_size_bytes": response_size,
            "optimizations": optimizations,
            "execution_time_ms": execution_time
        }
    
    async def optimize_batch_requests(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize batch API requests"""
        start_time = time.time()
        
        # Group requests by endpoint
        endpoint_groups = {}
        for request in requests:
            endpoint = request.get("endpoint", "unknown")
            if endpoint not in endpoint_groups:
                endpoint_groups[endpoint] = []
            endpoint_groups[endpoint].append(request)
        
        # Optimize each group
        optimizations = []
        for endpoint, group_requests in endpoint_groups.items():
            if len(group_requests) > 1:
                optimizations.append({
                    "endpoint": endpoint,
                    "request_count": len(group_requests),
                    "optimization": "batch_processing_recommended"
                })
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "timestamp": datetime.now(),
            "total_requests": len(requests),
            "endpoint_groups": len(endpoint_groups),
            "optimizations": optimizations,
            "execution_time_ms": execution_time
        }


class AsyncOptimizer:
    """Advanced async optimization"""
    
    def __init__(self):
        self.semaphore = Semaphore(100)  # Max 100 concurrent operations
        self.bounded_semaphore = BoundedSemaphore(50)  # Max 50 bounded operations
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.process_pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
        
    async def optimize_concurrency(self, tasks: List[Any]) -> Dict[str, Any]:
        """Optimize async concurrency"""
        start_time = time.time()
        
        # Analyze task types
        task_analysis = {
            "total_tasks": len(tasks),
            "io_bound_tasks": 0,
            "cpu_bound_tasks": 0,
            "mixed_tasks": 0
        }
        
        # Optimize task execution
        async with self.semaphore:
            # Execute tasks with optimal concurrency
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_tasks = sum(1 for r in results if not isinstance(r, Exception))
        failed_tasks = len(results) - successful_tasks
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "timestamp": datetime.now(),
            "task_analysis": task_analysis,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "execution_time_ms": execution_time
        }
    
    async def optimize_io_operations(self, operations: List[Any]) -> Dict[str, Any]:
        """Optimize I/O operations"""
        start_time = time.time()
        
        # Group operations by type
        operation_groups = {}
        for op in operations:
            op_type = type(op).__name__
            if op_type not in operation_groups:
                operation_groups[op_type] = []
            operation_groups[op_type].append(op)
        
        # Optimize each group
        optimizations = []
        for op_type, group_ops in operation_groups.items():
            if len(group_ops) > 1:
                optimizations.append({
                    "operation_type": op_type,
                    "operation_count": len(group_ops),
                    "optimization": "batch_io_processing"
                })
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "timestamp": datetime.now(),
            "total_operations": len(operations),
            "operation_groups": len(operation_groups),
            "optimizations": optimizations,
            "execution_time_ms": execution_time
        }


class PerformanceOptimizer:
    """Main Performance Optimizer Engine"""
    
    def __init__(self):
        self.memory_optimizer = MemoryOptimizer()
        self.db_optimizer = DatabaseOptimizer()
        self.api_optimizer = APIOptimizer()
        self.async_optimizer = AsyncOptimizer()
        
        self.optimization_rules: Dict[str, OptimizationRule] = {}
        self.optimization_history: List[OptimizationResult] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        
        # Initialize default optimization rules
        self._initialize_default_rules()
        
        # Start performance monitoring
        self._start_performance_monitoring()
    
    def _initialize_default_rules(self) -> None:
        """Initialize default optimization rules"""
        default_rules = [
            OptimizationRule(
                rule_id="memory_cleanup",
                name="Memory Cleanup",
                description="Clean up memory when usage exceeds threshold",
                condition="memory_percent > 80",
                action="run_garbage_collection",
                threshold=80.0,
                priority=1
            ),
            OptimizationRule(
                rule_id="cache_optimization",
                name="Cache Optimization",
                description="Optimize cache when hit rate is low",
                condition="cache_hit_rate < 70",
                action="optimize_cache_strategy",
                threshold=70.0,
                priority=2
            ),
            OptimizationRule(
                rule_id="db_connection_optimization",
                name="Database Connection Optimization",
                description="Optimize DB connections when pool is exhausted",
                condition="db_connection_pool_size > 90",
                action="optimize_connection_pool",
                threshold=90.0,
                priority=3
            ),
            OptimizationRule(
                rule_id="response_optimization",
                name="Response Optimization",
                description="Optimize API responses when size is large",
                condition="response_size_bytes > 1000000",
                action="optimize_response_compression",
                threshold=1000000.0,
                priority=4
            )
        ]
        
        for rule in default_rules:
            self.optimization_rules[rule.rule_id] = rule
    
    def _start_performance_monitoring(self) -> None:
        """Start performance monitoring"""
        # Start memory tracking
        tracemalloc.start()
        
        # Start background monitoring task
        asyncio.create_task(self._monitor_performance())
    
    async def _monitor_performance(self) -> None:
        """Monitor system performance"""
        while True:
            try:
                # Collect performance metrics
                metrics = await self._collect_performance_metrics()
                self.performance_metrics.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.performance_metrics) > 1000:
                    self.performance_metrics = self.performance_metrics[-1000:]
                
                # Check optimization rules
                await self._check_optimization_rules(metrics)
                
                # Wait 30 seconds before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Application metrics
        active_connections = len(asyncio.all_tasks())
        
        # GC metrics
        gc_stats = gc.get_stats()
        gc_collections = sum(stat['collections'] for stat in gc_stats)
        gc_time = sum(stat['collected'] for stat in gc_stats)
        
        return PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_available_mb=memory.available / 1024 / 1024,
            disk_usage_percent=disk.percent,
            network_io_bytes=network.bytes_sent + network.bytes_recv,
            active_connections=active_connections,
            gc_collections=gc_collections,
            gc_time_ms=gc_time
        )
    
    async def _check_optimization_rules(self, metrics: PerformanceMetrics) -> None:
        """Check optimization rules and trigger actions"""
        for rule in self.optimization_rules.values():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule.last_triggered:
                time_since_triggered = datetime.now() - rule.last_triggered
                if time_since_triggered.total_seconds() < rule.cooldown_seconds:
                    continue
            
            # Check condition
            if self._evaluate_condition(rule.condition, metrics):
                await self._execute_optimization_action(rule, metrics)
                rule.last_triggered = datetime.now()
    
    def _evaluate_condition(self, condition: str, metrics: PerformanceMetrics) -> bool:
        """Evaluate optimization rule condition"""
        try:
            # Simple condition evaluation
            if "memory_percent" in condition:
                return eval(condition.replace("memory_percent", str(metrics.memory_percent)))
            elif "cache_hit_rate" in condition:
                return eval(condition.replace("cache_hit_rate", str(metrics.cache_hit_rate)))
            elif "db_connection_pool_size" in condition:
                return eval(condition.replace("db_connection_pool_size", str(metrics.db_connection_pool_size)))
            elif "response_size_bytes" in condition:
                return eval(condition.replace("response_size_bytes", str(metrics.response_time_ms)))
            
            return False
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    async def _execute_optimization_action(self, rule: OptimizationRule, metrics: PerformanceMetrics) -> None:
        """Execute optimization action"""
        start_time = time.time()
        
        try:
            if rule.action == "run_garbage_collection":
                result = await self.memory_optimizer.optimize_memory()
            elif rule.action == "optimize_cache_strategy":
                result = await self.optimize_cache_strategy()
            elif rule.action == "optimize_connection_pool":
                result = await self.db_optimizer.optimize_connection_pool()
            elif rule.action == "optimize_response_compression":
                result = await self.optimize_response_compression()
            else:
                result = {"error": f"Unknown action: {rule.action}"}
            
            execution_time = (time.time() - start_time) * 1000
            
            optimization_result = OptimizationResult(
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                success=True,
                improvement_percent=0.0,  # Would need to measure actual improvement
                details=result,
                execution_time_ms=execution_time
            )
            
            self.optimization_history.append(optimization_result)
            
            logger.info(f"Optimization rule '{rule.name}' executed successfully")
            
        except Exception as e:
            logger.error(f"Error executing optimization action '{rule.action}': {e}")
            
            optimization_result = OptimizationResult(
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                success=False,
                improvement_percent=0.0,
                details={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
            self.optimization_history.append(optimization_result)
    
    async def optimize_system(self) -> Dict[str, Any]:
        """Optimize entire system"""
        start_time = time.time()
        
        optimizations = {}
        
        # Memory optimization
        optimizations["memory"] = await self.memory_optimizer.optimize_memory()
        
        # Database optimization
        optimizations["database"] = await self.db_optimizer.optimize_connection_pool()
        
        # API optimization
        optimizations["api"] = await self.api_optimizer.optimize_batch_requests([])
        
        # Async optimization
        optimizations["async"] = await self.async_optimizer.optimize_io_operations([])
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "timestamp": datetime.now(),
            "optimizations": optimizations,
            "execution_time_ms": execution_time,
            "total_optimizations": len(optimizations)
        }
    
    async def optimize_cache_strategy(self) -> Dict[str, Any]:
        """Optimize cache strategy"""
        start_time = time.time()
        
        # Analyze cache performance
        cache_stats = {
            "memory_cache_size": len(self.api_optimizer.response_cache),
            "db_cache_size": len(self.db_optimizer.query_cache),
            "memory_cache_hits": getattr(self.api_optimizer.response_cache, 'hits', 0),
            "memory_cache_misses": getattr(self.api_optimizer.response_cache, 'misses', 0)
        }
        
        # Calculate hit rates
        total_memory_requests = cache_stats["memory_cache_hits"] + cache_stats["memory_cache_misses"]
        memory_hit_rate = (cache_stats["memory_cache_hits"] / total_memory_requests * 100) if total_memory_requests > 0 else 0
        
        # Suggest optimizations
        suggestions = []
        
        if memory_hit_rate < 70:
            suggestions.append("Consider increasing cache size or TTL")
            suggestions.append("Review cache key strategy")
        
        if cache_stats["memory_cache_size"] > 4000:
            suggestions.append("Consider reducing cache size to free memory")
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "timestamp": datetime.now(),
            "cache_stats": cache_stats,
            "memory_hit_rate": memory_hit_rate,
            "suggestions": suggestions,
            "execution_time_ms": execution_time
        }
    
    async def optimize_response_compression(self) -> Dict[str, Any]:
        """Optimize response compression"""
        start_time = time.time()
        
        # Analyze response patterns
        response_analysis = {
            "compression_threshold": self.api_optimizer.compression_threshold,
            "batch_size": self.api_optimizer.batch_size,
            "max_concurrent_requests": self.api_optimizer.max_concurrent_requests
        }
        
        # Suggest optimizations
        suggestions = []
        
        if self.api_optimizer.compression_threshold > 2048:
            suggestions.append("Consider reducing compression threshold for better performance")
        
        if self.api_optimizer.batch_size < 50:
            suggestions.append("Consider increasing batch size for better throughput")
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "timestamp": datetime.now(),
            "response_analysis": response_analysis,
            "suggestions": suggestions,
            "execution_time_ms": execution_time
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if not self.performance_metrics:
            return {"error": "No performance metrics available"}
        
        latest_metrics = self.performance_metrics[-1]
        
        return {
            "timestamp": latest_metrics.timestamp,
            "cpu_percent": latest_metrics.cpu_percent,
            "memory_percent": latest_metrics.memory_percent,
            "memory_used_mb": latest_metrics.memory_used_mb,
            "memory_available_mb": latest_metrics.memory_available_mb,
            "disk_usage_percent": latest_metrics.disk_usage_percent,
            "active_connections": latest_metrics.active_connections,
            "gc_collections": latest_metrics.gc_collections,
            "gc_time_ms": latest_metrics.gc_time_ms
        }
    
    async def get_optimization_history(self) -> Dict[str, Any]:
        """Get optimization history"""
        return {
            "timestamp": datetime.now(),
            "total_optimizations": len(self.optimization_history),
            "successful_optimizations": sum(1 for r in self.optimization_history if r.success),
            "failed_optimizations": sum(1 for r in self.optimization_history if not r.success),
            "recent_optimizations": [
                {
                    "rule_id": r.rule_id,
                    "timestamp": r.timestamp,
                    "success": r.success,
                    "improvement_percent": r.improvement_percent,
                    "execution_time_ms": r.execution_time_ms
                }
                for r in self.optimization_history[-10:]  # Last 10 optimizations
            ]
        }
    
    async def get_optimization_rules(self) -> Dict[str, Any]:
        """Get optimization rules"""
        return {
            "timestamp": datetime.now(),
            "total_rules": len(self.optimization_rules),
            "enabled_rules": sum(1 for r in self.optimization_rules.values() if r.enabled),
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "condition": rule.condition,
                    "action": rule.action,
                    "threshold": rule.threshold,
                    "enabled": rule.enabled,
                    "priority": rule.priority,
                    "last_triggered": rule.last_triggered
                }
                for rule in self.optimization_rules.values()
            ]
        }
    
    async def add_optimization_rule(self, rule: OptimizationRule) -> Dict[str, Any]:
        """Add new optimization rule"""
        self.optimization_rules[rule.rule_id] = rule
        
        return {
            "timestamp": datetime.now(),
            "message": f"Optimization rule '{rule.name}' added successfully",
            "rule_id": rule.rule_id
        }
    
    async def update_optimization_rule(self, rule_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update optimization rule"""
        if rule_id not in self.optimization_rules:
            return {"error": f"Rule '{rule_id}' not found"}
        
        rule = self.optimization_rules[rule_id]
        
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        return {
            "timestamp": datetime.now(),
            "message": f"Optimization rule '{rule_id}' updated successfully"
        }
    
    async def delete_optimization_rule(self, rule_id: str) -> Dict[str, Any]:
        """Delete optimization rule"""
        if rule_id not in self.optimization_rules:
            return {"error": f"Rule '{rule_id}' not found"}
        
        rule_name = self.optimization_rules[rule_id].name
        del self.optimization_rules[rule_id]
        
        return {
            "timestamp": datetime.now(),
            "message": f"Optimization rule '{rule_name}' deleted successfully"
        }


# Global instance
performance_optimizer = PerformanceOptimizer()


async def initialize_performance_optimizer() -> None:
    """Initialize performance optimizer"""
    global performance_optimizer
    performance_optimizer = PerformanceOptimizer()
    logger.info("Performance Optimizer initialized successfully")


async def get_performance_optimizer() -> PerformanceOptimizer:
    """Get performance optimizer instance"""
    return performance_optimizer