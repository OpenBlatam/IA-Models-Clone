#!/usr/bin/env python3
"""
Enhanced Performance Optimizer
==============================

Provides advanced performance optimization capabilities including:
- Multi-level intelligent caching
- Advanced load balancing algorithms
- Resource optimization and monitoring
- Background task processing
- Performance analytics and metrics
"""

import asyncio
import logging
import time
import json
import statistics
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
import queue
import weakref

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache levels in the multi-level caching system."""
    L1_MEMORY = "l1_memory"      # Fastest, in-memory
    L2_REDIS = "l2_redis"        # Medium, Redis
    L3_CDN = "l3_cdn"            # Slowest, CDN

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    ADAPTIVE = "adaptive"

class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    DISK = "disk"
    DATABASE = "database"

@dataclass
class CacheEntry:
    """Represents a cache entry."""
    key: str
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ServerNode:
    """Represents a server node for load balancing."""
    id: str
    url: str
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    response_time: float = 0.0
    health_score: float = 1.0
    last_health_check: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    timestamp: float
    response_time: float
    throughput: float
    error_rate: float
    resource_usage: Dict[ResourceType, float]
    cache_hit_rate: float
    load_balancer_stats: Dict[str, Any]

class EnhancedPerformanceOptimizer:
    """
    Enhanced performance optimizer with advanced caching and load balancing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.cache_levels = self._initialize_cache_levels()
        self.load_balancer = self._initialize_load_balancer()
        self.background_workers = self._initialize_background_workers()
        self.performance_monitor = self._initialize_performance_monitor()
        self.resource_monitor = self._initialize_resource_monitor()
        
        # Start monitoring threads
        self._start_monitoring()
    
    def _initialize_cache_levels(self) -> Dict[CacheLevel, Dict[str, Any]]:
        """Initialize multi-level cache system."""
        return {
            CacheLevel.L1_MEMORY: {
                "cache": {},
                "max_size": self.config.get("l1_max_size", 1000),
                "ttl": self.config.get("l1_ttl", 300),  # 5 minutes
                "eviction_policy": "lru"
            },
            CacheLevel.L2_REDIS: {
                "cache": {},  # Simulated Redis
                "max_size": self.config.get("l2_max_size", 10000),
                "ttl": self.config.get("l2_ttl", 3600),  # 1 hour
                "eviction_policy": "lru"
            },
            CacheLevel.L3_CDN: {
                "cache": {},  # Simulated CDN
                "max_size": self.config.get("l3_max_size", 100000),
                "ttl": self.config.get("l3_ttl", 86400),  # 24 hours
                "eviction_policy": "lru"
            }
        }
    
    def _initialize_load_balancer(self) -> Dict[str, Any]:
        """Initialize load balancer."""
        return {
            "strategy": LoadBalancingStrategy.ROUND_ROBIN,
            "nodes": [],
            "current_node_index": 0,
            "health_check_interval": 30.0,
            "circuit_breaker_threshold": 5,
            "circuit_breaker_timeout": 60.0
        }
    
    def _initialize_background_workers(self) -> Dict[str, Any]:
        """Initialize background worker system."""
        return {
            "worker_pool": [],
            "task_queue": queue.Queue(),
            "max_workers": self.config.get("max_workers", 10),
            "worker_timeout": self.config.get("worker_timeout", 300)
        }
    
    def _initialize_performance_monitor(self) -> Dict[str, Any]:
        """Initialize performance monitoring."""
        return {
            "metrics": deque(maxlen=1000),
            "alerts": [],
            "thresholds": {
                "response_time": 1.0,  # seconds
                "error_rate": 0.05,    # 5%
                "cpu_usage": 0.8,      # 80%
                "memory_usage": 0.8    # 80%
            }
        }
    
    def _initialize_resource_monitor(self) -> Dict[str, Any]:
        """Initialize resource monitoring."""
        return {
            "monitors": {},
            "history": defaultdict(lambda: deque(maxlen=100)),
            "alerts": []
        }
    
    def _start_monitoring(self):
        """Start background monitoring threads."""
        # Start performance monitoring
        self._monitor_thread = threading.Thread(target=self._performance_monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        # Start resource monitoring
        self._resource_thread = threading.Thread(target=self._resource_monitoring_loop, daemon=True)
        self._resource_thread.start()
        
        # Start background workers
        self._start_background_workers()
    
    def _start_background_workers(self):
        """Start background worker threads."""
        for i in range(self.background_workers["max_workers"]):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.background_workers["worker_pool"].append(worker)
    
    async def get_cached_value(self, key: str, default: Any = None) -> Any:
        """
        Get value from multi-level cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        try:
            # Try L1 cache first (fastest)
            l1_cache = self.cache_levels[CacheLevel.L1_MEMORY]["cache"]
            if key in l1_cache:
                entry = l1_cache[key]
                if not self._is_expired(entry):
                    self._update_cache_access(entry)
                    return entry.value
            
            # Try L2 cache
            l2_cache = self.cache_levels[CacheLevel.L2_REDIS]["cache"]
            if key in l2_cache:
                entry = l2_cache[key]
                if not self._is_expired(entry):
                    # Promote to L1 cache
                    await self._promote_to_l1(key, entry)
                    return entry.value
            
            # Try L3 cache
            l3_cache = self.cache_levels[CacheLevel.L3_CDN]["cache"]
            if key in l3_cache:
                entry = l3_cache[key]
                if not self._is_expired(entry):
                    # Promote to L2 cache
                    await self._promote_to_l2(key, entry)
                    return entry.value
            
            return default
            
        except Exception as e:
            logger.error(f"Error getting cached value: {e}")
            return default
    
    async def set_cached_value(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        level: CacheLevel = CacheLevel.L1_MEMORY,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            level: Cache level to use
            metadata: Additional metadata
        """
        try:
            cache_config = self.cache_levels[level]
            cache = cache_config["cache"]
            
            # Calculate TTL
            if ttl is None:
                ttl = cache_config["ttl"]
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=len(str(value)),
                metadata=metadata or {}
            )
            
            # Check cache size limits
            if len(cache) >= cache_config["max_size"]:
                await self._evict_cache_entries(level)
            
            # Store entry
            cache[key] = entry
            
            # Log cache operation
            logger.debug(f"Cached value at {level.value}: {key}")
            
        except Exception as e:
            logger.error(f"Error setting cached value: {e}")
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        return time.time() - entry.timestamp > entry.ttl
    
    def _update_cache_access(self, entry: CacheEntry):
        """Update cache access statistics."""
        entry.access_count += 1
        entry.last_access = time.time()
    
    async def _promote_to_l1(self, key: str, entry: CacheEntry):
        """Promote entry from L2 to L1 cache."""
        try:
            l1_cache = self.cache_levels[CacheLevel.L1_MEMORY]["cache"]
            
            # Check L1 cache size
            if len(l1_cache) >= self.cache_levels[CacheLevel.L1_MEMORY]["max_size"]:
                await self._evict_cache_entries(CacheLevel.L1_MEMORY)
            
            # Create new entry with L1 TTL
            l1_entry = CacheEntry(
                key=key,
                value=entry.value,
                timestamp=time.time(),
                ttl=self.cache_levels[CacheLevel.L1_MEMORY]["ttl"],
                access_count=entry.access_count,
                last_access=entry.last_access,
                size_bytes=entry.size_bytes,
                metadata=entry.metadata
            )
            
            l1_cache[key] = l1_entry
            
        except Exception as e:
            logger.error(f"Error promoting to L1 cache: {e}")
    
    async def _promote_to_l2(self, key: str, entry: CacheEntry):
        """Promote entry from L3 to L2 cache."""
        try:
            l2_cache = self.cache_levels[CacheLevel.L2_REDIS]["cache"]
            
            # Check L2 cache size
            if len(l2_cache) >= self.cache_levels[CacheLevel.L2_REDIS]["max_size"]:
                await self._evict_cache_entries(CacheLevel.L2_REDIS)
            
            # Create new entry with L2 TTL
            l2_entry = CacheEntry(
                key=key,
                value=entry.value,
                timestamp=time.time(),
                ttl=self.cache_levels[CacheLevel.L2_REDIS]["ttl"],
                access_count=entry.access_count,
                last_access=entry.last_access,
                size_bytes=entry.size_bytes,
                metadata=entry.metadata
            )
            
            l2_cache[key] = l2_entry
            
        except Exception as e:
            logger.error(f"Error promoting to L2 cache: {e}")
    
    async def _evict_cache_entries(self, level: CacheLevel):
        """Evict entries from cache based on eviction policy."""
        try:
            cache_config = self.cache_levels[level]
            cache = cache_config["cache"]
            policy = cache_config["eviction_policy"]
            
            if policy == "lru":
                # Remove least recently used entries
                entries = sorted(cache.values(), key=lambda x: x.last_access)
                entries_to_remove = len(cache) - cache_config["max_size"] + 1
                
                for entry in entries[:entries_to_remove]:
                    del cache[entry.key]
                    
            elif policy == "lfu":
                # Remove least frequently used entries
                entries = sorted(cache.values(), key=lambda x: x.access_count)
                entries_to_remove = len(cache) - cache_config["max_size"] + 1
                
                for entry in entries[:entries_to_remove]:
                    del cache[entry.key]
            
            logger.debug(f"Evicted entries from {level.value} cache")
            
        except Exception as e:
            logger.error(f"Error evicting cache entries: {e}")
    
    async def add_server_node(self, node: ServerNode):
        """Add a server node to the load balancer."""
        try:
            self.load_balancer["nodes"].append(node)
            logger.info(f"Added server node: {node.id}")
        except Exception as e:
            logger.error(f"Error adding server node: {e}")
    
    async def remove_server_node(self, node_id: str):
        """Remove a server node from the load balancer."""
        try:
            self.load_balancer["nodes"] = [
                node for node in self.load_balancer["nodes"] 
                if node.id != node_id
            ]
            logger.info(f"Removed server node: {node_id}")
        except Exception as e:
            logger.error(f"Error removing server node: {e}")
    
    async def get_next_server(self, strategy: Optional[LoadBalancingStrategy] = None) -> Optional[ServerNode]:
        """
        Get next server based on load balancing strategy.
        
        Args:
            strategy: Load balancing strategy to use
            
        Returns:
            Next server node or None if no nodes available
        """
        try:
            if not self.load_balancer["nodes"]:
                return None
            
            strategy = strategy or self.load_balancer["strategy"]
            
            if strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return await self._round_robin_balancing()
            elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return await self._least_connections_balancing()
            elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return await self._weighted_round_robin_balancing()
            elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return await self._least_response_time_balancing()
            elif strategy == LoadBalancingStrategy.IP_HASH:
                return await self._ip_hash_balancing()
            elif strategy == LoadBalancingStrategy.ADAPTIVE:
                return await self._adaptive_balancing()
            else:
                return await self._round_robin_balancing()
                
        except Exception as e:
            logger.error(f"Error getting next server: {e}")
            return None
    
    async def _round_robin_balancing(self) -> Optional[ServerNode]:
        """Round-robin load balancing."""
        nodes = self.load_balancer["nodes"]
        if not nodes:
            return None
        
        current_index = self.load_balancer["current_node_index"]
        node = nodes[current_index]
        
        # Update index for next request
        self.load_balancer["current_node_index"] = (current_index + 1) % len(nodes)
        
        return node
    
    async def _least_connections_balancing(self) -> Optional[ServerNode]:
        """Least connections load balancing."""
        nodes = self.load_balancer["nodes"]
        if not nodes:
            return None
        
        # Find node with least connections
        return min(nodes, key=lambda x: x.current_connections)
    
    async def _weighted_round_robin_balancing(self) -> Optional[ServerNode]:
        """Weighted round-robin load balancing."""
        nodes = self.load_balancer["nodes"]
        if not nodes:
            return None
        
        # Calculate total weight
        total_weight = sum(node.weight for node in nodes)
        
        # Use current index to determine which node
        current_index = self.load_balancer["current_node_index"]
        current_weight = 0
        
        for node in nodes:
            current_weight += node.weight
            if current_index < current_weight:
                # Update index for next request
                self.load_balancer["current_node_index"] = (current_index + 1) % total_weight
                return node
        
        # Fallback to first node
        return nodes[0]
    
    async def _least_response_time_balancing(self) -> Optional[ServerNode]:
        """Least response time load balancing."""
        nodes = self.load_balancer["nodes"]
        if not nodes:
            return None
        
        # Find node with best response time
        return min(nodes, key=lambda x: x.response_time)
    
    async def _ip_hash_balancing(self) -> Optional[ServerNode]:
        """IP hash load balancing."""
        nodes = self.load_balancer["nodes"]
        if not nodes:
            return None
        
        # For now, use a simple hash of current time
        # In a real implementation, you'd hash the client IP
        hash_value = hash(str(time.time()))
        node_index = hash_value % len(nodes)
        
        return nodes[node_index]
    
    async def _adaptive_balancing(self) -> Optional[ServerNode]:
        """Adaptive load balancing based on multiple factors."""
        nodes = self.load_balancer["nodes"]
        if not nodes:
            return None
        
        # Calculate score for each node
        node_scores = []
        for node in nodes:
            # Score based on health, connections, and response time
            health_score = node.health_score
            connection_score = 1.0 - (node.current_connections / node.max_connections)
            response_score = 1.0 / (1.0 + node.response_time)
            
            # Weighted combination
            total_score = (
                health_score * 0.4 +
                connection_score * 0.3 +
                response_score * 0.3
            )
            
            node_scores.append((node, total_score))
        
        # Return node with highest score
        return max(node_scores, key=lambda x: x[1])[0]
    
    async def submit_background_task(self, task_func: Callable, *args, **kwargs) -> str:
        """
        Submit a task for background processing.
        
        Args:
            task_func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        try:
            task_id = f"task_{int(time.time() * 1000)}"
            
            # Create task
            task = {
                "id": task_id,
                "func": task_func,
                "args": args,
                "kwargs": kwargs,
                "submitted_at": time.time(),
                "status": "pending"
            }
            
            # Add to task queue
            self.background_workers["task_queue"].put(task)
            
            logger.info(f"Submitted background task: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error submitting background task: {e}")
            return ""
    
    def _worker_loop(self, worker_id: int):
        """Background worker loop."""
        while True:
            try:
                # Get task from queue
                task = self.background_workers["task_queue"].get(timeout=1)
                
                # Execute task
                task["status"] = "running"
                start_time = time.time()
                
                try:
                    result = task["func"](*task["args"], **task["kwargs"])
                    task["status"] = "completed"
                    task["result"] = result
                    task["execution_time"] = time.time() - start_time
                    
                    logger.info(f"Worker {worker_id} completed task: {task['id']}")
                    
                except Exception as e:
                    task["status"] = "failed"
                    task["error"] = str(e)
                    task["execution_time"] = time.time() - start_time
                    
                    logger.error(f"Worker {worker_id} failed task: {task['id']}: {e}")
                
                # Mark task as done
                self.background_workers["task_queue"].task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                time.sleep(1)
    
    def _performance_monitoring_loop(self):
        """Performance monitoring loop."""
        while True:
            try:
                # Collect performance metrics
                metrics = self._collect_performance_metrics()
                
                # Add to metrics history
                self.performance_monitor["metrics"].append(metrics)
                
                # Check for alerts
                self._check_performance_alerts(metrics)
                
                # Sleep for monitoring interval
                time.sleep(10)  # 10 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(10)
    
    def _resource_monitoring_loop(self):
        """Resource monitoring loop."""
        while True:
            try:
                # Collect resource metrics
                resource_metrics = self._collect_resource_metrics()
                
                # Update resource history
                for resource_type, value in resource_metrics.items():
                    self.resource_monitor["history"][resource_type].append({
                        "timestamp": time.time(),
                        "value": value
                    })
                
                # Check for resource alerts
                self._check_resource_alerts(resource_metrics)
                
                # Sleep for monitoring interval
                time.sleep(30)  # 30 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(30)
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            # Calculate cache hit rate
            total_requests = sum(
                len(cache) for cache in self.cache_levels.values()
            )
            cache_hits = sum(
                sum(1 for entry in cache.values() if entry.access_count > 0)
                for cache in self.cache_levels.values()
            )
            cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
            
            # Calculate load balancer stats
            lb_stats = {
                "total_nodes": len(self.load_balancer["nodes"]),
                "active_nodes": sum(1 for node in self.load_balancer["nodes"] if node.health_score > 0.5),
                "total_connections": sum(node.current_connections for node in self.load_balancer["nodes"]),
                "avg_response_time": statistics.mean([node.response_time for node in self.load_balancer["nodes"]]) if self.load_balancer["nodes"] else 0.0
            }
            
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                response_time=lb_stats["avg_response_time"],
                throughput=lb_stats["total_connections"],
                error_rate=0.0,  # Would be calculated from actual request logs
                resource_usage={},
                cache_hit_rate=cache_hit_rate,
                load_balancer_stats=lb_stats
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return PerformanceMetrics(
                timestamp=time.time(),
                response_time=0.0,
                throughput=0.0,
                error_rate=0.0,
                resource_usage={},
                cache_hit_rate=0.0,
                load_balancer_stats={}
            )
    
    def _collect_resource_metrics(self) -> Dict[ResourceType, float]:
        """Collect current resource usage metrics."""
        try:
            # This is a simplified implementation
            # In a real system, you'd collect actual system metrics
            return {
                ResourceType.CPU: 0.3,      # 30% CPU usage
                ResourceType.MEMORY: 0.5,   # 50% memory usage
                ResourceType.NETWORK: 0.2,  # 20% network usage
                ResourceType.DISK: 0.4      # 40% disk usage
            }
        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")
            return {}
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts."""
        try:
            thresholds = self.performance_monitor["thresholds"]
            
            if metrics.response_time > thresholds["response_time"]:
                alert = {
                    "type": "high_response_time",
                    "message": f"Response time {metrics.response_time}s exceeds threshold {thresholds['response_time']}s",
                    "timestamp": time.time(),
                    "severity": "warning"
                }
                self.performance_monitor["alerts"].append(alert)
                logger.warning(alert["message"])
            
            if metrics.error_rate > thresholds["error_rate"]:
                alert = {
                    "type": "high_error_rate",
                    "message": f"Error rate {metrics.error_rate} exceeds threshold {thresholds['error_rate']}",
                    "timestamp": time.time(),
                    "severity": "critical"
                }
                self.performance_monitor["alerts"].append(alert)
                logger.error(alert["message"])
                
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    def _check_resource_alerts(self, resource_metrics: Dict[ResourceType, float]):
        """Check for resource alerts."""
        try:
            thresholds = self.performance_monitor["thresholds"]
            
            for resource_type, usage in resource_metrics.items():
                if resource_type in [ResourceType.CPU, ResourceType.MEMORY]:
                    threshold = thresholds.get(f"{resource_type.value}_usage", 0.8)
                    
                    if usage > threshold:
                        alert = {
                            "type": f"high_{resource_type.value}_usage",
                            "message": f"{resource_type.value} usage {usage:.1%} exceeds threshold {threshold:.1%}",
                            "timestamp": time.time(),
                            "severity": "warning"
                        }
                        self.resource_monitor["alerts"].append(alert)
                        logger.warning(alert["message"])
                        
        except Exception as e:
            logger.error(f"Error checking resource alerts: {e}")
    
    async def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        try:
            # Cache statistics
            cache_stats = {}
            for level, config in self.cache_levels.items():
                cache = config["cache"]
                cache_stats[level.value] = {
                    "size": len(cache),
                    "max_size": config["max_size"],
                    "utilization": len(cache) / config["max_size"] if config["max_size"] > 0 else 0.0
                }
            
            # Load balancer statistics
            lb_stats = {
                "total_nodes": len(self.load_balancer["nodes"]),
                "active_nodes": sum(1 for node in self.load_balancer["nodes"] if node.health_score > 0.5),
                "strategy": self.load_balancer["strategy"].value,
                "total_connections": sum(node.current_connections for node in self.load_balancer["nodes"])
            }
            
            # Background worker statistics
            worker_stats = {
                "total_workers": self.background_workers["max_workers"],
                "active_workers": len(self.background_workers["worker_pool"]),
                "queue_size": self.background_workers["task_queue"].qsize()
            }
            
            # Performance metrics
            if self.performance_monitor["metrics"]:
                recent_metrics = list(self.performance_monitor["metrics"])[-10:]
                avg_response_time = statistics.mean([m.response_time for m in recent_metrics])
                avg_cache_hit_rate = statistics.mean([m.cache_hit_rate for m in recent_metrics])
            else:
                avg_response_time = 0.0
                avg_cache_hit_rate = 0.0
            
            return {
                "cache_statistics": cache_stats,
                "load_balancer_statistics": lb_stats,
                "worker_statistics": worker_stats,
                "performance_metrics": {
                    "average_response_time": avg_response_time,
                    "average_cache_hit_rate": avg_cache_hit_rate,
                    "total_metrics_collected": len(self.performance_monitor["metrics"])
                },
                "alerts": {
                    "performance_alerts": len(self.performance_monitor["alerts"]),
                    "resource_alerts": len(self.resource_monitor["alerts"])
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance statistics: {e}")
            return {}
    
    async def clear_cache(self, level: Optional[CacheLevel] = None):
        """Clear cache at specified level or all levels."""
        try:
            if level:
                self.cache_levels[level]["cache"].clear()
                logger.info(f"Cleared {level.value} cache")
            else:
                for level_name, config in self.cache_levels.items():
                    config["cache"].clear()
                logger.info("Cleared all cache levels")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    async def optimize_performance(self):
        """Run performance optimization routines."""
        try:
            # Optimize cache levels
            await self._optimize_cache_levels()
            
            # Optimize load balancer
            await self._optimize_load_balancer()
            
            # Clean up expired entries
            await self._cleanup_expired_entries()
            
            logger.info("Performance optimization completed")
            
        except Exception as e:
            logger.error(f"Error during performance optimization: {e}")
    
    async def _optimize_cache_levels(self):
        """Optimize cache level configurations."""
        try:
            # Analyze cache hit rates and adjust TTLs
            for level, config in self.cache_levels.items():
                cache = config["cache"]
                if len(cache) > 0:
                    # Calculate average access frequency
                    avg_access_freq = statistics.mean([
                        entry.access_count / (time.time() - entry.timestamp + 1)
                        for entry in cache.values()
                    ])
                    
                    # Adjust TTL based on access frequency
                    if avg_access_freq > 0.1:  # High frequency access
                        config["ttl"] = min(config["ttl"] * 1.2, 3600)  # Increase TTL
                    elif avg_access_freq < 0.01:  # Low frequency access
                        config["ttl"] = max(config["ttl"] * 0.8, 60)    # Decrease TTL
                        
        except Exception as e:
            logger.error(f"Error optimizing cache levels: {e}")
    
    async def _optimize_load_balancer(self):
        """Optimize load balancer configuration."""
        try:
            # Update health scores based on recent performance
            for node in self.load_balancer["nodes"]:
                # Simple health score calculation
                if node.response_time < 0.1:
                    node.health_score = min(1.0, node.health_score + 0.1)
                elif node.response_time > 1.0:
                    node.health_score = max(0.0, node.health_score - 0.1)
                
                # Connection-based health adjustment
                connection_ratio = node.current_connections / node.max_connections
                if connection_ratio > 0.8:
                    node.health_score = max(0.0, node.health_score - 0.05)
                    
        except Exception as e:
            logger.error(f"Error optimizing load balancer: {e}")
    
    async def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        try:
            for level, config in self.cache_levels.items():
                cache = config["cache"]
                expired_keys = [
                    key for key, entry in cache.items()
                    if self._is_expired(entry)
                ]
                
                for key in expired_keys:
                    del cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired entries from {level.value}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")

