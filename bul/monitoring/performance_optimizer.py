"""
Ultimate BUL System - Performance Optimization & Monitoring
Advanced performance optimization with real-time monitoring and auto-scaling
"""

import asyncio
import time
import psutil
import gc
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
from collections import deque, defaultdict
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiohttp
import redis
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

class PerformanceMetric(str, Enum):
    """Performance metrics to track"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    QUEUE_SIZE = "queue_size"
    CONNECTION_POOL = "connection_pool"

class OptimizationStrategy(str, Enum):
    """Optimization strategies"""
    AUTO_SCALING = "auto_scaling"
    CACHE_OPTIMIZATION = "cache_optimization"
    CONNECTION_POOLING = "connection_pooling"
    QUERY_OPTIMIZATION = "query_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    CPU_OPTIMIZATION = "cpu_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"
    DATABASE_OPTIMIZATION = "database_optimization"

@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric: PerformanceMetric
    warning_threshold: float
    critical_threshold: float
    optimization_threshold: float
    unit: str = "ms"
    enabled: bool = True

@dataclass
class PerformanceAlert:
    """Performance alert"""
    id: str
    metric: PerformanceMetric
    value: float
    threshold: float
    severity: str
    timestamp: datetime
    message: str
    resolved: bool = False

@dataclass
class OptimizationAction:
    """Optimization action"""
    id: str
    strategy: OptimizationStrategy
    description: str
    parameters: Dict[str, Any]
    expected_improvement: float
    risk_level: str
    executed: bool = False
    result: Optional[Dict[str, Any]] = None

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_usage_percent: float
    disk_io_read: int
    disk_io_write: int
    network_sent: int
    network_recv: int
    process_count: int
    thread_count: int
    load_average: Tuple[float, float, float]

@dataclass
class ApplicationMetrics:
    """Application performance metrics"""
    timestamp: datetime
    active_connections: int
    request_count: int
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    error_count: int
    cache_hits: int
    cache_misses: int
    queue_size: int
    worker_count: int

class PerformanceOptimizer:
    """Advanced performance optimizer with real-time monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history = deque(maxlen=1000)
        self.alerts = []
        self.optimization_actions = []
        self.thresholds = self._initialize_thresholds()
        self.monitoring_active = False
        self.optimization_active = False
        
        # Prometheus metrics
        self.prometheus_metrics = self._initialize_prometheus_metrics()
        
        # Performance tracking
        self.performance_data = defaultdict(list)
        self.baseline_metrics = {}
        self.optimization_history = []
        
        # Auto-scaling configuration
        self.auto_scaling_config = {
            "enabled": True,
            "min_instances": 1,
            "max_instances": 10,
            "scale_up_threshold": 0.8,
            "scale_down_threshold": 0.3,
            "cooldown_period": 300,  # 5 minutes
            "last_scale_time": None
        }
        
        # Cache optimization
        self.cache_optimization_config = {
            "enabled": True,
            "target_hit_rate": 0.9,
            "eviction_policy": "lru",
            "max_memory_usage": 0.8,
            "optimization_interval": 60  # 1 minute
        }
        
        # Connection pooling
        self.connection_pool_config = {
            "enabled": True,
            "max_connections": 100,
            "min_connections": 10,
            "connection_timeout": 30,
            "idle_timeout": 300
        }
        
        # Start monitoring
        self.start_monitoring()
    
    def _initialize_thresholds(self) -> List[PerformanceThreshold]:
        """Initialize performance thresholds"""
        return [
            PerformanceThreshold(
                metric=PerformanceMetric.RESPONSE_TIME,
                warning_threshold=1000,  # 1 second
                critical_threshold=5000,  # 5 seconds
                optimization_threshold=500,  # 500ms
                unit="ms"
            ),
            PerformanceThreshold(
                metric=PerformanceMetric.CPU_USAGE,
                warning_threshold=70,  # 70%
                critical_threshold=90,  # 90%
                optimization_threshold=50,  # 50%
                unit="%"
            ),
            PerformanceThreshold(
                metric=PerformanceMetric.MEMORY_USAGE,
                warning_threshold=80,  # 80%
                critical_threshold=95,  # 95%
                optimization_threshold=60,  # 60%
                unit="%"
            ),
            PerformanceThreshold(
                metric=PerformanceMetric.ERROR_RATE,
                warning_threshold=5,  # 5%
                critical_threshold=10,  # 10%
                optimization_threshold=1,  # 1%
                unit="%"
            ),
            PerformanceThreshold(
                metric=PerformanceMetric.CACHE_HIT_RATE,
                warning_threshold=70,  # 70%
                critical_threshold=50,  # 50%
                optimization_threshold=90,  # 90%
                unit="%"
            )
        ]
    
    def _initialize_prometheus_metrics(self) -> Dict[str, Any]:
        """Initialize Prometheus metrics"""
        return {
            "response_time": Histogram(
                "bul_response_time_seconds",
                "Response time in seconds",
                ["endpoint", "method", "status_code"]
            ),
            "request_count": Counter(
                "bul_requests_total",
                "Total number of requests",
                ["endpoint", "method", "status_code"]
            ),
            "active_connections": Gauge(
                "bul_active_connections",
                "Number of active connections"
            ),
            "cpu_usage": Gauge(
                "bul_cpu_usage_percent",
                "CPU usage percentage"
            ),
            "memory_usage": Gauge(
                "bul_memory_usage_percent",
                "Memory usage percentage"
            ),
            "cache_hit_rate": Gauge(
                "bul_cache_hit_rate",
                "Cache hit rate"
            ),
            "error_rate": Gauge(
                "bul_error_rate",
                "Error rate percentage"
            ),
            "queue_size": Gauge(
                "bul_queue_size",
                "Queue size"
            ),
            "optimization_actions": Counter(
                "bul_optimization_actions_total",
                "Total optimization actions",
                ["strategy", "result"]
            )
        }
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting performance monitoring")
        
        # Start Prometheus metrics server
        start_http_server(9090)
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_system_metrics())
        asyncio.create_task(self._monitor_application_metrics())
        asyncio.create_task(self._analyze_performance())
        asyncio.create_task(self._optimize_performance())
        asyncio.create_task(self._cleanup_old_data())
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("Stopping performance monitoring")
    
    async def _monitor_system_metrics(self):
        """Monitor system-level metrics"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()
                load_avg = psutil.getloadavg()
                
                system_metrics = SystemMetrics(
                    timestamp=datetime.utcnow(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available=memory.available,
                    disk_usage_percent=disk.percent,
                    disk_io_read=disk_io.read_bytes if disk_io else 0,
                    disk_io_write=disk_io.write_bytes if disk_io else 0,
                    network_sent=network_io.bytes_sent,
                    network_recv=network_io.bytes_recv,
                    process_count=len(psutil.pids()),
                    thread_count=threading.active_count(),
                    load_average=load_avg
                )
                
                # Update Prometheus metrics
                self.prometheus_metrics["cpu_usage"].set(cpu_percent)
                self.prometheus_metrics["memory_usage"].set(memory.percent)
                
                # Store metrics
                self.metrics_history.append(system_metrics)
                self.performance_data["system"].append(system_metrics)
                
                # Check thresholds
                await self._check_thresholds(PerformanceMetric.CPU_USAGE, cpu_percent)
                await self._check_thresholds(PerformanceMetric.MEMORY_USAGE, memory.percent)
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_application_metrics(self):
        """Monitor application-level metrics"""
        while self.monitoring_active:
            try:
                # Collect application metrics
                app_metrics = ApplicationMetrics(
                    timestamp=datetime.utcnow(),
                    active_connections=await self._get_active_connections(),
                    request_count=await self._get_request_count(),
                    response_time_avg=await self._get_avg_response_time(),
                    response_time_p95=await self._get_p95_response_time(),
                    response_time_p99=await self._get_p99_response_time(),
                    error_count=await self._get_error_count(),
                    cache_hits=await self._get_cache_hits(),
                    cache_misses=await self._get_cache_misses(),
                    queue_size=await self._get_queue_size(),
                    worker_count=await self._get_worker_count()
                )
                
                # Update Prometheus metrics
                self.prometheus_metrics["active_connections"].set(app_metrics.active_connections)
                self.prometheus_metrics["queue_size"].set(app_metrics.queue_size)
                
                # Calculate cache hit rate
                total_cache_requests = app_metrics.cache_hits + app_metrics.cache_misses
                if total_cache_requests > 0:
                    cache_hit_rate = (app_metrics.cache_hits / total_cache_requests) * 100
                    self.prometheus_metrics["cache_hit_rate"].set(cache_hit_rate)
                    await self._check_thresholds(PerformanceMetric.CACHE_HIT_RATE, cache_hit_rate)
                
                # Calculate error rate
                total_requests = app_metrics.request_count
                if total_requests > 0:
                    error_rate = (app_metrics.error_count / total_requests) * 100
                    self.prometheus_metrics["error_rate"].set(error_rate)
                    await self._check_thresholds(PerformanceMetric.ERROR_RATE, error_rate)
                
                # Store metrics
                self.performance_data["application"].append(app_metrics)
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring application metrics: {e}")
                await asyncio.sleep(15)
    
    async def _analyze_performance(self):
        """Analyze performance trends and patterns"""
        while self.monitoring_active:
            try:
                if len(self.performance_data["system"]) < 10:
                    await asyncio.sleep(30)
                    continue
                
                # Analyze system performance
                system_metrics = self.performance_data["system"][-100:]  # Last 100 samples
                cpu_values = [m.cpu_percent for m in system_metrics]
                memory_values = [m.memory_percent for m in system_metrics]
                
                # Detect trends
                cpu_trend = self._calculate_trend(cpu_values)
                memory_trend = self._calculate_trend(memory_values)
                
                # Detect anomalies
                cpu_anomalies = self._detect_anomalies(cpu_values)
                memory_anomalies = self._detect_anomalies(memory_values)
                
                # Generate insights
                insights = []
                if cpu_trend > 0.1:  # 10% increase
                    insights.append({
                        "type": "trend",
                        "metric": "cpu_usage",
                        "message": f"CPU usage trending upward: {cpu_trend:.2%}",
                        "severity": "warning"
                    })
                
                if memory_trend > 0.1:  # 10% increase
                    insights.append({
                        "type": "trend",
                        "metric": "memory_usage",
                        "message": f"Memory usage trending upward: {memory_trend:.2%}",
                        "severity": "warning"
                    })
                
                if cpu_anomalies:
                    insights.append({
                        "type": "anomaly",
                        "metric": "cpu_usage",
                        "message": f"CPU usage anomaly detected: {len(cpu_anomalies)} instances",
                        "severity": "critical"
                    })
                
                if memory_anomalies:
                    insights.append({
                        "type": "anomaly",
                        "metric": "memory_usage",
                        "message": f"Memory usage anomaly detected: {len(memory_anomalies)} instances",
                        "severity": "critical"
                    })
                
                # Store insights
                for insight in insights:
                    logger.info(f"Performance insight: {insight['message']}")
                
                await asyncio.sleep(60)  # Analyze every minute
                
            except Exception as e:
                logger.error(f"Error analyzing performance: {e}")
                await asyncio.sleep(60)
    
    async def _optimize_performance(self):
        """Optimize performance based on metrics"""
        while self.monitoring_active:
            try:
                if not self.optimization_active:
                    await asyncio.sleep(30)
                    continue
                
                # Check if optimization is needed
                optimization_needed = await self._check_optimization_needed()
                
                if optimization_needed:
                    # Generate optimization actions
                    actions = await self._generate_optimization_actions()
                    
                    # Execute optimization actions
                    for action in actions:
                        await self._execute_optimization_action(action)
                
                await asyncio.sleep(120)  # Optimize every 2 minutes
                
            except Exception as e:
                logger.error(f"Error optimizing performance: {e}")
                await asyncio.sleep(120)
    
    async def _check_thresholds(self, metric: PerformanceMetric, value: float):
        """Check if metrics exceed thresholds"""
        threshold = next((t for t in self.thresholds if t.metric == metric), None)
        if not threshold or not threshold.enabled:
            return
        
        severity = None
        if value >= threshold.critical_threshold:
            severity = "critical"
        elif value >= threshold.warning_threshold:
            severity = "warning"
        
        if severity:
            alert = PerformanceAlert(
                id=f"{metric}_{int(time.time())}",
                metric=metric,
                value=value,
                threshold=threshold.critical_threshold if severity == "critical" else threshold.warning_threshold,
                severity=severity,
                timestamp=datetime.utcnow(),
                message=f"{metric.value} is {value:.2f}{threshold.unit}, exceeding {severity} threshold of {threshold.critical_threshold if severity == 'critical' else threshold.warning_threshold}{threshold.unit}"
            )
            
            self.alerts.append(alert)
            logger.warning(f"Performance alert: {alert.message}")
    
    async def _check_optimization_needed(self) -> bool:
        """Check if performance optimization is needed"""
        if len(self.performance_data["system"]) < 5:
            return False
        
        recent_metrics = self.performance_data["system"][-5:]
        avg_cpu = statistics.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = statistics.mean([m.memory_percent for m in recent_metrics])
        
        # Check if metrics exceed optimization thresholds
        cpu_threshold = next((t for t in self.thresholds if t.metric == PerformanceMetric.CPU_USAGE), None)
        memory_threshold = next((t for t in self.thresholds if t.metric == PerformanceMetric.MEMORY_USAGE), None)
        
        if cpu_threshold and avg_cpu > cpu_threshold.optimization_threshold:
            return True
        
        if memory_threshold and avg_memory > memory_threshold.optimization_threshold:
            return True
        
        return False
    
    async def _generate_optimization_actions(self) -> List[OptimizationAction]:
        """Generate optimization actions based on current metrics"""
        actions = []
        
        # Analyze current performance
        if len(self.performance_data["system"]) >= 5:
            recent_metrics = self.performance_data["system"][-5:]
            avg_cpu = statistics.mean([m.cpu_percent for m in recent_metrics])
            avg_memory = statistics.mean([m.memory_percent for m in recent_metrics])
            
            # CPU optimization
            if avg_cpu > 70:
                actions.append(OptimizationAction(
                    id=f"cpu_opt_{int(time.time())}",
                    strategy=OptimizationStrategy.CPU_OPTIMIZATION,
                    description="Optimize CPU usage by adjusting worker processes",
                    parameters={"worker_count": max(1, multiprocessing.cpu_count() - 1)},
                    expected_improvement=0.2,
                    risk_level="low"
                ))
            
            # Memory optimization
            if avg_memory > 80:
                actions.append(OptimizationAction(
                    id=f"memory_opt_{int(time.time())}",
                    strategy=OptimizationStrategy.MEMORY_OPTIMIZATION,
                    description="Optimize memory usage by garbage collection and cache cleanup",
                    parameters={"gc_threshold": 0.8, "cache_cleanup": True},
                    expected_improvement=0.15,
                    risk_level="low"
                ))
            
            # Auto-scaling
            if avg_cpu > 80 or avg_memory > 85:
                actions.append(OptimizationAction(
                    id=f"scale_up_{int(time.time())}",
                    strategy=OptimizationStrategy.AUTO_SCALING,
                    description="Scale up instances due to high resource usage",
                    parameters={"scale_factor": 1.5, "target_instances": 3},
                    expected_improvement=0.4,
                    risk_level="medium"
                ))
        
        return actions
    
    async def _execute_optimization_action(self, action: OptimizationAction):
        """Execute an optimization action"""
        try:
            logger.info(f"Executing optimization action: {action.description}")
            
            if action.strategy == OptimizationStrategy.CPU_OPTIMIZATION:
                result = await self._optimize_cpu(action.parameters)
            elif action.strategy == OptimizationStrategy.MEMORY_OPTIMIZATION:
                result = await self._optimize_memory(action.parameters)
            elif action.strategy == OptimizationStrategy.AUTO_SCALING:
                result = await self._auto_scale(action.parameters)
            elif action.strategy == OptimizationStrategy.CACHE_OPTIMIZATION:
                result = await self._optimize_cache(action.parameters)
            elif action.strategy == OptimizationStrategy.CONNECTION_POOLING:
                result = await self._optimize_connection_pool(action.parameters)
            else:
                result = {"status": "not_implemented"}
            
            action.executed = True
            action.result = result
            
            # Update Prometheus metrics
            self.prometheus_metrics["optimization_actions"].labels(
                strategy=action.strategy.value,
                result=result.get("status", "unknown")
            ).inc()
            
            # Store optimization history
            self.optimization_history.append({
                "action": action,
                "timestamp": datetime.utcnow(),
                "result": result
            })
            
            logger.info(f"Optimization action completed: {result}")
            
        except Exception as e:
            logger.error(f"Error executing optimization action: {e}")
            action.result = {"status": "error", "error": str(e)}
    
    async def _optimize_cpu(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize CPU usage"""
        try:
            # Adjust worker processes
            worker_count = parameters.get("worker_count", multiprocessing.cpu_count())
            
            # This would typically involve adjusting the application configuration
            # For now, we'll simulate the optimization
            await asyncio.sleep(1)  # Simulate optimization time
            
            return {
                "status": "success",
                "worker_count": worker_count,
                "message": f"Adjusted worker count to {worker_count}"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _optimize_memory(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory usage"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clean up caches if needed
            if parameters.get("cache_cleanup", False):
                # This would clean up application caches
                pass
            
            # Get memory usage after optimization
            memory_after = psutil.virtual_memory().percent
            
            return {
                "status": "success",
                "memory_usage_after": memory_after,
                "message": "Memory optimization completed"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _auto_scale(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-scale instances"""
        try:
            if not self.auto_scaling_config["enabled"]:
                return {"status": "disabled", "message": "Auto-scaling is disabled"}
            
            # Check cooldown period
            last_scale = self.auto_scaling_config["last_scale_time"]
            if last_scale and (datetime.utcnow() - last_scale).seconds < self.auto_scaling_config["cooldown_period"]:
                return {"status": "cooldown", "message": "Auto-scaling in cooldown period"}
            
            # This would typically involve calling a container orchestration API
            # For now, we'll simulate the scaling
            target_instances = parameters.get("target_instances", 2)
            scale_factor = parameters.get("scale_factor", 1.5)
            
            await asyncio.sleep(2)  # Simulate scaling time
            
            self.auto_scaling_config["last_scale_time"] = datetime.utcnow()
            
            return {
                "status": "success",
                "target_instances": target_instances,
                "scale_factor": scale_factor,
                "message": f"Scaled to {target_instances} instances"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _optimize_cache(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize cache performance"""
        try:
            # This would typically involve adjusting cache settings
            # For now, we'll simulate the optimization
            await asyncio.sleep(1)
            
            return {
                "status": "success",
                "message": "Cache optimization completed"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _optimize_connection_pool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize connection pool"""
        try:
            # This would typically involve adjusting connection pool settings
            # For now, we'll simulate the optimization
            await asyncio.sleep(1)
            
            return {
                "status": "success",
                "message": "Connection pool optimization completed"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values"""
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope / values[0] if values[0] != 0 else 0.0
    
    def _detect_anomalies(self, values: List[float]) -> List[int]:
        """Detect anomalies in values using statistical methods"""
        if len(values) < 10:
            return []
        
        # Use Z-score method
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        
        if stdev == 0:
            return []
        
        anomalies = []
        for i, value in enumerate(values):
            z_score = abs((value - mean) / stdev)
            if z_score > 2:  # 2 standard deviations
                anomalies.append(i)
        
        return anomalies
    
    async def _cleanup_old_data(self):
        """Cleanup old performance data"""
        while self.monitoring_active:
            try:
                # Clean up old metrics (keep last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                for metric_type in self.performance_data:
                    self.performance_data[metric_type] = [
                        m for m in self.performance_data[metric_type]
                        if m.timestamp > cutoff_time
                    ]
                
                # Clean up old alerts (keep last 7 days)
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
                
                # Clean up old optimization history (keep last 30 days)
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                self.optimization_history = [
                    h for h in self.optimization_history
                    if h["timestamp"] > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up old data: {e}")
                await asyncio.sleep(3600)
    
    # Helper methods for getting application metrics
    async def _get_active_connections(self) -> int:
        """Get number of active connections"""
        # This would typically query the application server
        return 10  # Mock value
    
    async def _get_request_count(self) -> int:
        """Get request count"""
        # This would typically query the application metrics
        return 100  # Mock value
    
    async def _get_avg_response_time(self) -> float:
        """Get average response time"""
        # This would typically query the application metrics
        return 150.0  # Mock value in ms
    
    async def _get_p95_response_time(self) -> float:
        """Get 95th percentile response time"""
        # This would typically query the application metrics
        return 500.0  # Mock value in ms
    
    async def _get_p99_response_time(self) -> float:
        """Get 99th percentile response time"""
        # This would typically query the application metrics
        return 1000.0  # Mock value in ms
    
    async def _get_error_count(self) -> int:
        """Get error count"""
        # This would typically query the application metrics
        return 5  # Mock value
    
    async def _get_cache_hits(self) -> int:
        """Get cache hits"""
        # This would typically query the cache metrics
        return 80  # Mock value
    
    async def _get_cache_misses(self) -> int:
        """Get cache misses"""
        # This would typically query the cache metrics
        return 20  # Mock value
    
    async def _get_queue_size(self) -> int:
        """Get queue size"""
        # This would typically query the queue metrics
        return 5  # Mock value
    
    async def _get_worker_count(self) -> int:
        """Get worker count"""
        # This would typically query the worker metrics
        return 4  # Mock value
    
    # Public methods for external access
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_data["system"]:
            return {"status": "no_data"}
        
        recent_system = self.performance_data["system"][-1]
        recent_app = self.performance_data["application"][-1] if self.performance_data["application"] else None
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": recent_system.cpu_percent,
                "memory_percent": recent_system.memory_percent,
                "disk_usage_percent": recent_system.disk_usage_percent,
                "load_average": recent_system.load_average
            },
            "application": {
                "active_connections": recent_app.active_connections if recent_app else 0,
                "response_time_avg": recent_app.response_time_avg if recent_app else 0,
                "error_count": recent_app.error_count if recent_app else 0,
                "queue_size": recent_app.queue_size if recent_app else 0
            },
            "alerts": len([a for a in self.alerts if not a.resolved]),
            "optimization_actions": len([a for a in self.optimization_actions if a.executed])
        }
    
    def get_alerts(self, severity: Optional[str] = None) -> List[PerformanceAlert]:
        """Get performance alerts"""
        if severity:
            return [a for a in self.alerts if a.severity == severity and not a.resolved]
        return [a for a in self.alerts if not a.resolved]
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history"""
        return self.optimization_history[-50:]  # Last 50 optimizations
    
    def enable_optimization(self):
        """Enable performance optimization"""
        self.optimization_active = True
        logger.info("Performance optimization enabled")
    
    def disable_optimization(self):
        """Disable performance optimization"""
        self.optimization_active = False
        logger.info("Performance optimization disabled")
    
    def update_threshold(self, metric: PerformanceMetric, warning: float, critical: float, optimization: float):
        """Update performance threshold"""
        threshold = next((t for t in self.thresholds if t.metric == metric), None)
        if threshold:
            threshold.warning_threshold = warning
            threshold.critical_threshold = critical
            threshold.optimization_threshold = optimization
            logger.info(f"Updated threshold for {metric.value}")
    
    def get_metrics_history(self, metric_type: str, limit: int = 100) -> List[Any]:
        """Get metrics history"""
        return list(self.performance_data.get(metric_type, []))[-limit:]

# Global performance optimizer instance
performance_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance"""
    global performance_optimizer
    if performance_optimizer is None:
        config = {
            "monitoring_interval": 5,
            "optimization_interval": 120,
            "cleanup_interval": 3600
        }
        performance_optimizer = PerformanceOptimizer(config)
    return performance_optimizer

if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "monitoring_interval": 5,
            "optimization_interval": 120,
            "cleanup_interval": 3600
        }
        
        optimizer = PerformanceOptimizer(config)
        optimizer.enable_optimization()
        
        # Run for 5 minutes
        await asyncio.sleep(300)
        
        # Get performance summary
        summary = optimizer.get_performance_summary()
        print("Performance Summary:")
        print(json.dumps(summary, indent=2))
        
        # Get alerts
        alerts = optimizer.get_alerts()
        print(f"\nActive Alerts: {len(alerts)}")
        for alert in alerts:
            print(f"- {alert.message}")
        
        # Get optimization history
        history = optimizer.get_optimization_history()
        print(f"\nOptimization History: {len(history)} actions")
        
        await optimizer.stop_monitoring()
    
    asyncio.run(main())













