"""
Advanced Performance Optimization for Microservices
Features: Intelligent load balancing, auto-scaling, performance monitoring, resource optimization
"""

import asyncio
import time
import statistics
import psutil
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import json
import math

# Performance monitoring imports
try:
    import aiohttp
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CONSISTENT_HASH = "consistent_hash"
    AI_POWERED = "ai_powered"

class ScalingTrigger(Enum):
    """Scaling triggers"""
    CPU_THRESHOLD = "cpu_threshold"
    MEMORY_THRESHOLD = "memory_threshold"
    RESPONSE_TIME_THRESHOLD = "response_time_threshold"
    REQUEST_RATE_THRESHOLD = "request_rate_threshold"
    ERROR_RATE_THRESHOLD = "error_rate_threshold"
    QUEUE_DEPTH_THRESHOLD = "queue_depth_threshold"

@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    response_time: float
    request_rate: float
    error_rate: float
    active_connections: int
    queue_depth: int
    throughput: float

@dataclass
class ServiceInstance:
    """Service instance with performance data"""
    id: str
    host: str
    port: int
    weight: int = 1
    active_connections: int = 0
    response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    last_health_check: float = 0.0
    is_healthy: bool = True
    metrics_history: deque = field(default_factory=lambda: deque(maxlen=100))

@dataclass
class ScalingConfig:
    """Auto-scaling configuration"""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu: float = 70.0
    target_memory: float = 80.0
    target_response_time: float = 500.0  # ms
    target_error_rate: float = 5.0  # %
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    scale_up_step: int = 2
    scale_down_step: int = 1

class PerformanceMonitor:
    """
    Advanced performance monitoring system
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis = redis_client
        self.metrics_history: deque = deque(maxlen=1000)
        self.service_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.alerts: List[Dict[str, Any]] = []
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self, interval: float = 10.0):
        """Start performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Store in Redis if available
                if self.redis:
                    await self._store_metrics(metrics)
                
                # Check for alerts
                await self._check_alerts(metrics)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Network I/O
            network = psutil.net_io_counters()
            
            # Disk I/O
            disk = psutil.disk_io_counters()
            
            # Calculate response time (simplified)
            response_time = await self._measure_response_time()
            
            # Calculate request rate (simplified)
            request_rate = await self._calculate_request_rate()
            
            # Calculate error rate (simplified)
            error_rate = await self._calculate_error_rate()
            
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                response_time=response_time,
                request_rate=request_rate,
                error_rate=error_rate,
                active_connections=0,  # Would be tracked in real implementation
                queue_depth=0,  # Would be tracked in real implementation
                throughput=network.bytes_sent + network.bytes_recv
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=0.0,
                memory_usage=0.0,
                response_time=0.0,
                request_rate=0.0,
                error_rate=0.0,
                active_connections=0,
                queue_depth=0,
                throughput=0.0
            )
    
    async def _measure_response_time(self) -> float:
        """Measure average response time"""
        # This would measure actual response times in a real implementation
        # For now, return a simulated value
        return 100.0 + (time.time() % 100)  # Simulate 100-200ms response time
    
    async def _calculate_request_rate(self) -> float:
        """Calculate requests per second"""
        # This would calculate actual request rate in a real implementation
        return 50.0 + (time.time() % 50)  # Simulate 50-100 req/s
    
    async def _calculate_error_rate(self) -> float:
        """Calculate error rate percentage"""
        # This would calculate actual error rate in a real implementation
        return 1.0 + (time.time() % 5)  # Simulate 1-5% error rate
    
    async def _store_metrics(self, metrics: PerformanceMetrics):
        """Store metrics in Redis"""
        try:
            if not self.redis:
                return
            
            metrics_data = {
                "timestamp": metrics.timestamp,
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "response_time": metrics.response_time,
                "request_rate": metrics.request_rate,
                "error_rate": metrics.error_rate,
                "active_connections": metrics.active_connections,
                "queue_depth": metrics.queue_depth,
                "throughput": metrics.throughput
            }
            
            # Store in time-series format
            await self.redis.zadd(
                "performance_metrics",
                {json.dumps(metrics_data): metrics.timestamp}
            )
            
            # Keep only last 24 hours of data
            cutoff_time = time.time() - 86400
            await self.redis.zremrangebyscore("performance_metrics", 0, cutoff_time)
            
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    async def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts"""
        alerts = []
        
        # CPU alert
        if metrics.cpu_usage > 90:
            alerts.append({
                "type": "cpu_high",
                "severity": "critical",
                "value": metrics.cpu_usage,
                "threshold": 90,
                "timestamp": metrics.timestamp
            })
        
        # Memory alert
        if metrics.memory_usage > 90:
            alerts.append({
                "type": "memory_high",
                "severity": "critical",
                "value": metrics.memory_usage,
                "threshold": 90,
                "timestamp": metrics.timestamp
            })
        
        # Response time alert
        if metrics.response_time > 1000:
            alerts.append({
                "type": "response_time_high",
                "severity": "warning",
                "value": metrics.response_time,
                "threshold": 1000,
                "timestamp": metrics.timestamp
            })
        
        # Error rate alert
        if metrics.error_rate > 10:
            alerts.append({
                "type": "error_rate_high",
                "severity": "critical",
                "value": metrics.error_rate,
                "threshold": 10,
                "timestamp": metrics.timestamp
            })
        
        # Store alerts
        for alert in alerts:
            self.alerts.append(alert)
            logger.warning(f"Performance alert: {alert}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        
        return {
            "cpu_usage": {
                "current": recent_metrics[-1].cpu_usage,
                "average": statistics.mean([m.cpu_usage for m in recent_metrics]),
                "max": max([m.cpu_usage for m in recent_metrics]),
                "min": min([m.cpu_usage for m in recent_metrics])
            },
            "memory_usage": {
                "current": recent_metrics[-1].memory_usage,
                "average": statistics.mean([m.memory_usage for m in recent_metrics]),
                "max": max([m.memory_usage for m in recent_metrics]),
                "min": min([m.memory_usage for m in recent_metrics])
            },
            "response_time": {
                "current": recent_metrics[-1].response_time,
                "average": statistics.mean([m.response_time for m in recent_metrics]),
                "max": max([m.response_time for m in recent_metrics]),
                "min": min([m.response_time for m in recent_metrics])
            },
            "request_rate": {
                "current": recent_metrics[-1].request_rate,
                "average": statistics.mean([m.request_rate for m in recent_metrics]),
                "max": max([m.request_rate for m in recent_metrics]),
                "min": min([m.request_rate for m in recent_metrics])
            },
            "error_rate": {
                "current": recent_metrics[-1].error_rate,
                "average": statistics.mean([m.error_rate for m in recent_metrics]),
                "max": max([m.error_rate for m in recent_metrics]),
                "min": min([m.error_rate for m in recent_metrics])
            },
            "alerts_count": len(self.alerts),
            "monitoring_active": self.monitoring_active
        }

class IntelligentLoadBalancer:
    """
    AI-powered intelligent load balancer
    """
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.AI_POWERED):
        self.strategy = strategy
        self.instances: List[ServiceInstance] = []
        self.current_index = 0
        self.instance_weights: Dict[str, float] = {}
        self.performance_monitor = PerformanceMonitor()
        self.load_balancing_history: deque = deque(maxlen=1000)
        
    async def add_instance(self, instance: ServiceInstance):
        """Add service instance to load balancer"""
        self.instances.append(instance)
        self.instance_weights[instance.id] = 1.0
        logger.info(f"Added instance {instance.id} to load balancer")
    
    async def remove_instance(self, instance_id: str):
        """Remove service instance from load balancer"""
        self.instances = [i for i in self.instances if i.id != instance_id]
        self.instance_weights.pop(instance_id, None)
        logger.info(f"Removed instance {instance_id} from load balancer")
    
    async def get_best_instance(self) -> Optional[ServiceInstance]:
        """Get the best instance based on load balancing strategy"""
        if not self.instances:
            return None
        
        # Filter healthy instances
        healthy_instances = [i for i in self.instances if i.is_healthy]
        if not healthy_instances:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return await self._round_robin_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return await self._least_connections_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return await self._least_response_time_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return await self._weighted_round_robin_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.AI_POWERED:
            return await self._ai_powered_selection(healthy_instances)
        else:
            return healthy_instances[0]
    
    async def _round_robin_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin selection"""
        instance = instances[self.current_index % len(instances)]
        self.current_index += 1
        return instance
    
    async def _least_connections_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections selection"""
        return min(instances, key=lambda x: x.active_connections)
    
    async def _least_response_time_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least response time selection"""
        return min(instances, key=lambda x: x.response_time)
    
    async def _weighted_round_robin_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round robin selection"""
        # Calculate total weight
        total_weight = sum(instance.weight for instance in instances)
        
        # Select based on weights
        random_value = time.time() % total_weight
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.weight
            if random_value <= current_weight:
                return instance
        
        return instances[0]
    
    async def _ai_powered_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """AI-powered instance selection"""
        try:
            # Calculate scores for each instance
            scores = []
            for instance in instances:
                score = await self._calculate_instance_score(instance)
                scores.append((instance, score))
            
            # Sort by score (higher is better)
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select the best instance
            selected_instance = scores[0][0]
            
            # Log selection
            self.load_balancing_history.append({
                "timestamp": time.time(),
                "selected_instance": selected_instance.id,
                "scores": {instance.id: score for instance, score in scores},
                "strategy": "ai_powered"
            })
            
            return selected_instance
            
        except Exception as e:
            logger.error(f"AI-powered selection failed: {e}")
            # Fallback to least connections
            return await self._least_connections_selection(instances)
    
    async def _calculate_instance_score(self, instance: ServiceInstance) -> float:
        """Calculate AI-powered score for instance"""
        try:
            # Base score
            score = 1.0
            
            # CPU factor (lower is better)
            cpu_factor = max(0.1, 1.0 - (instance.cpu_usage / 100.0))
            score *= cpu_factor
            
            # Memory factor (lower is better)
            memory_factor = max(0.1, 1.0 - (instance.memory_usage / 100.0))
            score *= memory_factor
            
            # Response time factor (lower is better)
            response_time_factor = max(0.1, 1.0 - (instance.response_time / 1000.0))
            score *= response_time_factor
            
            # Error rate factor (lower is better)
            error_rate_factor = max(0.1, 1.0 - (instance.error_rate / 100.0))
            score *= error_rate_factor
            
            # Connection factor (fewer connections is better)
            connection_factor = max(0.1, 1.0 - (instance.active_connections / 100.0))
            score *= connection_factor
            
            # Weight factor
            weight_factor = instance.weight / 10.0
            score *= weight_factor
            
            # Health factor
            health_factor = 1.0 if instance.is_healthy else 0.0
            score *= health_factor
            
            return score
            
        except Exception as e:
            logger.error(f"Failed to calculate instance score: {e}")
            return 0.1
    
    async def update_instance_metrics(self, instance_id: str, metrics: Dict[str, Any]):
        """Update instance performance metrics"""
        for instance in self.instances:
            if instance.id == instance_id:
                instance.cpu_usage = metrics.get("cpu_usage", instance.cpu_usage)
                instance.memory_usage = metrics.get("memory_usage", instance.memory_usage)
                instance.response_time = metrics.get("response_time", instance.response_time)
                instance.error_rate = metrics.get("error_rate", instance.error_rate)
                instance.active_connections = metrics.get("active_connections", instance.active_connections)
                instance.last_health_check = time.time()
                break
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        if not self.instances:
            return {"status": "no_instances"}
        
        healthy_instances = [i for i in self.instances if i.is_healthy]
        
        return {
            "total_instances": len(self.instances),
            "healthy_instances": len(healthy_instances),
            "strategy": self.strategy.value,
            "instances": [
                {
                    "id": instance.id,
                    "host": instance.host,
                    "port": instance.port,
                    "weight": instance.weight,
                    "cpu_usage": instance.cpu_usage,
                    "memory_usage": instance.memory_usage,
                    "response_time": instance.response_time,
                    "error_rate": instance.error_rate,
                    "active_connections": instance.active_connections,
                    "is_healthy": instance.is_healthy
                }
                for instance in self.instances
            ],
            "selection_history": len(self.load_balancing_history)
        }

class AutoScaler:
    """
    Intelligent auto-scaling system
    """
    
    def __init__(self, config: ScalingConfig, load_balancer: IntelligentLoadBalancer):
        self.config = config
        self.load_balancer = load_balancer
        self.performance_monitor = PerformanceMonitor()
        self.scaling_history: List[Dict[str, Any]] = []
        self.last_scale_up = 0
        self.last_scale_down = 0
        self.current_instances = config.min_instances
        self.scaling_active = False
        self.scaling_task: Optional[asyncio.Task] = None
    
    async def start_auto_scaling(self, interval: float = 30.0):
        """Start auto-scaling"""
        if self.scaling_active:
            return
        
        self.scaling_active = True
        await self.performance_monitor.start_monitoring(interval)
        self.scaling_task = asyncio.create_task(self._scaling_loop(interval))
        logger.info("Auto-scaling started")
    
    async def stop_auto_scaling(self):
        """Stop auto-scaling"""
        self.scaling_active = False
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        await self.performance_monitor.stop_monitoring()
        logger.info("Auto-scaling stopped")
    
    async def _scaling_loop(self, interval: float):
        """Main scaling loop"""
        while self.scaling_active:
            try:
                # Get current metrics
                metrics_summary = self.performance_monitor.get_metrics_summary()
                
                if metrics_summary.get("status") != "no_data":
                    # Check if scaling is needed
                    scaling_action = await self._evaluate_scaling_need(metrics_summary)
                    
                    if scaling_action:
                        await self._execute_scaling_action(scaling_action)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(interval)
    
    async def _evaluate_scaling_need(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate if scaling is needed"""
        try:
            current_time = time.time()
            
            # Check scale-up conditions
            if await self._should_scale_up(metrics):
                if current_time - self.last_scale_up > self.config.scale_up_cooldown:
                    return {
                        "action": "scale_up",
                        "reason": "high_load",
                        "current_instances": self.current_instances,
                        "target_instances": min(
                            self.current_instances + self.config.scale_up_step,
                            self.config.max_instances
                        )
                    }
            
            # Check scale-down conditions
            elif await self._should_scale_down(metrics):
                if current_time - self.last_scale_down > self.config.scale_down_cooldown:
                    return {
                        "action": "scale_down",
                        "reason": "low_load",
                        "current_instances": self.current_instances,
                        "target_instances": max(
                            self.current_instances - self.config.scale_down_step,
                            self.config.min_instances
                        )
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to evaluate scaling need: {e}")
            return None
    
    async def _should_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """Check if should scale up"""
        cpu_avg = metrics.get("cpu_usage", {}).get("average", 0)
        memory_avg = metrics.get("memory_usage", {}).get("average", 0)
        response_time_avg = metrics.get("response_time", {}).get("average", 0)
        error_rate_avg = metrics.get("error_rate", {}).get("average", 0)
        
        # Scale up if any metric exceeds threshold
        return (
            cpu_avg > self.config.scale_up_threshold or
            memory_avg > self.config.scale_up_threshold or
            response_time_avg > self.config.target_response_time * 1.5 or
            error_rate_avg > self.config.target_error_rate * 1.5
        )
    
    async def _should_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """Check if should scale down"""
        cpu_avg = metrics.get("cpu_usage", {}).get("average", 0)
        memory_avg = metrics.get("memory_usage", {}).get("average", 0)
        response_time_avg = metrics.get("response_time", {}).get("average", 0)
        error_rate_avg = metrics.get("error_rate", {}).get("average", 0)
        
        # Scale down if all metrics are below threshold
        return (
            cpu_avg < self.config.scale_down_threshold and
            memory_avg < self.config.scale_down_threshold and
            response_time_avg < self.config.target_response_time * 0.5 and
            error_rate_avg < self.config.target_error_rate * 0.5 and
            self.current_instances > self.config.min_instances
        )
    
    async def _execute_scaling_action(self, action: Dict[str, Any]):
        """Execute scaling action"""
        try:
            action_type = action["action"]
            target_instances = action["target_instances"]
            
            if action_type == "scale_up":
                await self._scale_up(target_instances)
                self.last_scale_up = time.time()
            elif action_type == "scale_down":
                await self._scale_down(target_instances)
                self.last_scale_down = time.time()
            
            # Record scaling action
            self.scaling_history.append({
                "timestamp": time.time(),
                "action": action_type,
                "from_instances": self.current_instances,
                "to_instances": target_instances,
                "reason": action["reason"]
            })
            
            logger.info(f"Scaling {action_type}: {self.current_instances} -> {target_instances}")
            
        except Exception as e:
            logger.error(f"Failed to execute scaling action: {e}")
    
    async def _scale_up(self, target_instances: int):
        """Scale up instances"""
        instances_to_add = target_instances - self.current_instances
        
        for i in range(instances_to_add):
            # Create new instance
            new_instance = ServiceInstance(
                id=f"instance-{int(time.time())}-{i}",
                host="localhost",  # Would be actual host in real implementation
                port=8000 + self.current_instances + i,
                weight=1
            )
            
            # Add to load balancer
            await self.load_balancer.add_instance(new_instance)
            
            # Simulate instance startup
            await asyncio.sleep(1)
        
        self.current_instances = target_instances
    
    async def _scale_down(self, target_instances: int):
        """Scale down instances"""
        instances_to_remove = self.current_instances - target_instances
        
        # Remove instances (start with least healthy ones)
        instances_to_remove_list = sorted(
            self.load_balancer.instances,
            key=lambda x: (x.is_healthy, x.cpu_usage, x.memory_usage),
            reverse=True
        )[:instances_to_remove]
        
        for instance in instances_to_remove_list:
            await self.load_balancer.remove_instance(instance.id)
            await asyncio.sleep(1)  # Simulate graceful shutdown
        
        self.current_instances = target_instances
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics"""
        return {
            "current_instances": self.current_instances,
            "min_instances": self.config.min_instances,
            "max_instances": self.config.max_instances,
            "scaling_active": self.scaling_active,
            "last_scale_up": self.last_scale_up,
            "last_scale_down": self.last_scale_down,
            "scaling_history": self.scaling_history[-10:],  # Last 10 scaling actions
            "performance_metrics": self.performance_monitor.get_metrics_summary()
        }

class PerformanceOptimizer:
    """
    Main performance optimization manager
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis = redis_client
        self.load_balancer = IntelligentLoadBalancer()
        self.auto_scaler = AutoScaler(
            ScalingConfig(),
            self.load_balancer
        )
        self.optimization_active = False
    
    async def start_optimization(self):
        """Start performance optimization"""
        if self.optimization_active:
            return
        
        self.optimization_active = True
        await self.auto_scaler.start_auto_scaling()
        logger.info("Performance optimization started")
    
    async def stop_optimization(self):
        """Stop performance optimization"""
        self.optimization_active = False
        await self.auto_scaler.stop_auto_scaling()
        logger.info("Performance optimization stopped")
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            "optimization_active": self.optimization_active,
            "load_balancer": self.load_balancer.get_load_balancer_stats(),
            "auto_scaler": self.auto_scaler.get_scaling_stats()
        }

# Global performance optimizer
performance_optimizer = PerformanceOptimizer()

# Decorator for performance monitoring
def monitor_performance(func_name: str = None):
    """Decorator for performance monitoring"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_cpu = psutil.cpu_percent()
            start_memory = psutil.virtual_memory().percent
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                end_cpu = psutil.cpu_percent()
                end_memory = psutil.virtual_memory().percent
                
                # Log performance metrics
                logger.info(
                    f"Performance metrics for {func_name or func.__name__}",
                    extra={
                        "execution_time": end_time - start_time,
                        "cpu_usage": end_cpu - start_cpu,
                        "memory_usage": end_memory - start_memory
                    }
                )
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            start_cpu = psutil.cpu_percent()
            start_memory = psutil.virtual_memory().percent
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                end_cpu = psutil.cpu_percent()
                end_memory = psutil.virtual_memory().percent
                
                # Log performance metrics
                logger.info(
                    f"Performance metrics for {func_name or func.__name__}",
                    extra={
                        "execution_time": end_time - start_time,
                        "cpu_usage": end_cpu - start_cpu,
                        "memory_usage": end_memory - start_memory
                    }
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator






























