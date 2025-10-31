"""
Advanced Performance Optimization System
=======================================

Comprehensive performance optimization system with real-time monitoring,
automatic tuning, and intelligent resource management.
"""

import logging
import asyncio
import json
import psutil
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from pathlib import Path
import threading
import queue
import statistics
import gc
import os
import sys

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Optimization levels"""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class ResourceType(Enum):
    """Resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    CACHE = "cache"
    DATABASE = "database"

class PerformanceMetric(Enum):
    """Performance metrics"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUEUE_LENGTH = "queue_length"

@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    active_connections: int
    response_time_ms: float
    throughput_rps: float
    error_rate: float
    cache_hit_rate: float
    queue_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationRule:
    """Optimization rule"""
    id: str
    name: str
    condition: str
    action: str
    resource_type: ResourceType
    threshold: float
    optimization_level: OptimizationLevel
    enabled: bool
    created_at: datetime
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationAction:
    """Optimization action"""
    id: str
    rule_id: str
    action_type: str
    parameters: Dict[str, Any]
    executed_at: datetime
    success: bool
    impact_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceReport:
    """Performance analysis report"""
    id: str
    period_start: datetime
    period_end: datetime
    average_metrics: Dict[PerformanceMetric, float]
    peak_metrics: Dict[PerformanceMetric, float]
    optimization_opportunities: List[str]
    recommendations: List[str]
    generated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedPerformanceOptimizer:
    """
    Advanced performance optimization system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance optimizer
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Performance monitoring
        self.performance_history: List[PerformanceSnapshot] = []
        self.optimization_rules: Dict[str, OptimizationRule] = {}
        self.optimization_actions: List[OptimizationAction] = []
        
        # Monitoring settings
        self.monitoring_interval = self.config.get("monitoring_interval", 5)  # seconds
        self.history_retention = self.config.get("history_retention", 1000)  # snapshots
        self.optimization_enabled = self.config.get("optimization_enabled", True)
        
        # Performance thresholds
        self.thresholds = {
            PerformanceMetric.CPU_USAGE: 80.0,
            PerformanceMetric.MEMORY_USAGE: 85.0,
            PerformanceMetric.RESPONSE_TIME: 1000.0,  # ms
            PerformanceMetric.ERROR_RATE: 5.0,  # percentage
            PerformanceMetric.CACHE_HIT_RATE: 70.0,  # percentage
            PerformanceMetric.QUEUE_LENGTH: 100
        }
        
        # Initialize optimization rules
        self._initialize_optimization_rules()
        
        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        self.metrics_queue = queue.Queue()
        
        # Performance tracking
        self.request_times: List[float] = []
        self.error_counts: Dict[str, int] = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        
    def _initialize_optimization_rules(self):
        """Initialize default optimization rules"""
        default_rules = [
            {
                "id": "high_cpu_usage",
                "name": "High CPU Usage",
                "condition": "cpu_percent > 80",
                "action": "scale_workers",
                "resource_type": ResourceType.CPU,
                "threshold": 80.0,
                "optimization_level": OptimizationLevel.MODERATE
            },
            {
                "id": "high_memory_usage",
                "name": "High Memory Usage",
                "condition": "memory_percent > 85",
                "action": "clear_cache",
                "resource_type": ResourceType.MEMORY,
                "threshold": 85.0,
                "optimization_level": OptimizationLevel.AGGRESSIVE
            },
            {
                "id": "slow_response_time",
                "name": "Slow Response Time",
                "condition": "response_time_ms > 1000",
                "action": "optimize_queries",
                "resource_type": ResourceType.CPU,
                "threshold": 1000.0,
                "optimization_level": OptimizationLevel.MODERATE
            },
            {
                "id": "high_error_rate",
                "name": "High Error Rate",
                "condition": "error_rate > 5",
                "action": "enable_circuit_breaker",
                "resource_type": ResourceType.CPU,
                "threshold": 5.0,
                "optimization_level": OptimizationLevel.AGGRESSIVE
            },
            {
                "id": "low_cache_hit_rate",
                "name": "Low Cache Hit Rate",
                "condition": "cache_hit_rate < 70",
                "action": "increase_cache_size",
                "resource_type": ResourceType.CACHE,
                "threshold": 70.0,
                "optimization_level": OptimizationLevel.MODERATE
            },
            {
                "id": "long_queue",
                "name": "Long Processing Queue",
                "condition": "queue_length > 100",
                "action": "scale_workers",
                "resource_type": ResourceType.CPU,
                "threshold": 100.0,
                "optimization_level": OptimizationLevel.AGGRESSIVE
            }
        ]
        
        for rule_data in default_rules:
            rule = OptimizationRule(
                id=rule_data["id"],
                name=rule_data["name"],
                condition=rule_data["condition"],
                action=rule_data["action"],
                resource_type=rule_data["resource_type"],
                threshold=rule_data["threshold"],
                optimization_level=rule_data["optimization_level"],
                enabled=True,
                created_at=datetime.now()
            )
            self.optimization_rules[rule.id] = rule
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                snapshot = self._collect_performance_snapshot()
                self.performance_history.append(snapshot)
                
                # Maintain history size
                if len(self.performance_history) > self.history_retention:
                    self.performance_history = self.performance_history[-self.history_retention:]
                
                # Check optimization rules
                if self.optimization_enabled:
                    self._check_optimization_rules(snapshot)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_performance_snapshot(self) -> PerformanceSnapshot:
        """Collect current performance metrics"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Application metrics
        response_time = self._calculate_average_response_time()
        throughput = self._calculate_throughput()
        error_rate = self._calculate_error_rate()
        cache_hit_rate = self._calculate_cache_hit_rate()
        queue_length = self._get_queue_length()
        
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            disk_io_read_mb=disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
            disk_io_write_mb=disk_io.write_bytes / (1024 * 1024) if disk_io else 0,
            network_io_sent_mb=network_io.bytes_sent / (1024 * 1024) if network_io else 0,
            network_io_recv_mb=network_io.bytes_recv / (1024 * 1024) if network_io else 0,
            active_connections=self._get_active_connections(),
            response_time_ms=response_time,
            throughput_rps=throughput,
            error_rate=error_rate,
            cache_hit_rate=cache_hit_rate,
            queue_length=queue_length,
            metadata={
                "process_count": len(psutil.pids()),
                "load_average": os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
            }
        )
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time"""
        if not self.request_times:
            return 0.0
        
        # Keep only recent request times (last 100)
        if len(self.request_times) > 100:
            self.request_times = self.request_times[-100:]
        
        return statistics.mean(self.request_times) if self.request_times else 0.0
    
    def _calculate_throughput(self) -> float:
        """Calculate requests per second"""
        if len(self.request_times) < 2:
            return 0.0
        
        # Calculate based on recent requests
        recent_times = self.request_times[-10:] if len(self.request_times) >= 10 else self.request_times
        if len(recent_times) < 2:
            return 0.0
        
        time_span = recent_times[-1] - recent_times[0]
        if time_span > 0:
            return len(recent_times) / time_span
        return 0.0
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate percentage"""
        total_errors = sum(self.error_counts.values())
        total_requests = len(self.request_times)
        
        if total_requests == 0:
            return 0.0
        
        return (total_errors / total_requests) * 100
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        
        if total_requests == 0:
            return 0.0
        
        return (self.cache_stats["hits"] / total_requests) * 100
    
    def _get_queue_length(self) -> int:
        """Get current queue length"""
        return self.metrics_queue.qsize()
    
    def _get_active_connections(self) -> int:
        """Get number of active connections"""
        try:
            connections = psutil.net_connections()
            return len([conn for conn in connections if conn.status == 'ESTABLISHED'])
        except:
            return 0
    
    def _check_optimization_rules(self, snapshot: PerformanceSnapshot):
        """Check optimization rules against current snapshot"""
        for rule in self.optimization_rules.values():
            if not rule.enabled:
                continue
            
            if self._evaluate_rule_condition(rule, snapshot):
                self._execute_optimization_action(rule, snapshot)
    
    def _evaluate_rule_condition(self, rule: OptimizationRule, snapshot: PerformanceSnapshot) -> bool:
        """Evaluate if rule condition is met"""
        try:
            # Create evaluation context
            context = {
                "cpu_percent": snapshot.cpu_percent,
                "memory_percent": snapshot.memory_percent,
                "response_time_ms": snapshot.response_time_ms,
                "error_rate": snapshot.error_rate,
                "cache_hit_rate": snapshot.cache_hit_rate,
                "queue_length": snapshot.queue_length,
                "threshold": rule.threshold
            }
            
            # Evaluate condition (simplified - in practice, use a proper expression evaluator)
            if rule.condition == "cpu_percent > 80":
                return snapshot.cpu_percent > rule.threshold
            elif rule.condition == "memory_percent > 85":
                return snapshot.memory_percent > rule.threshold
            elif rule.condition == "response_time_ms > 1000":
                return snapshot.response_time_ms > rule.threshold
            elif rule.condition == "error_rate > 5":
                return snapshot.error_rate > rule.threshold
            elif rule.condition == "cache_hit_rate < 70":
                return snapshot.cache_hit_rate < rule.threshold
            elif rule.condition == "queue_length > 100":
                return snapshot.queue_length > rule.threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating rule condition: {e}")
            return False
    
    def _execute_optimization_action(self, rule: OptimizationRule, snapshot: PerformanceSnapshot):
        """Execute optimization action"""
        try:
            action_id = str(uuid.uuid4())
            success = False
            impact_score = 0.0
            
            # Execute action based on type
            if rule.action == "scale_workers":
                success, impact_score = self._scale_workers(rule.optimization_level)
            elif rule.action == "clear_cache":
                success, impact_score = self._clear_cache(rule.optimization_level)
            elif rule.action == "optimize_queries":
                success, impact_score = self._optimize_queries(rule.optimization_level)
            elif rule.action == "enable_circuit_breaker":
                success, impact_score = self._enable_circuit_breaker(rule.optimization_level)
            elif rule.action == "increase_cache_size":
                success, impact_score = self._increase_cache_size(rule.optimization_level)
            
            # Record action
            action = OptimizationAction(
                id=action_id,
                rule_id=rule.id,
                action_type=rule.action,
                parameters={"optimization_level": rule.optimization_level.value},
                executed_at=datetime.now(),
                success=success,
                impact_score=impact_score,
                metadata={"snapshot": snapshot.__dict__}
            )
            
            self.optimization_actions.append(action)
            
            # Update rule statistics
            rule.last_triggered = datetime.now()
            rule.trigger_count += 1
            
            logger.info(f"Executed optimization action: {rule.action} (success: {success}, impact: {impact_score:.2f})")
            
        except Exception as e:
            logger.error(f"Error executing optimization action: {e}")
    
    def _scale_workers(self, level: OptimizationLevel) -> Tuple[bool, float]:
        """Scale worker processes"""
        try:
            # Simulate worker scaling
            if level == OptimizationLevel.MINIMAL:
                impact = 0.1
            elif level == OptimizationLevel.MODERATE:
                impact = 0.3
            elif level == OptimizationLevel.AGGRESSIVE:
                impact = 0.6
            else:  # MAXIMUM
                impact = 0.9
            
            logger.info(f"Scaling workers with {level.value} optimization")
            return True, impact
            
        except Exception as e:
            logger.error(f"Error scaling workers: {e}")
            return False, 0.0
    
    def _clear_cache(self, level: OptimizationLevel) -> Tuple[bool, float]:
        """Clear system cache"""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Calculate impact based on level
            if level == OptimizationLevel.MINIMAL:
                impact = 0.2
            elif level == OptimizationLevel.MODERATE:
                impact = 0.4
            elif level == OptimizationLevel.AGGRESSIVE:
                impact = 0.7
            else:  # MAXIMUM
                impact = 0.9
            
            logger.info(f"Cleared cache with {level.value} optimization (collected {collected} objects)")
            return True, impact
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False, 0.0
    
    def _optimize_queries(self, level: OptimizationLevel) -> Tuple[bool, float]:
        """Optimize database queries"""
        try:
            # Simulate query optimization
            impact = 0.3 if level in [OptimizationLevel.MODERATE, OptimizationLevel.AGGRESSIVE] else 0.1
            
            logger.info(f"Optimizing queries with {level.value} optimization")
            return True, impact
            
        except Exception as e:
            logger.error(f"Error optimizing queries: {e}")
            return False, 0.0
    
    def _enable_circuit_breaker(self, level: OptimizationLevel) -> Tuple[bool, float]:
        """Enable circuit breaker pattern"""
        try:
            # Simulate circuit breaker activation
            impact = 0.5 if level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.MAXIMUM] else 0.2
            
            logger.info(f"Enabling circuit breaker with {level.value} optimization")
            return True, impact
            
        except Exception as e:
            logger.error(f"Error enabling circuit breaker: {e}")
            return False, 0.0
    
    def _increase_cache_size(self, level: OptimizationLevel) -> Tuple[bool, float]:
        """Increase cache size"""
        try:
            # Simulate cache size increase
            impact = 0.4 if level in [OptimizationLevel.MODERATE, OptimizationLevel.AGGRESSIVE] else 0.2
            
            logger.info(f"Increasing cache size with {level.value} optimization")
            return True, impact
            
        except Exception as e:
            logger.error(f"Error increasing cache size: {e}")
            return False, 0.0
    
    def record_request_time(self, response_time: float):
        """Record request response time"""
        self.request_times.append(response_time)
    
    def record_error(self, error_type: str):
        """Record an error occurrence"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.cache_stats["hits"] += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_stats["misses"] += 1
    
    async def generate_performance_report(self, period_hours: int = 24) -> PerformanceReport:
        """Generate performance analysis report"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=period_hours)
        
        # Filter snapshots for the period
        period_snapshots = [
            snapshot for snapshot in self.performance_history
            if start_time <= snapshot.timestamp <= end_time
        ]
        
        if not period_snapshots:
            raise ValueError("No performance data available for the specified period")
        
        # Calculate average and peak metrics
        average_metrics = {
            PerformanceMetric.CPU_USAGE: statistics.mean([s.cpu_percent for s in period_snapshots]),
            PerformanceMetric.MEMORY_USAGE: statistics.mean([s.memory_percent for s in period_snapshots]),
            PerformanceMetric.RESPONSE_TIME: statistics.mean([s.response_time_ms for s in period_snapshots]),
            PerformanceMetric.ERROR_RATE: statistics.mean([s.error_rate for s in period_snapshots]),
            PerformanceMetric.CACHE_HIT_RATE: statistics.mean([s.cache_hit_rate for s in period_snapshots]),
            PerformanceMetric.QUEUE_LENGTH: statistics.mean([s.queue_length for s in period_snapshots])
        }
        
        peak_metrics = {
            PerformanceMetric.CPU_USAGE: max([s.cpu_percent for s in period_snapshots]),
            PerformanceMetric.MEMORY_USAGE: max([s.memory_percent for s in period_snapshots]),
            PerformanceMetric.RESPONSE_TIME: max([s.response_time_ms for s in period_snapshots]),
            PerformanceMetric.ERROR_RATE: max([s.error_rate for s in period_snapshots]),
            PerformanceMetric.CACHE_HIT_RATE: max([s.cache_hit_rate for s in period_snapshots]),
            PerformanceMetric.QUEUE_LENGTH: max([s.queue_length for s in period_snapshots])
        }
        
        # Identify optimization opportunities
        opportunities = []
        recommendations = []
        
        if average_metrics[PerformanceMetric.CPU_USAGE] > 70:
            opportunities.append("High CPU usage detected")
            recommendations.append("Consider scaling horizontally or optimizing algorithms")
        
        if average_metrics[PerformanceMetric.MEMORY_USAGE] > 80:
            opportunities.append("High memory usage detected")
            recommendations.append("Implement memory optimization and garbage collection tuning")
        
        if average_metrics[PerformanceMetric.RESPONSE_TIME] > 500:
            opportunities.append("Slow response times detected")
            recommendations.append("Optimize database queries and implement caching")
        
        if average_metrics[PerformanceMetric.ERROR_RATE] > 2:
            opportunities.append("High error rate detected")
            recommendations.append("Implement better error handling and circuit breakers")
        
        if average_metrics[PerformanceMetric.CACHE_HIT_RATE] < 80:
            opportunities.append("Low cache hit rate")
            recommendations.append("Increase cache size and improve cache strategies")
        
        # Create report
        report = PerformanceReport(
            id=str(uuid.uuid4()),
            period_start=start_time,
            period_end=end_time,
            average_metrics=average_metrics,
            peak_metrics=peak_metrics,
            optimization_opportunities=opportunities,
            recommendations=recommendations,
            generated_at=datetime.now(),
            metadata={
                "snapshots_analyzed": len(period_snapshots),
                "optimization_actions_taken": len([a for a in self.optimization_actions if start_time <= a.executed_at <= end_time])
            }
        )
        
        logger.info(f"Generated performance report for {period_hours} hours")
        
        return report
    
    def get_current_performance(self) -> Optional[PerformanceSnapshot]:
        """Get current performance snapshot"""
        if self.performance_history:
            return self.performance_history[-1]
        return None
    
    def get_optimization_rules(self) -> List[OptimizationRule]:
        """Get all optimization rules"""
        return list(self.performance_history.values())
    
    def get_optimization_actions(self, limit: int = 100) -> List[OptimizationAction]:
        """Get recent optimization actions"""
        return self.optimization_actions[-limit:]
    
    def get_optimizer_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        total_actions = len(self.optimization_actions)
        successful_actions = len([a for a in self.optimization_actions if a.success])
        
        # Action types
        action_types = {}
        for action in self.optimization_actions:
            action_types[action.action_type] = action_types.get(action.action_type, 0) + 1
        
        # Rule trigger counts
        rule_triggers = {rule.name: rule.trigger_count for rule in self.optimization_rules.values()}
        
        return {
            "monitoring_active": self.monitoring_active,
            "total_snapshots": len(self.performance_history),
            "total_optimization_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": (successful_actions / total_actions * 100) if total_actions > 0 else 0,
            "action_types": action_types,
            "rule_triggers": rule_triggers,
            "optimization_rules_count": len(self.optimization_rules),
            "enabled_rules_count": len([r for r in self.optimization_rules.values() if r.enabled])
        }

# Example usage
if __name__ == "__main__":
    # Initialize performance optimizer
    optimizer = AdvancedPerformanceOptimizer({
        "monitoring_interval": 5,
        "optimization_enabled": True
    })
    
    # Start monitoring
    asyncio.run(optimizer.start_monitoring())
    
    # Simulate some activity
    for i in range(10):
        optimizer.record_request_time(100 + i * 10)  # Simulate increasing response times
        if i % 3 == 0:
            optimizer.record_error("timeout")
        if i % 2 == 0:
            optimizer.record_cache_hit()
        else:
            optimizer.record_cache_miss()
        
        time.sleep(1)
    
    # Get current performance
    current = optimizer.get_current_performance()
    if current:
        print("Current Performance:")
        print(f"CPU Usage: {current.cpu_percent:.1f}%")
        print(f"Memory Usage: {current.memory_percent:.1f}%")
        print(f"Response Time: {current.response_time_ms:.1f}ms")
        print(f"Error Rate: {current.error_rate:.1f}%")
        print(f"Cache Hit Rate: {current.cache_hit_rate:.1f}%")
    
    # Generate performance report
    report = asyncio.run(optimizer.generate_performance_report(1))  # Last hour
    
    print(f"\nPerformance Report:")
    print(f"Period: {report.period_start} to {report.period_end}")
    print(f"Average CPU Usage: {report.average_metrics[PerformanceMetric.CPU_USAGE]:.1f}%")
    print(f"Average Response Time: {report.average_metrics[PerformanceMetric.RESPONSE_TIME]:.1f}ms")
    print(f"Optimization Opportunities: {len(report.optimization_opportunities)}")
    print(f"Recommendations: {len(report.recommendations)}")
    
    # Get statistics
    stats = optimizer.get_optimizer_statistics()
    print(f"\nOptimizer Statistics:")
    print(f"Total Snapshots: {stats['total_snapshots']}")
    print(f"Total Actions: {stats['total_optimization_actions']}")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    print(f"Enabled Rules: {stats['enabled_rules_count']}")
    
    # Stop monitoring
    asyncio.run(optimizer.stop_monitoring())
    
    print("\nAdvanced Performance Optimizer initialized successfully")

























