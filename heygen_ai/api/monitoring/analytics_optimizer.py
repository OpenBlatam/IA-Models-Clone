from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import structlog
import psutil
import numpy as np
from fastapi import Request, Response
    import torch
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Advanced Analytics and Monitoring Optimizer for HeyGen AI FastAPI
Real-time performance analytics, bottleneck detection, and optimization insights.
"""


try:
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = structlog.get_logger()

# =============================================================================
# Analytics Types
# =============================================================================

class MetricType(Enum):
    """Metric type enumeration."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    CACHE_PERFORMANCE = "cache_performance"
    DATABASE_PERFORMANCE = "database_performance"
    AI_MODEL_PERFORMANCE = "ai_model_performance"

class AlertLevel(Enum):
    """Alert level enumeration."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    alert_id: str
    level: AlertLevel
    metric_type: MetricType
    message: str
    threshold_value: float
    current_value: float
    timestamp: datetime
    endpoint: Optional[str] = None
    suggested_actions: List[str] = field(default_factory=list)

@dataclass
class BottleneckAnalysis:
    """Bottleneck analysis result."""
    component: str
    severity: float  # 0-1 scale
    impact_estimate: str
    root_cause: str
    optimization_suggestions: List[str]
    priority: int  # 1=high, 2=medium, 3=low

# =============================================================================
# Real-Time Metrics Collector
# =============================================================================

class RealTimeMetricsCollector:
    """Collect and analyze real-time performance metrics."""
    
    def __init__(self) -> Any:
        # Time-series data storage
        self.metrics_history: Dict[str, deque] = {
            "response_times": deque(maxlen=10000),
            "throughput": deque(maxlen=1000),
            "error_rates": deque(maxlen=1000),
            "cpu_usage": deque(maxlen=1000),
            "memory_usage": deque(maxlen=1000),
            "gpu_usage": deque(maxlen=1000),
            "cache_hit_rates": deque(maxlen=1000),
            "database_query_times": deque(maxlen=5000),
            "ai_inference_times": deque(maxlen=5000)
        }
        
        # Endpoint-specific metrics
        self.endpoint_metrics: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {
                "response_times": deque(maxlen=1000),
                "request_counts": deque(maxlen=1000),
                "error_counts": deque(maxlen=1000)
            }
        )
        
        # Real-time aggregations
        self.current_aggregates: Dict[str, float] = {}
        self.alert_thresholds: Dict[str, float] = {
            "avg_response_time": 1000,  # 1 second
            "p95_response_time": 5000,  # 5 seconds
            "error_rate": 0.05,  # 5%
            "cpu_usage": 80,  # 80%
            "memory_usage": 85,  # 85%
            "cache_hit_rate": 0.7  # 70%
        }
        
        # Background processing
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self) -> Any:
        """Start the metrics collector."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        logger.info("Real-time metrics collector started")
    
    async def stop(self) -> Any:
        """Stop the metrics collector."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Real-time metrics collector stopped")
    
    async def _processing_loop(self) -> Any:
        """Background processing loop for metrics aggregation."""
        while self.is_running:
            try:
                # Update real-time aggregates
                self._update_aggregates()
                
                # Collect system metrics
                self._collect_system_metrics()
                
                # Update throughput calculations
                self._update_throughput_metrics()
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Metrics processing error: {e}")
                await asyncio.sleep(1)
    
    def record_request_metric(
        self,
        endpoint: str,
        response_time_ms: float,
        status_code: int,
        response_size_bytes: Optional[int] = None
    ):
        """Record a request metric."""
        timestamp = time.time()
        
        # Global metrics
        self.metrics_history["response_times"].append((timestamp, response_time_ms))
        
        # Endpoint-specific metrics
        self.endpoint_metrics[endpoint]["response_times"].append((timestamp, response_time_ms))
        self.endpoint_metrics[endpoint]["request_counts"].append((timestamp, 1))
        
        if status_code >= 400:
            self.endpoint_metrics[endpoint]["error_counts"].append((timestamp, 1))
    
    def record_database_metric(self, query_time_ms: float, query_type: str = "unknown"):
        """Record database query metric."""
        timestamp = time.time()
        self.metrics_history["database_query_times"].append((timestamp, query_time_ms))
    
    def record_ai_inference_metric(self, inference_time_ms: float, model_name: str = "unknown"):
        """Record AI model inference metric."""
        timestamp = time.time()
        self.metrics_history["ai_inference_times"].append((timestamp, inference_time_ms))
    
    def record_cache_metric(self, hit: bool):
        """Record cache hit/miss metric."""
        timestamp = time.time()
        hit_rate = 1.0 if hit else 0.0
        self.metrics_history["cache_hit_rates"].append((timestamp, hit_rate))
    
    def _collect_system_metrics(self) -> Any:
        """Collect system resource metrics."""
        timestamp = time.time()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics_history["cpu_usage"].append((timestamp, cpu_percent))
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics_history["memory_usage"].append((timestamp, memory.percent))
            
            # GPU usage (if available)
            if HAS_TORCH and torch.cuda.is_available():
                try:
                    gpu_percent = torch.cuda.utilization()
                    self.metrics_history["gpu_usage"].append((timestamp, gpu_percent))
                except:
                    pass
            
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    def _update_aggregates(self) -> Any:
        """Update real-time aggregate metrics."""
        current_time = time.time()
        window_size = 60  # 1 minute window
        
        # Response time aggregates
        recent_response_times = [
            value for timestamp, value in self.metrics_history["response_times"]
            if current_time - timestamp <= window_size
        ]
        
        if recent_response_times:
            self.current_aggregates["avg_response_time"] = statistics.mean(recent_response_times)
            self.current_aggregates["p95_response_time"] = np.percentile(recent_response_times, 95)
            self.current_aggregates["p99_response_time"] = np.percentile(recent_response_times, 99)
        
        # Error rate aggregate
        total_requests = 0
        total_errors = 0
        
        for endpoint_data in self.endpoint_metrics.values():
            recent_requests = [
                value for timestamp, value in endpoint_data["request_counts"]
                if current_time - timestamp <= window_size
            ]
            recent_errors = [
                value for timestamp, value in endpoint_data["error_counts"]
                if current_time - timestamp <= window_size
            ]
            
            total_requests += sum(recent_requests)
            total_errors += sum(recent_errors)
        
        self.current_aggregates["error_rate"] = (
            total_errors / total_requests if total_requests > 0 else 0
        )
        
        # Cache hit rate aggregate
        recent_cache_hits = [
            value for timestamp, value in self.metrics_history["cache_hit_rates"]
            if current_time - timestamp <= window_size
        ]
        
        if recent_cache_hits:
            self.current_aggregates["cache_hit_rate"] = statistics.mean(recent_cache_hits)
        
        # Resource usage aggregates
        if self.metrics_history["cpu_usage"]:
            recent_cpu = [
                value for timestamp, value in self.metrics_history["cpu_usage"]
                if current_time - timestamp <= window_size
            ]
            if recent_cpu:
                self.current_aggregates["cpu_usage"] = statistics.mean(recent_cpu)
        
        if self.metrics_history["memory_usage"]:
            recent_memory = [
                value for timestamp, value in self.metrics_history["memory_usage"]
                if current_time - timestamp <= window_size
            ]
            if recent_memory:
                self.current_aggregates["memory_usage"] = statistics.mean(recent_memory)
    
    def _update_throughput_metrics(self) -> Any:
        """Update throughput calculations."""
        current_time = time.time()
        window_size = 60  # 1 minute window
        
        total_requests = 0
        for endpoint_data in self.endpoint_metrics.values():
            recent_requests = [
                value for timestamp, value in endpoint_data["request_counts"]
                if current_time - timestamp <= window_size
            ]
            total_requests += sum(recent_requests)
        
        # Calculate requests per second
        rps = total_requests / window_size
        self.metrics_history["throughput"].append((current_time, rps))
        self.current_aggregates["throughput_rps"] = rps
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values."""
        return self.current_aggregates.copy()
    
    def get_historical_data(
        self,
        metric_name: str,
        time_range_minutes: int = 60
    ) -> List[Tuple[float, float]]:
        """Get historical data for a specific metric."""
        if metric_name not in self.metrics_history:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - (time_range_minutes * 60)
        
        return [
            (timestamp, value) for timestamp, value in self.metrics_history[metric_name]
            if timestamp >= cutoff_time
        ]

# =============================================================================
# Bottleneck Detector
# =============================================================================

class BottleneckDetector:
    """Detect and analyze performance bottlenecks."""
    
    def __init__(self, metrics_collector: RealTimeMetricsCollector):
        
    """__init__ function."""
self.metrics_collector = metrics_collector
        self.bottleneck_history: List[BottleneckAnalysis] = []
        self.detection_rules = self._initialize_detection_rules()
    
    def _initialize_detection_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize bottleneck detection rules."""
        return {
            "high_response_time": {
                "threshold": 2000,  # 2 seconds
                "metric": "avg_response_time",
                "component": "application",
                "severity_multiplier": 2.0
            },
            "high_error_rate": {
                "threshold": 0.1,  # 10%
                "metric": "error_rate",
                "component": "application",
                "severity_multiplier": 3.0
            },
            "high_cpu_usage": {
                "threshold": 85,  # 85%
                "metric": "cpu_usage",
                "component": "system",
                "severity_multiplier": 1.5
            },
            "high_memory_usage": {
                "threshold": 90,  # 90%
                "metric": "memory_usage",
                "component": "system",
                "severity_multiplier": 2.0
            },
            "low_cache_hit_rate": {
                "threshold": 0.5,  # 50%
                "metric": "cache_hit_rate",
                "component": "cache",
                "severity_multiplier": 1.5,
                "invert": True  # Lower values are worse
            },
            "low_throughput": {
                "threshold": 10,  # 10 RPS
                "metric": "throughput_rps",
                "component": "application",
                "severity_multiplier": 1.0,
                "invert": True
            }
        }
    
    def analyze_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyze current metrics for bottlenecks."""
        current_metrics = self.metrics_collector.get_current_metrics()
        bottlenecks = []
        
        for rule_name, rule_config in self.detection_rules.items():
            metric_name = rule_config["metric"]
            threshold = rule_config["threshold"]
            component = rule_config["component"]
            severity_multiplier = rule_config["severity_multiplier"]
            invert = rule_config.get("invert", False)
            
            if metric_name not in current_metrics:
                continue
            
            current_value = current_metrics[metric_name]
            
            # Check if bottleneck condition is met
            is_bottleneck = (
                (not invert and current_value > threshold) or
                (invert and current_value < threshold)
            )
            
            if is_bottleneck:
                # Calculate severity (0-1 scale)
                if invert:
                    severity = (threshold - current_value) / threshold
                else:
                    severity = min(1.0, (current_value - threshold) / threshold)
                
                severity = min(1.0, severity * severity_multiplier)
                
                # Generate bottleneck analysis
                bottleneck = self._generate_bottleneck_analysis(
                    rule_name, component, current_value, threshold, severity
                )
                bottlenecks.append(bottleneck)
        
        # Sort by severity (highest first)
        bottlenecks.sort(key=lambda x: x.severity, reverse=True)
        
        # Store in history
        self.bottleneck_history.extend(bottlenecks)
        if len(self.bottleneck_history) > 100:
            self.bottleneck_history = self.bottleneck_history[-100:]
        
        return bottlenecks
    
    def _generate_bottleneck_analysis(
        self,
        rule_name: str,
        component: str,
        current_value: float,
        threshold: float,
        severity: float
    ) -> BottleneckAnalysis:
        """Generate detailed bottleneck analysis."""
        
        # Define root causes and suggestions based on component and rule
        analysis_data = {
            "high_response_time": {
                "root_cause": "Application processing time is elevated",
                "suggestions": [
                    "Check database query performance",
                    "Review CPU-intensive operations",
                    "Consider implementing caching",
                    "Optimize algorithm complexity",
                    "Scale horizontally"
                ]
            },
            "high_error_rate": {
                "root_cause": "Application is experiencing increased errors",
                "suggestions": [
                    "Review error logs for patterns",
                    "Check external service dependencies",
                    "Validate input data processing",
                    "Implement circuit breakers",
                    "Add more comprehensive error handling"
                ]
            },
            "high_cpu_usage": {
                "root_cause": "CPU resources are over-utilized",
                "suggestions": [
                    "Profile CPU-intensive code paths",
                    "Optimize algorithms and data structures",
                    "Consider async processing for I/O operations",
                    "Scale to more CPU cores",
                    "Implement CPU-based load balancing"
                ]
            },
            "high_memory_usage": {
                "root_cause": "Memory consumption is approaching limits",
                "suggestions": [
                    "Check for memory leaks",
                    "Optimize data structures",
                    "Implement object pooling",
                    "Add garbage collection tuning",
                    "Scale to higher memory instances"
                ]
            },
            "low_cache_hit_rate": {
                "root_cause": "Cache efficiency is below optimal levels",
                "suggestions": [
                    "Review cache key strategies",
                    "Optimize cache TTL settings",
                    "Implement cache warming",
                    "Increase cache size if memory allows",
                    "Review cache invalidation patterns"
                ]
            },
            "low_throughput": {
                "root_cause": "System throughput is below expected levels",
                "suggestions": [
                    "Check for blocking operations",
                    "Optimize database connections",
                    "Review request processing pipeline",
                    "Implement request batching",
                    "Scale horizontally"
                ]
            }
        }
        
        data = analysis_data.get(rule_name, {
            "root_cause": f"Performance degradation in {component}",
            "suggestions": ["Investigate component performance", "Consider scaling"]
        })
        
        # Determine impact estimate
        if severity > 0.8:
            impact_estimate = "Critical - Immediate action required"
            priority = 1
        elif severity > 0.6:
            impact_estimate = "High - Action needed soon"
            priority = 1
        elif severity > 0.4:
            impact_estimate = "Medium - Monitor closely"
            priority = 2
        else:
            impact_estimate = "Low - Track trends"
            priority = 3
        
        return BottleneckAnalysis(
            component=component,
            severity=severity,
            impact_estimate=impact_estimate,
            root_cause=data["root_cause"],
            optimization_suggestions=data["suggestions"],
            priority=priority
        )

# =============================================================================
# Performance Alert System
# =============================================================================

class PerformanceAlertSystem:
    """Generate and manage performance alerts."""
    
    def __init__(self, metrics_collector: RealTimeMetricsCollector):
        
    """__init__ function."""
self.metrics_collector = metrics_collector
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        self.alert_cooldown = 300  # 5 minutes between similar alerts
        self.last_alert_times: Dict[str, float] = {}
    
    def check_and_generate_alerts(self) -> List[PerformanceAlert]:
        """Check current metrics and generate alerts if needed."""
        current_metrics = self.metrics_collector.get_current_metrics()
        new_alerts = []
        current_time = time.time()
        
        # Define alert rules
        alert_rules = [
            {
                "metric": "avg_response_time",
                "threshold": 1000,
                "level": AlertLevel.WARNING,
                "message": "Average response time is elevated"
            },
            {
                "metric": "avg_response_time",
                "threshold": 3000,
                "level": AlertLevel.CRITICAL,
                "message": "Average response time is critically high"
            },
            {
                "metric": "p95_response_time",
                "threshold": 5000,
                "level": AlertLevel.CRITICAL,
                "message": "95th percentile response time is unacceptable"
            },
            {
                "metric": "error_rate",
                "threshold": 0.05,
                "level": AlertLevel.WARNING,
                "message": "Error rate is elevated"
            },
            {
                "metric": "error_rate",
                "threshold": 0.15,
                "level": AlertLevel.CRITICAL,
                "message": "Error rate is critically high"
            },
            {
                "metric": "cpu_usage",
                "threshold": 85,
                "level": AlertLevel.WARNING,
                "message": "CPU usage is high"
            },
            {
                "metric": "cpu_usage",
                "threshold": 95,
                "level": AlertLevel.CRITICAL,
                "message": "CPU usage is critically high"
            },
            {
                "metric": "memory_usage",
                "threshold": 90,
                "level": AlertLevel.WARNING,
                "message": "Memory usage is high"
            },
            {
                "metric": "memory_usage",
                "threshold": 98,
                "level": AlertLevel.CRITICAL,
                "message": "Memory usage is critically high"
            }
        ]
        
        for rule in alert_rules:
            metric_name = rule["metric"]
            threshold = rule["threshold"]
            level = rule["level"]
            message = rule["message"]
            
            if metric_name not in current_metrics:
                continue
            
            current_value = current_metrics[metric_name]
            
            # Check if alert condition is met
            if current_value > threshold:
                alert_key = f"{metric_name}_{level.value}"
                
                # Check cooldown
                if (alert_key in self.last_alert_times and
                    current_time - self.last_alert_times[alert_key] < self.alert_cooldown):
                    continue
                
                # Generate alert
                alert = PerformanceAlert(
                    alert_id=f"{alert_key}_{int(current_time)}",
                    level=level,
                    metric_type=self._metric_to_type(metric_name),
                    message=f"{message}: {current_value:.2f} (threshold: {threshold})",
                    threshold_value=threshold,
                    current_value=current_value,
                    timestamp=datetime.now(timezone.utc),
                    suggested_actions=self._get_suggested_actions(metric_name, level)
                )
                
                new_alerts.append(alert)
                self.active_alerts[alert.alert_id] = alert
                self.last_alert_times[alert_key] = current_time
        
        # Add to history
        self.alert_history.extend(new_alerts)
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        return new_alerts
    
    def _metric_to_type(self, metric_name: str) -> MetricType:
        """Convert metric name to metric type."""
        if "response_time" in metric_name:
            return MetricType.LATENCY
        elif "error_rate" in metric_name:
            return MetricType.ERROR_RATE
        elif "cpu_usage" in metric_name or "memory_usage" in metric_name:
            return MetricType.RESOURCE_USAGE
        elif "cache" in metric_name:
            return MetricType.CACHE_PERFORMANCE
        elif "throughput" in metric_name:
            return MetricType.THROUGHPUT
        else:
            return MetricType.LATENCY
    
    def _get_suggested_actions(self, metric_name: str, level: AlertLevel) -> List[str]:
        """Get suggested actions for specific metric alerts."""
        actions = {
            "avg_response_time": [
                "Check database query performance",
                "Review application bottlenecks",
                "Consider horizontal scaling"
            ],
            "error_rate": [
                "Review error logs",
                "Check external dependencies",
                "Implement circuit breakers"
            ],
            "cpu_usage": [
                "Profile CPU-intensive operations",
                "Optimize algorithms",
                "Scale to more cores"
            ],
            "memory_usage": [
                "Check for memory leaks",
                "Optimize data structures",
                "Scale to higher memory"
            ]
        }
        
        return actions.get(metric_name, ["Investigate performance issue"])
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get currently active alerts."""
        return list(self.active_alerts.values())
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]

# =============================================================================
# Analytics Optimizer
# =============================================================================

class AnalyticsOptimizer:
    """Main analytics and optimization system."""
    
    def __init__(self) -> Any:
        self.metrics_collector = RealTimeMetricsCollector()
        self.bottleneck_detector = BottleneckDetector(self.metrics_collector)
        self.alert_system = PerformanceAlertSystem(self.metrics_collector)
        self.optimization_recommendations: List[str] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self) -> Any:
        """Start the analytics optimizer."""
        if self.is_running:
            return
        
        await self.metrics_collector.start()
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Analytics optimizer started")
    
    async def stop(self) -> Any:
        """Stop the analytics optimizer."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        await self.metrics_collector.stop()
        logger.info("Analytics optimizer stopped")
    
    async def _monitoring_loop(self) -> Any:
        """Background monitoring and analysis loop."""
        while self.is_running:
            try:
                # Analyze bottlenecks
                bottlenecks = self.bottleneck_detector.analyze_bottlenecks()
                
                # Check for alerts
                new_alerts = self.alert_system.check_and_generate_alerts()
                
                # Generate optimization recommendations
                self._update_optimization_recommendations(bottlenecks)
                
                # Log critical issues
                for alert in new_alerts:
                    if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                        logger.warning(f"Performance alert: {alert.message}")
                
                await asyncio.sleep(30)  # Analyze every 30 seconds
                
            except Exception as e:
                logger.error(f"Analytics monitoring error: {e}")
                await asyncio.sleep(30)
    
    def _update_optimization_recommendations(self, bottlenecks: List[BottleneckAnalysis]):
        """Update optimization recommendations based on bottlenecks."""
        recommendations = set()
        
        for bottleneck in bottlenecks:
            if bottleneck.severity > 0.5:  # Only for significant bottlenecks
                recommendations.update(bottleneck.optimization_suggestions)
        
        self.optimization_recommendations = list(recommendations)
    
    def record_request(
        self,
        endpoint: str,
        response_time_ms: float,
        status_code: int,
        response_size_bytes: Optional[int] = None
    ):
        """Record request metrics."""
        self.metrics_collector.record_request_metric(
            endpoint, response_time_ms, status_code, response_size_bytes
        )
    
    def record_database_query(self, query_time_ms: float, query_type: str = "unknown"):
        """Record database query metrics."""
        self.metrics_collector.record_database_metric(query_time_ms, query_type)
    
    def record_ai_inference(self, inference_time_ms: float, model_name: str = "unknown"):
        """Record AI inference metrics."""
        self.metrics_collector.record_ai_inference_metric(inference_time_ms, model_name)
    
    def record_cache_hit(self, hit: bool):
        """Record cache hit/miss."""
        self.metrics_collector.record_cache_metric(hit)
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive performance analysis report."""
        current_metrics = self.metrics_collector.get_current_metrics()
        bottlenecks = self.bottleneck_detector.analyze_bottlenecks()
        active_alerts = self.alert_system.get_active_alerts()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_metrics": current_metrics,
            "bottlenecks": [
                {
                    "component": b.component,
                    "severity": b.severity,
                    "impact": b.impact_estimate,
                    "root_cause": b.root_cause,
                    "suggestions": b.optimization_suggestions,
                    "priority": b.priority
                }
                for b in bottlenecks
            ],
            "active_alerts": [
                {
                    "id": a.alert_id,
                    "level": a.level.value,
                    "metric_type": a.metric_type.value,
                    "message": a.message,
                    "threshold": a.threshold_value,
                    "current_value": a.current_value,
                    "timestamp": a.timestamp.isoformat(),
                    "suggested_actions": a.suggested_actions
                }
                for a in active_alerts
            ],
            "optimization_recommendations": self.optimization_recommendations,
            "summary": {
                "total_bottlenecks": len(bottlenecks),
                "critical_alerts": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
                "avg_severity": statistics.mean([b.severity for b in bottlenecks]) if bottlenecks else 0,
                "recommendations_count": len(self.optimization_recommendations)
            }
        }

# =============================================================================
# Factory Function
# =============================================================================

async def create_analytics_optimizer() -> AnalyticsOptimizer:
    """Create and initialize analytics optimizer."""
    optimizer = AnalyticsOptimizer()
    await optimizer.start()
    return optimizer 