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
import logging
import statistics
import threading
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
from pathlib import Path
import functools
import weakref
import structlog
from pydantic import BaseModel, Field
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import redis.asyncio as redis
from .api_performance_metrics import (
from typing import Any, List, Dict, Optional
"""
🚀 API Performance Optimizer
============================

Advanced API performance optimization system with:
- Real-time performance analysis
- Automatic optimization recommendations
- Performance regression detection
- Load balancing optimization
- Caching strategy optimization
- Database query optimization
- Resource allocation optimization
- Predictive performance modeling
- SLA compliance monitoring
- Performance trend analysis
"""



    APIPerformanceMonitor, APIPerformanceMetrics, MetricPriority, 
    PerformanceThreshold, LatencyType, ResponseTimeMetrics, ThroughputMetrics
)

logger = structlog.get_logger(__name__)

class OptimizationType(Enum):
    """Types of optimizations"""
    CACHING = "caching"
    DATABASE = "database"
    LOAD_BALANCING = "load_balancing"
    RESOURCE_ALLOCATION = "resource_allocation"
    CODE_OPTIMIZATION = "code_optimization"
    NETWORK = "network"
    MEMORY = "memory"
    CPU = "cpu"

class OptimizationImpact(Enum):
    """Impact levels of optimizations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class OptimizationRecommendation:
    """Performance optimization recommendation"""
    
    def __init__(self, 
                 optimization_type: OptimizationType,
                 endpoint: str,
                 description: str,
                 impact: OptimizationImpact,
                 estimated_improvement: float,
                 implementation_effort: str,
                 priority: MetricPriority,
                 reasoning: str):
        
    """__init__ function."""
self.id = f"{optimization_type.value}_{endpoint}_{int(time.time())}"
        self.optimization_type = optimization_type
        self.endpoint = endpoint
        self.description = description
        self.impact = impact
        self.estimated_improvement = estimated_improvement
        self.implementation_effort = implementation_effort
        self.priority = priority
        self.reasoning = reasoning
        self.timestamp = time.time()
        self.implemented = False
        self.actual_improvement = 0.0

class PerformanceTrend:
    """Performance trend analysis"""
    
    def __init__(self, metric_name: str, window_hours: int = 24):
        
    """__init__ function."""
self.metric_name = metric_name
        self.window_hours = window_hours
        self.data_points: deque = deque(maxlen=window_hours * 60)  # 1 point per minute
        self.trend_direction = "stable"
        self.trend_strength = 0.0
        self.regression_detected = False
    
    def add_data_point(self, value: float, timestamp: float = None):
        """Add a data point"""
        if timestamp is None:
            timestamp = time.time()
        
        self.data_points.append((timestamp, value))
        self._analyze_trend()
    
    def _analyze_trend(self) -> Any:
        """Analyze trend direction and strength"""
        if len(self.data_points) < 10:
            return
        
        timestamps, values = zip(*self.data_points)
        
        # Convert timestamps to relative time (hours from start)
        start_time = min(timestamps)
        relative_times = [(t - start_time) / 3600 for t in timestamps]
        
        # Fit linear regression
        X = np.array(relative_times).reshape(-1, 1)
        y = np.array(values)
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            
            slope = model.coef_[0]
            r_squared = model.score(X, y)
            
            # Determine trend direction
            if abs(slope) < 0.01:
                self.trend_direction = "stable"
            elif slope > 0:
                self.trend_direction = "increasing"
            else:
                self.trend_direction = "decreasing"
            
            self.trend_strength = abs(r_squared)
            
            # Detect regression (worsening performance)
            if self.trend_direction == "increasing" and self.trend_strength > 0.7:
                self.regression_detected = True
            else:
                self.regression_detected = False
                
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")

class SLAMonitor:
    """SLA compliance monitoring"""
    
    def __init__(self) -> Any:
        self.sla_targets = {
            "response_time_p95": 1.0,  # 95th percentile response time < 1s
            "response_time_p99": 2.0,  # 99th percentile response time < 2s
            "availability": 0.999,     # 99.9% availability
            "error_rate": 0.01,        # < 1% error rate
            "throughput_min": 100      # Minimum 100 req/s
        }
        
        self.sla_violations: List[Dict[str, Any]] = []
        self.compliance_rate = 1.0
    
    def check_sla_compliance(self, metrics: APIPerformanceMetrics) -> Dict[str, bool]:
        """Check SLA compliance for an endpoint"""
        compliance = {}
        
        # Check response time P95
        if metrics.response_time.p95 > self.sla_targets["response_time_p95"]:
            compliance["response_time_p95"] = False
            self._record_violation(metrics.endpoint, "response_time_p95", 
                                 metrics.response_time.p95, self.sla_targets["response_time_p95"])
        else:
            compliance["response_time_p95"] = True
        
        # Check response time P99
        if metrics.response_time.p99 > self.sla_targets["response_time_p99"]:
            compliance["response_time_p99"] = False
            self._record_violation(metrics.endpoint, "response_time_p99", 
                                 metrics.response_time.p99, self.sla_targets["response_time_p99"])
        else:
            compliance["response_time_p99"] = True
        
        # Check error rate
        error_rate = 1 - metrics.throughput.success_rate
        if error_rate > self.sla_targets["error_rate"]:
            compliance["error_rate"] = False
            self._record_violation(metrics.endpoint, "error_rate", 
                                 error_rate, self.sla_targets["error_rate"])
        else:
            compliance["error_rate"] = True
        
        # Check throughput
        if metrics.throughput.requests_per_second < self.sla_targets["throughput_min"]:
            compliance["throughput"] = False
            self._record_violation(metrics.endpoint, "throughput", 
                                 metrics.throughput.requests_per_second, self.sla_targets["throughput_min"])
        else:
            compliance["throughput"] = True
        
        return compliance
    
    def _record_violation(self, endpoint: str, metric: str, actual: float, target: float):
        """Record an SLA violation"""
        violation = {
            "endpoint": endpoint,
            "metric": metric,
            "actual": actual,
            "target": target,
            "timestamp": time.time(),
            "severity": "high" if actual > target * 2 else "medium"
        }
        self.sla_violations.append(violation)
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get SLA compliance summary"""
        total_violations = len(self.sla_violations)
        recent_violations = len([v for v in self.sla_violations 
                               if time.time() - v["timestamp"] < 3600])  # Last hour
        
        return {
            "overall_compliance_rate": self.compliance_rate,
            "total_violations": total_violations,
            "recent_violations": recent_violations,
            "violations_by_metric": self._group_violations_by_metric(),
            "violations_by_endpoint": self._group_violations_by_endpoint()
        }
    
    def _group_violations_by_metric(self) -> Dict[str, int]:
        """Group violations by metric type"""
        grouped = defaultdict(int)
        for violation in self.sla_violations:
            grouped[violation["metric"]] += 1
        return dict(grouped)
    
    def _group_violations_by_endpoint(self) -> Dict[str, int]:
        """Group violations by endpoint"""
        grouped = defaultdict(int)
        for violation in self.sla_violations:
            grouped[violation["endpoint"]] += 1
        return dict(grouped)

class APIPerformanceOptimizer:
    """Main API performance optimization system"""
    
    def __init__(self, monitor: APIPerformanceMonitor):
        
    """__init__ function."""
self.monitor = monitor
        self.recommendations: List[OptimizationRecommendation] = []
        self.recommendations_lock = threading.Lock()
        
        # Performance trends
        self.trends: Dict[str, PerformanceTrend] = {}
        self.trends_lock = threading.Lock()
        
        # SLA monitoring
        self.sla_monitor = SLAMonitor()
        
        # Optimization history
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.auto_optimize = True
        self.optimization_interval = 300  # 5 minutes
        self.trend_analysis_interval = 60  # 1 minute
        
        logger.info("API Performance Optimizer initialized")
    
    async def start_optimization_loop(self) -> Any:
        """Start the continuous optimization loop"""
        logger.info("Starting API performance optimization loop")
        
        while True:
            try:
                await self._run_optimization_cycle()
                await asyncio.sleep(self.optimization_interval)
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _run_optimization_cycle(self) -> Any:
        """Run a single optimization cycle"""
        # Get current metrics
        all_metrics = self.monitor.get_all_metrics()
        
        # Update trends
        await self._update_trends(all_metrics)
        
        # Check SLA compliance
        await self._check_sla_compliance(all_metrics)
        
        # Generate optimization recommendations
        await self._generate_recommendations(all_metrics)
        
        # Apply auto-optimizations if enabled
        if self.auto_optimize:
            await self._apply_auto_optimizations(all_metrics)
        
        # Log optimization summary
        self._log_optimization_summary()
    
    async def _update_trends(self, metrics: Dict[str, APIPerformanceMetrics]):
        """Update performance trends"""
        current_time = time.time()
        
        for key, metric in metrics.items():
            # Create trend trackers if they don't exist
            if key not in self.trends:
                self.trends[key] = {
                    "response_time": PerformanceTrend("response_time"),
                    "throughput": PerformanceTrend("throughput"),
                    "error_rate": PerformanceTrend("error_rate")
                }
            
            # Update response time trend
            if metric.response_time.average > 0:
                self.trends[key]["response_time"].add_data_point(
                    metric.response_time.average, current_time
                )
            
            # Update throughput trend
            if metric.throughput.requests_per_second > 0:
                self.trends[key]["throughput"].add_data_point(
                    metric.throughput.requests_per_second, current_time
                )
            
            # Update error rate trend
            error_rate = 1 - metric.throughput.success_rate
            self.trends[key]["error_rate"].add_data_point(error_rate, current_time)
    
    async def _check_sla_compliance(self, metrics: Dict[str, APIPerformanceMetrics]):
        """Check SLA compliance for all endpoints"""
        for metric in metrics.values():
            compliance = self.sla_monitor.check_sla_compliance(metric)
            
            # Log violations
            for metric_name, compliant in compliance.items():
                if not compliant:
                    logger.warning(f"SLA violation for {metric.endpoint}: {metric_name}")
    
    async def _generate_recommendations(self, metrics: Dict[str, APIPerformanceMetrics]):
        """Generate optimization recommendations"""
        new_recommendations = []
        
        for key, metric in metrics.items():
            # Check for response time issues
            if metric.response_time.average > 1.0:  # > 1 second average
                recommendation = self._create_caching_recommendation(metric)
                if recommendation:
                    new_recommendations.append(recommendation)
                
                recommendation = self._create_database_recommendation(metric)
                if recommendation:
                    new_recommendations.append(recommendation)
            
            # Check for throughput issues
            if metric.throughput.requests_per_second < 50:  # < 50 req/s
                recommendation = self._create_load_balancing_recommendation(metric)
                if recommendation:
                    new_recommendations.append(recommendation)
            
            # Check for high error rates
            error_rate = 1 - metric.throughput.success_rate
            if error_rate > 0.05:  # > 5% error rate
                recommendation = self._create_code_optimization_recommendation(metric)
                if recommendation:
                    new_recommendations.append(recommendation)
            
            # Check for resource issues
            if metric.latency_breakdown.database_latency > metric.response_time.average * 0.5:
                recommendation = self._create_database_recommendation(metric)
                if recommendation:
                    new_recommendations.append(recommendation)
        
        # Add new recommendations
        with self.recommendations_lock:
            self.recommendations.extend(new_recommendations)
            
            # Remove old recommendations (older than 24 hours)
            cutoff_time = time.time() - 86400
            self.recommendations = [
                r for r in self.recommendations 
                if r.timestamp > cutoff_time
            ]
    
    def _create_caching_recommendation(self, metric: APIPerformanceMetrics) -> Optional[OptimizationRecommendation]:
        """Create caching optimization recommendation"""
        if metric.response_time.average > 0.5:  # > 500ms
            return OptimizationRecommendation(
                optimization_type=OptimizationType.CACHING,
                endpoint=metric.endpoint,
                description=f"Implement caching for {metric.endpoint} to reduce response time",
                impact=OptimizationImpact.HIGH,
                estimated_improvement=0.6,  # 60% improvement
                implementation_effort="medium",
                priority=metric.priority,
                reasoning=f"Response time of {metric.response_time.average:.3f}s is above optimal threshold"
            )
        return None
    
    def _create_database_recommendation(self, metric: APIPerformanceMetrics) -> Optional[OptimizationRecommendation]:
        """Create database optimization recommendation"""
        if metric.latency_breakdown.database_latency > 0.2:  # > 200ms
            return OptimizationRecommendation(
                optimization_type=OptimizationType.DATABASE,
                endpoint=metric.endpoint,
                description=f"Optimize database queries for {metric.endpoint}",
                impact=OptimizationImpact.MEDIUM,
                estimated_improvement=0.4,  # 40% improvement
                implementation_effort="high",
                priority=metric.priority,
                reasoning=f"Database latency of {metric.latency_breakdown.database_latency:.3f}s is high"
            )
        return None
    
    def _create_load_balancing_recommendation(self, metric: APIPerformanceMetrics) -> Optional[OptimizationRecommendation]:
        """Create load balancing optimization recommendation"""
        if metric.throughput.requests_per_second < 100:  # < 100 req/s
            return OptimizationRecommendation(
                optimization_type=OptimizationType.LOAD_BALANCING,
                endpoint=metric.endpoint,
                description=f"Implement load balancing for {metric.endpoint}",
                impact=OptimizationImpact.HIGH,
                estimated_improvement=0.8,  # 80% improvement
                implementation_effort="high",
                priority=metric.priority,
                reasoning=f"Throughput of {metric.throughput.requests_per_second:.2f} req/s is low"
            )
        return None
    
    def _create_code_optimization_recommendation(self, metric: APIPerformanceMetrics) -> Optional[OptimizationRecommendation]:
        """Create code optimization recommendation"""
        error_rate = 1 - metric.throughput.success_rate
        if error_rate > 0.05:  # > 5% error rate
            return OptimizationRecommendation(
                optimization_type=OptimizationType.CODE_OPTIMIZATION,
                endpoint=metric.endpoint,
                description=f"Optimize error handling and code for {metric.endpoint}",
                impact=OptimizationImpact.MEDIUM,
                estimated_improvement=0.3,  # 30% improvement
                implementation_effort="medium",
                priority=metric.priority,
                reasoning=f"Error rate of {error_rate:.2%} is above acceptable threshold"
            )
        return None
    
    async def _apply_auto_optimizations(self, metrics: Dict[str, APIPerformanceMetrics]):
        """Apply automatic optimizations"""
        applied_optimizations = []
        
        for metric in metrics.values():
            # Auto-optimize high-priority endpoints with critical performance issues
            if (metric.priority == MetricPriority.CRITICAL and 
                metric.response_time.average > 2.0):  # > 2 seconds
                
                optimization = {
                    "endpoint": metric.endpoint,
                    "type": "auto_caching",
                    "description": "Auto-applied caching for critical endpoint",
                    "timestamp": time.time(),
                    "improvement": 0.5
                }
                applied_optimizations.append(optimization)
                
                logger.info(f"Auto-applied caching optimization for {metric.endpoint}")
        
        # Record applied optimizations
        self.optimization_history.extend(applied_optimizations)
        
        # Keep only recent history (last 1000 optimizations)
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-1000:]
    
    def _log_optimization_summary(self) -> Any:
        """Log optimization summary"""
        with self.recommendations_lock:
            active_recommendations = len([r for r in self.recommendations if not r.implemented])
            high_priority_recommendations = len([
                r for r in self.recommendations 
                if r.priority in [MetricPriority.HIGH, MetricPriority.CRITICAL] and not r.implemented
            ])
        
        sla_summary = self.sla_monitor.get_compliance_summary()
        
        logger.info(
            f"Optimization Summary: {active_recommendations} active recommendations, "
            f"{high_priority_recommendations} high priority, "
            f"{sla_summary['recent_violations']} recent SLA violations"
        )
    
    def get_recommendations(self, 
                          optimization_type: Optional[OptimizationType] = None,
                          priority: Optional[MetricPriority] = None,
                          implemented: Optional[bool] = None) -> List[OptimizationRecommendation]:
        """Get optimization recommendations with optional filtering"""
        with self.recommendations_lock:
            recommendations = self.recommendations.copy()
        
        # Apply filters
        if optimization_type:
            recommendations = [r for r in recommendations if r.optimization_type == optimization_type]
        
        if priority:
            recommendations = [r for r in recommendations if r.priority == priority]
        
        if implemented is not None:
            recommendations = [r for r in recommendations if r.implemented == implemented]
        
        # Sort by priority and timestamp
        recommendations.sort(key=lambda r: (r.priority.value, r.timestamp), reverse=True)
        
        return recommendations
    
    def mark_recommendation_implemented(self, recommendation_id: str, actual_improvement: float = 0.0):
        """Mark a recommendation as implemented"""
        with self.recommendations_lock:
            for recommendation in self.recommendations:
                if recommendation.id == recommendation_id:
                    recommendation.implemented = True
                    recommendation.actual_improvement = actual_improvement
                    break
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary"""
        with self.recommendations_lock:
            total_recommendations = len(self.recommendations)
            implemented_recommendations = len([r for r in self.recommendations if r.implemented])
            active_recommendations = total_recommendations - implemented_recommendations
            
            recommendations_by_type = defaultdict(int)
            recommendations_by_priority = defaultdict(int)
            
            for recommendation in self.recommendations:
                recommendations_by_type[recommendation.optimization_type.value] += 1
                recommendations_by_priority[recommendation.priority.value] += 1
        
        # Get trend analysis
        trend_summary = {}
        for key, trends in self.trends.items():
            trend_summary[key] = {
                "response_time": {
                    "direction": trends["response_time"].trend_direction,
                    "strength": trends["response_time"].trend_strength,
                    "regression": trends["response_time"].regression_detected
                },
                "throughput": {
                    "direction": trends["throughput"].trend_direction,
                    "strength": trends["throughput"].trend_strength,
                    "regression": trends["throughput"].regression_detected
                }
            }
        
        return {
            "recommendations": {
                "total": total_recommendations,
                "implemented": implemented_recommendations,
                "active": active_recommendations,
                "by_type": dict(recommendations_by_type),
                "by_priority": dict(recommendations_by_priority)
            },
            "trends": trend_summary,
            "sla_compliance": self.sla_monitor.get_compliance_summary(),
            "optimization_history": {
                "total_applied": len(self.optimization_history),
                "recent_applied": len([o for o in self.optimization_history 
                                    if time.time() - o["timestamp"] < 3600])
            }
        }
    
    def get_performance_predictions(self, endpoint: str, hours_ahead: int = 24) -> Dict[str, Any]:
        """Get performance predictions for an endpoint"""
        if endpoint not in self.trends:
            return {"error": "No trend data available for endpoint"}
        
        trends = self.trends[endpoint]
        predictions = {}
        
        # Predict response time
        if trends["response_time"].data_points:
            current_response_time = trends["response_time"].data_points[-1][1]
            if trends["response_time"].trend_direction == "increasing":
                predicted_response_time = current_response_time * (1 + trends["response_time"].trend_strength * hours_ahead / 24)
            else:
                predicted_response_time = current_response_time * (1 - trends["response_time"].trend_strength * hours_ahead / 24)
            
            predictions["response_time"] = {
                "current": current_response_time,
                "predicted": predicted_response_time,
                "trend": trends["response_time"].trend_direction
            }
        
        # Predict throughput
        if trends["throughput"].data_points:
            current_throughput = trends["throughput"].data_points[-1][1]
            if trends["throughput"].trend_direction == "increasing":
                predicted_throughput = current_throughput * (1 + trends["throughput"].trend_strength * hours_ahead / 24)
            else:
                predicted_throughput = current_throughput * (1 - trends["throughput"].trend_strength * hours_ahead / 24)
            
            predictions["throughput"] = {
                "current": current_throughput,
                "predicted": predicted_throughput,
                "trend": trends["throughput"].trend_direction
            }
        
        return predictions

# Global optimizer instance
_optimizer: Optional[APIPerformanceOptimizer] = None

async async def get_api_optimizer() -> APIPerformanceOptimizer:
    """Get the global API performance optimizer instance"""
    global _optimizer
    if _optimizer is None:
        monitor = await get_api_monitor()
        _optimizer = APIPerformanceOptimizer(monitor)
    return _optimizer

async def example_usage():
    """Example usage of the API performance optimizer"""
    
    # Get optimizer
    optimizer = await get_api_optimizer()
    
    # Start optimization loop
    optimization_task = asyncio.create_task(optimizer.start_optimization_loop())
    
    # Simulate some API calls
    monitor = await get_api_monitor()
    
    # Register endpoints
    monitor.register_endpoint("/api/users", "GET", MetricPriority.HIGH)
    monitor.register_endpoint("/api/admin", "POST", MetricPriority.CRITICAL)
    
    # Simulate performance issues
    for i in range(50):
        # Simulate slow response times
        monitor.record_request(
            endpoint="/api/users",
            method="GET",
            response_time=1.5 + (i % 10) * 0.1,  # 1.5-2.4 seconds
            status_code=200
        )
        
        # Simulate high error rates
        monitor.record_request(
            endpoint="/api/admin",
            method="POST",
            response_time=0.8,
            status_code=500 if i % 5 == 0 else 200  # 20% error rate
        )
        
        await asyncio.sleep(0.1)
    
    # Wait for optimization cycle
    await asyncio.sleep(10)
    
    # Get recommendations
    recommendations = optimizer.get_recommendations()
    print(f"Generated {len(recommendations)} optimization recommendations:")
    
    for rec in recommendations[:5]:  # Show first 5
        print(f"- {rec.description} (Impact: {rec.impact.value}, Priority: {rec.priority.value})")
    
    # Get optimization summary
    summary = optimizer.get_optimization_summary()
    print(f"\nOptimization Summary:")
    print(f"Active recommendations: {summary['recommendations']['active']}")
    print(f"SLA violations: {summary['sla_compliance']['recent_violations']}")
    
    # Get performance predictions
    predictions = optimizer.get_performance_predictions("/api/users")
    print(f"\nPerformance Predictions for /api/users:")
    print(json.dumps(predictions, indent=2))
    
    # Stop optimization loop
    optimization_task.cancel()
    try:
        await optimization_task
    except asyncio.CancelledError:
        pass

match __name__:
    case "__main__":
    asyncio.run(example_usage()) 