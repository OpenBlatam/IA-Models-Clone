#!/usr/bin/env python3
"""
ADVANCED MONITORING AND ANALYTICS SYSTEM v4.0
=============================================

Comprehensive monitoring, analytics, and observability system
consolidating all previous monitoring features with modern practices.

Features:
- Real-time performance monitoring
- Advanced analytics and insights
- Predictive analytics
- Alert management
- Dashboard generation
- Performance optimization recommendations
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import aiohttp
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
import psutil
import gc
from collections import defaultdict, deque
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertLevel(str, Enum):
    """Alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricData:
    """Metric data structure"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class PerformanceInsight:
    """Performance insight data structure"""
    insight_type: str
    description: str
    impact: str
    recommendation: str
    confidence: float
    timestamp: datetime


class AdvancedMonitoringSystem:
    """
    Advanced monitoring and analytics system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the monitoring system"""
        self.config = config or self._get_default_config()
        self.metrics_buffer = deque(maxlen=10000)
        self.alerts = []
        self.insights = []
        self.performance_history = defaultdict(list)
        self.alert_rules = {}
        
        # Prometheus metrics
        self.prometheus_metrics = self._initialize_prometheus_metrics()
        
        # Services
        self.redis_client = None
        self.http_session = None
        self.monitoring_tasks = []
        
        logger.info("Advanced Monitoring System initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "monitoring": {
                "metrics_interval": 5.0,
                "alert_check_interval": 10.0,
                "insight_generation_interval": 60.0,
                "retention_days": 30,
                "buffer_size": 10000
            },
            "alerts": {
                "cpu_threshold": 85.0,
                "memory_threshold": 90.0,
                "error_rate_threshold": 0.05,
                "latency_threshold": 1000.0,
                "throughput_threshold": 100.0
            },
            "analytics": {
                "trend_analysis_window": 24,  # hours
                "anomaly_detection": True,
                "predictive_analytics": True,
                "correlation_analysis": True
            },
            "storage": {
                "redis_url": "redis://localhost:6379",
                "metrics_ttl": 86400,  # 24 hours
                "alerts_ttl": 604800,  # 7 days
                "insights_ttl": 2592000  # 30 days
            }
        }
    
    def _initialize_prometheus_metrics(self) -> Dict[str, Any]:
        """Initialize Prometheus metrics"""
        return {
            "request_total": Counter(
                'facebook_posts_requests_total',
                'Total requests',
                ['method', 'endpoint', 'status_code']
            ),
            "request_duration": Histogram(
                'facebook_posts_request_duration_seconds',
                'Request duration',
                ['method', 'endpoint'],
                buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
            ),
            "active_connections": Gauge(
                'facebook_posts_active_connections',
                'Active connections'
            ),
            "memory_usage": Gauge(
                'facebook_posts_memory_usage_bytes',
                'Memory usage in bytes'
            ),
            "cpu_usage": Gauge(
                'facebook_posts_cpu_usage_percent',
                'CPU usage percentage'
            ),
            "cache_hits": Counter(
                'facebook_posts_cache_hits_total',
                'Cache hits'
            ),
            "cache_misses": Counter(
                'facebook_posts_cache_misses_total',
                'Cache misses'
            ),
            "ai_generation_time": Histogram(
                'facebook_posts_ai_generation_seconds',
                'AI generation time',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            ),
            "optimization_time": Histogram(
                'facebook_posts_optimization_seconds',
                'Optimization time',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            ),
            "error_rate": Gauge(
                'facebook_posts_error_rate',
                'Error rate'
            ),
            "throughput": Gauge(
                'facebook_posts_throughput_rps',
                'Throughput in requests per second'
            )
        }
    
    async def initialize(self) -> None:
        """Initialize the monitoring system"""
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(
                self.config["storage"]["redis_url"],
                max_connections=50
            )
            
            # Initialize HTTP session
            connector = aiohttp.TCPConnector(limit=50, limit_per_host=20)
            self.http_session = aiohttp.ClientSession(connector=connector)
            
            # Start monitoring tasks
            self.monitoring_tasks = [
                asyncio.create_task(self._metrics_collection_loop()),
                asyncio.create_task(self._alert_monitoring_loop()),
                asyncio.create_task(self._insight_generation_loop()),
                asyncio.create_task(self._data_retention_loop())
            ]
            
            logger.info("Advanced Monitoring System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring system: {e}")
            raise
    
    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop"""
        while True:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.config["monitoring"]["metrics_interval"])
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system performance metrics"""
        try:
            timestamp = datetime.utcnow()
            
            # System metrics
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk_usage = psutil.disk_usage('/')
            
            # Update Prometheus metrics
            self.prometheus_metrics["memory_usage"].set(memory_info.used)
            self.prometheus_metrics["cpu_usage"].set(cpu_percent)
            
            # Store metrics
            metrics = [
                MetricData("memory_usage_percent", memory_info.percent, timestamp),
                MetricData("memory_usage_bytes", memory_info.used, timestamp),
                MetricData("cpu_usage_percent", cpu_percent, timestamp),
                MetricData("disk_usage_percent", disk_usage.percent, timestamp),
                MetricData("disk_usage_bytes", disk_usage.used, timestamp),
                MetricData("active_connections", len(asyncio.all_tasks()), timestamp)
            ]
            
            # Add to buffer
            for metric in metrics:
                self.metrics_buffer.append(metric)
                self.performance_history[metric.name].append(metric)
            
            # Store in Redis
            await self._store_metrics_in_redis(metrics)
            
            # Check for alerts
            await self._check_alert_conditions(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def _store_metrics_in_redis(self, metrics: List[MetricData]) -> None:
        """Store metrics in Redis"""
        if not self.redis_client:
            return
        
        try:
            for metric in metrics:
                key = f"metrics:{metric.name}:{int(metric.timestamp.timestamp())}"
                data = {
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "labels": metric.labels,
                    "metric_type": metric.metric_type.value
                }
                await self.redis_client.setex(
                    key,
                    self.config["storage"]["metrics_ttl"],
                    json.dumps(data)
                )
        except Exception as e:
            logger.error(f"Failed to store metrics in Redis: {e}")
    
    async def _check_alert_conditions(self, metrics: List[MetricData]) -> None:
        """Check for alert conditions"""
        try:
            alert_rules = self.config["alerts"]
            
            for metric in metrics:
                if metric.name == "cpu_usage_percent" and metric.value > alert_rules["cpu_threshold"]:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        f"High CPU usage: {metric.value}%",
                        metric.name,
                        alert_rules["cpu_threshold"],
                        metric.value
                    )
                
                elif metric.name == "memory_usage_percent" and metric.value > alert_rules["memory_threshold"]:
                    await self._create_alert(
                        AlertLevel.ERROR,
                        f"High memory usage: {metric.value}%",
                        metric.name,
                        alert_rules["memory_threshold"],
                        metric.value
                    )
                
                elif metric.name == "error_rate" and metric.value > alert_rules["error_rate_threshold"]:
                    await self._create_alert(
                        AlertLevel.CRITICAL,
                        f"High error rate: {metric.value}",
                        metric.name,
                        alert_rules["error_rate_threshold"],
                        metric.value
                    )
        
        except Exception as e:
            logger.error(f"Failed to check alert conditions: {e}")
    
    async def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        metric_name: str,
        threshold: float,
        current_value: float
    ) -> None:
        """Create a new alert"""
        alert_id = f"alert_{int(time.time() * 1000)}"
        
        alert = Alert(
            id=alert_id,
            level=level,
            message=message,
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value,
            timestamp=datetime.utcnow()
        )
        
        self.alerts.append(alert)
        
        # Store in Redis
        if self.redis_client:
            try:
                key = f"alerts:{alert_id}"
                data = {
                    "id": alert.id,
                    "level": alert.level.value,
                    "message": alert.message,
                    "metric_name": alert.metric_name,
                    "threshold": alert.threshold,
                    "current_value": alert.current_value,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved
                }
                await self.redis_client.setex(
                    key,
                    self.config["storage"]["alerts_ttl"],
                    json.dumps(data)
                )
            except Exception as e:
                logger.error(f"Failed to store alert in Redis: {e}")
        
        logger.warning(f"Alert created: {message}", alert_id=alert_id, level=level.value)
    
    async def _alert_monitoring_loop(self) -> None:
        """Background alert monitoring loop"""
        while True:
            try:
                await self._process_alerts()
                await asyncio.sleep(self.config["monitoring"]["alert_check_interval"])
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _process_alerts(self) -> None:
        """Process and manage alerts"""
        try:
            # Check for resolved alerts
            for alert in self.alerts:
                if not alert.resolved:
                    # Check if alert condition is still active
                    if await self._is_alert_resolved(alert):
                        alert.resolved = True
                        alert.resolved_at = datetime.utcnow()
                        logger.info(f"Alert resolved: {alert.id}")
            
            # Clean up old alerts
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
            
        except Exception as e:
            logger.error(f"Failed to process alerts: {e}")
    
    async def _is_alert_resolved(self, alert: Alert) -> bool:
        """Check if an alert is resolved"""
        try:
            # Get recent metrics for the alert's metric
            recent_metrics = self.performance_history.get(alert.metric_name, [])
            if not recent_metrics:
                return False
            
            # Check last 5 data points
            recent_values = [m.value for m in recent_metrics[-5:]]
            if not recent_values:
                return False
            
            # Alert is resolved if all recent values are below threshold
            return all(value < alert.threshold for value in recent_values)
            
        except Exception as e:
            logger.error(f"Failed to check if alert is resolved: {e}")
            return False
    
    async def _insight_generation_loop(self) -> None:
        """Background insight generation loop"""
        while True:
            try:
                await self._generate_insights()
                await asyncio.sleep(self.config["monitoring"]["insight_generation_interval"])
            except Exception as e:
                logger.error(f"Insight generation error: {e}")
                await asyncio.sleep(30)
    
    async def _generate_insights(self) -> None:
        """Generate performance insights"""
        try:
            insights = []
            
            # Analyze performance trends
            trend_insights = await self._analyze_performance_trends()
            insights.extend(trend_insights)
            
            # Detect anomalies
            anomaly_insights = await self._detect_anomalies()
            insights.extend(anomaly_insights)
            
            # Generate optimization recommendations
            optimization_insights = await self._generate_optimization_recommendations()
            insights.extend(optimization_insights)
            
            # Store insights
            for insight in insights:
                self.insights.append(insight)
                
                # Store in Redis
                if self.redis_client:
                    try:
                        key = f"insights:{int(time.time() * 1000)}"
                        data = {
                            "insight_type": insight.insight_type,
                            "description": insight.description,
                            "impact": insight.impact,
                            "recommendation": insight.recommendation,
                            "confidence": insight.confidence,
                            "timestamp": insight.timestamp.isoformat()
                        }
                        await self.redis_client.setex(
                            key,
                            self.config["storage"]["insights_ttl"],
                            json.dumps(data)
                        )
                    except Exception as e:
                        logger.error(f"Failed to store insight in Redis: {e}")
            
            if insights:
                logger.info(f"Generated {len(insights)} insights")
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
    
    async def _analyze_performance_trends(self) -> List[PerformanceInsight]:
        """Analyze performance trends"""
        insights = []
        
        try:
            # Analyze CPU usage trend
            cpu_metrics = self.performance_history.get("cpu_usage_percent", [])
            if len(cpu_metrics) > 10:
                recent_avg = np.mean([m.value for m in cpu_metrics[-10:]])
                older_avg = np.mean([m.value for m in cpu_metrics[-20:-10]])
                
                if recent_avg > older_avg * 1.2:
                    insights.append(PerformanceInsight(
                        insight_type="trend",
                        description=f"CPU usage has increased by {((recent_avg / older_avg - 1) * 100):.1f}%",
                        impact="high",
                        recommendation="Consider scaling resources or optimizing CPU-intensive operations",
                        confidence=0.8,
                        timestamp=datetime.utcnow()
                    ))
            
            # Analyze memory usage trend
            memory_metrics = self.performance_history.get("memory_usage_percent", [])
            if len(memory_metrics) > 10:
                recent_avg = np.mean([m.value for m in memory_metrics[-10:]])
                older_avg = np.mean([m.value for m in memory_metrics[-20:-10]])
                
                if recent_avg > older_avg * 1.15:
                    insights.append(PerformanceInsight(
                        insight_type="trend",
                        description=f"Memory usage has increased by {((recent_avg / older_avg - 1) * 100):.1f}%",
                        impact="medium",
                        recommendation="Consider memory optimization or increasing available memory",
                        confidence=0.75,
                        timestamp=datetime.utcnow()
                    ))
        
        except Exception as e:
            logger.error(f"Failed to analyze performance trends: {e}")
        
        return insights
    
    async def _detect_anomalies(self) -> List[PerformanceInsight]:
        """Detect performance anomalies"""
        insights = []
        
        try:
            # Simple anomaly detection using z-score
            for metric_name, metrics in self.performance_history.items():
                if len(metrics) < 20:
                    continue
                
                values = [m.value for m in metrics[-20:]]
                mean = np.mean(values)
                std = np.std(values)
                
                if std > 0:
                    z_scores = [(v - mean) / std for v in values]
                    max_z_score = max(abs(z) for z in z_scores)
                    
                    if max_z_score > 2.5:  # Threshold for anomaly
                        insights.append(PerformanceInsight(
                            insight_type="anomaly",
                            description=f"Anomaly detected in {metric_name} (z-score: {max_z_score:.2f})",
                            impact="medium",
                            recommendation="Investigate the cause of the anomaly",
                            confidence=0.7,
                            timestamp=datetime.utcnow()
                        ))
        
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
        
        return insights
    
    async def _generate_optimization_recommendations(self) -> List[PerformanceInsight]:
        """Generate optimization recommendations"""
        insights = []
        
        try:
            # CPU optimization recommendations
            cpu_metrics = self.performance_history.get("cpu_usage_percent", [])
            if cpu_metrics and np.mean([m.value for m in cpu_metrics[-10:]]) > 70:
                insights.append(PerformanceInsight(
                    insight_type="optimization",
                    description="High CPU usage detected",
                    impact="high",
                    recommendation="Consider implementing async processing, caching, or horizontal scaling",
                    confidence=0.9,
                    timestamp=datetime.utcnow()
                ))
            
            # Memory optimization recommendations
            memory_metrics = self.performance_history.get("memory_usage_percent", [])
            if memory_metrics and np.mean([m.value for m in memory_metrics[-10:]]) > 80:
                insights.append(PerformanceInsight(
                    insight_type="optimization",
                    description="High memory usage detected",
                    impact="high",
                    recommendation="Consider memory optimization, garbage collection tuning, or increasing available memory",
                    confidence=0.85,
                    timestamp=datetime.utcnow()
                ))
        
        except Exception as e:
            logger.error(f"Failed to generate optimization recommendations: {e}")
        
        return insights
    
    async def _data_retention_loop(self) -> None:
        """Background data retention loop"""
        while True:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Data retention error: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old data based on retention policy"""
        try:
            retention_days = self.config["monitoring"]["retention_days"]
            cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
            
            # Clean up metrics buffer
            self.metrics_buffer = deque(
                [m for m in self.metrics_buffer if m.timestamp > cutoff_time],
                maxlen=self.config["monitoring"]["buffer_size"]
            )
            
            # Clean up performance history
            for metric_name in list(self.performance_history.keys()):
                self.performance_history[metric_name] = [
                    m for m in self.performance_history[metric_name]
                    if m.timestamp > cutoff_time
                ]
                if not self.performance_history[metric_name]:
                    del self.performance_history[metric_name]
            
            # Clean up insights
            self.insights = [i for i in self.insights if i.timestamp > cutoff_time]
            
            logger.info("Data cleanup completed")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data"""
        try:
            # Get recent metrics
            recent_metrics = {}
            for metric_name, metrics in self.performance_history.items():
                if metrics:
                    recent_metrics[metric_name] = {
                        "current": metrics[-1].value,
                        "average": np.mean([m.value for m in metrics[-10:]]),
                        "min": min(m.value for m in metrics[-10:]),
                        "max": max(m.value for m in metrics[-10:])
                    }
            
            # Get active alerts
            active_alerts = [alert for alert in self.alerts if not alert.resolved]
            
            # Get recent insights
            recent_insights = self.insights[-10:] if self.insights else []
            
            return {
                "metrics": recent_metrics,
                "alerts": {
                    "active": len(active_alerts),
                    "total": len(self.alerts),
                    "recent": active_alerts[-5:] if active_alerts else []
                },
                "insights": {
                    "total": len(self.insights),
                    "recent": recent_insights
                },
                "system_health": await self._get_system_health(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {}
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            # Calculate health score
            health_score = 1.0
            
            # Check CPU usage
            cpu_metrics = self.performance_history.get("cpu_usage_percent", [])
            if cpu_metrics:
                avg_cpu = np.mean([m.value for m in cpu_metrics[-5:]])
                if avg_cpu > 90:
                    health_score -= 0.3
                elif avg_cpu > 70:
                    health_score -= 0.1
            
            # Check memory usage
            memory_metrics = self.performance_history.get("memory_usage_percent", [])
            if memory_metrics:
                avg_memory = np.mean([m.value for m in memory_metrics[-5:]])
                if avg_memory > 95:
                    health_score -= 0.3
                elif avg_memory > 80:
                    health_score -= 0.1
            
            # Check active alerts
            active_alerts = len([alert for alert in self.alerts if not alert.resolved])
            if active_alerts > 5:
                health_score -= 0.2
            elif active_alerts > 0:
                health_score -= 0.1
            
            return {
                "score": max(health_score, 0.0),
                "status": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.5 else "unhealthy",
                "active_alerts": active_alerts,
                "uptime": time.time() - getattr(self, '_start_time', time.time())
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {"score": 0.0, "status": "unknown", "error": str(e)}
    
    async def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics"""
        return generate_latest()
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Close connections
            if self.redis_client:
                await self.redis_client.close()
            
            if self.http_session:
                await self.http_session.close()
            
            logger.info("Advanced Monitoring System cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Example usage
async def main():
    """Example usage of the Advanced Monitoring System"""
    monitoring = AdvancedMonitoringSystem()
    await monitoring.initialize()
    
    # Simulate some metrics
    for i in range(10):
        await monitoring._collect_system_metrics()
        await asyncio.sleep(1)
    
    # Get dashboard data
    dashboard = await monitoring.get_dashboard_data()
    print(f"Dashboard data: {json.dumps(dashboard, indent=2, default=str)}")
    
    # Get Prometheus metrics
    prometheus_metrics = await monitoring.get_prometheus_metrics()
    print(f"Prometheus metrics:\n{prometheus_metrics}")
    
    await monitoring.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

