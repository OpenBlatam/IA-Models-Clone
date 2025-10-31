from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from functools import wraps
from prometheus_client import (
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.metrics import Counter as OTelCounter, Histogram as OTelHistogram
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
import structlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from ...shared.logging import get_logger
from typing import Any, List, Dict, Optional
import logging
"""
Advanced Monitoring System
==========================

Comprehensive monitoring and observability system for LinkedIn posts
with Prometheus metrics, OpenTelemetry tracing, and advanced analytics.
"""


    Counter, Histogram, Gauge, Summary, Info,
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
)


logger = get_logger(__name__)

# Initialize OpenTelemetry
trace.set_tracer_provider(TracerProvider())
metrics.set_meter_provider(MeterProvider())

# Get tracer and meter
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


class AdvancedMonitoring:
    """
    Advanced monitoring system with comprehensive metrics and tracing.
    
    Features:
    - Prometheus metrics collection
    - OpenTelemetry distributed tracing
    - Performance monitoring
    - Error tracking
    - Business metrics
    - Custom dashboards
    - Alerting capabilities
    """
    
    def __init__(self, service_name: str = "linkedin-posts-service"):
        """Initialize the monitoring system."""
        self.service_name = service_name
        self.registry = CollectorRegistry()
        
        # Initialize Prometheus metrics
        self._initialize_prometheus_metrics()
        
        # Initialize OpenTelemetry metrics
        self._initialize_opentelemetry_metrics()
        
        # Initialize structured logging
        self._initialize_structured_logging()
        
        # Performance tracking
        self.performance_data = {}
        
        # Business metrics
        self.business_metrics = {
            "posts_generated": 0,
            "posts_optimized": 0,
            "posts_published": 0,
            "engagement_rate": 0.0,
            "average_generation_time": 0.0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0,
        }
        
        # Console for rich output
        self.console = Console()
    
    def _initialize_prometheus_metrics(self) -> Any:
        """Initialize Prometheus metrics."""
        # Request metrics
        self.request_counter = Counter(
            'linkedin_posts_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'linkedin_posts_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Business metrics
        self.posts_generated = Counter(
            'linkedin_posts_generated_total',
            'Total number of posts generated',
            ['tone', 'post_type', 'industry'],
            registry=self.registry
        )
        
        self.posts_optimized = Counter(
            'linkedin_posts_optimized_total',
            'Total number of posts optimized',
            ['optimization_type'],
            registry=self.registry
        )
        
        self.posts_published = Counter(
            'linkedin_posts_published_total',
            'Total number of posts published',
            ['platform', 'status'],
            registry=self.registry
        )
        
        # Performance metrics
        self.generation_duration = Histogram(
            'linkedin_posts_generation_duration_seconds',
            'Post generation duration in seconds',
            ['model', 'tone'],
            registry=self.registry
        )
        
        self.optimization_duration = Histogram(
            'linkedin_posts_optimization_duration_seconds',
            'Post optimization duration in seconds',
            ['optimization_type'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'linkedin_posts_cache_hits_total',
            'Total cache hits',
            ['cache_layer'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'linkedin_posts_cache_misses_total',
            'Total cache misses',
            ['cache_layer'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_counter = Counter(
            'linkedin_posts_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # System metrics
        self.active_connections = Gauge(
            'linkedin_posts_active_connections',
            'Number of active connections',
            ['connection_type'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'linkedin_posts_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )
        
        # Engagement metrics
        self.engagement_rate = Gauge(
            'linkedin_posts_engagement_rate',
            'Average engagement rate',
            ['post_type', 'industry'],
            registry=self.registry
        )
        
        # Service info
        self.service_info = Info(
            'linkedin_posts_service',
            'Service information',
            registry=self.registry
        )
        self.service_info.info({
            'name': self.service_name,
            'version': '1.0.0',
            'environment': 'production'
        })
    
    def _initialize_opentelemetry_metrics(self) -> Any:
        """Initialize OpenTelemetry metrics."""
        # OpenTelemetry counters
        self.otel_posts_generated = meter.create_counter(
            name="linkedin_posts_generated",
            description="Number of posts generated",
            unit="posts"
        )
        
        self.otel_generation_duration = meter.create_histogram(
            name="linkedin_posts_generation_duration",
            description="Post generation duration",
            unit="seconds"
        )
        
        self.otel_cache_operations = meter.create_counter(
            name="linkedin_posts_cache_operations",
            description="Cache operations",
            unit="operations"
        )
    
    def _initialize_structured_logging(self) -> Any:
        """Initialize structured logging."""
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
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations."""
        span = tracer.start_span(operation_name, attributes=attributes or {})
        start_time = time.time()
        
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            duration = time.time() - start_time
            span.set_attribute("duration", duration)
            span.end()
    
    def track_request(self, method: str, endpoint: str, status: int, duration: float):
        """Track HTTP request metrics."""
        self.request_counter.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def track_post_generation(self, tone: str, post_type: str, industry: str, duration: float):
        """Track post generation metrics."""
        self.posts_generated.labels(tone=tone, post_type=post_type, industry=industry).inc()
        self.generation_duration.labels(model="gpt-4", tone=tone).observe(duration)
        
        # Update OpenTelemetry metrics
        self.otel_posts_generated.add(1, {"tone": tone, "post_type": post_type, "industry": industry})
        self.otel_generation_duration.record(duration, {"tone": tone, "post_type": post_type})
        
        # Update business metrics
        self.business_metrics["posts_generated"] += 1
        self.business_metrics["average_generation_time"] = (
            (self.business_metrics["average_generation_time"] * (self.business_metrics["posts_generated"] - 1) + duration) /
            self.business_metrics["posts_generated"]
        )
    
    def track_post_optimization(self, optimization_type: str, duration: float):
        """Track post optimization metrics."""
        self.posts_optimized.labels(optimization_type=optimization_type).inc()
        self.optimization_duration.labels(optimization_type=optimization_type).observe(duration)
        
        self.business_metrics["posts_optimized"] += 1
    
    def track_post_publication(self, platform: str, status: str):
        """Track post publication metrics."""
        self.posts_published.labels(platform=platform, status=status).inc()
        
        if status == "success":
            self.business_metrics["posts_published"] += 1
    
    def track_cache_operation(self, operation: str, cache_layer: str, hit: bool):
        """Track cache operation metrics."""
        if hit:
            self.cache_hits.labels(cache_layer=cache_layer).inc()
        else:
            self.cache_misses.labels(cache_layer=cache_layer).inc()
        
        # Update OpenTelemetry metrics
        self.otel_cache_operations.add(1, {"operation": operation, "cache_layer": cache_layer, "hit": hit})
        
        # Update cache hit rate
        total_hits = self.cache_hits.labels(cache_layer=cache_layer)._value.get()
        total_misses = self.cache_misses.labels(cache_layer=cache_layer)._value.get()
        total_operations = total_hits + total_misses
        
        if total_operations > 0:
            self.business_metrics["cache_hit_rate"] = total_hits / total_operations
    
    def track_error(self, error_type: str, component: str, error_message: str):
        """Track error metrics."""
        self.error_counter.labels(error_type=error_type, component=component).inc()
        
        # Update error rate
        total_requests = sum(self.request_counter._metrics.values())
        total_errors = sum(self.error_counter._metrics.values())
        
        if total_requests > 0:
            self.business_metrics["error_rate"] = total_errors / total_requests
        
        # Log error with structured logging
        logger.error(
            "Error occurred",
            error_type=error_type,
            component=component,
            error_message=error_message
        )
    
    def track_engagement(self, post_type: str, industry: str, engagement_rate: float):
        """Track engagement metrics."""
        self.engagement_rate.labels(post_type=post_type, industry=industry).set(engagement_rate)
        self.business_metrics["engagement_rate"] = engagement_rate
    
    def track_system_metrics(self, memory_usage: int, active_connections: int):
        """Track system metrics."""
        self.memory_usage.labels(component="main").set(memory_usage)
        self.active_connections.labels(connection_type="database").set(active_connections)
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics."""
        return generate_latest(self.registry)
    
    def get_business_metrics(self) -> Dict[str, Any]:
        """Get business metrics summary."""
        return {
            **self.business_metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "service_name": self.service_name,
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "average_generation_time": self.business_metrics["average_generation_time"],
            "cache_hit_rate": self.business_metrics["cache_hit_rate"],
            "error_rate": self.business_metrics["error_rate"],
            "total_posts_generated": self.business_metrics["posts_generated"],
            "total_posts_optimized": self.business_metrics["posts_optimized"],
            "total_posts_published": self.business_metrics["posts_published"],
            "engagement_rate": self.business_metrics["engagement_rate"],
        }
    
    def display_metrics_dashboard(self) -> Any:
        """Display metrics dashboard using Rich."""
        # Create tables
        business_table = Table(title="Business Metrics")
        business_table.add_column("Metric", style="cyan")
        business_table.add_column("Value", style="green")
        business_table.add_column("Status", style="yellow")
        
        performance_table = Table(title="Performance Metrics")
        performance_table.add_column("Metric", style="cyan")
        performance_table.add_column("Value", style="green")
        performance_table.add_column("Trend", style="blue")
        
        # Add business metrics
        business_table.add_row(
            "Posts Generated",
            str(self.business_metrics["posts_generated"]),
            "✅" if self.business_metrics["posts_generated"] > 0 else "❌"
        )
        business_table.add_row(
            "Posts Optimized",
            str(self.business_metrics["posts_optimized"]),
            "✅" if self.business_metrics["posts_optimized"] > 0 else "❌"
        )
        business_table.add_row(
            "Posts Published",
            str(self.business_metrics["posts_published"]),
            "✅" if self.business_metrics["posts_published"] > 0 else "❌"
        )
        business_table.add_row(
            "Engagement Rate",
            f"{self.business_metrics['engagement_rate']:.2f}%",
            "✅" if self.business_metrics["engagement_rate"] > 5.0 else "⚠️"
        )
        
        # Add performance metrics
        performance_table.add_row(
            "Avg Generation Time",
            f"{self.business_metrics['average_generation_time']:.2f}s",
            "📈" if self.business_metrics["average_generation_time"] < 10 else "📉"
        )
        performance_table.add_row(
            "Cache Hit Rate",
            f"{self.business_metrics['cache_hit_rate']:.2f}%",
            "📈" if self.business_metrics["cache_hit_rate"] > 80 else "📉"
        )
        performance_table.add_row(
            "Error Rate",
            f"{self.business_metrics['error_rate']:.2f}%",
            "📉" if self.business_metrics["error_rate"] < 1 else "📈"
        )
        
        # Display dashboard
        self.console.print(Panel(f"[bold blue]{self.service_name} Dashboard[/bold blue]"))
        self.console.print(business_table)
        self.console.print(performance_table)
    
    def monitor_function(self, function_name: str, component: str = "general"):
        """Decorator for monitoring function performance."""
        def decorator(func) -> Any:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                
                with self.trace_operation(f"{component}.{function_name}") as span:
                    try:
                        result = await func(*args, **kwargs)
                        
                        # Track success
                        duration = time.time() - start_time
                        span.set_attribute("duration", duration)
                        span.set_attribute("status", "success")
                        
                        return result
                        
                    except Exception as e:
                        # Track error
                        duration = time.time() - start_time
                        span.set_attribute("duration", duration)
                        span.set_attribute("status", "error")
                        span.set_attribute("error.message", str(e))
                        
                        self.track_error(
                            error_type=type(e).__name__,
                            component=component,
                            error_message=str(e)
                        )
                        
                        raise
                
            return wrapper
        return decorator
    
    async def export_metrics(self, interval: int = 60):
        """Export metrics periodically."""
        while True:
            try:
                # Export Prometheus metrics
                metrics_data = self.get_metrics()
                
                # Export business metrics
                business_metrics = self.get_business_metrics()
                
                # Export performance summary
                performance_summary = self.get_performance_summary()
                
                # Log metrics
                logger.info(
                    "Metrics exported",
                    business_metrics=business_metrics,
                    performance_summary=performance_summary
                )
                
                # Display dashboard periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    self.display_metrics_dashboard()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error exporting metrics: {e}")
                await asyncio.sleep(interval)
    
    def create_alert_rule(self, metric_name: str, threshold: float, condition: str = "gt"):
        """Create alert rule for monitoring."""
        alert_rule = {
            "metric": metric_name,
            "threshold": threshold,
            "condition": condition,
            "service": self.service_name,
            "created_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Alert rule created: {alert_rule}")
        return alert_rule
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for triggered alerts."""
        alerts = []
        
        # Check error rate
        if self.business_metrics["error_rate"] > 0.05:  # 5%
            alerts.append({
                "type": "high_error_rate",
                "message": f"Error rate is {self.business_metrics['error_rate']:.2f}%",
                "severity": "warning",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Check cache hit rate
        if self.business_metrics["cache_hit_rate"] < 0.7:  # 70%
            alerts.append({
                "type": "low_cache_hit_rate",
                "message": f"Cache hit rate is {self.business_metrics['cache_hit_rate']:.2f}%",
                "severity": "info",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Check generation time
        if self.business_metrics["average_generation_time"] > 30:  # 30 seconds
            alerts.append({
                "type": "slow_generation",
                "message": f"Average generation time is {self.business_metrics['average_generation_time']:.2f}s",
                "severity": "warning",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return alerts


class MonitoringMiddleware:
    """FastAPI middleware for automatic monitoring."""
    
    def __init__(self, monitoring: AdvancedMonitoring):
        
    """__init__ function."""
self.monitoring = monitoring
    
    async def __call__(self, request, call_next) -> Any:
        start_time = time.time()
        
        # Track request start
        method = request.method
        endpoint = request.url.path
        
        try:
            response = await call_next(request)
            
            # Track successful request
            duration = time.time() - start_time
            self.monitoring.track_request(method, endpoint, response.status_code, duration)
            
            return response
            
        except Exception as e:
            # Track failed request
            duration = time.time() - start_time
            self.monitoring.track_request(method, endpoint, 500, duration)
            
            self.monitoring.track_error(
                error_type=type(e).__name__,
                component="api",
                error_message=str(e)
            )
            
            raise


# Global monitoring instance
monitoring = AdvancedMonitoring()


def get_monitoring() -> AdvancedMonitoring:
    """Get global monitoring instance."""
    return monitoring 