"""
Performance Monitoring and Logging for Email Sequence System

This module provides comprehensive monitoring, metrics collection,
and structured logging for the email sequence system.
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import structlog
from loguru import logger as loguru_logger

from .config import get_settings

settings = get_settings()


# Prometheus Metrics
REQUEST_COUNT = Counter(
    'email_sequence_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'email_sequence_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'email_sequence_active_connections',
    'Number of active connections'
)

EMAIL_SENT_COUNT = Counter(
    'email_sequence_emails_sent_total',
    'Total number of emails sent',
    ['sequence_id', 'status']
)

EMAIL_OPEN_COUNT = Counter(
    'email_sequence_emails_opened_total',
    'Total number of emails opened',
    ['sequence_id']
)

EMAIL_CLICK_COUNT = Counter(
    'email_sequence_emails_clicked_total',
    'Total number of emails clicked',
    ['sequence_id']
)

SEQUENCE_ACTIVE_COUNT = Gauge(
    'email_sequence_active_sequences',
    'Number of active sequences'
)

SUBSCRIBER_COUNT = Gauge(
    'email_sequence_subscribers_total',
    'Total number of subscribers',
    ['status']
)

CACHE_HIT_COUNT = Counter(
    'email_sequence_cache_hits_total',
    'Total number of cache hits',
    ['cache_type']
)

CACHE_MISS_COUNT = Counter(
    'email_sequence_cache_misses_total',
    'Total number of cache misses',
    ['cache_type']
)

DATABASE_QUERY_DURATION = Histogram(
    'email_sequence_database_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type']
)

REDIS_OPERATION_DURATION = Histogram(
    'email_sequence_redis_operation_duration_seconds',
    'Redis operation duration in seconds',
    ['operation_type']
)

ERROR_COUNT = Counter(
    'email_sequence_errors_total',
    'Total number of errors',
    ['error_type', 'component']
)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    request_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float('inf')
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_types: Dict[str, int] = field(default_factory=dict)
    endpoint_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.start_time = time.time()
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self) -> None:
        """Start performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        # Start Prometheus metrics server
        if settings.enable_metrics:
            start_http_server(settings.metrics_port)
            loguru_logger.info(f"Prometheus metrics server started on port {settings.metrics_port}")
        
        # Start background monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        loguru_logger.info("Performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        loguru_logger.info("Performance monitoring stopped")
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
    ) -> None:
        """Record request metrics"""
        # Update Prometheus metrics
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        
        # Update internal metrics
        self.metrics.request_count += 1
        self.metrics.response_times.append(duration)
        
        # Update response time statistics
        if duration > self.metrics.max_response_time:
            self.metrics.max_response_time = duration
        if duration < self.metrics.min_response_time:
            self.metrics.min_response_time = duration
        
        # Update endpoint statistics
        if endpoint not in self.metrics.endpoint_stats:
            self.metrics.endpoint_stats[endpoint] = {
                'count': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'error_count': 0
            }
        
        endpoint_stats = self.metrics.endpoint_stats[endpoint]
        endpoint_stats['count'] += 1
        endpoint_stats['total_time'] += duration
        endpoint_stats['avg_time'] = endpoint_stats['total_time'] / endpoint_stats['count']
        
        if status_code >= 400:
            endpoint_stats['error_count'] += 1
            self.metrics.error_count += 1
        
        # Update average response time
        if self.metrics.response_times:
            self.metrics.avg_response_time = sum(self.metrics.response_times) / len(self.metrics.response_times)
        
        self.metrics.last_updated = datetime.utcnow()
    
    def record_error(self, error_type: str, component: str) -> None:
        """Record error metrics"""
        ERROR_COUNT.labels(error_type=error_type, component=component).inc()
        
        if error_type not in self.metrics.error_types:
            self.metrics.error_types[error_type] = 0
        self.metrics.error_types[error_type] += 1
    
    def record_email_sent(self, sequence_id: str, status: str) -> None:
        """Record email sent metrics"""
        EMAIL_SENT_COUNT.labels(sequence_id=sequence_id, status=status).inc()
    
    def record_email_opened(self, sequence_id: str) -> None:
        """Record email opened metrics"""
        EMAIL_OPEN_COUNT.labels(sequence_id=sequence_id).inc()
    
    def record_email_clicked(self, sequence_id: str) -> None:
        """Record email clicked metrics"""
        EMAIL_CLICK_COUNT.labels(sequence_id=sequence_id).inc()
    
    def update_sequence_count(self, count: int) -> None:
        """Update active sequence count"""
        SEQUENCE_ACTIVE_COUNT.set(count)
    
    def update_subscriber_count(self, status: str, count: int) -> None:
        """Update subscriber count"""
        SUBSCRIBER_COUNT.labels(status=status).set(count)
    
    def record_cache_hit(self, cache_type: str) -> None:
        """Record cache hit"""
        CACHE_HIT_COUNT.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str) -> None:
        """Record cache miss"""
        CACHE_MISS_COUNT.labels(cache_type=cache_type).inc()
    
    def record_database_query(self, query_type: str, duration: float) -> None:
        """Record database query metrics"""
        DATABASE_QUERY_DURATION.labels(query_type=query_type).observe(duration)
    
    def record_redis_operation(self, operation_type: str, duration: float) -> None:
        """Record Redis operation metrics"""
        REDIS_OPERATION_DURATION.labels(operation_type=operation_type).observe(duration)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'uptime_human': str(timedelta(seconds=int(uptime))),
            'request_count': self.metrics.request_count,
            'error_count': self.metrics.error_count,
            'error_rate': (self.metrics.error_count / self.metrics.request_count * 100) if self.metrics.request_count > 0 else 0,
            'avg_response_time': self.metrics.avg_response_time,
            'max_response_time': self.metrics.max_response_time,
            'min_response_time': self.metrics.min_response_time if self.metrics.min_response_time != float('inf') else 0,
            'requests_per_second': self.metrics.request_count / uptime if uptime > 0 else 0,
            'error_types': self.metrics.error_types,
            'endpoint_stats': self.metrics.endpoint_stats,
            'last_updated': self.metrics.last_updated.isoformat()
        }
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                # Log performance metrics every 5 minutes
                await asyncio.sleep(300)
                
                summary = self.get_metrics_summary()
                loguru_logger.info(
                    "Performance metrics",
                    **summary
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                loguru_logger.error(f"Error in monitoring loop: {e}")


class StructuredLogger:
    """Structured logging with context"""
    
    def __init__(self):
        self.logger = structlog.get_logger()
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup structured logging"""
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
    
    def log_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_id: str,
        user_id: Optional[str] = None
    ) -> None:
        """Log request with context"""
        self.logger.info(
            "Request processed",
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            duration=duration,
            request_id=request_id,
            user_id=user_id
        )
    
    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> None:
        """Log error with context"""
        self.logger.error(
            "Error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            request_id=request_id,
            exc_info=True
        )
    
    def log_email_event(
        self,
        event_type: str,
        sequence_id: str,
        subscriber_id: str,
        email_address: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log email event"""
        self.logger.info(
            "Email event",
            event_type=event_type,
            sequence_id=sequence_id,
            subscriber_id=subscriber_id,
            email_address=email_address,
            metadata=metadata or {}
        )
    
    def log_sequence_event(
        self,
        event_type: str,
        sequence_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log sequence event"""
        self.logger.info(
            "Sequence event",
            event_type=event_type,
            sequence_id=sequence_id,
            metadata=metadata or {}
        )
    
    def log_performance(
        self,
        operation: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log performance metrics"""
        self.logger.info(
            "Performance metric",
            operation=operation,
            duration=duration,
            metadata=metadata or {}
        )


class HealthChecker:
    """Health check utilities"""
    
    def __init__(self):
        self.checks: Dict[str, callable] = {}
        self.last_checks: Dict[str, datetime] = {}
        self.check_results: Dict[str, Dict[str, Any]] = {}
    
    def register_check(self, name: str, check_func: callable) -> None:
        """Register a health check"""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                result = await check_func()
                duration = time.time() - start_time
                
                is_healthy = result.get('healthy', False)
                if not is_healthy:
                    overall_healthy = False
                
                results[name] = {
                    'healthy': is_healthy,
                    'duration': duration,
                    'details': result.get('details', {}),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                self.last_checks[name] = datetime.utcnow()
                self.check_results[name] = results[name]
                
            except Exception as e:
                overall_healthy = False
                results[name] = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                loguru_logger.error(f"Health check {name} failed: {e}")
        
        return {
            'overall_healthy': overall_healthy,
            'checks': results,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_last_check_results(self) -> Dict[str, Any]:
        """Get last health check results"""
        return self.check_results


# Global instances
performance_monitor = PerformanceMonitor()
structured_logger = StructuredLogger()
health_checker = HealthChecker()


# Context managers for monitoring
@asynccontextmanager
async def monitor_operation(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for monitoring operations"""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        structured_logger.log_performance(operation_name, duration, metadata)
        performance_monitor.record_database_query(operation_name, duration)


@asynccontextmanager
async def monitor_redis_operation(operation_type: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for monitoring Redis operations"""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        performance_monitor.record_redis_operation(operation_type, duration)


# Initialize monitoring
async def init_monitoring() -> None:
    """Initialize monitoring systems"""
    await performance_monitor.start_monitoring()
    
    # Register default health checks
    from .dependencies import check_database_health, check_redis_health, check_services_health
    
    health_checker.register_check("database", check_database_health)
    health_checker.register_check("redis", check_redis_health)
    health_checker.register_check("services", check_services_health)
    
    loguru_logger.info("Monitoring systems initialized")


async def close_monitoring() -> None:
    """Close monitoring systems"""
    await performance_monitor.stop_monitoring()
    loguru_logger.info("Monitoring systems closed")


def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary"""
    return performance_monitor.get_metrics_summary()


async def get_health_status() -> Dict[str, Any]:
    """Get health status"""
    return await health_checker.run_checks()






























