"""
Prometheus Metrics Exporter
Exports metrics for Prometheus/Grafana monitoring
"""

import time
import logging
from typing import Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest,
        CONTENT_TYPE_LATEST, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = None

logger = logging.getLogger(__name__)


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """
    Prometheus metrics middleware
    
    Exports metrics at /metrics endpoint for Prometheus scraping
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Install with: pip install prometheus-client")
            return
        
        # Define metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.request_size = Histogram(
            'http_request_size_bytes',
            'HTTP request size in bytes',
            ['method', 'endpoint']
        )
        
        self.response_size = Histogram(
            'http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'endpoint']
        )
        
        self.active_requests = Gauge(
            'http_active_requests',
            'Number of active HTTP requests',
            ['method', 'endpoint']
        )
        
        self.error_count = Counter(
            'http_errors_total',
            'Total HTTP errors',
            ['method', 'endpoint', 'error_type']
        )
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and collect metrics"""
        if not PROMETHEUS_AVAILABLE:
            return await call_next(request)
        
        # Skip metrics endpoint
        if request.url.path == "/metrics":
            return await call_next(request)
        
        method = request.method
        endpoint = request.url.path
        
        # Track active requests
        self.active_requests.labels(method=method, endpoint=endpoint).inc()
        
        start_time = time.time()
        error_type = None
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            self.request_count.labels(
                method=method,
                endpoint=endpoint,
                status=response.status_code
            ).inc()
            
            self.request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            # Estimate response size (if available)
            if hasattr(response, 'body'):
                response_size = len(response.body) if response.body else 0
                self.response_size.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(response_size)
            
            # Track errors
            if response.status_code >= 400:
                error_type = f"http_{response.status_code}"
                self.error_count.labels(
                    method=method,
                    endpoint=endpoint,
                    error_type=error_type
                ).inc()
            
            return response
            
        except Exception as e:
            error_type = type(e).__name__
            self.error_count.labels(
                method=method,
                endpoint=endpoint,
                error_type=error_type
            ).inc()
            raise
            
        finally:
            # Decrement active requests
            self.active_requests.labels(method=method, endpoint=endpoint).dec()


def get_metrics_endpoint():
    """Get /metrics endpoint handler"""
    async def metrics(request: Request) -> Response:
        """Prometheus metrics endpoint"""
        if not PROMETHEUS_AVAILABLE:
            return Response(
                content="Prometheus client not available",
                status_code=503
            )
        
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST
        )
    
    return metrics


class CustomMetrics:
    """Custom business metrics"""
    
    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Business metrics
        self.recovery_assessments = Counter(
            'recovery_assessments_total',
            'Total recovery assessments',
            ['addiction_type', 'severity']
        )
        
        self.recovery_plans_created = Counter(
            'recovery_plans_created_total',
            'Total recovery plans created',
            ['addiction_type']
        )
        
        self.milestones_achieved = Counter(
            'milestones_achieved_total',
            'Total milestones achieved',
            ['milestone_type']
        )
        
        self.relapse_risks_detected = Counter(
            'relapse_risks_detected_total',
            'Total relapse risks detected',
            ['risk_level']
        )
        
        self.active_users = Gauge(
            'active_users',
            'Number of active users',
            ['addiction_type']
        )
        
        self.recovery_progress = Histogram(
            'recovery_progress_percentage',
            'Recovery progress percentage',
            ['addiction_type', 'days_sober_range']
        )
    
    def record_assessment(self, addiction_type: str, severity: str) -> None:
        """Record recovery assessment"""
        if PROMETHEUS_AVAILABLE:
            self.recovery_assessments.labels(
                addiction_type=addiction_type,
                severity=severity
            ).inc()
    
    def record_plan_created(self, addiction_type: str) -> None:
        """Record recovery plan creation"""
        if PROMETHEUS_AVAILABLE:
            self.recovery_plans_created.labels(
                addiction_type=addiction_type
            ).inc()
    
    def record_milestone(self, milestone_type: str) -> None:
        """Record milestone achievement"""
        if PROMETHEUS_AVAILABLE:
            self.milestones_achieved.labels(
                milestone_type=milestone_type
            ).inc()
    
    def record_relapse_risk(self, risk_level: str) -> None:
        """Record relapse risk detection"""
        if PROMETHEUS_AVAILABLE:
            self.relapse_risks_detected.labels(
                risk_level=risk_level
            ).inc()
    
    def update_active_users(self, addiction_type: str, count: int) -> None:
        """Update active users count"""
        if PROMETHEUS_AVAILABLE:
            self.active_users.labels(
                addiction_type=addiction_type
            ).set(count)
    
    def record_progress(self, addiction_type: str, progress: float, days_sober: int) -> None:
        """Record recovery progress"""
        if PROMETHEUS_AVAILABLE:
            days_range = f"{days_sober // 30 * 30}-{(days_sober // 30 + 1) * 30}"
            self.recovery_progress.labels(
                addiction_type=addiction_type,
                days_sober_range=days_range
            ).observe(progress)















