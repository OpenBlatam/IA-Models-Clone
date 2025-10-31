"""
Metrics Middleware
==================

Middleware for collecting HTTP request metrics.
"""

from __future__ import annotations
import time
import logging
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..services.metrics_collector import (
    record_http_request,
    increment_counter,
    observe_histogram,
    set_gauge
)


logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting HTTP request metrics"""
    
    def __init__(
        self,
        app: ASGIApp,
        enable_detailed_metrics: bool = True,
        enable_response_size_metrics: bool = True,
        enable_duration_metrics: bool = True
    ):
        super().__init__(app)
        self.enable_detailed_metrics = enable_detailed_metrics
        self.enable_response_size_metrics = enable_response_size_metrics
        self.enable_duration_metrics = enable_duration_metrics
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics"""
        start_time = time.time()
        
        # Extract request information
        method = request.method
        url_path = request.url.path
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Record request start
        if self.enable_detailed_metrics:
            increment_counter(
                "http_requests_started_total",
                labels={
                    "method": method,
                    "endpoint": self._normalize_endpoint(url_path),
                    "client_ip": client_ip
                }
            )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate metrics
            duration = time.time() - start_time
            status_code = response.status_code
            
            # Record HTTP request metrics
            record_http_request(
                method=method,
                endpoint=self._normalize_endpoint(url_path),
                status_code=status_code,
                duration=duration
            )
            
            # Record detailed metrics
            if self.enable_detailed_metrics:
                self._record_detailed_metrics(
                    method, url_path, status_code, duration, response, client_ip, user_agent
                )
            
            # Record response size metrics
            if self.enable_response_size_metrics:
                self._record_response_size_metrics(
                    method, url_path, status_code, response
                )
            
            # Record duration metrics
            if self.enable_duration_metrics:
                self._record_duration_metrics(
                    method, url_path, status_code, duration
                )
            
            return response
        
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            status_code = 500
            
            record_http_request(
                method=method,
                endpoint=self._normalize_endpoint(url_path),
                status_code=status_code,
                duration=duration
            )
            
            # Record error details
            if self.enable_detailed_metrics:
                increment_counter(
                    "http_errors_total",
                    labels={
                        "method": method,
                        "endpoint": self._normalize_endpoint(url_path),
                        "error_type": type(e).__name__,
                        "client_ip": client_ip
                    }
                )
            
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return "unknown"
    
    def _normalize_endpoint(self, url_path: str) -> str:
        """Normalize endpoint path for metrics"""
        # Remove query parameters
        if "?" in url_path:
            url_path = url_path.split("?")[0]
        
        # Replace dynamic segments with placeholders
        import re
        
        # Replace UUIDs
        url_path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', url_path)
        
        # Replace numeric IDs
        url_path = re.sub(r'/\d+', '/{id}', url_path)
        
        # Replace other common patterns
        url_path = re.sub(r'/[a-zA-Z0-9_-]{20,}', '/{token}', url_path)
        
        return url_path
    
    def _record_detailed_metrics(
        self,
        method: str,
        url_path: str,
        status_code: int,
        duration: float,
        response: Response,
        client_ip: str,
        user_agent: str
    ):
        """Record detailed metrics"""
        labels = {
            "method": method,
            "endpoint": self._normalize_endpoint(url_path),
            "status_code": str(status_code),
            "client_ip": client_ip
        }
        
        # Record request completion
        increment_counter("http_requests_completed_total", labels=labels)
        
        # Record status code distribution
        increment_counter(
            "http_status_codes_total",
            labels={
                "status_code": str(status_code),
                "status_class": f"{status_code // 100}xx"
            }
        )
        
        # Record method distribution
        increment_counter(
            "http_methods_total",
            labels={"method": method}
        )
        
        # Record endpoint popularity
        increment_counter(
            "http_endpoints_total",
            labels={"endpoint": self._normalize_endpoint(url_path)}
        )
        
        # Record client IP distribution (anonymized)
        anonymized_ip = self._anonymize_ip(client_ip)
        increment_counter(
            "http_clients_total",
            labels={"client_ip": anonymized_ip}
        )
        
        # Record user agent distribution
        browser = self._extract_browser(user_agent)
        increment_counter(
            "http_browsers_total",
            labels={"browser": browser}
        )
    
    def _record_response_size_metrics(
        self,
        method: str,
        url_path: str,
        status_code: int,
        response: Response
    ):
        """Record response size metrics"""
        # Get response size
        response_size = 0
        if hasattr(response, 'body'):
            if isinstance(response.body, bytes):
                response_size = len(response.body)
            elif isinstance(response.body, str):
                response_size = len(response.body.encode('utf-8'))
        
        # Record response size
        observe_histogram(
            "http_response_size_bytes",
            response_size,
            labels={
                "method": method,
                "endpoint": self._normalize_endpoint(url_path),
                "status_code": str(status_code)
            }
        )
        
        # Record response size by status code
        observe_histogram(
            "http_response_size_by_status_bytes",
            response_size,
            labels={"status_code": str(status_code)}
        )
    
    def _record_duration_metrics(
        self,
        method: str,
        url_path: str,
        status_code: int,
        duration: float
    ):
        """Record duration metrics"""
        labels = {
            "method": method,
            "endpoint": self._normalize_endpoint(url_path),
            "status_code": str(status_code)
        }
        
        # Record duration by endpoint
        observe_histogram("http_request_duration_by_endpoint_seconds", duration, labels)
        
        # Record duration by method
        observe_histogram(
            "http_request_duration_by_method_seconds",
            duration,
            labels={"method": method}
        )
        
        # Record duration by status code
        observe_histogram(
            "http_request_duration_by_status_seconds",
            duration,
            labels={"status_code": str(status_code)}
        )
        
        # Record slow requests
        if duration > 1.0:  # Requests taking more than 1 second
            increment_counter(
                "http_slow_requests_total",
                labels=labels
            )
        
        # Record very slow requests
        if duration > 5.0:  # Requests taking more than 5 seconds
            increment_counter(
                "http_very_slow_requests_total",
                labels=labels
            )
    
    def _anonymize_ip(self, ip: str) -> str:
        """Anonymize IP address for privacy"""
        if ip == "unknown":
            return ip
        
        try:
            parts = ip.split(".")
            if len(parts) == 4:  # IPv4
                return f"{parts[0]}.{parts[1]}.{parts[2]}.xxx"
            else:  # IPv6 or other
                return f"{ip[:8]}...{ip[-4:]}"
        except:
            return "unknown"
    
    def _extract_browser(self, user_agent: str) -> str:
        """Extract browser name from user agent"""
        user_agent_lower = user_agent.lower()
        
        if "chrome" in user_agent_lower:
            return "chrome"
        elif "firefox" in user_agent_lower:
            return "firefox"
        elif "safari" in user_agent_lower:
            return "safari"
        elif "edge" in user_agent_lower:
            return "edge"
        elif "opera" in user_agent_lower:
            return "opera"
        elif "curl" in user_agent_lower:
            return "curl"
        elif "wget" in user_agent_lower:
            return "wget"
        elif "python" in user_agent_lower:
            return "python"
        elif "postman" in user_agent_lower:
            return "postman"
        else:
            return "other"


class CustomMetricsMiddleware:
    """Custom metrics middleware for specific use cases"""
    
    def __init__(self, app: ASGIApp):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """ASGI application callable"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        start_time = time.time()
        
        # Custom metrics collection
        self._record_custom_metrics(request, "start")
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Record response start metrics
                duration = time.time() - start_time
                self._record_custom_metrics(request, "response_start", duration)
            
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            self._record_custom_metrics(request, "error", duration, error=str(e))
            raise
    
    def _record_custom_metrics(self, request: Request, event_type: str, duration: Optional[float] = None, **kwargs):
        """Record custom metrics"""
        labels = {
            "method": request.method,
            "endpoint": self._normalize_endpoint(request.url.path),
            "event_type": event_type
        }
        
        # Add additional labels from kwargs
        labels.update(kwargs)
        
        # Record event
        increment_counter("http_custom_events_total", labels=labels)
        
        # Record duration if provided
        if duration is not None:
            observe_histogram("http_custom_duration_seconds", duration, labels=labels)
    
    def _normalize_endpoint(self, url_path: str) -> str:
        """Normalize endpoint path for metrics"""
        # Remove query parameters
        if "?" in url_path:
            url_path = url_path.split("?")[0]
        
        # Replace dynamic segments with placeholders
        import re
        url_path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', url_path)
        url_path = re.sub(r'/\d+', '/{id}', url_path)
        url_path = re.sub(r'/[a-zA-Z0-9_-]{20,}', '/{token}', url_path)
        
        return url_path


# Utility functions for metrics
def record_workflow_metrics(workflow_id: str, operation: str, duration: float, success: bool):
    """Record workflow-specific metrics"""
    from ..services.metrics_collector import record_workflow_operation
    
    record_workflow_operation(operation, duration, success)
    
    # Additional workflow-specific metrics
    increment_counter(
        "workflow_operations_by_id_total",
        labels={
            "workflow_id": workflow_id,
            "operation": operation,
            "success": str(success).lower()
        }
    )


def record_node_metrics(node_id: str, operation: str, duration: float, success: bool):
    """Record node-specific metrics"""
    increment_counter(
        "node_operations_total",
        labels={
            "node_id": node_id,
            "operation": operation,
            "success": str(success).lower()
        }
    )
    
    observe_histogram(
        "node_operation_duration_seconds",
        duration,
        labels={
            "node_id": node_id,
            "operation": operation
        }
    )


def record_ai_metrics(provider: str, model: str, duration: float, tokens_used: int, success: bool):
    """Record AI service metrics"""
    labels = {
        "provider": provider,
        "model": model,
        "success": str(success).lower()
    }
    
    increment_counter("ai_requests_total", labels=labels)
    observe_histogram("ai_request_duration_seconds", duration, labels=labels)
    observe_histogram("ai_tokens_used", tokens_used, labels=labels)
    
    if success:
        set_gauge("ai_last_successful_request", time.time(), labels=labels)
    else:
        increment_counter("ai_errors_total", labels=labels)


def record_cache_metrics(operation: str, hit: bool, duration: float, key_pattern: str):
    """Record cache metrics"""
    labels = {
        "operation": operation,
        "hit": str(hit).lower(),
        "key_pattern": key_pattern
    }
    
    increment_counter("cache_operations_total", labels=labels)
    observe_histogram("cache_operation_duration_seconds", duration, labels=labels)
    
    if hit:
        increment_counter("cache_hits_total", labels=labels)
    else:
        increment_counter("cache_misses_total", labels=labels)




