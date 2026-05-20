from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import uuid
import re
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Logging Middleware for HeyGen AI API
Comprehensive logging with structured output, performance metrics, and security monitoring.
"""


logger = structlog.get_logger()

# =============================================================================
# Logging Configuration
# =============================================================================

class LogLevel(Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class LogCategory(Enum):
    """Log categories."""
    REQUEST = "request"
    RESPONSE = "response"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"
    SYSTEM = "system"
    AUDIT = "audit"

@dataclass
class LogContext:
    """Context information for logging."""
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: datetime = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    duration_ms: Optional[float] = None
    status_code: Optional[int] = None
    
    def __post_init__(self) -> Any:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

# =============================================================================
# Request/Response Logging
# =============================================================================

class RequestResponseLogger:
    """Log request and response details."""
    
    def __init__(
        self,
        log_request_body: bool = False,
        log_response_body: bool = False,
        log_headers: bool = False,
        sensitive_headers: List[str] = None,
        sensitive_fields: List[str] = None,
        max_body_size: int = 1024 * 1024  # 1MB
    ):
        
    """__init__ function."""
self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.log_headers = log_headers
        self.sensitive_headers = sensitive_headers or [
            "authorization", "cookie", "x-api-key", "x-auth-token"
        ]
        self.sensitive_fields = sensitive_fields or [
            "password", "token", "secret", "key", "credential"
        ]
        self.max_body_size = max_body_size
    
    async def log_request(self, request: Request, context: LogContext):
        """Log request details."""
        log_data = {
            "category": LogCategory.REQUEST.value,
            "request_id": context.request_id,
            "method": request.method,
            "url": str(request.url),
            "endpoint": context.endpoint,
            "ip_address": context.ip_address,
            "user_agent": context.user_agent,
            "timestamp": context.timestamp.isoformat()
        }
        
        # Add headers if enabled
        if self.log_headers:
            headers = dict(request.headers)
            log_data["headers"] = self._sanitize_headers(headers)
        
        # Add request body if enabled
        if self.log_request_body:
            try:
                body = await self._get_request_body(request)
                if body:
                    log_data["request_body"] = self._sanitize_body(body)
                    context.request_size = len(body)
            except Exception as e:
                log_data["request_body_error"] = str(e)
        
        logger.info("Request received", **log_data)
    
    async def log_response(self, response: Response, context: LogContext):
        """Log response details."""
        log_data = {
            "category": LogCategory.RESPONSE.value,
            "request_id": context.request_id,
            "status_code": response.status_code,
            "duration_ms": context.duration_ms,
            "timestamp": context.timestamp.isoformat()
        }
        
        # Add response headers if enabled
        if self.log_headers:
            headers = dict(response.headers)
            log_data["headers"] = self._sanitize_headers(headers)
        
        # Add response body if enabled
        if self.log_response_body and hasattr(response, 'body'):
            try:
                body = response.body
                if body and len(body) <= self.max_body_size:
                    log_data["response_body"] = self._sanitize_body(body.decode())
                    context.response_size = len(body)
            except Exception as e:
                log_data["response_body_error"] = str(e)
        
        # Log based on status code
        if response.status_code >= 400:
            logger.warning("Response with error status", **log_data)
        else:
            logger.info("Response sent", **log_data)
    
    async async def _get_request_body(self, request: Request) -> Optional[str]:
        """Get request body safely."""
        try:
            body = await request.body()
            if body and len(body) <= self.max_body_size:
                return body.decode()
        except Exception:
            pass
        return None
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize sensitive headers."""
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        return sanitized
    
    def _sanitize_body(self, body: str) -> str:
        """Sanitize sensitive fields in body."""
        try:
            # Try to parse as JSON
            data = json.loads(body)
            sanitized_data = self._sanitize_json(data)
            return json.dumps(sanitized_data)
        except (json.JSONDecodeError, TypeError):
            # If not JSON, sanitize as string
            return self._sanitize_string(body)
    
    def _sanitize_json(self, data: Any) -> Any:
        """Sanitize sensitive fields in JSON data."""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(field in key.lower() for field in self.sensitive_fields):
                    sanitized[key] = "[REDACTED]"
                else:
                    sanitized[key] = self._sanitize_json(value)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize_json(item) for item in data]
        else:
            return data
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize sensitive patterns in string."""
        # Replace common sensitive patterns
        patterns = [
            (r'password["\']?\s*[:=]\s*["\']?[^"\s]+["\']?', 'password": "[REDACTED]"'),
            (r'token["\']?\s*[:=]\s*["\']?[^"\s]+["\']?', 'token": "[REDACTED]"'),
            (r'secret["\']?\s*[:=]\s*["\']?[^"\s]+["\']?', 'secret": "[REDACTED]"'),
        ]
        
        sanitized = text
        for pattern, replacement in patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized

# =============================================================================
# Performance Monitoring
# =============================================================================

class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self) -> Any:
        self.slow_request_threshold_ms = 1000  # 1 second
        self.performance_metrics: List[Dict[str, Any]] = []
        self.max_metrics = 1000
    
    async def log_performance(
        self,
        request: Request,
        response: Response,
        context: LogContext,
        additional_metrics: Dict[str, Any] = None
    ):
        """Log performance metrics."""
        metrics = {
            "category": LogCategory.PERFORMANCE.value,
            "request_id": context.request_id,
            "endpoint": context.endpoint,
            "method": request.method,
            "duration_ms": context.duration_ms,
            "status_code": response.status_code,
            "request_size": context.request_size,
            "response_size": context.response_size,
            "timestamp": context.timestamp.isoformat()
        }
        
        # Add additional metrics
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Log slow requests
        if context.duration_ms and context.duration_ms > self.slow_request_threshold_ms:
            logger.warning(
                "Slow request detected",
                **metrics,
                threshold_ms=self.slow_request_threshold_ms
            )
        
        # Store metrics
        self.performance_metrics.append(metrics)
        
        # Trim metrics if too many
        if len(self.performance_metrics) > self.max_metrics:
            self.performance_metrics = self.performance_metrics[-self.max_metrics:]
        
        # Log performance
        logger.info("Performance metrics", **metrics)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_metrics:
            return {"error": "No performance metrics available"}
        
        recent_metrics = [
            m for m in self.performance_metrics
            if m["timestamp"] > (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        ]
        
        if not recent_metrics:
            return {"error": "No recent performance metrics"}
        
        durations = [m["duration_ms"] for m in recent_metrics if m["duration_ms"]]
        
        return {
            "total_requests": len(recent_metrics),
            "average_duration_ms": sum(durations) / len(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "slow_requests": len([m for m in recent_metrics if m.get("duration_ms", 0) > self.slow_request_threshold_ms]),
            "error_requests": len([m for m in recent_metrics if m.get("status_code", 200) >= 400])
        }

# =============================================================================
# Security Monitoring
# =============================================================================

class SecurityMonitor:
    """Monitor and log security events."""
    
    def __init__(self) -> Any:
        self.suspicious_patterns = [
            r"<script[^>]*>",  # XSS attempts
            r"javascript:",     # JavaScript injection
            r"union\s+select",  # SQL injection
            r"drop\s+table",    # SQL injection
            r"exec\s*\(",       # Command injection
            r"system\s*\(",     # Command injection
            r"eval\s*\(",       # Code injection
            r"\.\./",           # Path traversal
            r"\.\.\\",          # Path traversal
        ]
        self.rate_limit_patterns = {}
        self.max_requests_per_minute = 100
    
    async def check_security(
        self,
        request: Request,
        context: LogContext
    ) -> Dict[str, Any]:
        """Check for security issues."""
        security_issues = []
        
        # Check for suspicious patterns in URL
        url = str(request.url)
        for pattern in self.suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                security_issues.append({
                    "type": "suspicious_pattern",
                    "pattern": pattern,
                    "location": "url",
                    "value": url
                })
        
        # Check for suspicious patterns in headers
        for header_name, header_value in request.headers.items():
            for pattern in self.suspicious_patterns:
                if re.search(pattern, header_value, re.IGNORECASE):
                    security_issues.append({
                        "type": "suspicious_pattern",
                        "pattern": pattern,
                        "location": f"header:{header_name}",
                        "value": header_value
                    })
        
        # Check rate limiting
        rate_limit_issue = await self._check_rate_limit(context.ip_address)
        if rate_limit_issue:
            security_issues.append(rate_limit_issue)
        
        # Log security issues
        if security_issues:
            logger.warning(
                "Security issues detected",
                category=LogCategory.SECURITY.value,
                request_id=context.request_id,
                ip_address=context.ip_address,
                issues=security_issues
            )
        
        return {
            "issues_found": len(security_issues),
            "issues": security_issues
        }
    
    async def _check_rate_limit(self, ip_address: str) -> Optional[Dict[str, Any]]:
        """Check rate limiting for IP address."""
        current_time = datetime.now(timezone.utc)
        minute_key = current_time.strftime("%Y-%m-%d %H:%M")
        key = f"{ip_address}:{minute_key}"
        
        if key not in self.rate_limit_patterns:
            self.rate_limit_patterns[key] = {
                "count": 0,
                "first_request": current_time
            }
        
        self.rate_limit_patterns[key]["count"] += 1
        
        # Clean old entries
        await self._clean_rate_limit_patterns()
        
        # Check if limit exceeded
        if self.rate_limit_patterns[key]["count"] > self.max_requests_per_minute:
            return {
                "type": "rate_limit_exceeded",
                "ip_address": ip_address,
                "count": self.rate_limit_patterns[key]["count"],
                "limit": self.max_requests_per_minute
            }
        
        return None
    
    async def _clean_rate_limit_patterns(self) -> Any:
        """Clean old rate limit patterns."""
        current_time = datetime.now(timezone.utc)
        keys_to_remove = []
        
        for key, data in self.rate_limit_patterns.items():
            if (current_time - data["first_request"]) > timedelta(minutes=5):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.rate_limit_patterns[key]

# =============================================================================
# Business Logic Logging
# =============================================================================

class BusinessLogger:
    """Log business logic events."""
    
    def __init__(self) -> Any:
        self.business_events: List[Dict[str, Any]] = []
        self.max_events = 1000
    
    async def log_business_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        context: LogContext,
        user_id: Optional[str] = None
    ):
        """Log business logic event."""
        event = {
            "category": LogCategory.BUSINESS.value,
            "event_type": event_type,
            "event_data": event_data,
            "request_id": context.request_id,
            "user_id": user_id or context.user_id,
            "timestamp": context.timestamp.isoformat(),
            "endpoint": context.endpoint,
            "method": context.method
        }
        
        # Store event
        self.business_events.append(event)
        
        # Trim events if too many
        if len(self.business_events) > self.max_events:
            self.business_events = self.business_events[-self.max_events:]
        
        # Log event
        logger.info("Business event", **event)
    
    async def log_video_creation(
        self,
        video_id: str,
        template_id: str,
        script_length: int,
        context: LogContext,
        user_id: Optional[str] = None
    ):
        """Log video creation event."""
        await self.log_business_event(
            "video_created",
            {
                "video_id": video_id,
                "template_id": template_id,
                "script_length": script_length,
                "status": "created"
            },
            context,
            user_id
        )
    
    async def log_user_action(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        context: LogContext,
        user_id: Optional[str] = None
    ):
        """Log user action event."""
        await self.log_business_event(
            "user_action",
            {
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id
            },
            context,
            user_id
        )

# =============================================================================
# Audit Logging
# =============================================================================

class AuditLogger:
    """Log audit events for compliance and security."""
    
    def __init__(self) -> Any:
        self.audit_events: List[Dict[str, Any]] = []
        self.max_audit_events = 1000
    
    async def log_audit_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        context: LogContext,
        user_id: Optional[str] = None,
        sensitive: bool = False
    ):
        """Log audit event."""
        event = {
            "category": LogCategory.AUDIT.value,
            "event_type": event_type,
            "event_data": event_data,
            "request_id": context.request_id,
            "user_id": user_id or context.user_id,
            "ip_address": context.ip_address,
            "user_agent": context.user_agent,
            "timestamp": context.timestamp.isoformat(),
            "sensitive": sensitive
        }
        
        # Store audit event
        self.audit_events.append(event)
        
        # Trim events if too many
        if len(self.audit_events) > self.max_audit_events:
            self.audit_events = self.audit_events[-self.max_audit_events:]
        
        # Log audit event
        logger.info("Audit event", **event)
    
    async def log_authentication(
        self,
        success: bool,
        user_id: Optional[str] = None,
        context: LogContext = None
    ):
        """Log authentication event."""
        if context:
            await self.log_audit_event(
                "authentication",
                {
                    "success": success,
                    "user_id": user_id
                },
                context,
                user_id,
                sensitive=True
            )
    
    async def log_authorization(
        self,
        resource: str,
        action: str,
        granted: bool,
        context: LogContext,
        user_id: Optional[str] = None
    ):
        """Log authorization event."""
        await self.log_audit_event(
            "authorization",
            {
                "resource": resource,
                "action": action,
                "granted": granted
            },
            context,
            user_id
        )

# =============================================================================
# Logging Middleware
# =============================================================================

class LoggingMiddleware(BaseHTTPMiddleware):
    """Comprehensive logging middleware."""
    
    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        log_headers: bool = False,
        enable_performance_monitoring: bool = True,
        enable_security_monitoring: bool = True,
        enable_business_logging: bool = True,
        enable_audit_logging: bool = True
    ):
        
    """__init__ function."""
super().__init__(app)
        
        self.request_response_logger = RequestResponseLogger(
            log_request_body=log_request_body,
            log_response_body=log_response_body,
            log_headers=log_headers
        )
        
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        self.security_monitor = SecurityMonitor() if enable_security_monitoring else None
        self.business_logger = BusinessLogger() if enable_business_logging else None
        self.audit_logger = AuditLogger() if enable_audit_logging else None
    
    async def dispatch(self, request: Request, call_next):
        """Process request with comprehensive logging."""
        start_time = time.time()
        
        # Create log context
        context = LogContext(
            request_id=self._get_request_id(request),
            endpoint=str(request.url.path),
            method=request.method,
            ip_address=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add context to request state
        request.state.log_context = context
        
        try:
            # Log request
            await self.request_response_logger.log_request(request, context)
            
            # Check security
            if self.security_monitor:
                security_result = await self.security_monitor.check_security(request, context)
                if security_result["issues_found"] > 0:
                    # Continue processing but log security issues
                    pass
            
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            context.duration_ms = (time.time() - start_time) * 1000
            context.status_code = response.status_code
            
            # Log response
            await self.request_response_logger.log_response(response, context)
            
            # Log performance metrics
            if self.performance_monitor:
                await self.performance_monitor.log_performance(request, response, context)
            
            return response
            
        except Exception as e:
            # Calculate duration for error case
            context.duration_ms = (time.time() - start_time) * 1000
            context.status_code = 500
            
            # Log error
            logger.error(
                "Request processing error",
                category=LogCategory.SYSTEM.value,
                request_id=context.request_id,
                error=str(e),
                duration_ms=context.duration_ms
            )
            
            # Re-raise exception
            raise
    
    async def _get_request_id(self, request: Request) -> str:
        """Get or generate request ID."""
        return (
            request.headers.get("X-Request-ID") or
            str(uuid.uuid4())
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"

# =============================================================================
# Logging Utilities
# =============================================================================

class LoggingUtilities:
    """Utility functions for logging."""
    
    @staticmethod
    def setup_structured_logging(
        log_level: str = "INFO",
        log_format: str = "json",
        include_timestamp: bool = True
    ):
        """Setup structured logging configuration."""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
        ]
        
        if include_timestamp:
            processors.append(structlog.processors.TimeStamper(fmt="iso"))
        
        processors.extend([
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ])
        
        if log_format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    @staticmethod
    def create_correlation_id() -> str:
        """Create a correlation ID for request tracking."""
        return str(uuid.uuid4())
    
    @staticmethod
    def mask_sensitive_data(data: str, patterns: List[str] = None) -> str:
        """Mask sensitive data in strings."""
        if patterns is None:
            patterns = [
                r'password["\']?\s*[:=]\s*["\']?[^"\s]+["\']?',
                r'token["\']?\s*[:=]\s*["\']?[^"\s]+["\']?',
                r'secret["\']?\s*[:=]\s*["\']?[^"\s]+["\']?',
            ]
        
        masked = data
        for pattern in patterns:
            masked = re.sub(pattern, '[REDACTED]', masked, flags=re.IGNORECASE)
        
        return masked

# =============================================================================
# Usage Examples
# =============================================================================

def create_logging_middleware(
    log_request_body: bool = False,
    log_response_body: bool = False,
    log_headers: bool = False,
    enable_performance_monitoring: bool = True,
    enable_security_monitoring: bool = True,
    enable_business_logging: bool = True,
    enable_audit_logging: bool = True
) -> LoggingMiddleware:
    """Create logging middleware with configuration."""
    return LoggingMiddleware(
        app=None,  # Will be set by FastAPI
        log_request_body=log_request_body,
        log_response_body=log_response_body,
        log_headers=log_headers,
        enable_performance_monitoring=enable_performance_monitoring,
        enable_security_monitoring=enable_security_monitoring,
        enable_business_logging=enable_business_logging,
        enable_audit_logging=enable_audit_logging
    )

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "LogLevel",
    "LogCategory",
    "LogContext",
    "RequestResponseLogger",
    "PerformanceMonitor",
    "SecurityMonitor",
    "BusinessLogger",
    "AuditLogger",
    "LoggingMiddleware",
    "LoggingUtilities",
    "create_logging_middleware",
] 