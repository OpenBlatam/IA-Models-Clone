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
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import hashlib
from ..exceptions.http_exceptions import (
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Error Monitoring Middleware for HeyGen AI API
Real-time error tracking, alerting, metrics collection, and error analysis.
"""


    BaseHTTPException, ErrorCategory, ErrorSeverity
)

logger = structlog.get_logger()

# =============================================================================
# Error Monitoring Types
# =============================================================================

class ErrorType(Enum):
    """Error types for monitoring."""
    HTTP_EXCEPTION = "http_exception"
    UNEXPECTED_EXCEPTION = "unexpected_exception"
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    DATABASE_ERROR = "database_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    CONFIGURATION_ERROR = "configuration_error"
    NETWORK_ERROR = "network_error"
    CACHE_ERROR = "cache_error"
    UNKNOWN_ERROR = "unknown_error"

class AlertLevel(Enum):
    """Alert levels for error monitoring."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ErrorEvent:
    """Error event data structure."""
    id: str
    timestamp: datetime
    error_type: ErrorType
    error_message: str
    exception_type: str
    request_id: str
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    alert_level: AlertLevel = AlertLevel.ERROR
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

# =============================================================================
# Error Metrics Collector
# =============================================================================

class ErrorMetricsCollector:
    """Collect and track error metrics."""
    
    def __init__(self, retention_hours: int = 24):
        
    """__init__ function."""
self.retention_hours = retention_hours
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_timeline: deque = deque(maxlen=10000)
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.user_error_counts: Dict[str, int] = defaultdict(int)
        self.endpoint_error_counts: Dict[str, int] = defaultdict(int)
        self.severity_counts: Dict[str, int] = defaultdict(int)
        
    def record_error(self, error_event: ErrorEvent):
        """Record an error event."""
        # Increment error counts
        self.error_counts[error_event.error_type.value] += 1
        self.severity_counts[error_event.severity.value] += 1
        
        if error_event.user_id:
            self.user_error_counts[error_event.user_id] += 1
        
        if error_event.endpoint:
            self.endpoint_error_counts[error_event.endpoint] += 1
        
        # Record error pattern
        pattern_key = f"{error_event.error_type.value}:{error_event.exception_type}"
        self.error_patterns[pattern_key] += 1
        
        # Add to timeline
        self.error_timeline.append(error_event)
        
        # Clean old data
        self._clean_old_data()
    
    def _clean_old_data(self) -> Any:
        """Clean old error data."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        
        # Clean timeline
        while self.error_timeline and self.error_timeline[0].timestamp < cutoff_time:
            old_event = self.error_timeline.popleft()
            
            # Decrement counts
            self.error_counts[old_event.error_type.value] = max(0, self.error_counts[old_event.error_type.value] - 1)
            self.severity_counts[old_event.severity.value] = max(0, self.severity_counts[old_event.severity.value] - 1)
            
            if old_event.user_id:
                self.user_error_counts[old_event.user_id] = max(0, self.user_error_counts[old_event.user_id] - 1)
            
            if old_event.endpoint:
                self.endpoint_error_counts[old_event.endpoint] = max(0, self.endpoint_error_counts[old_event.endpoint] - 1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current error metrics."""
        recent_errors = [
            event for event in self.error_timeline
            if event.timestamp > datetime.now(timezone.utc) - timedelta(hours=1)
        ]
        
        return {
            "total_errors": len(self.error_timeline),
            "recent_errors": len(recent_errors),
            "error_counts": dict(self.error_counts),
            "severity_counts": dict(self.severity_counts),
            "error_patterns": dict(self.error_patterns),
            "user_error_counts": dict(self.user_error_counts),
            "endpoint_error_counts": dict(self.endpoint_error_counts),
            "error_rate_per_minute": len(recent_errors) / 60 if recent_errors else 0
        }
    
    def get_error_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get error trends over time."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_errors = [
            event for event in self.error_timeline
            if event.timestamp > cutoff_time
        ]
        
        # Group by hour
        hourly_counts = defaultdict(int)
        for error in recent_errors:
            hour_key = error.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_counts[hour_key] += 1
        
        return {
            "hourly_counts": dict(hourly_counts),
            "total_errors": len(recent_errors),
            "average_errors_per_hour": len(recent_errors) / hours if hours > 0 else 0
        }

# =============================================================================
# Error Alerting System
# =============================================================================

class ErrorAlertingSystem:
    """Alert system for error monitoring."""
    
    def __init__(self) -> Any:
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_channels: Dict[str, Callable] = {}
        self.max_alerts = 1000
        
    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[ErrorEvent], bool],
        alert_level: AlertLevel,
        message_template: str,
        cooldown_minutes: int = 5
    ):
        """Add an alert rule."""
        self.alert_rules.append({
            "name": name,
            "condition": condition,
            "alert_level": alert_level,
            "message_template": message_template,
            "cooldown_minutes": cooldown_minutes,
            "last_triggered": None
        })
    
    def add_alert_channel(self, name: str, channel_func: Callable[[Dict[str, Any]], None]):
        """Add an alert channel."""
        self.alert_channels[name] = channel_func
    
    def check_alerts(self, error_event: ErrorEvent):
        """Check and trigger alerts for an error event."""
        for rule in self.alert_rules:
            # Check cooldown
            if rule["last_triggered"]:
                cooldown_time = rule["last_triggered"] + timedelta(minutes=rule["cooldown_minutes"])
                if datetime.now(timezone.utc) < cooldown_time:
                    continue
            
            # Check condition
            if rule["condition"](error_event):
                alert = self._create_alert(rule, error_event)
                self._trigger_alert(alert)
                rule["last_triggered"] = datetime.now(timezone.utc)
    
    def _create_alert(self, rule: Dict[str, Any], error_event: ErrorEvent) -> Dict[str, Any]:
        """Create an alert from a rule and error event."""
        alert = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc),
            "rule_name": rule["name"],
            "alert_level": rule["alert_level"].value,
            "message": rule["message_template"f"]",
            "error_event": error_event.to_dict()
        }
        
        self.alert_history.append(alert)
        
        # Trim alert history
        if len(self.alert_history) > self.max_alerts:
            self.alert_history = self.alert_history[-self.max_alerts:]
        
        return alert
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger alerts through all channels."""
        for channel_name, channel_func in self.alert_channels.items():
            try:
                channel_func(alert)
            except Exception as e:
                logger.error(f"Failed to send alert through channel {channel_name}", error=str(e))
    
    def setup_default_alerts(self) -> Any:
        """Setup default alert rules."""
        # High error rate alert
        self.add_alert_rule(
            name="high_error_rate",
            condition=lambda event: True,  # Will be checked by metrics
            alert_level=AlertLevel.WARNING,
            message_template="High error rate detected: {error_type} - {error_message}",
            cooldown_minutes=10
        )
        
        # Critical errors alert
        self.add_alert_rule(
            name="critical_errors",
            condition=lambda event: event.severity == ErrorSeverity.CRITICAL,
            alert_level=AlertLevel.CRITICAL,
            message_template="Critical error: {error_type} at {endpoint}",
            cooldown_minutes=1
        )
        
        # Database errors alert
        self.add_alert_rule(
            name="database_errors",
            condition=lambda event: event.error_type == ErrorType.DATABASE_ERROR,
            alert_level=AlertLevel.ERROR,
            message_template="Database error: {error_message}",
            cooldown_minutes=5
        )
        
        # External service errors alert
        self.add_alert_rule(
            name="external_service_errors",
            condition=lambda event: event.error_type == ErrorType.EXTERNAL_SERVICE_ERROR,
            alert_level=AlertLevel.ERROR,
            message_template="External service error: {error_message}",
            cooldown_minutes=5
        )
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert["timestamp"] > cutoff_time
        ]

# =============================================================================
# Error Analysis Engine
# =============================================================================

class ErrorAnalysisEngine:
    """Analyze errors for patterns and insights."""
    
    def __init__(self) -> Any:
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        self.error_correlations: Dict[str, List[str]] = defaultdict(list)
        self.root_cause_analysis: Dict[str, str] = {}
    
    def analyze_error(self, error_event: ErrorEvent):
        """Analyze an error event for patterns."""
        # Extract error signature
        signature = self._extract_error_signature(error_event)
        
        if signature not in self.error_patterns:
            self.error_patterns[signature] = {
                "count": 0,
                "first_seen": error_event.timestamp,
                "last_seen": error_event.timestamp,
                "users_affected": set(),
                "endpoints_affected": set(),
                "severity_distribution": defaultdict(int),
                "common_context": defaultdict(int)
            }
        
        pattern = self.error_patterns[signature]
        pattern["count"] += 1
        pattern["last_seen"] = error_event.timestamp
        
        if error_event.user_id:
            pattern["users_affected"].add(error_event.user_id)
        
        if error_event.endpoint:
            pattern["endpoints_affected"].add(error_event.endpoint)
        
        pattern["severity_distribution"][error_event.severity.value] += 1
        
        # Analyze context
        if error_event.context:
            for key, value in error_event.context.items():
                pattern["common_context"][f"{key}:{value}"] += 1
    
    def _extract_error_signature(self, error_event: ErrorEvent) -> str:
        """Extract a unique signature for the error."""
        signature_parts = [
            error_event.error_type.value,
            error_event.exception_type,
            error_event.error_message[:100]  # First 100 chars
        ]
        return hashlib.md5("|".join(signature_parts).encode()).hexdigest()
    
    def get_error_insights(self) -> Dict[str, Any]:
        """Get insights from error analysis."""
        insights = {
            "total_patterns": len(self.error_patterns),
            "most_common_errors": [],
            "users_most_affected": [],
            "endpoints_most_affected": [],
            "error_trends": {},
            "recommendations": []
        }
        
        # Most common errors
        sorted_patterns = sorted(
            self.error_patterns.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        insights["most_common_errors"] = [
            {
                "signature": signature,
                "count": pattern["count"],
                "first_seen": pattern["first_seen"].isoformat(),
                "last_seen": pattern["last_seen"].isoformat(),
                "users_affected": len(pattern["users_affected"]),
                "endpoints_affected": len(pattern["endpoints_affected"])
            }
            for signature, pattern in sorted_patterns[:10]
        ]
        
        # Generate recommendations
        insights["recommendations"] = self._generate_recommendations()
        
        return insights
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on error analysis."""
        recommendations = []
        
        for signature, pattern in self.error_patterns.items():
            if pattern["count"] > 10:
                recommendations.append(
                    f"High frequency error pattern detected: {pattern['count']} occurrences. "
                    f"Consider implementing retry logic or circuit breaker."
                )
            
            if len(pattern["users_affected"]) > 5:
                recommendations.append(
                    f"Error affecting multiple users: {len(pattern['users_affected'])} users. "
                    f"Check for systemic issues."
                )
            
            if pattern["severity_distribution"].get("critical", 0) > 0:
                recommendations.append(
                    f"Critical errors detected in pattern. "
                    f"Immediate attention required."
                )
        
        return recommendations[:5]  # Top 5 recommendations

# =============================================================================
# Error Recovery Strategies
# =============================================================================

class ErrorRecoveryStrategies:
    """Error recovery strategies and fallbacks."""
    
    def __init__(self) -> Any:
        self.recovery_strategies: Dict[ErrorType, Callable] = {
            ErrorType.DATABASE_ERROR: self._database_recovery,
            ErrorType.CACHE_ERROR: self._cache_recovery,
            ErrorType.EXTERNAL_SERVICE_ERROR: self._external_service_recovery,
            ErrorType.TIMEOUT_ERROR: self._timeout_recovery,
            ErrorType.NETWORK_ERROR: self._network_recovery
        }
    
    async def attempt_recovery(self, error_event: ErrorEvent) -> Optional[Dict[str, Any]]:
        """Attempt to recover from an error."""
        recovery_strategy = self.recovery_strategies.get(error_event.error_type)
        
        if recovery_strategy:
            try:
                return await recovery_strategy(error_event)
            except Exception as e:
                logger.error(
                    "Recovery attempt failed",
                    error_event_id=error_event.id,
                    recovery_error=str(e)
                )
        
        return None
    
    async def _database_recovery(self, error_event: ErrorEvent) -> Dict[str, Any]:
        """Database recovery strategy."""
        # Try to reconnect to database
        # Return cached data if available
        return {
            "strategy": "database_recovery",
            "action": "reconnect_and_retry",
            "fallback": "use_cached_data"
        }
    
    async def _cache_recovery(self, error_event: ErrorEvent) -> Dict[str, Any]:
        """Cache recovery strategy."""
        # Try alternative cache
        # Fall back to database
        return {
            "strategy": "cache_recovery",
            "action": "use_alternative_cache",
            "fallback": "use_database"
        }
    
    async def _external_service_recovery(self, error_event: ErrorEvent) -> Dict[str, Any]:
        """External service recovery strategy."""
        # Try alternative service
        # Use cached data
        return {
            "strategy": "external_service_recovery",
            "action": "use_alternative_service",
            "fallback": "use_cached_data"
        }
    
    async def _timeout_recovery(self, error_event: ErrorEvent) -> Dict[str, Any]:
        """Timeout recovery strategy."""
        # Increase timeout
        # Use cached results
        return {
            "strategy": "timeout_recovery",
            "action": "increase_timeout",
            "fallback": "use_cached_results"
        }
    
    async def _network_recovery(self, error_event: ErrorEvent) -> Dict[str, Any]:
        """Network recovery strategy."""
        # Retry with exponential backoff
        # Try alternative endpoints
        return {
            "strategy": "network_recovery",
            "action": "retry_with_backoff",
            "fallback": "use_alternative_endpoint"
        }

# =============================================================================
# Error Monitoring Middleware
# =============================================================================

class ErrorMonitoringMiddleware(BaseHTTPMiddleware):
    """Comprehensive error monitoring middleware."""
    
    def __init__(
        self,
        app: ASGIApp,
        enable_metrics: bool = True,
        enable_alerting: bool = True,
        enable_analysis: bool = True,
        enable_recovery: bool = True,
        retention_hours: int = 24
    ):
        
    """__init__ function."""
super().__init__(app)
        
        self.enable_metrics = enable_metrics
        self.enable_alerting = enable_alerting
        self.enable_analysis = enable_analysis
        self.enable_recovery = enable_recovery
        
        # Initialize components
        self.metrics_collector = ErrorMetricsCollector(retention_hours) if enable_metrics else None
        self.alerting_system = ErrorAlertingSystem() if enable_alerting else None
        self.analysis_engine = ErrorAnalysisEngine() if enable_analysis else None
        self.recovery_strategies = ErrorRecoveryStrategies() if enable_recovery else None
        
        # Setup default alerts
        if self.alerting_system:
            self.alerting_system.setup_default_alerts()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with error monitoring."""
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Monitor for error status codes
            if response.status_code >= 400:
                await self._handle_error_response(request, response, start_time)
            
            return response
            
        except BaseHTTPException as http_exc:
            # Handle known HTTP exceptions
            await self._handle_http_exception(http_exc, request, start_time)
            raise
            
        except Exception as exc:
            # Handle unexpected exceptions
            await self._handle_unexpected_exception(exc, request, start_time)
            raise
    
    async def _handle_error_response(
        self,
        request: Request,
        response: Response,
        start_time: float
    ):
        """Handle error responses."""
        error_event = self._create_error_event(
            error_type=ErrorType.HTTP_EXCEPTION,
            exception=Exception(f"HTTP {response.status_code}"),
            request=request,
            start_time=start_time,
            status_code=response.status_code
        )
        
        await self._process_error_event(error_event)
    
    async def _handle_http_exception(
        self,
        exception: BaseHTTPException,
        request: Request,
        start_time: float
    ):
        """Handle HTTP exceptions."""
        error_event = self._create_error_event(
            error_type=ErrorType.HTTP_EXCEPTION,
            exception=exception,
            request=request,
            start_time=start_time,
            status_code=exception.status_code
        )
        
        await self._process_error_event(error_event)
    
    async def _handle_unexpected_exception(
        self,
        exception: Exception,
        request: Request,
        start_time: float
    ):
        """Handle unexpected exceptions."""
        error_type = self._classify_exception(exception)
        
        error_event = self._create_error_event(
            error_type=error_type,
            exception=exception,
            request=request,
            start_time=start_time
        )
        
        await self._process_error_event(error_event)
    
    def _create_error_event(
        self,
        error_type: ErrorType,
        exception: Exception,
        request: Request,
        start_time: float,
        status_code: Optional[int] = None
    ) -> ErrorEvent:
        """Create an error event."""
        return ErrorEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            error_type=error_type,
            error_message=str(exception),
            exception_type=type(exception).__name__,
            request_id=getattr(request.state, 'request_id', 'unknown'),
            user_id=getattr(request.state, 'user_id', None),
            endpoint=str(request.url.path),
            method=request.method,
            status_code=status_code,
            ip_address=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            stack_trace=traceback.format_exc(),
            context=self._extract_context(request),
            severity=self._determine_severity(error_type, exception),
            alert_level=self._determine_alert_level(error_type, exception)
        )
    
    def _classify_exception(self, exception: Exception) -> ErrorType:
        """Classify exception type."""
        exception_type = type(exception).__name__.lower()
        exception_message = str(exception).lower()
        
        if "database" in exception_type or "sql" in exception_type:
            return ErrorType.DATABASE_ERROR
        elif "timeout" in exception_type or "timeout" in exception_message:
            return ErrorType.TIMEOUT_ERROR
        elif "memory" in exception_type:
            return ErrorType.MEMORY_ERROR
        elif "config" in exception_type:
            return ErrorType.CONFIGURATION_ERROR
        elif "network" in exception_type or "connection" in exception_message:
            return ErrorType.NETWORK_ERROR
        elif "cache" in exception_type:
            return ErrorType.CACHE_ERROR
        elif "api" in exception_message or "service" in exception_message:
            return ErrorType.EXTERNAL_SERVICE_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def _determine_severity(self, error_type: ErrorType, exception: Exception) -> ErrorSeverity:
        """Determine error severity."""
        if error_type in [ErrorType.MEMORY_ERROR, ErrorType.DATABASE_ERROR]:
            return ErrorSeverity.CRITICAL
        elif error_type in [ErrorType.EXTERNAL_SERVICE_ERROR, ErrorType.NETWORK_ERROR]:
            return ErrorSeverity.HIGH
        elif error_type in [ErrorType.TIMEOUT_ERROR, ErrorType.CONFIGURATION_ERROR]:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _determine_alert_level(self, error_type: ErrorType, exception: Exception) -> AlertLevel:
        """Determine alert level."""
        if error_type in [ErrorType.MEMORY_ERROR, ErrorType.DATABASE_ERROR]:
            return AlertLevel.CRITICAL
        elif error_type in [ErrorType.EXTERNAL_SERVICE_ERROR, ErrorType.NETWORK_ERROR]:
            return AlertLevel.ERROR
        elif error_type in [ErrorType.TIMEOUT_ERROR, ErrorType.CONFIGURATION_ERROR]:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO
    
    def _extract_context(self, request: Request) -> Dict[str, Any]:
        """Extract context from request."""
        return {
            "url": str(request.url),
            "headers": dict(request.headers),
            "query_params": dict(request.query_params),
            "path_params": dict(request.path_params)
        }
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    async def _process_error_event(self, error_event: ErrorEvent):
        """Process an error event through all monitoring systems."""
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_error(error_event)
        
        # Check alerts
        if self.alerting_system:
            self.alerting_system.check_alerts(error_event)
        
        # Analyze error
        if self.analysis_engine:
            self.analysis_engine.analyze_error(error_event)
        
        # Attempt recovery
        if self.recovery_strategies:
            recovery_result = await self.recovery_strategies.attempt_recovery(error_event)
            if recovery_result:
                logger.info(
                    "Error recovery attempted",
                    error_event_id=error_event.id,
                    recovery_result=recovery_result
                )
        
        # Log error event
        logger.error(
            "Error event recorded",
            error_event_id=error_event.id,
            error_type=error_event.error_type.value,
            error_message=error_event.error_message,
            severity=error_event.severity.value,
            alert_level=error_event.alert_level.value
        )

# =============================================================================
# Monitoring Endpoints
# =============================================================================

class ErrorMonitoringEndpoints:
    """Endpoints for error monitoring and management."""
    
    def __init__(self, middleware: ErrorMonitoringMiddleware):
        
    """__init__ function."""
self.middleware = middleware
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get error metrics."""
        if self.middleware.metrics_collector:
            return self.middleware.metrics_collector.get_metrics()
        return {"error": "Metrics collection not enabled"}
    
    def get_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get error trends."""
        if self.middleware.metrics_collector:
            return self.middleware.metrics_collector.get_error_trends(hours)
        return {"error": "Metrics collection not enabled"}
    
    def get_insights(self) -> Dict[str, Any]:
        """Get error insights."""
        if self.middleware.analysis_engine:
            return self.middleware.analysis_engine.get_error_insights()
        return {"error": "Analysis not enabled"}
    
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history."""
        if self.middleware.alerting_system:
            return self.middleware.alerting_system.get_alert_history(hours)
        return []
    
    def add_alert_channel(self, name: str, channel_func: Callable[[Dict[str, Any]], None]):
        """Add an alert channel."""
        if self.middleware.alerting_system:
            self.middleware.alerting_system.add_alert_channel(name, channel_func)

# =============================================================================
# Usage Examples
# =============================================================================

def create_error_monitoring_middleware(
    enable_metrics: bool = True,
    enable_alerting: bool = True,
    enable_analysis: bool = True,
    enable_recovery: bool = True,
    retention_hours: int = 24
) -> ErrorMonitoringMiddleware:
    """Create error monitoring middleware with configuration."""
    return ErrorMonitoringMiddleware(
        app=None,  # Will be set by FastAPI
        enable_metrics=enable_metrics,
        enable_alerting=enable_alerting,
        enable_analysis=enable_analysis,
        enable_recovery=enable_recovery,
        retention_hours=retention_hours
    )

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "ErrorType",
    "AlertLevel",
    "ErrorEvent",
    "ErrorMetricsCollector",
    "ErrorAlertingSystem",
    "ErrorAnalysisEngine",
    "ErrorRecoveryStrategies",
    "ErrorMonitoringMiddleware",
    "ErrorMonitoringEndpoints",
    "create_error_monitoring_middleware",
] 