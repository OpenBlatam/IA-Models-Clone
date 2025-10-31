"""
Logging Middleware
==================

Advanced logging middleware with structured logging and performance monitoring.
"""

from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...shared.config import get_settings


logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Log entry data"""
    request_id: str
    timestamp: datetime
    method: str
    path: str
    query_params: Dict[str, Any]
    headers: Dict[str, str]
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    status_code: int
    response_time_ms: float
    response_size_bytes: int
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoggingMiddleware:
    """
    Advanced logging middleware
    
    Provides structured logging with performance monitoring,
    request/response tracking, and error handling.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._statistics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "total_response_time": 0.0,
            "by_status_code": {},
            "by_method": {},
            "by_path": {}
        }
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Configure structured logging
        logging.basicConfig(
            level=getattr(logging, self.settings.log_level.value),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create request logger
        self.request_logger = logging.getLogger("request")
        self.error_logger = logging.getLogger("error")
        self.performance_logger = logging.getLogger("performance")
    
    async def process_request(self, request: Request) -> Request:
        """Process incoming request"""
        try:
            # Generate request ID
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            request.state.start_time = time.time()
            
            # Extract request information
            request_info = {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "ip_address": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
                "user_id": self._extract_user_id(request)
            }
            
            # Log request
            self.request_logger.info(
                f"Request started: {request.method} {request.url.path}",
                extra=request_info
            )
            
            return request
            
        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            return request
    
    async def process_response(
        self,
        request: Request,
        response: Response
    ) -> Response:
        """Process outgoing response"""
        try:
            # Get request information
            request_id = getattr(request.state, "request_id", "unknown")
            start_time = getattr(request.state, "start_time", time.time())
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Extract response information
            response_info = {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "response_time_ms": round(response_time, 2),
                "response_size_bytes": len(response.body) if response.body else 0,
                "ip_address": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
                "user_id": self._extract_user_id(request)
            }
            
            # Update statistics
            self._update_statistics(response_info)
            
            # Log response
            log_level = self._get_log_level(response.status_code)
            self.request_logger.log(
                getattr(logging, log_level.value),
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra=response_info
            )
            
            # Log performance metrics
            if response_time > 1000:  # Log slow requests (>1s)
                self.performance_logger.warning(
                    f"Slow request detected: {request.method} {request.url.path} - {response_time:.2f}ms",
                    extra=response_info
                )
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{response_time:.2f}ms"
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process response: {e}")
            return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request"""
        # In a real implementation, extract from JWT token or session
        # For now, return None
        return None
    
    def _get_log_level(self, status_code: int) -> LogLevel:
        """Get log level based on status code"""
        if status_code < 400:
            return LogLevel.INFO
        elif status_code < 500:
            return LogLevel.WARNING
        else:
            return LogLevel.ERROR
    
    def _update_statistics(self, response_info: Dict[str, Any]):
        """Update request statistics"""
        self._statistics["total_requests"] += 1
        self._statistics["total_response_time"] += response_info["response_time_ms"]
        self._statistics["average_response_time"] = (
            self._statistics["total_response_time"] / self._statistics["total_requests"]
        )
        
        # Update by status code
        status_code = response_info["status_code"]
        if status_code not in self._statistics["by_status_code"]:
            self._statistics["by_status_code"][status_code] = 0
        self._statistics["by_status_code"][status_code] += 1
        
        # Update by method
        method = response_info["method"]
        if method not in self._statistics["by_method"]:
            self._statistics["by_method"][method] = 0
        self._statistics["by_method"][method] += 1
        
        # Update by path
        path = response_info["path"]
        if path not in self._statistics["by_path"]:
            self._statistics["by_path"][path] = 0
        self._statistics["by_path"][path] += 1
        
        # Update success/failure counts
        if 200 <= status_code < 400:
            self._statistics["successful_requests"] += 1
        else:
            self._statistics["failed_requests"] += 1
    
    def log_error(
        self,
        request: Request,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log error with context"""
        try:
            request_id = getattr(request.state, "request_id", "unknown")
            
            error_info = {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "ip_address": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
                "user_id": self._extract_user_id(request),
                "context": context or {}
            }
            
            self.error_logger.error(
                f"Request error: {request.method} {request.url.path} - {type(error).__name__}",
                extra=error_info,
                exc_info=True
            )
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "ms",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log performance metric"""
        try:
            metric_info = {
                "metric_name": metric_name,
                "value": value,
                "unit": unit,
                "timestamp": datetime.utcnow().isoformat(),
                "context": context or {}
            }
            
            self.performance_logger.info(
                f"Performance metric: {metric_name} = {value}{unit}",
                extra=metric_info
            )
            
        except Exception as e:
            logger.error(f"Failed to log performance metric: {e}")
    
    def log_business_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> None:
        """Log business event"""
        try:
            event_info = {
                "event_type": event_type,
                "event_data": event_data,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.request_logger.info(
                f"Business event: {event_type}",
                extra=event_info
            )
            
        except Exception as e:
            logger.error(f"Failed to log business event: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging middleware statistics"""
        total_requests = self._statistics["total_requests"]
        success_rate = (
            (self._statistics["successful_requests"] / total_requests * 100)
            if total_requests > 0 else 0
        )
        
        return {
            **self._statistics,
            "success_rate": success_rate,
            "config": {
                "log_level": self.settings.log_level.value,
                "log_format": self.settings.log_format
            }
        }
    
    def get_recent_requests(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent request logs"""
        # In a real implementation, this would query a log storage
        # For now, return empty list
        return []
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for specified hours"""
        # In a real implementation, this would query error logs
        # For now, return mock data
        return {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_path": {},
            "most_common_errors": []
        }


# Global logging middleware instance
_logging_middleware: Optional[LoggingMiddleware] = None


def get_logging_middleware() -> LoggingMiddleware:
    """Get global logging middleware instance"""
    global _logging_middleware
    if _logging_middleware is None:
        _logging_middleware = LoggingMiddleware()
    return _logging_middleware


# FastAPI dependency
async def get_logging_middleware_dependency() -> LoggingMiddleware:
    """FastAPI dependency for logging middleware"""
    return get_logging_middleware()




