"""
AWS Observability Middleware
CloudWatch metrics, X-Ray tracing, and structured logging
"""

import time
import json
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

try:
    from aws_xray_sdk.core import xray_recorder
    from aws_xray_sdk.core.models import http
    XRAY_AVAILABLE = True
except ImportError:
    XRAY_AVAILABLE = False
    xray_recorder = None

from config.aws_settings import get_aws_settings
from aws.aws_services import CloudWatchService

logger = logging.getLogger(__name__)
aws_settings = get_aws_settings()
cloudwatch = CloudWatchService()


class AWSObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for AWS observability:
    - CloudWatch metrics
    - X-Ray tracing
    - Structured logging
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.cloudwatch = CloudWatchService()
        
        # Initialize X-Ray if available
        if XRAY_AVAILABLE and aws_settings.enable_xray:
            try:
                xray_recorder.configure(
                    service='addiction-recovery-ai',
                    sampling=False  # Sample all requests in Lambda
                )
            except Exception as e:
                logger.warning(f"Failed to configure X-Ray: {str(e)}")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with observability"""
        start_time = time.time()
        request_id = request.headers.get("x-request-id", "unknown")
        
        # Start X-Ray segment if available
        if XRAY_AVAILABLE and aws_settings.enable_xray:
            segment = xray_recorder.begin_segment(name=request.url.path)
            segment.put_http_meta(http.URL, str(request.url))
            segment.put_http_meta(http.METHOD, request.method)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log request
            self._log_request(request, response, duration, request_id)
            
            # Send metrics to CloudWatch
            self._send_metrics(request, response, duration)
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}"
            
            return response
            
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            self._log_error(request, e, duration, request_id)
            self._send_error_metrics(request, duration)
            raise
            
        finally:
            # End X-Ray segment
            if XRAY_AVAILABLE and aws_settings.enable_xray:
                try:
                    xray_recorder.end_segment()
                except Exception:
                    pass
    
    def _log_request(self, request: Request, response: Response, duration: float, request_id: str) -> None:
        """Log structured request information"""
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": str(request.url.path),
            "status_code": response.status_code,
            "duration_ms": duration * 1000,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
        
        # Add query parameters if present
        if request.query_params:
            log_data["query_params"] = dict(request.query_params)
        
        # Log at appropriate level
        if response.status_code >= 500:
            logger.error(json.dumps(log_data))
        elif response.status_code >= 400:
            logger.warning(json.dumps(log_data))
        else:
            logger.info(json.dumps(log_data))
    
    def _log_error(self, request: Request, error: Exception, duration: float, request_id: str) -> None:
        """Log error information"""
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": str(request.url.path),
            "error": str(error),
            "error_type": type(error).__name__,
            "duration_ms": duration * 1000,
        }
        logger.error(json.dumps(log_data), exc_info=True)
    
    def _send_metrics(self, request: Request, response: Response, duration: float) -> None:
        """Send metrics to CloudWatch"""
        try:
            # Request count
            self.cloudwatch.put_metric(
                metric_name="RequestCount",
                value=1,
                unit="Count",
                dimensions={
                    "Method": request.method,
                    "Path": request.url.path,
                    "Status": str(response.status_code)
                }
            )
            
            # Response time
            self.cloudwatch.put_metric(
                metric_name="ResponseTime",
                value=duration * 1000,  # Convert to milliseconds
                unit="Milliseconds",
                dimensions={
                    "Method": request.method,
                    "Path": request.url.path
                }
            )
            
            # Status code distribution
            self.cloudwatch.put_metric(
                metric_name="StatusCode",
                value=1,
                unit="Count",
                dimensions={
                    "Status": str(response.status_code)
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to send metrics to CloudWatch: {str(e)}")
    
    def _send_error_metrics(self, request: Request, duration: float) -> None:
        """Send error metrics to CloudWatch"""
        try:
            self.cloudwatch.put_metric(
                metric_name="ErrorCount",
                value=1,
                unit="Count",
                dimensions={
                    "Method": request.method,
                    "Path": request.url.path
                }
            )
        except Exception as e:
            logger.warning(f"Failed to send error metrics to CloudWatch: {str(e)}")















