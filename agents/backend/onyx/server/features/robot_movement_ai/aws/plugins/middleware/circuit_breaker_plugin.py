"""
Circuit Breaker Plugin
======================
"""

import logging
import time
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from aws.core.interfaces import MiddlewarePlugin

logger = logging.getLogger(__name__)


class CircuitBreakerMiddlewarePlugin(MiddlewarePlugin):
    """Circuit breaker middleware plugin."""
    
    def get_name(self) -> str:
        return "circuit_breaker"
    
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        middleware_config = config.get("middleware", {})
        return middleware_config.get("enable_circuit_breaker", True)
    
    def setup(self, app: FastAPI, config: Dict[str, Any]) -> FastAPI:
        """Setup circuit breaker middleware."""
        middleware_config = config.get("middleware", {})
        failure_threshold = middleware_config.get("failure_threshold", 5)
        recovery_timeout = middleware_config.get("recovery_timeout", 60)
        
        class CircuitBreakerMiddleware(BaseHTTPMiddleware):
            def __init__(self, app):
                super().__init__(app)
                self.failure_count = {}
                self.last_failure_time = {}
                self.circuit_open = {}
                self.failure_threshold = failure_threshold
                self.recovery_timeout = recovery_timeout
            
            async def dispatch(self, request: Request, call_next):
                # Only apply to external service calls
                if request.url.path.startswith("/api/v1/external/"):
                    service_name = request.url.path.split("/")[3]
                    
                    if self._is_circuit_open(service_name):
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"Circuit breaker is open for {service_name}"
                        )
                    
                    try:
                        response = await call_next(request)
                        if service_name in self.failure_count:
                            self.failure_count[service_name] = 0
                        return response
                    except Exception as e:
                        self._record_failure(service_name)
                        raise
                
                return await call_next(request)
            
            def _is_circuit_open(self, service_name: str) -> bool:
                if service_name not in self.circuit_open:
                    return False
                if not self.circuit_open[service_name]:
                    return False
                if service_name in self.last_failure_time:
                    if time.time() - self.last_failure_time[service_name] > self.recovery_timeout:
                        self.circuit_open[service_name] = False
                        self.failure_count[service_name] = 0
                        return False
                return True
            
            def _record_failure(self, service_name: str):
                if service_name not in self.failure_count:
                    self.failure_count[service_name] = 0
                self.failure_count[service_name] += 1
                self.last_failure_time[service_name] = time.time()
                if self.failure_count[service_name] >= self.failure_threshold:
                    self.circuit_open[service_name] = True
                    logger.warning(f"Circuit breaker opened for {service_name}")
        
        app.add_middleware(CircuitBreakerMiddleware)
        logger.info("Circuit breaker enabled")
        
        return app















