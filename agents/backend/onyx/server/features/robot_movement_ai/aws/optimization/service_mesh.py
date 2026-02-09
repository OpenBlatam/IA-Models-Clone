"""
Service Mesh
============

Service mesh patterns for microservices:
- Service-to-service communication
- Load balancing
- Circuit breakers
- Retries
- Timeouts
- Distributed tracing
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import httpx

logger = logging.getLogger(__name__)


@dataclass
class MeshConfig:
    """Service mesh configuration."""
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_circuit_breaker: bool = True
    default_timeout: float = 30.0
    default_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60


class ServiceMesh:
    """Service mesh for inter-service communication."""
    
    def __init__(self, config: Optional[MeshConfig] = None):
        self.config = config or MeshConfig()
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}
    
    async def call_service(
        self,
        service_name: str,
        method: str,
        path: str,
        **kwargs
    ) -> httpx.Response:
        """
        Call a service through the mesh.
        
        Features:
        - Automatic retries
        - Circuit breakers
        - Timeouts
        - Distributed tracing
        - Metrics collection
        """
        # Check circuit breaker
        if self.config.enable_circuit_breaker:
            if not self._can_call(service_name):
                raise Exception(f"Circuit breaker is open for {service_name}")
        
        # Start trace
        trace_id = None
        if self.config.enable_tracing:
            trace_id = self._start_trace(service_name, method, path)
        
        start_time = datetime.utcnow()
        last_exception = None
        
        # Retry logic
        for attempt in range(self.config.default_retries):
            try:
                # Make request
                response = await self._make_request(
                    service_name,
                    method,
                    path,
                    timeout=self.config.default_timeout,
                    **kwargs
                )
                
                # Record success
                if self.config.enable_circuit_breaker:
                    self._record_success(service_name)
                
                if self.config.enable_metrics:
                    self._record_metric(
                        service_name,
                        method,
                        path,
                        response.status_code,
                        (datetime.utcnow() - start_time).total_seconds(),
                        success=True
                    )
                
                if self.config.enable_tracing:
                    self._end_trace(trace_id, success=True)
                
                return response
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.default_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                else:
                    # All retries failed
                    if self.config.enable_circuit_breaker:
                        self._record_failure(service_name)
                    
                    if self.config.enable_metrics:
                        self._record_metric(
                            service_name,
                            method,
                            path,
                            0,
                            (datetime.utcnow() - start_time).total_seconds(),
                            success=False
                        )
                    
                    if self.config.enable_tracing:
                        self._end_trace(trace_id, success=False, error=str(e))
        
        raise last_exception or Exception("Service call failed")
    
    async def _make_request(
        self,
        service_name: str,
        method: str,
        path: str,
        timeout: float,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request."""
        # Get service URL from registry
        from aws.services.service_registry import get_service_registry
        registry = get_service_registry()
        instance = registry.get_instance(service_name)
        
        if not instance:
            raise Exception(f"Service {service_name} not available")
        
        url = f"http://{instance.host}:{instance.port}{path}"
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.request(method, url, **kwargs)
    
    def _can_call(self, service_name: str) -> bool:
        """Check if service can be called (circuit breaker)."""
        if service_name not in self._circuit_breakers:
            return True
        
        cb = self._circuit_breakers[service_name]
        
        if cb["state"] == "closed":
            return True
        
        if cb["state"] == "open":
            # Check if timeout has passed
            if datetime.utcnow() - cb["last_failure"] > timedelta(seconds=self.config.circuit_breaker_timeout):
                cb["state"] = "half_open"
                return True
            return False
        
        # half_open
        return True
    
    def _record_success(self, service_name: str):
        """Record successful call."""
        if service_name in self._circuit_breakers:
            self._circuit_breakers[service_name]["failure_count"] = 0
            self._circuit_breakers[service_name]["state"] = "closed"
    
    def _record_failure(self, service_name: str):
        """Record failed call."""
        if service_name not in self._circuit_breakers:
            self._circuit_breakers[service_name] = {
                "failure_count": 0,
                "state": "closed",
                "last_failure": datetime.utcnow()
            }
        
        cb = self._circuit_breakers[service_name]
        cb["failure_count"] += 1
        cb["last_failure"] = datetime.utcnow()
        
        if cb["failure_count"] >= self.config.circuit_breaker_threshold:
            cb["state"] = "open"
            logger.warning(f"Circuit breaker opened for {service_name}")
    
    def _start_trace(self, service_name: str, method: str, path: str) -> str:
        """Start distributed trace."""
        trace_id = f"trace-{int(datetime.utcnow().timestamp() * 1000)}"
        logger.debug(f"Started trace {trace_id} for {service_name}:{method}:{path}")
        return trace_id
    
    def _end_trace(self, trace_id: str, success: bool, error: Optional[str] = None):
        """End distributed trace."""
        status = "success" if success else "failure"
        logger.debug(f"Ended trace {trace_id}: {status}" + (f" - {error}" if error else ""))
    
    def _record_metric(
        self,
        service_name: str,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        success: bool
    ):
        """Record service call metric."""
        if service_name not in self._metrics:
            self._metrics[service_name] = []
        
        self._metrics[service_name].append({
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration": duration,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only last 1000 metrics
        if len(self._metrics[service_name]) > 1000:
            self._metrics[service_name] = self._metrics[service_name][-1000:]
    
    def get_metrics(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get service metrics."""
        if service_name:
            return self._metrics.get(service_name, [])
        return self._metrics
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get circuit breaker status."""
        return {
            name: {
                "state": cb["state"],
                "failure_count": cb["failure_count"],
                "last_failure": cb["last_failure"].isoformat()
            }
            for name, cb in self._circuit_breakers.items()
        }


# Global service mesh instance
_mesh: Optional[ServiceMesh] = None


def get_service_mesh(config: Optional[MeshConfig] = None) -> ServiceMesh:
    """Get global service mesh."""
    global _mesh
    if _mesh is None:
        _mesh = ServiceMesh(config)
    return _mesh















