"""
Service Mesh Middleware
Integración con service mesh (Istio, Linkerd, Consul Connect)
"""

import logging
import time
from typing import Dict, Any, Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ServiceMeshMiddleware(BaseHTTPMiddleware):
    """
    Middleware para integración con service mesh
    Maneja headers de tracing, routing, y circuit breaking
    """
    
    def __init__(self, app, service_name: str = "suno-clone-ai", **kwargs):
        super().__init__(app)
        self.service_name = service_name
        self.enable_istio = kwargs.get('enable_istio', True)
        self.enable_consul = kwargs.get('enable_consul', False)
        self.enable_linkerd = kwargs.get('enable_linkerd', False)
    
    async def dispatch(self, request: Request, call_next):
        """Procesa request con service mesh headers"""
        start_time = time.time()
        
        # Extraer headers de service mesh
        trace_id = self._extract_trace_id(request)
        span_id = self._extract_span_id(request)
        parent_span_id = request.headers.get("x-b3-parentspanid")
        
        # Agregar headers de contexto
        request.state.trace_id = trace_id
        request.state.span_id = span_id
        request.state.parent_span_id = parent_span_id
        request.state.service_name = self.service_name
        
        # Headers Istio
        if self.enable_istio:
            request.state.istio_request_id = request.headers.get("x-request-id")
            request.state.istio_route = request.headers.get("x-forwarded-for")
        
        # Headers Consul
        if self.enable_consul:
            request.state.consul_service = request.headers.get("x-consul-service")
            request.state.consul_node = request.headers.get("x-consul-node")
        
        # Headers Linkerd
        if self.enable_linkerd:
            request.state.linkerd_dst = request.headers.get("l5d-dst-override")
        
        # Procesar request
        response = await call_next(request)
        
        # Agregar headers de respuesta para service mesh
        duration = time.time() - start_time
        
        # Propagar trace ID
        if trace_id:
            response.headers["X-Trace-ID"] = trace_id
        if span_id:
            response.headers["X-Span-ID"] = span_id
        
        # Headers de métricas
        response.headers["X-Response-Time"] = f"{duration:.4f}"
        response.headers["X-Service-Name"] = self.service_name
        
        return response
    
    def _extract_trace_id(self, request: Request) -> Optional[str]:
        """Extrae trace ID de diferentes formatos"""
        # B3 format (Zipkin)
        trace_id = request.headers.get("x-b3-traceid")
        if trace_id:
            return trace_id
        
        # W3C Trace Context
        trace_id = request.headers.get("traceparent")
        if trace_id:
            return trace_id.split("-")[1] if "-" in trace_id else trace_id
        
        # Jaeger
        trace_id = request.headers.get("uber-trace-id")
        if trace_id:
            return trace_id.split(":")[0] if ":" in trace_id else trace_id
        
        # Istio
        trace_id = request.headers.get("x-request-id")
        if trace_id:
            return trace_id
        
        return None
    
    def _extract_span_id(self, request: Request) -> Optional[str]:
        """Extrae span ID"""
        # B3 format
        span_id = request.headers.get("x-b3-spanid")
        if span_id:
            return span_id
        
        return None















