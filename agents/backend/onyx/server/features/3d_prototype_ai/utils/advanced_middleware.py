"""
Advanced Middleware - Middleware avanzado para FastAPI
======================================================

Incluye:
- OpenTelemetry para distributed tracing
- Structured logging
- Request/Response logging
- Performance monitoring
- Security headers
"""

import time
import logging
import json
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from contextlib import asynccontextmanager
import uuid

# OpenTelemetry imports (optional)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None

logger = logging.getLogger(__name__)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware para logging estructurado"""
    
    def __init__(self, app: ASGIApp, service_name: str = "music_analyzer_ai"):
        super().__init__(app)
        self.service_name = service_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesa la request y genera logs estructurados"""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Agregar request_id al request state
        request.state.request_id = request_id
        
        # Log de request
        log_data = {
            "timestamp": time.time(),
            "service": self.service_name,
            "request_id": request_id,
            "method": request.method,
            "path": str(request.url.path),
            "query_params": dict(request.query_params),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "type": "request"
        }
        
        logger.info("Incoming request", extra={"structured_data": log_data})
        
        try:
            response = await call_next(request)
            
            # Calcular tiempo de procesamiento
            process_time = time.time() - start_time
            
            # Log de response
            response_log = {
                "timestamp": time.time(),
                "service": self.service_name,
                "request_id": request_id,
                "method": request.method,
                "path": str(request.url.path),
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2),
                "type": "response"
            }
            
            logger.info("Request completed", extra={"structured_data": response_log})
            
            # Agregar headers de respuesta
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time, 3))
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            error_log = {
                "timestamp": time.time(),
                "service": self.service_name,
                "request_id": request_id,
                "method": request.method,
                "path": str(request.url.path),
                "error": str(e),
                "error_type": type(e).__name__,
                "process_time_ms": round(process_time * 1000, 2),
                "type": "error"
            }
            
            logger.error("Request failed", extra={"structured_data": error_log}, exc_info=True)
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware para agregar headers de seguridad"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Agrega headers de seguridad a la respuesta"""
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware para monitoreo de performance"""
    
    def __init__(self, app: ASGIApp, metrics_collector: Optional[Callable] = None):
        super().__init__(app)
        self.metrics_collector = metrics_collector
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitorea el performance de las requests"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        end_memory = self._get_memory_usage()
        memory_delta = end_memory - start_memory
        
        # Recopilar métricas
        metrics = {
            "path": str(request.url.path),
            "method": request.method,
            "status_code": response.status_code,
            "process_time": process_time,
            "memory_delta": memory_delta
        }
        
        if self.metrics_collector:
            self.metrics_collector(metrics)
        
        # Agregar métricas a headers
        response.headers["X-Process-Time"] = str(round(process_time, 3))
        response.headers["X-Memory-Delta"] = str(memory_delta)
        
        return response
    
    def _get_memory_usage(self) -> float:
        """Obtiene el uso de memoria actual"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0


class OpenTelemetryMiddleware:
    """Configuración de OpenTelemetry para distributed tracing"""
    
    @staticmethod
    def setup_tracing(service_name: str = "3d_prototype_ai", 
                     endpoint: Optional[str] = None,
                     enable_console: bool = False):
        """Configura OpenTelemetry tracing"""
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")
            return None
        
        try:
            # Crear resource
            resource = Resource.create({
                "service.name": service_name,
                "service.version": "2.0.0"
            })
            
            # Configurar tracer provider
            tracer_provider = TracerProvider(resource=resource)
            
            # Exporters
            exporters = []
            
            if enable_console:
                console_exporter = ConsoleSpanExporter()
                exporters.append(BatchSpanProcessor(console_exporter))
            
            if endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
                exporters.append(BatchSpanProcessor(otlp_exporter))
            
            for exporter in exporters:
                tracer_provider.add_span_processor(exporter)
            
            # Establecer tracer provider global
            trace.set_tracer_provider(tracer_provider)
            
            logger.info(f"OpenTelemetry tracing configured for {service_name}")
            return tracer_provider
            
        except Exception as e:
            logger.error(f"Failed to setup OpenTelemetry: {e}")
            return None
    
    @staticmethod
    def instrument_fastapi(app):
        """Instrumenta FastAPI con OpenTelemetry"""
        if not OPENTELEMETRY_AVAILABLE:
            return
        
        try:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI instrumented with OpenTelemetry")
        except Exception as e:
            logger.error(f"Failed to instrument FastAPI: {e}")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware para manejar contexto de request"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Agrega contexto a la request"""
        # Agregar información de contexto
        request.state.start_time = time.time()
        request.state.request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        response = await call_next(request)
        return response


def setup_advanced_middleware(app, service_name: str = "music_analyzer_ai", 
                             enable_opentelemetry: bool = True,
                             opentelemetry_endpoint: Optional[str] = None):
    """Configura todos los middlewares avanzados"""
    
    # Structured logging
    app.add_middleware(StructuredLoggingMiddleware, service_name=service_name)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Performance monitoring
    app.add_middleware(PerformanceMonitoringMiddleware)
    
    # Request context
    app.add_middleware(RequestContextMiddleware)
    
    # OpenTelemetry
    if enable_opentelemetry:
        OpenTelemetryMiddleware.setup_tracing(
            service_name=service_name,
            endpoint=opentelemetry_endpoint,
            enable_console=True
        )
        OpenTelemetryMiddleware.instrument_fastapi(app)
    
    logger.info("Advanced middleware configured successfully")

