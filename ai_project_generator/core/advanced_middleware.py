"""
Advanced Middleware - Middleware avanzado para logging, tracing y monitoring
============================================================================

Middleware para FastAPI que implementa:
- Logging estructurado
- Distributed tracing (OpenTelemetry)
- Request/response monitoring
- Performance tracking
- Security headers
"""

import time
import uuid
from typing import Callable, Optional
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import structlog
from ..core.json_utils import json_dumps_str

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

from .microservices_config import get_microservices_config

logger = structlog.get_logger(__name__)
config = get_microservices_config()

# Configurar structured logging
if config.structured_logging:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

# Configurar OpenTelemetry si está disponible
tracer = None
if OPENTELEMETRY_AVAILABLE and config.opentelemetry_enabled:
    try:
        resource = Resource.create({
            "service.name": "ai-project-generator",
            "service.version": "1.0.0",
        })
        
        provider = TracerProvider(resource=resource)
        
        if config.opentelemetry_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=config.opentelemetry_endpoint,
                insecure=True,
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(__name__)
    except Exception as e:
        logger.warning(f"Failed to initialize OpenTelemetry: {e}")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware para logging estructurado"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generar request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Tiempo de inicio
        start_time = time.time()
        
        # Log de request
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
        
        if config.structured_logging:
            structlog.get_logger().info("incoming_request", **log_data)
        else:
            logger.info(f"Request: {request.method} {request.url.path} [ID: {request_id}]")
        
        # Procesar request
        try:
            response = await call_next(request)
            
            # Calcular tiempo de procesamiento
            process_time = time.time() - start_time
            
            # Log de response
            response_log = {
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time": round(process_time, 4),
            }
            
            if config.structured_logging:
                structlog.get_logger().info("outgoing_response", **response_log)
            else:
                logger.info(
                    f"Response: {response.status_code} "
                    f"[ID: {request_id}] "
                    f"[Time: {process_time:.4f}s]"
                )
            
            # Agregar headers de performance
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            error_log = {
                "request_id": request_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "process_time": round(process_time, 4),
            }
            
            if config.structured_logging:
                structlog.get_logger().error("request_error", **error_log)
            else:
                logger.error(f"Request error [ID: {request_id}]: {e}")
            
            raise


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware para distributed tracing con OpenTelemetry"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not tracer:
            return await call_next(request)
        
        # Crear span
        span_name = f"{request.method} {request.url.path}"
        with tracer.start_as_current_span(span_name) as span:
            # Agregar atributos
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.path", request.url.path)
            span.set_attribute("http.client_ip", request.client.host if request.client else None)
            
            try:
                response = await call_next(request)
                span.set_attribute("http.status_code", response.status_code)
                span.set_status(trace.Status(trace.StatusCode.OK))
                return response
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware para agregar security headers"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline';"
            ),
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware para monitoreo de performance"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.request_times = []
        self.error_count = 0
        self.success_count = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Track metrics
            self.request_times.append(process_time)
            if response.status_code < 400:
                self.success_count += 1
            else:
                self.error_count += 1
            
            # Mantener solo últimos 1000 tiempos
            if len(self.request_times) > 1000:
                self.request_times = self.request_times[-1000:]
            
            # Agregar métricas a headers
            if len(self.request_times) > 0:
                avg_time = sum(self.request_times) / len(self.request_times)
                response.headers["X-Avg-Process-Time"] = str(round(avg_time, 4))
            
            response.headers["X-Success-Count"] = str(self.success_count)
            response.headers["X-Error-Count"] = str(self.error_count)
            
            return response
            
        except Exception as e:
            self.error_count += 1
            raise


def setup_advanced_middleware(app):
    """Configura todos los middlewares avanzados"""
    # Structured logging
    app.add_middleware(StructuredLoggingMiddleware)
    
    # Tracing
    if OPENTELEMETRY_AVAILABLE and config.opentelemetry_enabled:
        app.add_middleware(TracingMiddleware)
        try:
            FastAPIInstrumentor.instrument_app(app)
        except Exception as e:
            logger.warning(f"Failed to instrument FastAPI with OpenTelemetry: {e}")
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Performance monitoring
    app.add_middleware(PerformanceMonitoringMiddleware)
    
    logger.info("Advanced middleware configured successfully")




