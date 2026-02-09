"""
OpenTelemetry Middleware para Distributed Tracing
Integración con AWS X-Ray y OpenTelemetry
"""

import logging
from typing import Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Intentar importar OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPOTLPSpanExporter
    
    # AWS X-Ray
    try:
        from opentelemetry.sdk.extension.aws.trace import AwsXRayIdGenerator
        from opentelemetry.sdk.extension.aws.trace.propagation import AwsXRayPropagator
        AWS_XRAY_AVAILABLE = True
    except ImportError:
        AWS_XRAY_AVAILABLE = False
        logger.warning("AWS X-Ray extension not available")
    
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")


class OpenTelemetryMiddleware(BaseHTTPMiddleware):
    """
    Middleware para distributed tracing con OpenTelemetry
    Compatible con AWS X-Ray
    """
    
    def __init__(self, app, service_name: str = "suno-clone-ai", enabled: bool = True):
        super().__init__(app)
        self.service_name = service_name
        self.enabled = enabled and OPENTELEMETRY_AVAILABLE
        
        if self.enabled:
            self._setup_tracing()
    
    def _setup_tracing(self):
        """Configura OpenTelemetry tracing"""
        try:
            # Crear resource con información del servicio
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "1.0.0",
            })
            
            # Configurar tracer provider
            if AWS_XRAY_AVAILABLE:
                trace.set_tracer_provider(
                    TracerProvider(
                        resource=resource,
                        id_generator=AwsXRayIdGenerator()
                    )
                )
            else:
                trace.set_tracer_provider(TracerProvider(resource=resource))
            
            tracer_provider = trace.get_tracer_provider()
            
            # Configurar exportador (OTLP para AWS X-Ray o Collector)
            otlp_endpoint = self._get_otlp_endpoint()
            if otlp_endpoint:
                if otlp_endpoint.startswith("http"):
                    exporter = HTTPOTLPSpanExporter(endpoint=otlp_endpoint)
                else:
                    exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                
                span_processor = BatchSpanProcessor(exporter)
                tracer_provider.add_span_processor(span_processor)
            
            logger.info(f"OpenTelemetry tracing configured for {self.service_name}")
        except Exception as e:
            logger.error(f"Failed to setup OpenTelemetry: {e}", exc_info=True)
            self.enabled = False
    
    def _get_otlp_endpoint(self) -> Optional[str]:
        """Obtiene el endpoint OTLP desde variables de entorno"""
        import os
        return os.getenv("OTLP_ENDPOINT") or os.getenv("AWS_XRAY_DAEMON_ADDRESS")
    
    async def dispatch(self, request: Request, call_next):
        """Procesa request con tracing"""
        if not self.enabled:
            return await call_next(request)
        
        tracer = trace.get_tracer(__name__)
        
        # Crear span para la request
        with tracer.start_as_current_span(
            f"{request.method} {request.url.path}",
            kind=trace.SpanKind.SERVER
        ) as span:
            # Agregar atributos al span
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.route", request.url.path)
            
            # Headers relevantes
            if "user-agent" in request.headers:
                span.set_attribute("http.user_agent", request.headers["user-agent"])
            if "x-forwarded-for" in request.headers:
                span.set_attribute("http.client_ip", request.headers["x-forwarded-for"])
            
            try:
                # Procesar request
                response = await call_next(request)
                
                # Agregar información de respuesta
                span.set_attribute("http.status_code", response.status_code)
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                return response
            except Exception as e:
                # Registrar error en el span
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise


def instrument_fastapi(app, service_name: str = "suno-clone-ai"):
    """
    Instrumenta una aplicación FastAPI con OpenTelemetry
    
    Args:
        app: Aplicación FastAPI
        service_name: Nombre del servicio
    """
    if not OPENTELEMETRY_AVAILABLE:
        logger.warning("OpenTelemetry not available, skipping instrumentation")
        return
    
    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info(f"FastAPI instrumented with OpenTelemetry: {service_name}")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI: {e}", exc_info=True)















