"""
OpenTelemetry setup for distributed tracing
"""
import os
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.boto3sqs import Boto3SQSInstrumentor

def setup_opentelemetry(app=None):
    """
    Setup OpenTelemetry for distributed tracing
    
    Args:
        app: FastAPI application instance (optional)
    """
    if not os.getenv("ENABLE_OPENTELEMETRY", "true").lower() == "true":
        return
    
    # Create resource with service information
    resource = Resource.create({
        "service.name": "faceless-video-ai",
        "service.version": os.getenv("API_VERSION", "1.0.0"),
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    })
    
    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Configure exporter
    otlp_endpoint = os.getenv("OPENTELEMETRY_ENDPOINT")
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=os.getenv("OPENTELEMETRY_INSECURE", "false").lower() == "true"
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
    
    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    # Instrument FastAPI if app is provided
    if app:
        FastAPIInstrumentor.instrument_app(app)
    
    # Instrument HTTP clients
    HTTPXClientInstrumentor().instrument()
    
    # Instrument Redis
    try:
        RedisInstrumentor().instrument()
    except Exception:
        pass
    
    # Instrument AWS SQS if available
    try:
        Boto3SQSInstrumentor().instrument()
    except Exception:
        pass
    
    return tracer_provider

def get_tracer(name: str = "faceless-video-ai"):
    """Get a tracer instance"""
    return trace.get_tracer(name)




