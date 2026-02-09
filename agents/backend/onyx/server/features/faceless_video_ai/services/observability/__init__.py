"""
Observability module for distributed tracing and monitoring
"""
from .opentelemetry_setup import setup_opentelemetry, get_tracer

__all__ = ["setup_opentelemetry", "get_tracer"]




