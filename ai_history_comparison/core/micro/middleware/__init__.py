"""
Micro Middleware Module

Ultra-specialized middleware components for the AI History Comparison System.
Each middleware handles specific cross-cutting concerns in the request/response pipeline.
"""

from .base_middleware import BaseMiddleware, MiddlewareRegistry, MiddlewareChain
from .auth_middleware import AuthenticationMiddleware, AuthorizationMiddleware, TokenMiddleware
from .logging_middleware import LoggingMiddleware, AuditMiddleware, TraceMiddleware
from .performance_middleware import PerformanceMiddleware, TimingMiddleware, ProfilingMiddleware
from .security_middleware import SecurityMiddleware, CSRFMiddleware, RateLimitMiddleware
from .caching_middleware import CachingMiddleware, ResponseCacheMiddleware, RequestCacheMiddleware
from .validation_middleware import ValidationMiddleware, SchemaMiddleware, TypeCheckMiddleware
from .monitoring_middleware import MonitoringMiddleware, MetricsMiddleware, HealthMiddleware
from .error_middleware import ErrorMiddleware, ExceptionMiddleware, FallbackMiddleware
from .compression_middleware import CompressionMiddleware, GzipMiddleware, BrotliMiddleware
from .cors_middleware import CORSMiddleware, HeadersMiddleware, OptionsMiddleware

__all__ = [
    'BaseMiddleware', 'MiddlewareRegistry', 'MiddlewareChain',
    'AuthenticationMiddleware', 'AuthorizationMiddleware', 'TokenMiddleware',
    'LoggingMiddleware', 'AuditMiddleware', 'TraceMiddleware',
    'PerformanceMiddleware', 'TimingMiddleware', 'ProfilingMiddleware',
    'SecurityMiddleware', 'CSRFMiddleware', 'RateLimitMiddleware',
    'CachingMiddleware', 'ResponseCacheMiddleware', 'RequestCacheMiddleware',
    'ValidationMiddleware', 'SchemaMiddleware', 'TypeCheckMiddleware',
    'MonitoringMiddleware', 'MetricsMiddleware', 'HealthMiddleware',
    'ErrorMiddleware', 'ExceptionMiddleware', 'FallbackMiddleware',
    'CompressionMiddleware', 'GzipMiddleware', 'BrotliMiddleware',
    'CORSMiddleware', 'HeadersMiddleware', 'OptionsMiddleware'
]





















