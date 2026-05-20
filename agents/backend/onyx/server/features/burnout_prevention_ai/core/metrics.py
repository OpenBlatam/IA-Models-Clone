"""
Metrics and Monitoring Utilities
=================================
Utilities for collecting and exposing metrics.
"""

from typing import Dict, Any, Optional
from datetime import datetime

try:
    from prometheus_client import Counter, Histogram, Gauge
    _has_prometheus = True
except ImportError:
    _has_prometheus = False

# Metrics (if prometheus is available)
if _has_prometheus:
    # Request metrics
    api_requests_total = Counter(
        'burnout_ai_api_requests_total',
        'Total API requests',
        ['endpoint', 'method', 'status']
    )
    
    api_request_duration = Histogram(
        'burnout_ai_api_request_duration_seconds',
        'API request duration',
        ['endpoint', 'method']
    )
    
    # Cache metrics
    cache_hits_total = Counter(
        'burnout_ai_cache_hits_total',
        'Total cache hits'
    )
    
    cache_misses_total = Counter(
        'burnout_ai_cache_misses_total',
        'Total cache misses'
    )
    
    cache_size = Gauge(
        'burnout_ai_cache_size',
        'Current cache size'
    )
    
    # OpenRouter API metrics
    openrouter_requests_total = Counter(
        'burnout_ai_openrouter_requests_total',
        'Total OpenRouter API requests',
        ['model', 'status']
    )
    
    openrouter_request_duration = Histogram(
        'burnout_ai_openrouter_request_duration_seconds',
        'OpenRouter API request duration',
        ['model']
    )
else:
    # Dummy metrics if prometheus not available
    api_requests_total = None
    api_request_duration = None
    cache_hits_total = None
    cache_misses_total = None
    cache_size = None
    openrouter_requests_total = None
    openrouter_request_duration = None


def record_api_request(endpoint: str, method: str, status: int, duration: float) -> None:
    """
    Record API request metrics.
    
    Args:
        endpoint: API endpoint path
        method: HTTP method
        status: HTTP status code
        duration: Request duration in seconds
    """
    if not _has_prometheus or not api_requests_total:
        return
    
    try:
        api_requests_total.labels(endpoint=endpoint, method=method, status=str(status)).inc()
        if api_request_duration:
            api_request_duration.labels(endpoint=endpoint, method=method).observe(duration)
    except Exception:
        # Silently fail metrics recording to avoid breaking the application
        pass


def record_cache_hit() -> None:
    """Record cache hit metric."""
    if not _has_prometheus or not cache_hits_total:
        return
    try:
        cache_hits_total.inc()
    except Exception:
        # Silently fail metrics recording to avoid breaking the application
        pass


def record_cache_miss() -> None:
    """Record cache miss metric."""
    if not _has_prometheus or not cache_misses_total:
        return
    try:
        cache_misses_total.inc()
    except Exception:
        # Silently fail metrics recording to avoid breaking the application
        pass


def update_cache_size(size: int) -> None:
    """
    Update cache size metric.
    
    Args:
        size: Current cache size
    """
    if not _has_prometheus or not cache_size:
        return
    try:
        cache_size.set(size)
    except Exception:
        # Silently fail metrics recording to avoid breaking the application
        pass


def record_openrouter_request(model: str, status: str, duration: float) -> None:
    """
    Record OpenRouter API request metrics.
    
    Args:
        model: Model name used for the request
        status: Request status (success, error, etc.)
        duration: Request duration in seconds
    """
    if not _has_prometheus or not openrouter_requests_total:
        return
    try:
        openrouter_requests_total.labels(model=model, status=status).inc()
        if openrouter_request_duration:
            openrouter_request_duration.labels(model=model).observe(duration)
    except Exception:
        # Silently fail metrics recording to avoid breaking the application
        pass

