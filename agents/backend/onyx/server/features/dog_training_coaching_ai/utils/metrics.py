"""
Metrics Utilities
=================
"""

from prometheus_client import Counter, Histogram, Gauge
from functools import wraps

# Metrics
requests_total = Counter(
    'dog_training_requests_total',
    'Total number of requests',
    ['endpoint', 'method', 'status']
)

request_duration = Histogram(
    'dog_training_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint']
)

active_requests = Gauge(
    'dog_training_active_requests',
    'Number of active requests'
)


def track_metrics(endpoint_name: str):
    """Decorator para trackear métricas."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            active_requests.inc()
            try:
                with request_duration.labels(endpoint=endpoint_name).time():
                    result = await func(*args, **kwargs)
                requests_total.labels(
                    endpoint=endpoint_name,
                    method='POST',
                    status='success'
                ).inc()
                return result
            except Exception as e:
                requests_total.labels(
                    endpoint=endpoint_name,
                    method='POST',
                    status='error'
                ).inc()
                raise
            finally:
                active_requests.dec()
        return wrapper
    return decorator

