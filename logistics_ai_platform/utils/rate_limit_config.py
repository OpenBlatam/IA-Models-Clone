"""
Rate limiting configuration per endpoint

This module provides configuration for rate limiting per endpoint.
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class EndpointRateLimit:
    """Rate limit configuration for an endpoint"""
    max_requests: int
    window_seconds: int
    burst: Optional[int] = None  # Allow burst of requests


# Default rate limits per endpoint pattern
ENDPOINT_RATE_LIMITS: Dict[str, EndpointRateLimit] = {
    # Health and metrics endpoints - more lenient
    "/health": EndpointRateLimit(max_requests=100, window_seconds=60),
    "/ready": EndpointRateLimit(max_requests=100, window_seconds=60),
    "/metrics": EndpointRateLimit(max_requests=200, window_seconds=60),
    "/metrics/info": EndpointRateLimit(max_requests=100, window_seconds=60),
    
    # Read operations - moderate limits
    "/forwarding/quotes/{quote_id}": EndpointRateLimit(max_requests=200, window_seconds=60),
    "/forwarding/bookings/{booking_id}": EndpointRateLimit(max_requests=200, window_seconds=60),
    "/forwarding/shipments": EndpointRateLimit(max_requests=150, window_seconds=60),
    "/forwarding/shipments/{shipment_id}": EndpointRateLimit(max_requests=200, window_seconds=60),
    "/tracking/": EndpointRateLimit(max_requests=300, window_seconds=60),
    
    # Write operations - stricter limits
    "/forwarding/quotes": EndpointRateLimit(max_requests=50, window_seconds=60),
    "/forwarding/bookings": EndpointRateLimit(max_requests=30, window_seconds=60),
    "/forwarding/shipments": EndpointRateLimit(max_requests=30, window_seconds=60, burst=5),
    "/forwarding/containers": EndpointRateLimit(max_requests=50, window_seconds=60),
    
    # Document operations - moderate limits
    "/documents": EndpointRateLimit(max_requests=100, window_seconds=60),
    "/documents/": EndpointRateLimit(max_requests=200, window_seconds=60),
    
    # Invoice operations
    "/invoices": EndpointRateLimit(max_requests=100, window_seconds=60),
    "/invoices/": EndpointRateLimit(max_requests=200, window_seconds=60),
    
    # Alert operations
    "/alerts": EndpointRateLimit(max_requests=100, window_seconds=60),
    "/alerts/": EndpointRateLimit(max_requests=200, window_seconds=60),
}


def get_rate_limit_for_endpoint(endpoint_path: str) -> Optional[EndpointRateLimit]:
    """
    Get rate limit configuration for an endpoint
    
    Args:
        endpoint_path: Endpoint path (e.g., "/forwarding/quotes")
        
    Returns:
        Rate limit configuration or None if not configured
    """
    # Try exact match first
    if endpoint_path in ENDPOINT_RATE_LIMITS:
        return ENDPOINT_RATE_LIMITS[endpoint_path]
    
    # Try pattern matching (for paths with parameters)
    for pattern, rate_limit in ENDPOINT_RATE_LIMITS.items():
        if pattern.endswith("/") and endpoint_path.startswith(pattern):
            return rate_limit
        if pattern.endswith("{") and endpoint_path.startswith(pattern.split("{")[0]):
            return rate_limit
    
    return None


def get_default_rate_limit() -> EndpointRateLimit:
    """Get default rate limit configuration"""
    return EndpointRateLimit(max_requests=100, window_seconds=60)

