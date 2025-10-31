"""
Webhooks Extensions Module
Optional features and extensions with graceful fallbacks
"""

from typing import Optional

# Try to import optional extensions
try:
    from .validators import WebhookValidator, validate_webhook_endpoint
    VALIDATORS_AVAILABLE = True
except ImportError:
    VALIDATORS_AVAILABLE = False
    WebhookValidator = None
    validate_webhook_endpoint = None

try:
    from .rate_limiting import RateLimiter, RateLimitConfig
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False
    RateLimiter = None
    RateLimitConfig = None

try:
    from .health import (
        WebhookHealthChecker,
        HealthStatus,
        HealthCheckResult,
        health_checker,
        check_storage_health,
        check_queue_health,
        check_workers_health
    )
    HEALTH_MONITORING_AVAILABLE = True
except ImportError:
    HEALTH_MONITORING_AVAILABLE = False
    WebhookHealthChecker = None
    HealthStatus = None
    HealthCheckResult = None
    health_checker = None
    check_storage_health = None
    check_queue_health = None
    check_workers_health = None

try:
    from .enhanced_delivery import (
        EnhancedWebhookDelivery,
        DeliveryStatus,
        DeliveryMetrics,
        RetryStrategy,
        WebhookQueue
    )
    ENHANCED_DELIVERY_AVAILABLE = True
except ImportError:
    ENHANCED_DELIVERY_AVAILABLE = False
    EnhancedWebhookDelivery = None
    DeliveryStatus = None
    DeliveryMetrics = None
    RetryStrategy = None
    WebhookQueue = None

try:
    from .metrics import (
        WebhookMetricsCollector,
        DeliveryMetric,
        metrics_collector
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    WebhookMetricsCollector = None
    DeliveryMetric = None
    metrics_collector = None

try:
    from .utils_functions import (
        generate_webhook_signature,
        verify_webhook_signature,
        normalize_endpoint_url,
        calculate_retry_delay,
        should_retry,
        create_webhook_headers,
        parse_webhook_headers,
        format_webhook_payload,
        is_valid_webhook_url,
        calculate_payload_size,
        sanitize_endpoint_url
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    generate_webhook_signature = None
    verify_webhook_signature = None
    normalize_endpoint_url = None
    calculate_retry_delay = None
    should_retry = None
    create_webhook_headers = None
    parse_webhook_headers = None
    format_webhook_payload = None
    is_valid_webhook_url = None
    calculate_payload_size = None
    sanitize_endpoint_url = None

__all__ = [
    # Validators
    "WebhookValidator",
    "validate_webhook_endpoint",
    "VALIDATORS_AVAILABLE",
    # Rate Limiting
    "RateLimiter",
    "RateLimitConfig",
    "RATE_LIMITING_AVAILABLE",
    # Health Monitoring
    "WebhookHealthChecker",
    "HealthStatus",
    "HealthCheckResult",
    "health_checker",
    "check_storage_health",
    "check_queue_health",
    "check_workers_health",
    "HEALTH_MONITORING_AVAILABLE",
    # Enhanced Delivery
    "EnhancedWebhookDelivery",
    "DeliveryStatus",
    "DeliveryMetrics",
    "RetryStrategy",
    "WebhookQueue",
    "ENHANCED_DELIVERY_AVAILABLE",
    # Metrics
    "WebhookMetricsCollector",
    "DeliveryMetric",
    "metrics_collector",
    "METRICS_AVAILABLE",
    # Utils
    "generate_webhook_signature",
    "verify_webhook_signature",
    "normalize_endpoint_url",
    "calculate_retry_delay",
    "should_retry",
    "create_webhook_headers",
    "parse_webhook_headers",
    "format_webhook_payload",
    "is_valid_webhook_url",
    "calculate_payload_size",
    "sanitize_endpoint_url",
    "UTILS_AVAILABLE",
]






