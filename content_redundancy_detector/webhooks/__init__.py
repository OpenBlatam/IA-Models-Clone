"""
Webhook Manager
Lightweight manager to register webhooks and dispatch events with retries and HMAC verification.
"""

from __future__ import annotations

import asyncio
import hmac
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


SUPPORTED_EVENTS = {
    "analysis.completed",
    "similarity.completed",
    "quality.completed",
    "batch.completed",
}


@dataclass
class WebhookSubscription:
    url: str
    events: List[str]
    secret: Optional[str] = None
    timeout_s: int = 30
    retry_attempts: int = 3
    backoff_s: float = 1.5

    def is_event_supported(self, event: str) -> bool:
        return event in self.events and event in SUPPORTED_EVENTS


class WebhookManager:
    def __init__(self) -> None:
        self._subscriptions: List[WebhookSubscription] = []
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._client is None:
            # HTTP/1.1 keep-alive with sensible limits
            limits = httpx.Limits(max_keepalive_connections=10, max_connections=50)
            self._client = httpx.AsyncClient(timeout=None, limits=limits)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def subscribe(self, url: str, events: List[str], secret: Optional[str] = None,
                        timeout_s: int = 30, retry_attempts: int = 3) -> Dict[str, Any]:
        filtered = [e for e in events if e in SUPPORTED_EVENTS]
        if not filtered:
            return {"success": False, "error": "No valid events provided"}
        sub = WebhookSubscription(
            url=url,
            events=filtered,
            secret=secret,
            timeout_s=timeout_s,
            retry_attempts=retry_attempts,
        )
        async with self._lock:
            self._subscriptions.append(sub)
        return {"success": True, "data": {"url": url, "events": filtered}}

    async def unsubscribe(self, url: str, events: Optional[List[str]] = None) -> Dict[str, Any]:
        async with self._lock:
            if events is None:
                before = len(self._subscriptions)
                self._subscriptions = [s for s in self._subscriptions if s.url != url]
                removed = before - len(self._subscriptions)
                return {"success": True, "data": {"removed": removed}}
            else:
                removed = 0
                keep: List[WebhookSubscription] = []
                for s in self._subscriptions:
                    if s.url != url:
                        keep.append(s)
                        continue
                    remaining = [e for e in s.events if e not in events]
                    if remaining:
                        s.events = remaining
                        keep.append(s)
                    else:
                        removed += 1
                self._subscriptions = keep
                return {"success": True, "data": {"removed": removed}}

    async def list_subscriptions(self) -> List[Dict[str, Any]]:
        return [
            {"url": s.url, "events": s.events, "has_secret": bool(s.secret)}
            for s in self._subscriptions
        ]

    def _signature(self, secret: str, payload_bytes: bytes) -> str:
        mac = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256)
        return mac.hexdigest()

    async def _post_with_retries(self, sub: WebhookSubscription, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert self._client is not None, "WebhookManager not started"
        attempt = 0
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Content-Redundancy-Detector/2.0"
        }
        if sub.secret:
            headers["X-Webhook-Signature"] = self._signature(sub.secret, payload_bytes)
        last_error: Optional[str] = None
        backoff = 0.5
        while attempt <= sub.retry_attempts:
            try:
                resp = await self._client.post(sub.url, content=payload_bytes, headers=headers, timeout=sub.timeout_s)
                if 200 <= resp.status_code < 300:
                    return {"success": True, "status_code": resp.status_code}
                last_error = f"HTTP {resp.status_code}"
            except Exception as e:
                last_error = str(e)
            attempt += 1
            if attempt <= sub.retry_attempts:
                await asyncio.sleep(backoff)
                backoff *= sub.backoff_s
        return {"success": False, "error": last_error}

    async def send_webhook(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a webhook event to all matching subscriptions."""
        if event_type not in SUPPORTED_EVENTS:
            return {"success": False, "error": f"Unsupported event: {event_type}"}

        # snapshot to avoid holding the lock during IO
        async with self._lock:
            targets = [s for s in self._subscriptions if s.is_event_supported(event_type)]
        if not targets:
            return {"success": True, "data": {"delivered": 0}}

        payload = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data,
        }
        results = await asyncio.gather(
            *[self._post_with_retries(s, payload) for s in targets], return_exceptions=True
        )
        delivered = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        return {"success": True, "data": {"delivered": delivered, "attempted": len(targets)}}


# Global instance
webhook_manager = WebhookManager()

"""
High-performance Webhooks package
"""

from .dispatcher import WebhookDispatcher, get_webhook_dispatcher
from .router import router as webhooks_router
from .workers import start_webhook_workers, stop_webhook_workers
from .optimization import setup_fastapi_optimization, run_optimized_server

__all__ = [
    "WebhookDispatcher",
    "get_webhook_dispatcher",
    "webhooks_router",
    "start_webhook_workers",
    "stop_webhook_workers",
    "setup_fastapi_optimization",
    "run_optimized_server",
]

"""
Webhooks Module - Modular Architecture
Advanced webhook system for real-time notifications
Optimized for microservices, serverless, and cloud-native deployments

Modular Structure:
- core/: Core manager and configuration
- api/: Public API functions
- factory/: Factory functions for creating instances
- extensions/: Optional features and extensions

Example Usage:
    ```python
    from webhooks import send_webhook, WebhookEvent
    
    # Send a webhook
    result = await send_webhook(
        event=WebhookEvent.ANALYSIS_COMPLETED,
        data={"content_id": "123", "status": "completed"}
    )
    ```
"""

import logging
from typing import TYPE_CHECKING, Optional, Any

# Core models and infrastructure classes
from .models import (
    WebhookEvent,
    WebhookPayload,
    WebhookEndpoint,
    WebhookDelivery
)
from .circuit_breaker import CircuitBreaker
from .storage import (
    StorageFactory,
    StorageBackend,
    RedisStorageBackend,
    InMemoryStorageBackend
)
from .observability import ObservabilityManager, observability_manager
from .config import WebhookConfig

# Core manager (via factory pattern)
from .factory import (
    create_webhook_manager,
    get_default_webhook_manager,
    reset_default_manager
)
from .core import WebhookManager

logger = logging.getLogger(__name__)

# Public API functions
from .api import (
    send_webhook,
    register_webhook_endpoint,
    unregister_webhook_endpoint,
    get_webhook_endpoints,
    get_webhook_deliveries,
    get_webhook_stats,
    get_webhook_health,
    get_rate_limit_status,
    configure_rate_limit
)

# Optional extensions (with graceful fallbacks)
_import_errors = []
try:
    from .extensions import (
        # Validators
        WebhookValidator,
        validate_webhook_endpoint,
        VALIDATORS_AVAILABLE,
        # Rate Limiting
        RateLimiter,
        RateLimitConfig,
        RATE_LIMITING_AVAILABLE,
        # Health Monitoring
        WebhookHealthChecker,
        HealthStatus,
        HealthCheckResult,
        health_checker,
        check_storage_health,
        check_queue_health,
        check_workers_health,
        HEALTH_MONITORING_AVAILABLE,
        # Enhanced Delivery
        EnhancedWebhookDelivery,
        DeliveryStatus,
        DeliveryMetrics,
        RetryStrategy,
        WebhookQueue,
        ENHANCED_DELIVERY_AVAILABLE,
        # Metrics
        WebhookMetricsCollector,
        DeliveryMetric,
        metrics_collector,
        METRICS_AVAILABLE,
        # Utils
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
        sanitize_endpoint_url,
        UTILS_AVAILABLE,
    )
except ImportError as e:
    # Fallback if extensions module not available
    _import_errors.append(f"extensions: {e}")
    logger.debug(f"Extensions module not available: {e}")
    VALIDATORS_AVAILABLE = False
    RATE_LIMITING_AVAILABLE = False
    HEALTH_MONITORING_AVAILABLE = False
    ENHANCED_DELIVERY_AVAILABLE = False
    METRICS_AVAILABLE = False
    UTILS_AVAILABLE = False
    WebhookValidator = None
    validate_webhook_endpoint = None
    RateLimiter = None
    RateLimitConfig = None
    WebhookHealthChecker = None
    HealthStatus = None
    HealthCheckResult = None
    health_checker = None
    check_storage_health = None
    check_queue_health = None
    check_workers_health = None
    EnhancedWebhookDelivery = None
    DeliveryStatus = None
    DeliveryMetrics = None
    RetryStrategy = None
    WebhookQueue = None
    WebhookMetricsCollector = None
    DeliveryMetric = None
    metrics_collector = None
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

# Prometheus instrumentation (optional)
try:
    from .prom_metrics import (
        PROM_AVAILABLE as PROMETHEUS_AVAILABLE,
        observe_delivery,
        set_queue_size,
        set_circuit_breaker_state,
        inc_retry,
        inc_error,
        inc_worker_delivery,
        inc_target_status,
        observe_payload_bytes,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False
    def observe_delivery(*args, **kwargs):
        return None
    def set_queue_size(*args, **kwargs):
        return None
    def set_circuit_breaker_state(*args, **kwargs):
        return None
    def inc_retry(*args, **kwargs):
        return None
    def inc_error(*args, **kwargs):
        return None
    def inc_worker_delivery(*args, **kwargs):
        return None
    def inc_target_status(*args, **kwargs):
        return None
    def observe_payload_bytes(*args, **kwargs):
        return None
# Integration with microservices patterns (optional)
try:
    from .integration import (
        wrap_webhook_delivery_with_resilience,
        record_webhook_metrics,
        RESILIENCE_AVAILABLE
    )
    INTEGRATION_AVAILABLE = True
    logger.debug("Integration module loaded successfully")
except ImportError as e:
    _import_errors.append(f"integration: {e}")
    logger.debug(f"Integration module not available: {e}")
    INTEGRATION_AVAILABLE = False
    RESILIENCE_AVAILABLE = False
    wrap_webhook_delivery_with_resilience = None
    record_webhook_metrics = None

# Default webhook manager instance (lazy initialization via factory)
# Created on first access to avoid circular imports and initialization issues
def _initialize_webhook_manager() -> WebhookManager:
    """Initialize webhook manager with resilience patterns if available"""
    manager = get_default_webhook_manager()
    
    # Apply resilience patterns if available
    if INTEGRATION_AVAILABLE and RESILIENCE_AVAILABLE:
        try:
            # Wrap delivery method with circuit breaker and retry
            original_deliver = manager.deliver_webhook
            
            async def resilient_deliver(*args, **kwargs):
                """Wrapped delivery with resilience patterns"""
                return await wrap_webhook_delivery_with_resilience(original_deliver)(*args, **kwargs)
            
            manager.deliver_webhook = resilient_deliver
            logger.debug("Resilience patterns applied to webhook manager")
        except Exception as e:
            logger.warning(f"Failed to apply resilience patterns: {e}")
    
    return manager


# Lazy initialization proxy
if TYPE_CHECKING:
    from typing import Any

class _WebhookManagerProxy:
    """Proxy for lazy initialization of webhook manager"""
    _instance: Optional[WebhookManager] = None
    
    def __getattr__(self, name: str) -> Any:
        """Lazy load webhook manager on first attribute access"""
        if self._instance is None:
            self._instance = _initialize_webhook_manager()
        return getattr(self._instance, name)

webhook_manager = _WebhookManagerProxy()


__all__ = [
    # Models
    "WebhookEvent",
    "WebhookPayload",
    "WebhookEndpoint",
    "WebhookDelivery",
    # Core Classes
    "CircuitBreaker",
    "WebhookManager",
    "StorageBackend",
    "RedisStorageBackend",
    "InMemoryStorageBackend",
    "StorageFactory",
    "ObservabilityManager",
    "WebhookConfig",
    # Factory
    "create_webhook_manager",
    "get_default_webhook_manager",
    "reset_default_manager",
    # Instances
    "webhook_manager",
    "observability_manager",
    # Public API
    "send_webhook",
    "register_webhook_endpoint",
    "unregister_webhook_endpoint",
    "get_webhook_endpoints",
    "get_webhook_deliveries",
    "get_webhook_stats",
    "get_webhook_health",
    "get_rate_limit_status",
    "configure_rate_limit",
    "get_metrics_summary",
    "get_analytics_report",
    "get_cache_stats",
    "get_system_health",
    # Optional Extensions
    "WebhookValidator",
    "validate_webhook_endpoint",
    "RateLimiter",
    "RateLimitConfig",
    "WebhookHealthChecker",
    "HealthStatus",
    "HealthCheckResult",
    "health_checker",
    "check_storage_health",
    "check_queue_health",
    "check_workers_health",
    "EnhancedWebhookDelivery",
    "DeliveryStatus",
    "DeliveryMetrics",
    "RetryStrategy",
    "WebhookQueue",
    "WebhookMetricsCollector",
    "DeliveryMetric",
    "metrics_collector",
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
    # Prometheus helpers
    "PROMETHEUS_AVAILABLE",
    "observe_delivery",
    "set_queue_size",
    "set_circuit_breaker_state",
    "inc_retry",
    "inc_error",
    "inc_worker_delivery",
    "inc_target_status",
    "observe_payload_bytes",
    # Availability Flags
    "VALIDATORS_AVAILABLE",
    "RATE_LIMITING_AVAILABLE",
    "HEALTH_MONITORING_AVAILABLE",
    "ENHANCED_DELIVERY_AVAILABLE",
    "METRICS_AVAILABLE",
    "UTILS_AVAILABLE",
    "INTEGRATION_AVAILABLE",
    "RESILIENCE_AVAILABLE",
    "wrap_webhook_delivery_with_resilience",
    "record_webhook_metrics",
]
