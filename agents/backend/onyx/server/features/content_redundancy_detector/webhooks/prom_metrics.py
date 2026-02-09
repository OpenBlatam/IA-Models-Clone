"""
Prometheus instrumentation for the webhooks subsystem (optional).

All functions are safe to import/use even if prometheus_client is missing.
"""

from typing import Optional

try:
	from prometheus_client import Counter, Histogram, Gauge  # type: ignore
	PROM_AVAILABLE = True
except Exception:
	PROM_AVAILABLE = False
	Counter = Histogram = Gauge = None  # type: ignore

# Metric definitions (created only if available)
if PROM_AVAILABLE:
	WEBHOOK_DELIVERIES = Counter(
		"webhook_deliveries_total",
		"Total webhook deliveries processed",
		["status", "event", "endpoint_id"],
	)
	WEBHOOK_DELIVERY_SECONDS = Histogram(
		"webhook_delivery_seconds",
		"Webhook delivery duration in seconds",
		buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
		labelnames=("event", "endpoint_id"),
	)
	WEBHOOK_QUEUE_SIZE = Gauge(
		"webhook_queue_size",
		"Current size of the webhook delivery queue",
	)
	WEBHOOK_CIRCUIT_BREAKER_STATE = Gauge(
		"webhook_circuit_breaker_state",
		"Circuit breaker state per endpoint (0=closed,1=half_open,2=open)",
		["endpoint_id"],
	)
    WEBHOOK_RETRIES_TOTAL = Counter(
        "webhook_retries_total",
        "Total webhook retry attempts",
        ["endpoint_id", "reason"],
    )
    WEBHOOK_ERRORS_TOTAL = Counter(
        "webhook_errors_total",
        "Total webhook errors by type",
        ["endpoint_id", "error_type"],
    )
    WEBHOOK_WORKER_DELIVERIES_TOTAL = Counter(
        "webhook_worker_deliveries_total",
        "Total deliveries processed per worker",
        ["worker_id"],
    )
    WEBHOOK_TARGET_STATUS_TOTAL = Counter(
        "webhook_target_status_total",
        "Total responses by target status code",
        ["endpoint_id", "status_code"],
    )
    WEBHOOK_PAYLOAD_BYTES = Histogram(
        "webhook_payload_bytes",
        "Size of webhook payloads in bytes",
        buckets=(128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 131072, 262144, 524288, 1048576),
        labelnames=("endpoint_id", "event"),
    )
else:
	WEBHOOK_DELIVERIES = None  # type: ignore
	WEBHOOK_DELIVERY_SECONDS = None  # type: ignore
	WEBHOOK_QUEUE_SIZE = None  # type: ignore
	WEBHOOK_CIRCUIT_BREAKER_STATE = None  # type: ignore
    WEBHOOK_RETRIES_TOTAL = None  # type: ignore
    WEBHOOK_ERRORS_TOTAL = None  # type: ignore
    WEBHOOK_WORKER_DELIVERIES_TOTAL = None  # type: ignore
    WEBHOOK_TARGET_STATUS_TOTAL = None  # type: ignore
    WEBHOOK_PAYLOAD_BYTES = None  # type: ignore


def observe_delivery(status: str, event: str, endpoint_id: str, duration_seconds: float) -> None:
	"""Record a webhook delivery event into Prometheus metrics (no-op if unavailable)."""
	if not PROM_AVAILABLE:
		return
	try:
		WEBHOOK_DELIVERIES.labels(status=status, event=event, endpoint_id=endpoint_id).inc()
		WEBHOOK_DELIVERY_SECONDS.labels(event=event, endpoint_id=endpoint_id).observe(max(0.0, duration_seconds))
	except Exception:
		# Never raise from metrics path
		pass


def set_queue_size(size: int) -> None:
	"""Set current queue size gauge (no-op if unavailable)."""
	if not PROM_AVAILABLE:
		return
	try:
		WEBHOOK_QUEUE_SIZE.set(float(size))
	except Exception:
		pass


def set_circuit_breaker_state(endpoint_id: str, state: str) -> None:
	"""Set circuit breaker state gauge (no-op if unavailable)."""
	if not PROM_AVAILABLE:
		return
	try:
		value = {"closed": 0, "half_open": 1, "open": 2}.get(state, 0)
		WEBHOOK_CIRCUIT_BREAKER_STATE.labels(endpoint_id=endpoint_id).set(value)
	except Exception:
		pass


def inc_retry(endpoint_id: str, reason: str) -> None:
    """Increment retry counter (no-op if unavailable)."""
    if not PROM_AVAILABLE:
        return
    try:
        WEBHOOK_RETRIES_TOTAL.labels(endpoint_id=endpoint_id, reason=reason).inc()
    except Exception:
        pass


def inc_error(endpoint_id: str, error_type: str) -> None:
    """Increment error counter (no-op if unavailable)."""
    if not PROM_AVAILABLE:
        return
    try:
        WEBHOOK_ERRORS_TOTAL.labels(endpoint_id=endpoint_id, error_type=error_type).inc()
    except Exception:
        pass


def inc_worker_delivery(worker_id: str) -> None:
    """Increment per-worker deliveries counter (no-op if unavailable)."""
    if not PROM_AVAILABLE:
        return
    try:
        WEBHOOK_WORKER_DELIVERIES_TOTAL.labels(worker_id=worker_id).inc()
    except Exception:
        pass


def inc_target_status(endpoint_id: str, status_code: int) -> None:
    """Increment counter for target HTTP status codes (no-op if unavailable)."""
    if not PROM_AVAILABLE:
        return
    try:
        WEBHOOK_TARGET_STATUS_TOTAL.labels(endpoint_id=endpoint_id, status_code=str(status_code)).inc()
    except Exception:
        pass


def observe_payload_bytes(endpoint_id: str, event: str, size_bytes: int) -> None:
    """Observe payload size in bytes (no-op if unavailable)."""
    if not PROM_AVAILABLE:
        return
    try:
        WEBHOOK_PAYLOAD_BYTES.labels(endpoint_id=endpoint_id, event=event).observe(float(max(0, size_bytes)))
    except Exception:
        pass
