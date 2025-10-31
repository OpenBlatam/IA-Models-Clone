"""
Webhook Manager - Main orchestration and management
Optimized for microservices, serverless, and cloud-native deployments
"""

import asyncio
import logging
import time
import uuid
import os
from typing import Dict, Any, List, Optional
from collections import defaultdict

import httpx

from .models import WebhookEvent, WebhookEndpoint, WebhookDelivery, WebhookPayload
from .circuit_breaker import CircuitBreaker
from .delivery import WebhookDeliveryService
from .storage import StorageFactory, StorageBackend
from .observability import observability_manager
# Optional Prometheus helpers (no-op if unavailable)
try:
    from .prom_metrics import (
        observe_delivery as prom_observe_delivery,
        set_queue_size as prom_set_queue_size,
        set_circuit_breaker_state as prom_set_cb_state,
        inc_retry as prom_inc_retry,
        inc_error as prom_inc_error,
        inc_worker_delivery as prom_inc_worker_delivery,
        inc_target_status as prom_inc_target_status,
        observe_payload_bytes as prom_observe_payload_bytes,
    )  # type: ignore
except Exception:
    def prom_observe_delivery(*args, **kwargs):
        return None
    def prom_set_queue_size(*args, **kwargs):
        return None
    def prom_set_cb_state(*args, **kwargs):
        return None
    def prom_inc_retry(*args, **kwargs):
        return None
    def prom_inc_error(*args, **kwargs):
        return None
    def prom_inc_worker_delivery(*args, **kwargs):
        return None
    def prom_inc_target_status(*args, **kwargs):
        return None
    def prom_observe_payload_bytes(*args, **kwargs):
        return None
from .config import WebhookConfig
try:
    from ..core.telemetry import set_request_context_attributes  # type: ignore
except Exception:
    def set_request_context_attributes(*args, **kwargs):
        return None
from .validators import WebhookValidator
from .rate_limiter import RateLimiter, RateLimitConfig
from .health import WebhookHealthChecker, HealthStatus

logger = logging.getLogger(__name__)


class WebhookManager:
    """
    Advanced webhook management system optimized for microservices and serverless
    
    Features:
    - Stateless design with external storage (Redis)
    - OpenTelemetry distributed tracing
    - Prometheus metrics
    - Cold start optimization for serverless
    - Circuit breaker pattern
    - Async worker pool
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        max_queue_size: int = 1000,
        storage_backend: Optional[StorageBackend] = None,
        enable_tracing: bool = True,
        enable_metrics: bool = True
    ):
        """
        Initialize webhook manager
        
        Args:
            max_workers: Maximum number of worker threads (auto-detected for serverless)
            max_queue_size: Maximum size of delivery queue
            storage_backend: Storage backend for stateless persistence
            enable_tracing: Enable OpenTelemetry tracing
            enable_metrics: Enable Prometheus metrics
        """
        # Serverless optimization: minimal workers for Lambda/Functions
        if max_workers is None:
            max_workers = WebhookConfig.detect_max_workers()
        
        self._max_workers = max_workers
        self._max_queue_size = max_queue_size
        
        # Stateless storage (Redis or in-memory)
        if storage_backend:
            self._storage = storage_backend
        else:
            self._storage = StorageFactory.create(
                storage_type=WebhookConfig.STORAGE_TYPE,
                redis_url=WebhookConfig.REDIS_URL,
                **WebhookConfig.get_redis_config()
            )
        
        # Initialize observability with config
        self._observability = observability_manager
        if not enable_tracing:
            self._observability.enable_tracing = False
        if not enable_metrics:
            self._observability.enable_metrics = False
        
        # In-memory state (backed by storage)
        self._endpoints: Dict[str, WebhookEndpoint] = {}
        self._deliveries: Dict[str, WebhookDelivery] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = defaultdict(lambda: CircuitBreaker())
        
        # Async components
        self._http_client: Optional[httpx.AsyncClient] = None
        self._delivery_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._is_running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._delivery_service: Optional[WebhookDeliveryService] = None
        
        # Metrics (also tracked in Prometheus)
        self._metrics = {
            "total_deliveries": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "circuit_breaker_tripped": 0,
            "queue_full_errors": 0,
            "average_delivery_time": 0.0,
        }
    
    @staticmethod
    def _detect_optimal_workers() -> int:
        """
        Detect optimal number of workers based on environment (deprecated, use WebhookConfig)
        
        Returns:
            Optimal worker count
        """
        return WebhookConfig.detect_max_workers()
    
    async def start(self) -> None:
        """
        Start webhook delivery system with worker pool
        Optimized for cold start in serverless environments
        """
        if self._is_running:
            return
        
        # Load state from storage (stateless recovery)
        await self._load_state_from_storage()
        
        self._is_running = True
        self._start_time = time.time()
        
        # Configure HTTP client optimized for serverless
        http_config = WebhookConfig.get_http_client_config()
        
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                http_config["timeout"],
                connect=http_config["connect_timeout"]
            ),
            limits=httpx.Limits(
                max_keepalive_connections=http_config["max_keepalive_connections"],
                max_connections=http_config["max_connections"]
            ),
            follow_redirects=http_config["follow_redirects"]
        )
        
        # Initialize delivery service
        self._delivery_service = WebhookDeliveryService(self._http_client)
        
        # Start multiple delivery workers for parallel processing
        self._worker_tasks = [
            asyncio.create_task(self._delivery_worker(f"worker-{i}"))
            for i in range(self._max_workers)
        ]
        
        is_serverless = WebhookConfig.is_serverless()
        logger.info(
            f"Webhook system started with {self._max_workers} workers "
            f"(serverless={is_serverless}, storage={type(self._storage).__name__}, "
            f"tracing={self._observability.enable_tracing}, metrics={self._observability.enable_metrics})"
        )
    
    async def _load_state_from_storage(self) -> None:
        """Load state from external storage (stateless recovery)"""
        try:
            # Load endpoints
            endpoints_data = await self._storage.get("webhooks:endpoints")
            if endpoints_data:
                for endpoint_data in endpoints_data:
                    # Reconstruct endpoints from storage
                    endpoint = WebhookEndpoint(**endpoint_data)
                    self._endpoints[endpoint.id] = endpoint
            
            logger.debug(f"Loaded {len(self._endpoints)} endpoints from storage")
        except Exception as e:
            logger.warning(f"Failed to load state from storage: {e}")
    
    async def _save_state_to_storage(self) -> None:
        """Save state to external storage"""
        try:
            # Save endpoints
            endpoints_data = [
                {
                    "id": ep.id,
                    "url": ep.url,
                    "events": [e.value for e in ep.events],
                    "secret": ep.secret,
                    "timeout": ep.timeout,
                    "retry_count": ep.retry_count,
                    "is_active": ep.is_active,
                    "created_at": ep.created_at,
                }
                for ep in self._endpoints.values()
            ]
            await self._storage.set("webhooks:endpoints", endpoints_data)
        except Exception as e:
            logger.warning(f"Failed to save state to storage: {e}")
    
    async def stop(self) -> None:
        """Stop webhook delivery system gracefully"""
        self._is_running = False
        
        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        # Close HTTP client
        if self._http_client:
            await self._http_client.aclose()
        
        logger.info("Webhook system stopped gracefully")
    
    async def register_endpoint(self, endpoint: WebhookEndpoint) -> None:
        """Register a webhook endpoint (stateless)"""
        # Validate endpoint using improved validator
        from .validators import validate_webhook_endpoint
        
        is_valid, error_msg = validate_webhook_endpoint(
            endpoint,
            allow_localhost=os.getenv("WEBHOOK_ALLOW_LOCALHOST", "false").lower() == "true"
        )
        if not is_valid:
            raise ValueError(f"Invalid webhook endpoint: {error_msg}")
        
        # Sanitize endpoint ID if needed
        try:
            from .validators import WebhookValidator
            sanitized_id = WebhookValidator.sanitize_endpoint_id(endpoint.id)
            if sanitized_id != endpoint.id:
                # Create new endpoint with sanitized ID
                endpoint = WebhookEndpoint(
                    id=sanitized_id,
                    url=endpoint.url.rstrip('/'),  # Normalize URL
                    events=endpoint.events,
                    secret=endpoint.secret,
                    timeout=max(WebhookValidator.MIN_TIMEOUT, min(endpoint.timeout, WebhookValidator.MAX_TIMEOUT)),
                    retry_count=max(WebhookValidator.MIN_RETRY_COUNT, min(endpoint.retry_count, WebhookValidator.MAX_RETRY_COUNT)),
                    is_active=endpoint.is_active
                )
        except Exception as e:
            logger.warning(f"Could not sanitize endpoint ID: {e}")
        
        # Register endpoint
        self._endpoints[endpoint.id] = endpoint
        await self._save_state_to_storage()
        
        # Configure rate limiter for endpoint
        self._rate_limiter.configure_endpoint(
            endpoint.id,
            max_requests=100,
            window_seconds=60
        )
        
        logger.info(f"Webhook endpoint registered: {endpoint.id}")
    
    def register_endpoint_sync(self, endpoint: WebhookEndpoint) -> None:
        """Synchronous wrapper for backward compatibility"""
        self._endpoints[endpoint.id] = endpoint
        # Save async in background
        asyncio.create_task(self._save_state_to_storage())
        logger.info(f"Webhook endpoint registered: {endpoint.id}")
    
    def unregister_endpoint(self, endpoint_id: str) -> bool:
        """Unregister a webhook endpoint"""
        if endpoint_id in self._endpoints:
            del self._endpoints[endpoint_id]
            # Clean up circuit breaker
            if endpoint_id in self._circuit_breakers:
                del self._circuit_breakers[endpoint_id]
            logger.info(f"Webhook endpoint unregistered: {endpoint_id}")
            return True
        return False
    
    async def send_webhook(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
        request_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send webhook for an event with improved error handling
        
        Args:
            event: Webhook event type
            data: Event data payload
            request_id: Optional request ID for tracking
            user_id: Optional user ID for tracking
            
        Returns:
            Dictionary with status information
        """
        if not self._is_running:
            logger.warning("Webhook system not running")
            return {"status": "error", "message": "Webhook system not running"}
        
        # Find endpoints that should receive this event
        target_endpoints = [
            endpoint for endpoint in self._endpoints.values()
            if endpoint.is_active and event in endpoint.events
        ]
        
        if not target_endpoints:
            logger.debug(f"No webhook endpoints for event: {event.value}")
            return {"status": "skipped", "message": "No endpoints registered for event"}
        
        # Validate event using improved validator
        is_valid_event, event_error = WebhookValidator.validate_event_type(event)
        if not is_valid_event:
            logger.error(f"Invalid webhook event: {event_error}")
            return {
                "status": "failed",
                "error": event_error,
                "queued": 0,
                "total": 0
            }
        
        # Validate payload using improved validator
        is_valid_payload, payload_error = WebhookValidator.validate_payload_size(data)
        if not is_valid_payload:
            logger.error(f"Invalid webhook payload: {payload_error}")
            return {
                "status": "failed",
                "error": payload_error,
                "queued": 0,
                "total": 0
            }
        
        # Create payload with improved structure
        payload = WebhookPayload(
            event=event.value if hasattr(event, 'value') else str(event),
            timestamp=time.time(),
            data=data,
            request_id=request_id or str(uuid.uuid4()),
            user_id=user_id
        )
        
        queued_count = 0
        errors = []
        
        # Update queue size metric (observability + Prometheus)
        queue_size = self._delivery_queue.qsize()
        self._observability.update_queue_size(queue_size)
        prom_set_queue_size(queue_size)
        
        # Create deliveries for each endpoint
        for endpoint in target_endpoints:
            try:
                # Check rate limit
                is_allowed, rate_limit_error = await self._rate_limiter.is_allowed(endpoint.id)
                if not is_allowed:
                    logger.warning(f"Rate limit exceeded for endpoint: {endpoint.id} - {rate_limit_error}")
                    errors.append(f"Rate limit exceeded for {endpoint.id}")
                    try:
                        prom_inc_error(endpoint.id, "rate_limit")
                    except Exception:
                        pass
                    continue
                
                # Check circuit breaker
                circuit_breaker = self._circuit_breakers[endpoint.id]
                if not circuit_breaker.can_proceed():
                    logger.warning(f"Circuit breaker open for endpoint: {endpoint.id}")
                    self._metrics["circuit_breaker_tripped"] += 1
                    errors.append(f"Circuit breaker open for {endpoint.id}")
                    continue
                
                delivery = WebhookDelivery(
                    id=f"{event.value}_{endpoint.id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
                    endpoint_id=endpoint.id,
                    event=event.value,
                    payload=payload
                )
                
                self._deliveries[delivery.id] = delivery
                
                # Try to queue (non-blocking with timeout)
                try:
                    self._delivery_queue.put_nowait(delivery)
                    queued_count += 1
                except asyncio.QueueFull:
                    logger.error(f"Webhook queue full, dropping delivery: {delivery.id}")
                    self._metrics["queue_full_errors"] += 1
                    errors.append(f"Queue full for {endpoint.id}")
                    try:
                        prom_inc_error(endpoint.id, "queue_full")
                    except Exception:
                        pass
                
            except Exception as e:
                logger.error(f"Error queuing webhook for {endpoint.id}: {e}")
                errors.append(f"Error queuing for {endpoint.id}: {str(e)}")
        
        logger.info(f"Webhook queued for {queued_count}/{len(target_endpoints)} endpoints: {event.value}")
        
        return {
            "status": "queued" if queued_count > 0 else "failed",
            "queued": queued_count,
            "total": len(target_endpoints),
            "errors": errors
        }
    
    async def _delivery_worker(self, worker_id: str) -> None:
        """Background worker for webhook delivery with observability"""
        logger.debug(f"Webhook delivery worker {worker_id} started")
        
        # Trace worker lifecycle (simplified approach)
        tracer = self._observability.get_tracer() if self._observability.enable_tracing else None
        
        while self._is_running:
            try:
                # Get delivery from queue with timeout
                delivery = await asyncio.wait_for(self._delivery_queue.get(), timeout=1.0)
                
                # Process delivery (tracing handled inside)
                await self._process_delivery(delivery)
                try:
                    prom_inc_worker_delivery(worker_id)
                except Exception:
                    pass
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info(f"Webhook delivery worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in delivery worker {worker_id}: {e}", exc_info=True)
    
    async def _process_delivery(self, delivery: WebhookDelivery) -> None:
        """Process a webhook delivery using delivery service with observability"""
        endpoint = self._endpoints.get(delivery.endpoint_id)
        if not endpoint:
            logger.warning(f"Endpoint not found: {delivery.endpoint_id}")
            return
        
        circuit_breaker = self._circuit_breakers[endpoint.id]
        
        if not self._delivery_service:
            logger.error("Delivery service not initialized")
            return
        
        # Trace delivery operation (simplified for reliability)
        start_time = time.time()
        # Attach request context to current span if tracing is enabled
        try:
            from opentelemetry import trace  # type: ignore
            span = trace.get_current_span()
            set_request_context_attributes(span, request_id=delivery.payload.request_id)
            # Additional attributes
            if span is not None:
                span.set_attribute("webhook.endpoint_id", delivery.endpoint_id)
                span.set_attribute("webhook.event", delivery.event)
        except Exception:
            pass
        
        # Deliver webhook using delivery service
        result = await self._delivery_service.deliver(delivery, endpoint, circuit_breaker)
        
        delivery_time = time.time() - start_time
        
        # Update metrics based on result
        self._metrics["total_deliveries"] += 1
        status = result.get("status", "unknown")
        
        # Record observability metrics
        self._observability.record_webhook_delivery(
            status=status,
            event_type=delivery.event,
            duration=delivery_time
        )
        # Prometheus delivery metrics
        try:
            prom_observe_delivery(status=status, event=delivery.event, endpoint_id=delivery.endpoint_id, duration_seconds=delivery_time)
        except Exception:
            pass
        # Record payload size if possible
        try:
            import json
            size_bytes = len(json.dumps(delivery.payload.data, default=str).encode("utf-8"))
            prom_observe_payload_bytes(delivery.endpoint_id, delivery.event, size_bytes)
        except Exception:
            pass
        # Record target status code distribution if present
        try:
            status_code = result.get("status_code")
            if isinstance(status_code, int):
                prom_inc_target_status(delivery.endpoint_id, status_code)
        except Exception:
            pass
        
        # Update circuit breaker state metric
        cb_state = circuit_breaker.get_state()
        self._observability.update_circuit_breaker_state(endpoint.id, cb_state)
        try:
            prom_set_cb_state(endpoint.id, cb_state)
        except Exception:
            pass
        
        if status == "delivered":
            self._metrics["successful_deliveries"] += 1
            self._update_delivery_metrics(delivery_time)
            logger.info(f"Webhook delivered successfully: {delivery.id} (took {delivery_time:.3f}s)")
        else:
            self._metrics["failed_deliveries"] += 1
            
            # Re-queue for retry if scheduled
            if status in ("failed_retry_scheduled", "timeout_retry_scheduled", "error_retry_scheduled"):
                try:
                    await self._delivery_queue.put(delivery)
                    try:
                        # Infer reason from status suffix
                        reason = status.replace("_retry_scheduled", "") if status.endswith("_retry_scheduled") else "retry"
                        prom_inc_retry(delivery.endpoint_id, reason)
                    except Exception:
                        pass
                except asyncio.QueueFull:
                    logger.error(f"Queue full, dropping retry for: {delivery.id}")
                    try:
                        prom_inc_error(delivery.endpoint_id, "queue_full")
                    except Exception:
                        pass
            
            error_msg = result.get("error", "Unknown error")
            logger.warning(f"Webhook delivery failed: {delivery.id} - {error_msg}")
    
    @staticmethod
    def _dummy_context():
        """Dummy context manager when tracing is disabled"""
        from contextlib import nullcontext
        return nullcontext()
    
    async def get_health(self) -> HealthStatus:
        """
        Get system health status
        
        Returns:
            HealthStatus object with detailed health information
        """
        return await self._health_checker.check_health()
    
    def get_rate_limit_status(self, endpoint_id: str) -> Dict[str, Any]:
        """
        Get rate limit status for an endpoint
        
        Args:
            endpoint_id: Endpoint identifier
            
        Returns:
            Rate limit status dictionary
        """
        return self._rate_limiter.get_rate_limit_status(endpoint_id)
    
    def configure_rate_limit(
        self,
        endpoint_id: str,
        max_requests: int,
        window_seconds: int = 60,
        burst_allowance: int = 10
    ) -> None:
        """
        Configure rate limit for an endpoint
        
        Args:
            endpoint_id: Endpoint identifier
            max_requests: Maximum requests in window
            window_seconds: Time window in seconds
            burst_allowance: Burst allowance
        """
        self._rate_limiter.configure_endpoint(
            endpoint_id,
            max_requests,
            window_seconds,
            burst_allowance
        )
        logger.info(f"Rate limit configured for {endpoint_id}: {max_requests}/{window_seconds}s")
    
    def get_endpoints(self) -> List[WebhookEndpoint]:
        """Get all registered endpoints"""
        return list(self._endpoints.values())
    
    def get_deliveries(self, limit: int = 100) -> List[WebhookDelivery]:
        """Get recent deliveries"""
        deliveries = list(self._deliveries.values())
        deliveries.sort(key=lambda x: x.created_at, reverse=True)
        return deliveries[:limit]
    
    def _update_delivery_metrics(self, delivery_time: float) -> None:
        """Update delivery time metrics"""
        total = self._metrics["total_deliveries"]
        current_avg = self._metrics["average_delivery_time"]
        self._metrics["average_delivery_time"] = (
            (current_avg * total + delivery_time) / (total + 1)
            if total > 0 else delivery_time
        )
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """Get comprehensive delivery statistics"""
        total_deliveries = len(self._deliveries)
        successful_deliveries = sum(
            1 for d in self._deliveries.values() if d.status == "delivered"
        )
        failed_deliveries = sum(
            1 for d in self._deliveries.values() if d.status == "failed"
        )
        pending_deliveries = sum(
            1 for d in self._deliveries.values() if d.status == "pending"
        )
        
        # Circuit breaker stats
        open_circuits = sum(
            1 for cb in self._circuit_breakers.values() if cb.state == "open"
        )
        half_open_circuits = sum(
            1 for cb in self._circuit_breakers.values() if cb.state == "half_open"
        )
        
        return {
            "total_deliveries": total_deliveries,
            "successful_deliveries": successful_deliveries,
            "failed_deliveries": failed_deliveries,
            "pending_deliveries": pending_deliveries,
            "success_rate": (
                (successful_deliveries / total_deliveries * 100)
                if total_deliveries > 0 else 0
            ),
            "active_endpoints": len([e for e in self._endpoints.values() if e.is_active]),
            "queue_size": self._delivery_queue.qsize(),
            "average_delivery_time": self._metrics["average_delivery_time"],
            "circuit_breakers": {
                "open": open_circuits,
                "half_open": half_open_circuits,
                "tripped_total": self._metrics["circuit_breaker_tripped"]
            },
            "metrics": {
                **self._metrics,
                "queue_full_errors": self._metrics["queue_full_errors"]
            }
        }

