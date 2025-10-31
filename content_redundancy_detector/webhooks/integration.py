"""
Webhooks - Microservices Integration
Integration with resilience patterns, tracing, and monitoring
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import resilience patterns from pdf_variantes or local
try:
    import sys
    import os
    
    # Try local first
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Try pdf_variantes patterns
    pdf_variantes_path = os.path.join(parent_dir, "..", "pdf_variantes")
    if os.path.exists(pdf_variantes_path):
        sys.path.insert(0, os.path.abspath(pdf_variantes_path))
        
        try:
            from core.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
            from core.retry import retry, RetryConfig
            from monitoring.tracing import get_tracing_service
            from monitoring.metrics import get_metrics_collector
            RESILIENCE_AVAILABLE = True
        except ImportError:
            RESILIENCE_AVAILABLE = False
            logger.warning("Resilience patterns not available from pdf_variantes")
    else:
        RESILIENCE_AVAILABLE = False
except Exception:
    RESILIENCE_AVAILABLE = False
    logger.warning("Failed to setup resilience patterns integration")


def wrap_webhook_delivery_with_resilience(delivery_func):
    """Wrap webhook delivery function with circuit breaker and retry"""
    if not RESILIENCE_AVAILABLE:
        return delivery_func
    
    async def resilient_delivery(*args, **kwargs):
        """Delivery function with resilience patterns"""
        # Extract endpoint URL from args (delivery and endpoint objects)
        endpoint_url = "unknown"
        if len(args) >= 2:
            endpoint = args[1]  # Second arg should be WebhookEndpoint
            if hasattr(endpoint, 'url'):
                endpoint_url = endpoint.url
            elif hasattr(endpoint, 'endpoint'):
                endpoint_url = endpoint.endpoint
        elif 'endpoint' in kwargs:
            endpoint = kwargs['endpoint']
            if hasattr(endpoint, 'url'):
                endpoint_url = endpoint.url
        
        # Create circuit breaker for webhook endpoint
        circuit_name = f"webhook_{hash(endpoint_url) % 10000}"
        
        circuit = get_circuit_breaker(circuit_name, CircuitBreakerConfig(
            failure_threshold=5,
            timeout_seconds=60,
            expected_exception=Exception
        ))
        
        # Retry configuration for webhooks
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=30.0
        )
        
        async def _deliver_with_tracing():
            """Delivery with tracing"""
            try:
                tracing = get_tracing_service()
                if tracing and tracing.enabled:
                    with tracing.span("webhook_delivery", attributes={
                        "webhook.endpoint": str(endpoint_url)[:100],  # Limit length
                        "webhook.event": getattr(args[0], 'event', 'unknown') if args else 'unknown'
                    }):
                        result = await delivery_func(*args, **kwargs)
                        # Record metrics
                        if hasattr(result, 'status'):
                            record_webhook_metrics(
                                event=getattr(args[0], 'event', 'unknown') if args else 'unknown',
                                status=result.status if hasattr(result, 'status') else 'success'
                            )
                        return result
                else:
                    result = await delivery_func(*args, **kwargs)
                    # Record metrics even without tracing
                    if hasattr(result, 'status'):
                        record_webhook_metrics(
                            event=getattr(args[0], 'event', 'unknown') if args else 'unknown',
                            status=result.status if hasattr(result, 'status') else 'success'
                        )
                    return result
            except Exception as e:
                # Record failure metric
                record_webhook_metrics(
                    event=getattr(args[0], 'event', 'unknown') if args else 'unknown',
                    status='failed'
                )
                raise
        
        # Use circuit breaker with retry
        try:
            return await circuit.call(
                lambda: retry(_deliver_with_tracing, config=retry_config)(),
                fallback=lambda: {
                    "status": "failed",
                    "error": "Circuit breaker open - service unavailable",
                    "retryable": True
                }
            )
        except Exception as e:
            logger.error(f"Webhook delivery failed after resilience patterns: {e}")
            raise
    
    return resilient_delivery


def record_webhook_metrics(event: str, status: str, duration: float = None):
    """Record webhook metrics"""
    if not RESILIENCE_AVAILABLE:
        return
    
    try:
        metrics = get_metrics_collector()
        if hasattr(metrics, 'record_http_request'):
            # Record as HTTP request metric
            metrics.record_http_request(
                method="POST",
                endpoint=f"/webhooks/{event}",
                status=200 if status == "success" else 500,
                duration=duration or 0.0
            )
    except Exception as e:
        logger.warning(f"Failed to record webhook metrics: {e}")

