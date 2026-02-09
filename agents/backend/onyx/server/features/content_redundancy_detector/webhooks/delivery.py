"""
Webhook Delivery - Delivery logic with retry and signature
Optimized for high throughput and serverless
"""

import asyncio
import json
import time
import logging
import hashlib
import hmac
import random
from typing import Dict, Any, Optional
from dataclasses import asdict

import httpx

from .models import WebhookEndpoint, WebhookDelivery, WebhookPayload
from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class WebhookDeliveryService:
    """Service for delivering webhooks"""
    
    def __init__(self, http_client: httpx.AsyncClient):
        """
        Initialize delivery service
        
        Args:
            http_client: HTTP client for making requests
        """
        self._http_client = http_client
    
    def _generate_signature(self, payload: WebhookPayload, secret: str) -> str:
        """
        Generate HMAC signature for webhook payload
        
        Args:
            payload: Webhook payload
            secret: Secret key for signing
            
        Returns:
            HMAC-SHA256 signature
        """
        payload_dict = asdict(payload)
        payload_json = json.dumps(payload_dict, sort_keys=True, default=str)
        signature = hmac.new(
            secret.encode(),
            payload_json.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    def _prepare_headers(self, delivery: WebhookDelivery, endpoint: WebhookEndpoint) -> Dict[str, str]:
        """
        Prepare headers for webhook request
        
        Args:
            delivery: Webhook delivery record
            endpoint: Webhook endpoint configuration
            
        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Content-Redundancy-Detector/2.0.0",
            "X-Webhook-Id": delivery.id,
            "X-Webhook-Event": delivery.event,
            "X-Webhook-Timestamp": str(int(delivery.payload.timestamp)),
        }
        
        # Add signature if secret is provided
        if endpoint.secret:
            signature = self._generate_signature(delivery.payload, endpoint.secret)
            headers["X-Webhook-Signature"] = signature
        
        return headers
    
    def _calculate_retry_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """
        Calculate retry delay with exponential backoff and jitter
        
        Args:
            attempt: Current attempt number
            base_delay: Base delay in seconds
            
        Returns:
            Delay in seconds
        """
        exponential_delay = base_delay * (2 ** attempt)
        jitter = random.uniform(0, 0.3 * exponential_delay)  # Up to 30% jitter
        return exponential_delay + jitter
    
    async def deliver(
        self,
        delivery: WebhookDelivery,
        endpoint: WebhookEndpoint,
        circuit_breaker: CircuitBreaker
    ) -> Dict[str, Any]:
        """
        Deliver a webhook
        
        Args:
            delivery: Webhook delivery record
            endpoint: Webhook endpoint configuration
            circuit_breaker: Circuit breaker for this endpoint
            
        Returns:
            Dictionary with delivery result
        """
        # Check if retry is ready
        if delivery.next_retry and time.time() < delivery.next_retry:
            return {
                "status": "pending",
                "reason": "retry_not_ready",
                "next_retry": delivery.next_retry
            }
        
        delivery.attempts += 1
        delivery.last_attempt = time.time()
        start_time = time.time()
        
        try:
            # Prepare headers
            headers = self._prepare_headers(delivery, endpoint)
            
            # Send webhook
            response = await self._http_client.post(
                endpoint.url,
                json=asdict(delivery.payload),
                headers=headers,
                timeout=endpoint.timeout
            )
            
            delivery_time = time.time() - start_time
            
            # Check response status
            if 200 <= response.status_code < 400:
                delivery.status = "delivered"
                circuit_breaker.record_success()
                return {
                    "status": "delivered",
                    "delivery_time": delivery_time,
                    "status_code": response.status_code
                }
            else:
                delivery.status = "failed"
                circuit_breaker.record_failure()
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                
                # Calculate retry delay
                if delivery.attempts < endpoint.retry_count:
                    delay = self._calculate_retry_delay(delivery.attempts)
                    delivery.next_retry = time.time() + delay
                    delivery.status = "pending"
                    
                    return {
                        "status": "failed_retry_scheduled",
                        "error": error_msg,
                        "next_retry": delivery.next_retry,
                        "attempt": delivery.attempts
                    }
                
                return {
                    "status": "failed",
                    "error": error_msg,
                    "max_attempts_reached": True
                }
        
        except httpx.TimeoutException:
            delivery.status = "failed"
            circuit_breaker.record_failure()
            
            if delivery.attempts < endpoint.retry_count:
                delay = self._calculate_retry_delay(delivery.attempts)
                delivery.next_retry = time.time() + delay
                delivery.status = "pending"
                
                return {
                    "status": "timeout_retry_scheduled",
                    "next_retry": delivery.next_retry,
                    "attempt": delivery.attempts
                }
            
            return {
                "status": "failed",
                "error": "Timeout",
                "max_attempts_reached": True
            }
        
        except Exception as e:
            delivery.status = "failed"
            circuit_breaker.record_failure()
            logger.error(f"Webhook delivery error: {delivery.id} - {e}", exc_info=True)
            
            if delivery.attempts < endpoint.retry_count:
                delay = self._calculate_retry_delay(delivery.attempts)
                delivery.next_retry = time.time() + delay
                delivery.status = "pending"
                
                return {
                    "status": "error_retry_scheduled",
                    "error": str(e),
                    "next_retry": delivery.next_retry,
                    "attempt": delivery.attempts
                }
            
            return {
                "status": "failed",
                "error": str(e),
                "max_attempts_reached": True
            }
        finally:
            delivery_time = time.time() - start_time
            return {"delivery_time": delivery_time}

