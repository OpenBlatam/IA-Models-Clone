"""
Integration Optimizations

Optimizations for:
- Webhooks
- Event system
- External API integration
- Message queues
- Service mesh
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import hashlib
import time

logger = logging.getLogger(__name__)


class WebhookManager:
    """Optimized webhook management."""
    
    def __init__(self, max_retries: int = 3, timeout: int = 30):
        """
        Initialize webhook manager.
        
        Args:
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.webhooks: Dict[str, Dict[str, Any]] = {}
    
    def register_webhook(
        self,
        event_type: str,
        url: str,
        secret: Optional[str] = None
    ) -> str:
        """
        Register webhook.
        
        Args:
            event_type: Type of event
            url: Webhook URL
            secret: Optional secret for signing
            
        Returns:
            Webhook ID
        """
        webhook_id = hashlib.md5(f"{event_type}:{url}:{time.time()}".encode()).hexdigest()
        
        self.webhooks[webhook_id] = {
            'event_type': event_type,
            'url': url,
            'secret': secret,
            'enabled': True,
            'created_at': time.time()
        }
        
        return webhook_id
    
    async def trigger_webhook(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Trigger webhooks for event type.
        
        Args:
            event_type: Event type
            data: Event data
            
        Returns:
            List of webhook results
        """
        import httpx
        
        # Find webhooks for this event type
        matching_webhooks = [
            (wid, wh) for wid, wh in self.webhooks.items()
            if wh['event_type'] == event_type and wh['enabled']
        ]
        
        if not matching_webhooks:
            return []
        
        # Trigger all webhooks in parallel
        async def trigger_one(webhook_id: str, webhook: Dict[str, Any]):
            url = webhook['url']
            secret = webhook.get('secret')
            
            # Sign payload if secret provided
            payload = data.copy()
            if secret:
                payload['signature'] = self._sign_payload(payload, secret)
            
            # Retry logic
            for attempt in range(self.max_retries):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(url, json=payload)
                        response.raise_for_status()
                        return {
                            'webhook_id': webhook_id,
                            'status': 'success',
                            'status_code': response.status_code
                        }
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Webhook {webhook_id} failed after {self.max_retries} attempts: {e}")
                        return {
                            'webhook_id': webhook_id,
                            'status': 'failed',
                            'error': str(e)
                        }
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None
        
        tasks = [trigger_one(wid, wh) for wid, wh in matching_webhooks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if r and not isinstance(r, Exception)]
    
    def _sign_payload(self, payload: Dict[str, Any], secret: str) -> str:
        """Sign webhook payload."""
        import hmac
        import json
        
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature


class EventBus:
    """Optimized event bus."""
    
    def __init__(self):
        """Initialize event bus."""
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """
        Subscribe to event type.
        
        Args:
            event_type: Event type
            handler: Handler function
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Publish event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        event = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Notify subscribers
        if event_type in self.subscribers:
            tasks = []
            for handler in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Event handler failed: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


class MessageQueue:
    """Optimized message queue."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize message queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.processors: List[Callable] = []
        self.running = False
    
    async def enqueue(self, message: Dict[str, Any]) -> bool:
        """
        Enqueue message.
        
        Args:
            message: Message data
            
        Returns:
            True if enqueued successfully
        """
        try:
            await asyncio.wait_for(self.queue.put(message), timeout=1.0)
            return True
        except asyncio.TimeoutError:
            logger.warning("Queue full, message dropped")
            return False
    
    async def start_processing(self, processor: Callable) -> None:
        """
        Start processing messages.
        
        Args:
            processor: Message processor function
        """
        self.running = True
        
        while self.running:
            try:
                message = await self.queue.get()
                
                if asyncio.iscoroutinefunction(processor):
                    await processor(message)
                else:
                    processor(message)
                
                self.queue.task_done()
            except Exception as e:
                logger.error(f"Message processing error: {e}")
    
    def stop_processing(self) -> None:
        """Stop processing messages."""
        self.running = False


class ServiceMesh:
    """Service mesh integration."""
    
    def __init__(self):
        """Initialize service mesh."""
        self.services: Dict[str, Dict[str, Any]] = {}
        self.health_checks: Dict[str, Callable] = {}
    
    def register_service(
        self,
        name: str,
        url: str,
        health_check: Optional[Callable] = None
    ) -> None:
        """
        Register service.
        
        Args:
            name: Service name
            url: Service URL
            health_check: Optional health check function
        """
        self.services[name] = {
            'url': url,
            'healthy': True,
            'last_check': None
        }
        
        if health_check:
            self.health_checks[name] = health_check
    
    async def call_service(
        self,
        service_name: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Call service.
        
        Args:
            service_name: Service name
            endpoint: Endpoint path
            method: HTTP method
            data: Request data
            
        Returns:
            Service response
        """
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not registered")
        
        service = self.services[service_name]
        
        # Check health
        if not service['healthy']:
            # Try health check
            if service_name in self.health_checks:
                try:
                    if asyncio.iscoroutinefunction(self.health_checks[service_name]):
                        healthy = await self.health_checks[service_name]()
                    else:
                        healthy = self.health_checks[service_name]()
                    
                    service['healthy'] = healthy
                except Exception:
                    service['healthy'] = False
            
            if not service['healthy']:
                raise Exception(f"Service {service_name} is unhealthy")
        
        # Make request
        import httpx
        
        url = f"{service['url']}{endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if method == "GET":
                    response = await client.get(url)
                elif method == "POST":
                    response = await client.post(url, json=data)
                else:
                    response = await client.request(method, url, json=data)
                
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Service call failed: {e}")
            service['healthy'] = False
            raise








