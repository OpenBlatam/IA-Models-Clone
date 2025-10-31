"""
External services infrastructure for the ads feature.

This module consolidates external service integrations from scattered implementations.
Provides unified management of AI providers, analytics, notifications, and other external services.
"""

import asyncio
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import aiohttp
try:
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    aioredis = None  # type: ignore[assignment]
from fastapi import HTTPException

try:
    from onyx.utils.logger import setup_logger  # type: ignore
except Exception:  # pragma: no cover - fallback minimal logger for tests
    import logging as _logging

    def setup_logger(name: str | None = None):  # type: ignore[override]
        logger = _logging.getLogger(name or __name__)
        if not _logging.getLogger().handlers:
            _logging.basicConfig(level=_logging.INFO)
        return logger
from ..config import get_optimized_settings

logger = setup_logger()

class ServiceType(Enum):
    """External service types."""
    AI_PROVIDER = "ai_provider"
    ANALYTICS = "analytics"
    NOTIFICATION = "notification"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    SMS = "sms"

class ServiceStatus(Enum):
    """Service status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"

@dataclass
class ExternalServiceConfig:
    """Configuration for external services."""
    service_type: ServiceType
    base_url: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    rate_limit: Optional[int] = None
    rate_limit_window: int = 3600  # 1 hour
    enabled: bool = True
    priority: int = 1  # Lower number = higher priority
    # Circuit breaker (optional)
    circuit_breaker_threshold: int = 5
    circuit_breaker_window_seconds: int = 60
    circuit_breaker_cooldown_seconds: int = 120

class ExternalServiceManager:
    """Manages external service integrations."""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._redis_client: Optional[aioredis.Redis] = None
        self._rate_limit_cache: Dict[str, Dict[str, Any]] = {}
        # Circuit breaker state per service
        self._circuit: Dict[str, Dict[str, Any]] = {}
    
    @property
    async def http_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            # Enable connection pooling and DNS caching
            connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300, enable_cleanup_closed=True)
            self._http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._http_session
    
    @property
    async def redis_client(self) -> Any:
        """Get Redis client for rate limiting."""
        if self._redis_client is None:
            settings = get_optimized_settings()
            if aioredis is None or not getattr(settings, "redis_url", None):
                self._redis_client = None
            else:
                self._redis_client = await aioredis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
        return self._redis_client
    
    def register_service(self, service_id: str, service: Any):
        """Register an external service."""
        self.services[service_id] = service
        logger.info(f"Registered external service: {service_id}")
    
    def get_service(self, service_id: str) -> Optional[Any]:
        """Get a registered service."""
        return self.services.get(service_id)
    
    async def check_rate_limit(self, service_id: str, client_id: str) -> bool:
        """Check if rate limit is exceeded for a service."""
        try:
            service = self.get_service(service_id)
            if not service or not hasattr(service, 'config'):
                return True
            
            config = service.config
            if not config.rate_limit:
                return True
            
            cache_key = f"rate_limit:{service_id}:{client_id}"
            redis_client = await self.redis_client
            if redis_client is None:
                return True
            
            current = await redis_client.incr(cache_key)
            if current == 1:
                await redis_client.expire(cache_key, config.rate_limit_window)
            
            return current <= config.rate_limit
            
        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}")
            return True
    
    async def make_request(
        self,
        service_id: str,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make a request to an external service."""
        try:
            service = self.get_service(service_id)
            if not service:
                raise HTTPException(status_code=400, detail=f"Service {service_id} not found")
            config: ExternalServiceConfig = service.config
            
            # Check rate limit
            if not await self.check_rate_limit(service_id, "default"):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            # Circuit breaker check
            now = datetime.utcnow()
            state = self._circuit.get(service_id) or {}
            opened_until: Optional[datetime] = state.get("opened_until")
            if opened_until and now < opened_until:
                raise HTTPException(status_code=503, detail="Service temporarily unavailable (circuit open)")
            
            # Make request
            session = await self.http_session
            url = f"{service.config.base_url}{endpoint}"
            
            request_headers = headers or {}
            if service.config.api_key:
                request_headers["Authorization"] = f"Bearer {service.config.api_key}"
            
            # Retry with exponential backoff on transient failures
            max_retries = config.max_retries
            backoff = 0.5
            last_error: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    async with session.request(
                        method=method,
                        url=url,
                        json=data,
                        headers=request_headers,
                        **kwargs
                    ) as response:
                        # Try to parse JSON, fallback to text
                        try:
                            response_data = await response.json()
                        except Exception:
                            response_text = await response.text()
                            response_data = {"text": response_text}

                        if response.status >= 500:
                            # Retry on server errors
                            last_error = HTTPException(status_code=response.status, detail=str(response_data))
                            await asyncio.sleep(backoff)
                            backoff *= 2
                            continue
                        if response.status >= 400:
                            logger.error(f"External service error: {response.status} - {response_data}")
                            raise HTTPException(
                                status_code=response.status,
                                detail=f"External service error: {response_data}"
                            )
                        # Success: reset circuit state for service
                        self._circuit.pop(service_id, None)
                        return response_data
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_error = e
                    await asyncio.sleep(backoff)
                    backoff *= 2
            # Exhausted retries
            if last_error:
                # Update circuit breaker state
                state = self._circuit.get(service_id) or {"failures": []}
                failures: List[datetime] = state.get("failures", [])
                # Keep only failures within window
                window_start = now - timedelta(seconds=config.circuit_breaker_window_seconds)
                failures = [ts for ts in failures if ts >= window_start]
                failures.append(datetime.utcnow())
                state["failures"] = failures
                if len(failures) >= max(1, config.circuit_breaker_threshold):
                    state["opened_until"] = datetime.utcnow() + timedelta(seconds=config.circuit_breaker_cooldown_seconds)
                    logger.warning(f"Circuit opened for service {service_id} until {state['opened_until']}")
                self._circuit[service_id] = state
                raise last_error
            raise HTTPException(status_code=502, detail="External service unavailable")
                
        except Exception as e:
            logger.error(f"External service request failed: {e}")
            raise
    
    async def close(self):
        """Close all connections."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        
        if self._redis_client:
            await self._redis_client.close()

    def get_health_snapshot(self) -> Dict[str, Any]:
        """Return a lightweight health snapshot of registered services and circuit state."""
        snapshot: Dict[str, Any] = {
            "services": {},
            "total": len(self.services),
        }
        for service_id, service in self.services.items():
            config: ExternalServiceConfig = getattr(service, "config", None)  # type: ignore
            state = self._circuit.get(service_id) or {}
            opened_until = state.get("opened_until")
            failures = state.get("failures", [])
            snapshot["services"][service_id] = {
                "enabled": getattr(config, "enabled", True) if config else True,
                "base_url": getattr(config, "base_url", None) if config else None,
                "status": "open" if opened_until else "closed",
                "opened_until": opened_until.isoformat() if opened_until else None,
                "recent_failures": len(failures),
                "rate_limit": getattr(config, "rate_limit", None) if config else None,
            }
        return snapshot

class AIProviderService:
    """Service for AI provider integrations."""
    
    def __init__(self, service_manager: ExternalServiceManager):
        self.service_manager = service_manager
        self.providers: Dict[str, Dict[str, Any]] = {}
    
    async def register_provider(self, provider_id: str, config: ExternalServiceConfig):
        """Register an AI provider."""
        self.providers[provider_id] = {
            "config": config,
            "status": ServiceStatus.ACTIVE,
            "last_used": None,
            "usage_count": 0,
            "error_count": 0
        }
        logger.info(f"Registered AI provider: {provider_id}")
    
    async def generate_content(
        self,
        provider_id: str,
        prompt: str,
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate content using an AI provider."""
        try:
            provider = self.providers.get(provider_id)
            if not provider:
                raise HTTPException(status_code=400, detail=f"Provider {provider_id} not found")
            
            if provider["status"] != ServiceStatus.ACTIVE:
                raise HTTPException(status_code=400, detail=f"Provider {provider_id} is not active")
            
            # Make request to provider
            response = await self.service_manager.make_request(
                service_id=provider_id,
                method="POST",
                endpoint="/v1/chat/completions",
                data={
                    "model": model or "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    **(parameters or {})
                }
            )
            
            # Update provider stats
            provider["last_used"] = datetime.now()
            provider["usage_count"] += 1
            
            return response
            
        except Exception as e:
            # Update error count
            if provider_id in self.providers:
                self.providers[provider_id]["error_count"] += 1
                if self.providers[provider_id]["error_count"] >= 5:
                    self.providers[provider_id]["status"] = ServiceStatus.ERROR
            
            logger.error(f"AI provider error: {e}")
            raise
    
    async def get_provider_status(self, provider_id: str) -> Dict[str, Any]:
        """Get status of an AI provider."""
        provider = self.providers.get(provider_id)
        if not provider:
            return {"status": "not_found"}
        
        return {
            "provider_id": provider_id,
            "status": provider["status"].value,
            "last_used": provider["last_used"],
            "usage_count": provider["usage_count"],
            "error_count": provider["error_count"]
        }
    
    async def list_providers(self) -> List[Dict[str, Any]]:
        """List all registered AI providers."""
        return [
            {
                "provider_id": provider_id,
                **await self.get_provider_status(provider_id)
            }
            for provider_id in self.providers.keys()
        ]

class AnalyticsService:
    """Service for analytics integrations."""
    
    def __init__(self, service_manager: ExternalServiceManager):
        self.service_manager = service_manager
        self.analytics_providers: Dict[str, Dict[str, Any]] = {}
    
    async def register_provider(self, provider_id: str, config: ExternalServiceConfig):
        """Register an analytics provider."""
        self.analytics_providers[provider_id] = {
            "config": config,
            "status": ServiceStatus.ACTIVE,
            "last_sync": None
        }
        logger.info(f"Registered analytics provider: {provider_id}")
    
    async def track_event(
        self,
        provider_id: str,
        event_name: str,
        event_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> bool:
        """Track an analytics event."""
        try:
            provider = self.analytics_providers.get(provider_id)
            if not provider:
                logger.warning(f"Analytics provider {provider_id} not found")
                return False
            
            if provider["status"] != ServiceStatus.ACTIVE:
                logger.warning(f"Analytics provider {provider_id} is not active")
                return False
            
            # Prepare event data
            event_payload = {
                "event": event_name,
                "properties": event_data,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id
            }
            
            # Send to analytics provider
            await self.service_manager.make_request(
                service_id=provider_id,
                method="POST",
                endpoint="/v1/events",
                data=event_payload
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Analytics tracking failed: {e}")
            return False
    
    async def get_analytics_data(
        self,
        provider_id: str,
        metric: str,
        start_date: datetime,
        end_date: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get analytics data from a provider."""
        try:
            provider = self.analytics_providers.get(provider_id)
            if not provider:
                raise HTTPException(status_code=400, detail=f"Provider {provider_id} not found")
            
            # Prepare query parameters
            params = {
                "metric": metric,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                **(filters or {})
            }
            
            # Get analytics data
            response = await self.service_manager.make_request(
                service_id=provider_id,
                method="GET",
                endpoint="/v1/analytics",
                data=params
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Analytics data retrieval failed: {e}")
            raise
    
    async def sync_analytics(self, provider_id: str) -> bool:
        """Sync analytics data from a provider."""
        try:
            provider = self.analytics_providers.get(provider_id)
            if not provider:
                return False
            
            # Perform sync
            await self.service_manager.make_request(
                service_id=provider_id,
                method="POST",
                endpoint="/v1/sync"
            )
            
            # Update last sync time
            provider["last_sync"] = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Analytics sync failed: {e}")
            return False

class NotificationService:
    """Service for notification integrations."""
    
    def __init__(self, service_manager: ExternalServiceManager):
        self.service_manager = service_manager
        self.notification_providers: Dict[str, Dict[str, Any]] = {}
    
    async def register_provider(self, provider_id: str, config: ExternalServiceConfig):
        """Register a notification provider."""
        self.notification_providers[provider_id] = {
            "config": config,
            "status": ServiceStatus.ACTIVE,
            "last_used": None,
            "success_count": 0,
            "error_count": 0
        }
        logger.info(f"Registered notification provider: {provider_id}")
    
    async def send_notification(
        self,
        provider_id: str,
        notification_type: str,
        recipient: str,
        content: Dict[str, Any],
        priority: str = "normal"
    ) -> bool:
        """Send a notification through a provider."""
        try:
            provider = self.notification_providers.get(provider_id)
            if not provider:
                logger.warning(f"Notification provider {provider_id} not found")
                return False
            
            if provider["status"] != ServiceStatus.ACTIVE:
                logger.warning(f"Notification provider {provider_id} is not active")
                return False
            
            # Prepare notification payload
            notification_payload = {
                "type": notification_type,
                "recipient": recipient,
                "content": content,
                "priority": priority,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send notification
            await self.service_manager.make_request(
                service_id=provider_id,
                method="POST",
                endpoint="/v1/notifications",
                data=notification_payload
            )
            
            # Update provider stats
            provider["last_used"] = datetime.now()
            provider["success_count"] += 1
            
            return True
            
        except Exception as e:
            # Update error count
            if provider_id in self.notification_providers:
                self.notification_providers[provider_id]["error_count"] += 1
                if self.notification_providers[provider_id]["error_count"] >= 10:
                    self.notification_providers[provider_id]["status"] = ServiceStatus.ERROR
            
            logger.error(f"Notification sending failed: {e}")
            return False
    
    async def send_bulk_notifications(
        self,
        provider_id: str,
        notifications: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Send multiple notifications in bulk."""
        try:
            results = {
                "success": 0,
                "failed": 0,
                "total": len(notifications)
            }
            
            for notification in notifications:
                success = await self.send_notification(
                    provider_id=provider_id,
                    **notification
                )
                
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Bulk notification failed: {e}")
            return {
                "success": 0,
                "failed": len(notifications),
                "total": len(notifications)
            }
    
    async def get_provider_stats(self, provider_id: str) -> Dict[str, Any]:
        """Get statistics for a notification provider."""
        provider = self.notification_providers.get(provider_id)
        if not provider:
            return {"status": "not_found"}
        
        return {
            "provider_id": provider_id,
            "status": provider["status"].value,
            "last_used": provider["last_used"],
            "success_count": provider["success_count"],
            "error_count": provider["error_count"],
            "success_rate": provider["success_count"] / max(1, provider["success_count"] + provider["error_count"])
        }

# Global service manager instance
_service_manager = None

def get_external_service_manager() -> ExternalServiceManager:
    """Get the global external service manager instance."""
    global _service_manager
    if _service_manager is None:
        _service_manager = ExternalServiceManager()
    return _service_manager

async def close_external_service_manager():
    """Close the global external service manager."""
    global _service_manager
    if _service_manager:
        await _service_manager.close()
        _service_manager = None
