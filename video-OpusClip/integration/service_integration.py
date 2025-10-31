#!/usr/bin/env python3
"""
Service Integration Hub

Advanced service integration with:
- External API integration
- Service orchestration
- Data synchronization
- Webhook management
- API versioning and compatibility
- Integration monitoring and health checks
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
import asyncio
import time
import json
import uuid
import httpx
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import defaultdict, deque
import aiohttp
from urllib.parse import urljoin, urlparse

logger = structlog.get_logger("service_integration")

# =============================================================================
# INTEGRATION MODELS
# =============================================================================

class IntegrationType(Enum):
    """Integration types."""
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    WEBHOOK = "webhook"
    MESSAGE_QUEUE = "message_queue"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    CUSTOM = "custom"

class IntegrationStatus(Enum):
    """Integration status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"

class AuthenticationType(Enum):
    """Authentication types for integrations."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"

@dataclass
class IntegrationConfig:
    """Integration configuration."""
    integration_id: str
    name: str
    description: str
    integration_type: IntegrationType
    base_url: str
    authentication: AuthenticationType
    auth_config: Dict[str, Any]
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 1
    rate_limit: Optional[int] = None
    headers: Dict[str, str] = None
    status: IntegrationStatus = IntegrationStatus.ACTIVE
    health_check_url: Optional[str] = None
    health_check_interval: int = 300
    webhook_secret: Optional[str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "integration_id": self.integration_id,
            "name": self.name,
            "description": self.description,
            "integration_type": self.integration_type.value,
            "base_url": self.base_url,
            "authentication": self.authentication.value,
            "auth_config": self.auth_config,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "rate_limit": self.rate_limit,
            "headers": self.headers,
            "status": self.status.value,
            "health_check_url": self.health_check_url,
            "health_check_interval": self.health_check_interval,
            "webhook_secret": self.webhook_secret
        }

@dataclass
class IntegrationRequest:
    """Integration request data."""
    request_id: str
    integration_id: str
    method: str
    endpoint: str
    headers: Dict[str, str]
    params: Dict[str, Any]
    data: Optional[Dict[str, Any]]
    timeout: Optional[int]
    retry_count: Optional[int]
    timestamp: datetime
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "integration_id": self.integration_id,
            "method": self.method,
            "endpoint": self.endpoint,
            "headers": self.headers,
            "params": self.params,
            "data": self.data,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class IntegrationResponse:
    """Integration response data."""
    request_id: str
    integration_id: str
    status_code: int
    headers: Dict[str, str]
    data: Any
    response_time: float
    success: bool
    error_message: Optional[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "integration_id": self.integration_id,
            "status_code": self.status_code,
            "headers": self.headers,
            "data": self.data,
            "response_time": self.response_time,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class WebhookConfig:
    """Webhook configuration."""
    webhook_id: str
    name: str
    url: str
    events: List[str]
    secret: Optional[str]
    headers: Dict[str, str]
    timeout: int = 30
    retry_count: int = 3
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "webhook_id": self.webhook_id,
            "name": self.name,
            "url": self.url,
            "events": self.events,
            "secret": self.secret,
            "headers": self.headers,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "enabled": self.enabled
        }

# =============================================================================
# INTEGRATION CLIENT
# =============================================================================

class IntegrationClient:
    """Client for external service integration."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.session: Optional[httpx.AsyncClient] = None
        self.rate_limiter: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.health_status: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'total_response_time': 0.0,
            'last_health_check': None
        }
    
    async def start(self) -> None:
        """Start the integration client."""
        # Create HTTP session
        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            headers=self.config.headers
        )
        
        # Start health checking if configured
        if self.config.health_check_url:
            asyncio.create_task(self._health_check_loop())
        
        logger.info(
            "Integration client started",
            integration_id=self.config.integration_id,
            name=self.config.name
        )
    
    async def stop(self) -> None:
        """Stop the integration client."""
        if self.session:
            await self.session.aclose()
        
        logger.info("Integration client stopped", integration_id=self.config.integration_id)
    
    async def request(self, request: IntegrationRequest) -> IntegrationResponse:
        """Make request to external service."""
        start_time = time.time()
        
        try:
            # Check rate limiting
            if not await self._check_rate_limit():
                raise RuntimeError("Rate limit exceeded")
            
            # Prepare request
            url = urljoin(self.config.base_url, request.endpoint)
            headers = {**self.config.headers, **request.headers}
            
            # Add authentication
            headers.update(await self._get_auth_headers())
            
            # Make request
            response = await self.session.request(
                method=request.method,
                url=url,
                headers=headers,
                params=request.params,
                json=request.data,
                timeout=request.timeout or self.config.timeout
            )
            
            # Process response
            response_time = time.time() - start_time
            
            try:
                response_data = response.json()
            except:
                response_data = response.text
            
            integration_response = IntegrationResponse(
                request_id=request.request_id,
                integration_id=self.config.integration_id,
                status_code=response.status_code,
                headers=dict(response.headers),
                data=response_data,
                response_time=response_time,
                success=200 <= response.status_code < 300,
                error_message=None,
                timestamp=datetime.utcnow()
            )
            
            # Update statistics
            self._update_stats(response_time, integration_response.success)
            
            logger.debug(
                "Integration request completed",
                integration_id=self.config.integration_id,
                request_id=request.request_id,
                status_code=response.status_code,
                response_time=response_time
            )
            
            return integration_response
            
        except Exception as e:
            response_time = time.time() - start_time
            
            integration_response = IntegrationResponse(
                request_id=request.request_id,
                integration_id=self.config.integration_id,
                status_code=0,
                headers={},
                data=None,
                response_time=response_time,
                success=False,
                error_message=str(e),
                timestamp=datetime.utcnow()
            )
            
            # Update statistics
            self._update_stats(response_time, False)
            
            logger.error(
                "Integration request failed",
                integration_id=self.config.integration_id,
                request_id=request.request_id,
                error=str(e)
            )
            
            return integration_response
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        auth_headers = {}
        
        if self.config.authentication == AuthenticationType.API_KEY:
            api_key = self.config.auth_config.get('api_key')
            if api_key:
                header_name = self.config.auth_config.get('header_name', 'X-API-Key')
                auth_headers[header_name] = api_key
        
        elif self.config.authentication == AuthenticationType.BEARER_TOKEN:
            token = self.config.auth_config.get('token')
            if token:
                auth_headers['Authorization'] = f'Bearer {token}'
        
        elif self.config.authentication == AuthenticationType.BASIC_AUTH:
            username = self.config.auth_config.get('username')
            password = self.config.auth_config.get('password')
            if username and password:
                import base64
                credentials = base64.b64encode(f'{username}:{password}'.encode()).decode()
                auth_headers['Authorization'] = f'Basic {credentials}'
        
        return auth_headers
    
    async def _check_rate_limit(self) -> bool:
        """Check rate limiting."""
        if not self.config.rate_limit:
            return True
        
        current_time = time.time()
        rate_queue = self.rate_limiter[self.config.integration_id]
        
        # Clean old entries
        while rate_queue and current_time - rate_queue[0] > 60:  # 1 minute window
            rate_queue.popleft()
        
        # Check if limit exceeded
        if len(rate_queue) >= self.config.rate_limit:
            return False
        
        # Add current request
        rate_queue.append(current_time)
        return True
    
    async def _health_check_loop(self) -> None:
        """Health check loop."""
        while True:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check error", integration_id=self.config.integration_id, error=str(e))
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _perform_health_check(self) -> None:
        """Perform health check."""
        try:
            start_time = time.time()
            
            response = await self.session.get(
                urljoin(self.config.base_url, self.config.health_check_url),
                timeout=10
            )
            
            response_time = time.time() - start_time
            
            self.health_status = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'status_code': response.status_code,
                'response_time': response_time,
                'last_check': datetime.utcnow().isoformat()
            }
            
            self.stats['last_health_check'] = datetime.utcnow()
            
            logger.debug(
                "Health check completed",
                integration_id=self.config.integration_id,
                status=self.health_status['status'],
                response_time=response_time
            )
            
        except Exception as e:
            self.health_status = {
                'status': 'error',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }
            
            logger.warning(
                "Health check failed",
                integration_id=self.config.integration_id,
                error=str(e)
            )
    
    def _update_stats(self, response_time: float, success: bool) -> None:
        """Update client statistics."""
        self.stats['total_requests'] += 1
        
        if success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
        
        self.stats['total_response_time'] += response_time
        
        # Update average response time
        total_requests = self.stats['successful_requests'] + self.stats['failed_requests']
        if total_requests > 0:
            self.stats['average_response_time'] = self.stats['total_response_time'] / total_requests
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self.stats,
            'health_status': self.health_status,
            'rate_limit': self.config.rate_limit,
            'current_rate': len(self.rate_limiter[self.config.integration_id])
        }

# =============================================================================
# INTEGRATION HUB
# =============================================================================

class IntegrationHub:
    """Central hub for managing all integrations."""
    
    def __init__(self):
        self.integrations: Dict[str, IntegrationClient] = {}
        self.integration_configs: Dict[str, IntegrationConfig] = {}
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.webhook_handlers: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = {
            'total_integrations': 0,
            'active_integrations': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'webhook_deliveries': 0,
            'webhook_failures': 0
        }
        
        # Request history
        self.request_history: deque = deque(maxlen=10000)
    
    async def start(self) -> None:
        """Start the integration hub."""
        # Start all integration clients
        for client in self.integrations.values():
            await client.start()
        
        logger.info("Integration hub started", integration_count=len(self.integrations))
    
    async def stop(self) -> None:
        """Stop the integration hub."""
        # Stop all integration clients
        for client in self.integrations.values():
            await client.stop()
        
        logger.info("Integration hub stopped")
    
    def register_integration(self, config: IntegrationConfig) -> IntegrationClient:
        """Register a new integration."""
        client = IntegrationClient(config)
        
        self.integrations[config.integration_id] = client
        self.integration_configs[config.integration_id] = config
        self.stats['total_integrations'] += 1
        
        if config.status == IntegrationStatus.ACTIVE:
            self.stats['active_integrations'] += 1
        
        logger.info(
            "Integration registered",
            integration_id=config.integration_id,
            name=config.name,
            type=config.integration_type.value
        )
        
        return client
    
    def unregister_integration(self, integration_id: str) -> bool:
        """Unregister an integration."""
        if integration_id in self.integrations:
            del self.integrations[integration_id]
            del self.integration_configs[integration_id]
            self.stats['total_integrations'] -= 1
            
            # Update active count
            active_count = len([
                config for config in self.integration_configs.values()
                if config.status == IntegrationStatus.ACTIVE
            ])
            self.stats['active_integrations'] = active_count
            
            logger.info("Integration unregistered", integration_id=integration_id)
            return True
        
        return False
    
    async def make_request(self, integration_id: str, request: IntegrationRequest) -> IntegrationResponse:
        """Make request through integration."""
        if integration_id not in self.integrations:
            raise ValueError(f"Integration {integration_id} not found")
        
        client = self.integrations[integration_id]
        
        # Check if integration is active
        config = self.integration_configs[integration_id]
        if config.status != IntegrationStatus.ACTIVE:
            raise RuntimeError(f"Integration {integration_id} is not active")
        
        # Make request
        response = await client.request(request)
        
        # Record request
        self.request_history.append({
            'request_id': request.request_id,
            'integration_id': integration_id,
            'timestamp': request.timestamp,
            'success': response.success,
            'response_time': response.response_time
        })
        
        # Update statistics
        self.stats['total_requests'] += 1
        if response.success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
        
        return response
    
    def register_webhook(self, config: WebhookConfig, handler: Callable) -> None:
        """Register a webhook."""
        self.webhooks[config.webhook_id] = config
        self.webhook_handlers[config.webhook_id] = handler
        
        logger.info(
            "Webhook registered",
            webhook_id=config.webhook_id,
            name=config.name,
            url=config.url
        )
    
    async def deliver_webhook(self, webhook_id: str, event: str, data: Dict[str, Any]) -> bool:
        """Deliver webhook."""
        if webhook_id not in self.webhooks:
            return False
        
        config = self.webhooks[webhook_id]
        
        if not config.enabled:
            return False
        
        if event not in config.events:
            return False
        
        try:
            # Prepare webhook payload
            payload = {
                'event': event,
                'data': data,
                'timestamp': datetime.utcnow().isoformat(),
                'webhook_id': webhook_id
            }
            
            # Add signature if secret is configured
            headers = dict(config.headers)
            if config.secret:
                import hmac
                import hashlib
                signature = hmac.new(
                    config.secret.encode(),
                    json.dumps(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                headers['X-Webhook-Signature'] = f'sha256={signature}'
            
            # Deliver webhook
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.post(
                    config.url,
                    json=payload,
                    headers=headers
                )
                
                success = 200 <= response.status_code < 300
                
                if success:
                    self.stats['webhook_deliveries'] += 1
                else:
                    self.stats['webhook_failures'] += 1
                
                logger.debug(
                    "Webhook delivered",
                    webhook_id=webhook_id,
                    event=event,
                    status_code=response.status_code,
                    success=success
                )
                
                return success
        
        except Exception as e:
            self.stats['webhook_failures'] += 1
            
            logger.error(
                "Webhook delivery failed",
                webhook_id=webhook_id,
                event=event,
                error=str(e)
            )
            
            return False
    
    def get_integration_stats(self, integration_id: str) -> Dict[str, Any]:
        """Get integration statistics."""
        if integration_id not in self.integrations:
            return {}
        
        client = self.integrations[integration_id]
        return client.get_client_stats()
    
    def get_hub_stats(self) -> Dict[str, Any]:
        """Get hub statistics."""
        return {
            **self.stats,
            'integrations': {
                integration_id: {
                    'name': config.name,
                    'type': config.integration_type.value,
                    'status': config.status.value,
                    'stats': self.get_integration_stats(integration_id)
                }
                for integration_id, config in self.integration_configs.items()
            },
            'webhooks': {
                webhook_id: {
                    'name': config.name,
                    'url': config.url,
                    'events': config.events,
                    'enabled': config.enabled
                }
                for webhook_id, config in self.webhooks.items()
            }
        }

# =============================================================================
# GLOBAL INTEGRATION INSTANCES
# =============================================================================

# Global integration hub
integration_hub = IntegrationHub()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'IntegrationType',
    'IntegrationStatus',
    'AuthenticationType',
    'IntegrationConfig',
    'IntegrationRequest',
    'IntegrationResponse',
    'WebhookConfig',
    'IntegrationClient',
    'IntegrationHub',
    'integration_hub'
]





























