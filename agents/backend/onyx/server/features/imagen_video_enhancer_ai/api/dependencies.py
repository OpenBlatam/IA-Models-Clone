"""
API Dependencies
================

Shared dependencies and utilities for API routes.
"""

import logging
from typing import Optional
from fastapi import HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..core.enhancer_agent import EnhancerAgent
from ..core.monitoring_dashboard import MonitoringDashboard
from ..core.auth import AuthManager
from ..core.notification_system import NotificationManager
from ..core.metrics_collector import MetricsCollector
from ..core.event_bus import EventBus
from ..core.rate_limiter import RateLimiter, RateLimitConfig
from ..constants import DEFAULT_RATE_LIMIT_RPS, DEFAULT_RATE_LIMIT_BURST

logger = logging.getLogger(__name__)

# Global instances (initialized in enhancer_api.py)
_agent: Optional[EnhancerAgent] = None
_dashboard: Optional[MonitoringDashboard] = None
_auth_manager: Optional[AuthManager] = None
_notification_manager: Optional[NotificationManager] = None
_metrics_collector: Optional[MetricsCollector] = None
_event_bus: Optional[EventBus] = None
_rate_limiter: Optional[RateLimiter] = None

# Security
security = HTTPBearer(auto_error=False)


def set_agent(agent: Optional[EnhancerAgent]):
    """Set global agent instance."""
    global _agent
    _agent = agent


def set_dashboard(dashboard: Optional[MonitoringDashboard]):
    """Set global dashboard instance."""
    global _dashboard
    _dashboard = dashboard


def set_managers(
    auth_manager: AuthManager,
    notification_manager: NotificationManager,
    metrics_collector: MetricsCollector,
    event_bus: EventBus,
    rate_limiter: RateLimiter
):
    """Set global manager instances."""
    global _auth_manager, _notification_manager, _metrics_collector, _event_bus, _rate_limiter
    _auth_manager = auth_manager
    _notification_manager = notification_manager
    _metrics_collector = metrics_collector
    _event_bus = event_bus
    _rate_limiter = rate_limiter


def get_agent() -> EnhancerAgent:
    """Get agent instance."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return _agent


def get_dashboard() -> MonitoringDashboard:
    """Get dashboard instance."""
    if _dashboard is None:
        raise HTTPException(status_code=503, detail="Dashboard not initialized")
    return _dashboard


def get_auth_manager() -> AuthManager:
    """Get auth manager instance."""
    if _auth_manager is None:
        raise HTTPException(status_code=503, detail="Auth manager not initialized")
    return _auth_manager


def get_notification_manager() -> NotificationManager:
    """Get notification manager instance."""
    if _notification_manager is None:
        raise HTTPException(status_code=503, detail="Notification manager not initialized")
    return _notification_manager


def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance."""
    if _metrics_collector is None:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")
    return _metrics_collector


def get_event_bus() -> EventBus:
    """Get event bus instance."""
    if _event_bus is None:
        raise HTTPException(status_code=503, detail="Event bus not initialized")
    return _event_bus


def get_rate_limiter() -> RateLimiter:
    """Get rate limiter instance."""
    if _rate_limiter is None:
        raise HTTPException(status_code=503, detail="Rate limiter not initialized")
    return _rate_limiter


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """Verify API key from header."""
    if _auth_manager is None:
        return None
    
    api_key = None
    
    if credentials:
        api_key = credentials.credentials
    elif x_api_key:
        api_key = x_api_key
    
    if api_key:
        key_info = _auth_manager.validate_key(api_key)
        if key_info:
            return api_key
    
    return None

