from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from .entities import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v13.0 - Domain Repositories

Repository interfaces for data persistence abstraction.
Following Clean Architecture principles - domain defines contracts.
"""

    CaptionRequest, CaptionResponse, BatchRequest, BatchResponse,
    RequestId, QualityMetrics, PerformanceMetrics
)


class ICacheRepository(ABC):
    """Repository interface for caching operations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CaptionResponse]:
        """Retrieve cached response by key."""
        pass
    
    @abstractmethod
    async def set(self, key: str, response: CaptionResponse, ttl: Optional[int] = None) -> None:
        """Store response in cache with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cached response."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached data."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class IMetricsRepository(ABC):
    """Repository interface for metrics storage."""
    
    @abstractmethod
    async async def record_request(
        self, 
        request: CaptionRequest, 
        response: CaptionResponse
    ) -> None:
        """Record a completed request with its response."""
        pass
    
    @abstractmethod
    async def record_batch(
        self, 
        batch_request: BatchRequest, 
        batch_response: BatchResponse
    ) -> None:
        """Record a completed batch operation."""
        pass
    
    @abstractmethod
    async def get_performance_metrics(
        self, 
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get performance metrics for specified time window."""
        pass
    
    @abstractmethod
    async def get_quality_trends(
        self, 
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get quality trends over time."""
        pass
    
    @abstractmethod
    async def get_usage_statistics(
        self, 
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get usage statistics, optionally filtered by tenant."""
        pass


class IAuditRepository(ABC):
    """Repository interface for audit logging."""
    
    @abstractmethod
    async async def log_request(
        self, 
        request: CaptionRequest, 
        response: CaptionResponse,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a request/response pair and return audit ID."""
        pass
    
    @abstractmethod
    async def log_batch_operation(
        self, 
        batch_request: BatchRequest, 
        batch_response: BatchResponse,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a batch operation and return audit ID."""
        pass
    
    @abstractmethod
    async def log_error(
        self, 
        request_id: RequestId,
        error_message: str,
        error_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an error event."""
        pass
    
    @abstractmethod
    async def get_audit_trail(
        self, 
        request_id: Optional[RequestId] = None,
        tenant_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve audit trail with optional filters."""
        pass


class IConfigRepository(ABC):
    """Repository interface for configuration management."""
    
    @abstractmethod
    async def get_tenant_config(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific tenant."""
        pass
    
    @abstractmethod
    async def set_tenant_config(self, tenant_id: str, config: Dict[str, Any]) -> None:
        """Set configuration for specific tenant."""
        pass
    
    @abstractmethod
    async def get_global_config(self) -> Dict[str, Any]:
        """Get global system configuration."""
        pass
    
    @abstractmethod
    async def update_global_config(self, config: Dict[str, Any]) -> None:
        """Update global system configuration."""
        pass
    
    @abstractmethod
    async def get_rate_limits(self, tenant_id: str) -> Dict[str, Any]:
        """Get rate limiting configuration for tenant."""
        pass


class ITemplateRepository(ABC):
    """Repository interface for template management."""
    
    @abstractmethod
    async def get_templates_for_style(self, style: str) -> List[str]:
        """Get caption templates for specific style."""
        pass
    
    @abstractmethod
    async def add_template(self, style: str, template: str) -> None:
        """Add new template for style."""
        pass
    
    @abstractmethod
    async def remove_template(self, style: str, template: str) -> bool:
        """Remove template from style."""
        pass
    
    @abstractmethod
    async def get_hashtag_sets(self, category: str) -> List[str]:
        """Get hashtag sets for category."""
        pass
    
    @abstractmethod
    async def update_hashtag_sets(self, category: str, hashtags: List[str]) -> None:
        """Update hashtag sets for category."""
        pass


class IHealthRepository(ABC):
    """Repository interface for health monitoring."""
    
    @abstractmethod
    async def record_health_check(
        self, 
        component: str, 
        status: str, 
        response_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record health check result."""
        pass
    
    @abstractmethod
    async def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status for specific component."""
        pass
    
    @abstractmethod
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        pass
    
    @abstractmethod
    async def get_uptime_stats(self) -> Dict[str, Any]:
        """Get system uptime statistics."""
        pass 