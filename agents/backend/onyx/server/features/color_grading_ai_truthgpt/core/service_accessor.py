"""
Service Accessor for Color Grading AI
======================================

Unified service accessor with lazy loading and caching.
"""

import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class ServiceAccessor:
    """
    Unified service accessor.
    
    Features:
    - Lazy service access
    - Service caching
    - Type checking
    - Error handling
    """
    
    def __init__(self, services: Dict[str, Any]):
        """
        Initialize service accessor.
        
        Args:
            services: Dictionary of services
        """
        self._services = services
        self._cache: Dict[str, Any] = {}
    
    def get(self, name: str, default: Any = None) -> Any:
        """
        Get service by name.
        
        Args:
            name: Service name
            default: Default value if not found
            
        Returns:
            Service instance
        """
        if name in self._cache:
            return self._cache[name]
        
        service = self._services.get(name, default)
        if service:
            self._cache[name] = service
        
        return service
    
    def get_group(self, group_name: str) -> Dict[str, Any]:
        """
        Get services by group.
        
        Groups:
        - processing
        - management
        - infrastructure
        - analytics
        - intelligence
        - collaboration
        
        Args:
            group_name: Group name
            
        Returns:
            Dictionary of services in group
        """
        groups = {
            "processing": [
                "video_processor", "image_processor", "color_analyzer",
                "color_matcher", "video_quality_analyzer"
            ],
            "management": [
                "template_manager", "preset_manager", "lut_manager",
                "cache_manager", "history_manager", "version_manager",
                "backup_manager"
            ],
            "infrastructure": [
                "event_bus", "security_manager", "telemetry_service",
                "task_queue", "cloud_integration"
            ],
            "analytics": [
                "metrics_collector", "performance_monitor",
                "performance_optimizer", "analytics_service"
            ],
            "intelligence": [
                "recommendation_engine", "ml_optimizer", "optimization_engine"
            ],
            "collaboration": [
                "webhook_manager", "notification_service",
                "collaboration_manager", "workflow_manager"
            ],
        }
        
        service_names = groups.get(group_name, [])
        return {
            name: self.get(name)
            for name in service_names
            if self.get(name) is not None
        }
    
    def has(self, name: str) -> bool:
        """Check if service exists."""
        return name in self._services
    
    def list_all(self) -> List[str]:
        """List all service names."""
        return list(self._services.keys())
    
    def clear_cache(self):
        """Clear service cache."""
        self._cache.clear()


def require_service(service_name: str):
    """
    Decorator to require a service.
    
    Args:
        service_name: Required service name
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'services') or not self.services.get(service_name):
                raise ValueError(f"Service {service_name} not available")
            return await func(self, *args, **kwargs)
        return wrapper
    return decorator




