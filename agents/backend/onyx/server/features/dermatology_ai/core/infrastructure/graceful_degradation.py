"""
Graceful Degradation
Provides fallback mechanisms when non-critical services fail
"""

from typing import Callable, Any, Optional, Dict
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)


class ServicePriority(Enum):
    """Service priority levels"""
    CRITICAL = "critical"  # Must be available
    HIGH = "high"  # Should be available
    MEDIUM = "medium"  # Nice to have
    LOW = "low"  # Optional


class GracefulDegradation:
    """Handle graceful degradation for services"""
    
    def __init__(self):
        self.service_status: Dict[str, bool] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.service_priorities: Dict[str, ServicePriority] = {}
    
    def register_service(
        self,
        service_name: str,
        priority: ServicePriority = ServicePriority.MEDIUM,
        fallback: Optional[Callable] = None
    ):
        """
        Register a service with priority and fallback
        
        Args:
            service_name: Name of the service
            priority: Service priority
            fallback: Fallback function if service fails
        """
        self.service_priorities[service_name] = priority
        if fallback:
            self.fallback_handlers[service_name] = fallback
        self.service_status[service_name] = True
        logger.info(f"Registered service: {service_name} (priority: {priority.value})")
    
    async def execute_with_fallback(
        self,
        service_name: str,
        primary_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with fallback if service is degraded
        
        Args:
            service_name: Name of the service
            primary_func: Primary function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
        """
        priority = self.service_priorities.get(service_name, ServicePriority.MEDIUM)
        
        # Critical services must work
        if priority == ServicePriority.CRITICAL:
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            return primary_func(*args, **kwargs)
        
        # Try primary function
        try:
            if asyncio.iscoroutinefunction(primary_func):
                result = await primary_func(*args, **kwargs)
            else:
                result = primary_func(*args, **kwargs)
            
            # Mark service as healthy
            self.service_status[service_name] = True
            return result
            
        except Exception as e:
            logger.warning(f"Service {service_name} failed, attempting fallback: {e}")
            self.service_status[service_name] = False
            
            # Try fallback
            fallback = self.fallback_handlers.get(service_name)
            if fallback:
                try:
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    return fallback(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback for {service_name} also failed: {fallback_error}")
            
            # If no fallback or fallback failed, and service is not critical
            if priority in [ServicePriority.MEDIUM, ServicePriority.LOW]:
                logger.info(f"Service {service_name} unavailable, continuing without it")
                return None
            
            # Re-raise for high priority services without fallback
            raise
    
    def is_service_available(self, service_name: str) -> bool:
        """Check if service is available"""
        return self.service_status.get(service_name, True)
    
    def get_degraded_services(self) -> list[str]:
        """Get list of degraded services"""
        return [
            name for name, status in self.service_status.items()
            if not status
        ]


# Global graceful degradation instance
_graceful_degradation = GracefulDegradation()


def get_graceful_degradation() -> GracefulDegradation:
    """Get global graceful degradation instance"""
    return _graceful_degradation


def with_graceful_degradation(
    service_name: str,
    priority: ServicePriority = ServicePriority.MEDIUM,
    fallback: Optional[Callable] = None
):
    """
    Decorator for graceful degradation
    
    Args:
        service_name: Name of the service
        priority: Service priority
        fallback: Fallback function
    """
    degradation = get_graceful_degradation()
    degradation.register_service(service_name, priority, fallback)
    
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            return await degradation.execute_with_fallback(
                service_name,
                func,
                *args,
                **kwargs
            )
        
        def sync_wrapper(*args, **kwargs):
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                degradation.execute_with_fallback(
                    service_name,
                    func,
                    *args,
                    **kwargs
                )
            )
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator















