"""
Service Manager
Utility functions for managing service lifecycle
"""

import logging
from typing import Optional, TypeVar, Callable, Any
from fastapi import HTTPException, status

T = TypeVar('T')

logger = logging.getLogger(__name__)

async def close_service(service: Optional[Any], service_name: str) -> None:
    """Close a service if it has a close method"""
    if not service:
        return
    
    if not hasattr(service, 'close'):
        return
    
    try:
        import inspect
        close_method = getattr(service, 'close')
        
        if inspect.iscoroutinefunction(close_method):
            await close_method()
        else:
            close_method()
        
        logger.debug(
            f"Service {service_name} closed successfully",
            extra={"service_name": service_name}
        )
    except Exception as e:
        logger.error(
            f"Error closing {service_name}",
            extra={"service_name": service_name, "error": str(e)},
            exc_info=True
        )

def create_service_dependency(
    get_instance: Callable[[], Optional[T]],
    service_name: str
) -> Callable[[], T]:
    """Create a dependency function for a service"""
    async def dependency() -> T:
        try:
            instance = get_instance()
            if not instance:
                logger.warning(
                    f"{service_name} not available",
                    extra={"service_name": service_name}
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"{service_name} not available"
                )
            return instance
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Error getting {service_name} instance",
                extra={"service_name": service_name, "error": str(e)},
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"{service_name} temporarily unavailable"
            )
    return dependency
