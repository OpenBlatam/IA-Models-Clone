"""
Interceptor Manager
==================

gRPC interceptor management.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class InterceptorType(Enum):
    """Interceptor types."""
    CLIENT = "client"
    SERVER = "server"
    UNARY = "unary"
    STREAM = "stream"


class InterceptorManager:
    """gRPC interceptor manager."""
    
    def __init__(self):
        self._interceptors: Dict[InterceptorType, List[Callable]] = {
            interceptor_type: []
            for interceptor_type in InterceptorType
        }
    
    def register_interceptor(
        self,
        interceptor_type: InterceptorType,
        interceptor: Callable
    ):
        """Register interceptor."""
        self._interceptors[interceptor_type].append(interceptor)
        logger.info(f"Registered {interceptor_type.value} interceptor")
    
    def get_interceptors(self, interceptor_type: InterceptorType) -> List[Callable]:
        """Get interceptors for type."""
        return self._interceptors[interceptor_type].copy()
    
    def apply_interceptors(
        self,
        interceptor_type: InterceptorType,
        handler: Callable
    ) -> Callable:
        """Apply interceptors to handler."""
        interceptors = self.get_interceptors(interceptor_type)
        
        # Apply interceptors in reverse order (last registered, first executed)
        for interceptor in reversed(interceptors):
            handler = interceptor(handler)
        
        return handler
    
    def get_interceptor_stats(self) -> Dict[str, Any]:
        """Get interceptor statistics."""
        return {
            interceptor_type.value: len(interceptors)
            for interceptor_type, interceptors in self._interceptors.items()
        }















