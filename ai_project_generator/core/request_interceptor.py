"""
Request Interceptor - Interceptor de Requests
============================================

Interceptores avanzados de requests:
- Request preprocessing
- Response postprocessing
- Middleware chain
- Interceptor pipeline
- Conditional interceptors
"""

import logging
from typing import Optional, Dict, Any, List, Callable, Awaitable
from enum import Enum

logger = logging.getLogger(__name__)


class InterceptorType(str, Enum):
    """Tipos de interceptor"""
    PRE_REQUEST = "pre_request"
    POST_REQUEST = "post_request"
    PRE_RESPONSE = "pre_response"
    POST_RESPONSE = "post_response"
    ERROR = "error"


class RequestInterceptor:
    """
    Interceptor de requests.
    """
    
    def __init__(self) -> None:
        self.interceptors: Dict[InterceptorType, List[Callable]] = {
            InterceptorType.PRE_REQUEST: [],
            InterceptorType.POST_REQUEST: [],
            InterceptorType.PRE_RESPONSE: [],
            InterceptorType.POST_RESPONSE: [],
            InterceptorType.ERROR: []
        }
        self.conditions: Dict[str, Callable] = {}
    
    def register_interceptor(
        self,
        interceptor_type: InterceptorType,
        interceptor: Callable,
        condition: Optional[Callable] = None,
        priority: int = 0
    ) -> None:
        """Registra interceptor"""
        if condition:
            condition_id = f"{interceptor_type.value}_{len(self.interceptors[interceptor_type])}"
            self.conditions[condition_id] = condition
        
        # Insertar según prioridad
        interceptors = self.interceptors[interceptor_type]
        insert_index = 0
        for i, (existing_priority, _) in enumerate(interceptors):
            if priority > existing_priority:
                insert_index = i
                break
            insert_index = i + 1
        
        interceptors.insert(insert_index, (priority, interceptor, condition_id if condition else None))
        logger.info(f"Interceptor registered: {interceptor_type.value} (priority: {priority})")
    
    async def execute_pre_request(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecuta interceptores pre-request"""
        result = request.copy()
        
        for priority, interceptor, condition_id in self.interceptors[InterceptorType.PRE_REQUEST]:
            if condition_id:
                condition = self.conditions.get(condition_id)
                if condition and not condition(result):
                    continue
            
            try:
                if asyncio.iscoroutinefunction(interceptor):
                    result = await interceptor(result)
                else:
                    result = interceptor(result)
            except Exception as e:
                logger.error(f"Pre-request interceptor error: {e}")
        
        return result
    
    async def execute_post_request(
        self,
        request: Dict[str, Any],
        response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecuta interceptores post-request"""
        result = response.copy()
        
        for priority, interceptor, condition_id in self.interceptors[InterceptorType.POST_REQUEST]:
            if condition_id:
                condition = self.conditions.get(condition_id)
                if condition and not condition(request, result):
                    continue
            
            try:
                if asyncio.iscoroutinefunction(interceptor):
                    result = await interceptor(request, result)
                else:
                    result = interceptor(request, result)
            except Exception as e:
                logger.error(f"Post-request interceptor error: {e}")
        
        return result
    
    async def execute_pre_response(
        self,
        response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecuta interceptores pre-response"""
        result = response.copy()
        
        for priority, interceptor, condition_id in self.interceptors[InterceptorType.PRE_RESPONSE]:
            if condition_id:
                condition = self.conditions.get(condition_id)
                if condition and not condition(result):
                    continue
            
            try:
                if asyncio.iscoroutinefunction(interceptor):
                    result = await interceptor(result)
                else:
                    result = interceptor(result)
            except Exception as e:
                logger.error(f"Pre-response interceptor error: {e}")
        
        return result
    
    async def execute_post_response(
        self,
        request: Dict[str, Any],
        response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecuta interceptores post-response"""
        result = response.copy()
        
        for priority, interceptor, condition_id in self.interceptors[InterceptorType.POST_RESPONSE]:
            if condition_id:
                condition = self.conditions.get(condition_id)
                if condition and not condition(request, result):
                    continue
            
            try:
                if asyncio.iscoroutinefunction(interceptor):
                    result = await interceptor(request, result)
                else:
                    result = interceptor(request, result)
            except Exception as e:
                logger.error(f"Post-response interceptor error: {e}")
        
        return result
    
    async def execute_error(
        self,
        request: Dict[str, Any],
        error: Exception
    ) -> Optional[Dict[str, Any]]:
        """Ejecuta interceptores de error"""
        for priority, interceptor, condition_id in self.interceptors[InterceptorType.ERROR]:
            try:
                if asyncio.iscoroutinefunction(interceptor):
                    result = await interceptor(request, error)
                    if result:
                        return result
                else:
                    result = interceptor(request, error)
                    if result:
                        return result
            except Exception as e:
                logger.error(f"Error interceptor error: {e}")
        
        return None


import asyncio


def get_request_interceptor() -> RequestInterceptor:
    """Obtiene interceptor de requests"""
    return RequestInterceptor()















