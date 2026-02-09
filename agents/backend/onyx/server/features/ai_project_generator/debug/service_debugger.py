"""
Service Debugger - Debugger de servicios
=======================================

Herramientas para debugging de servicios.
"""

import logging
import time
from typing import Optional, Dict, Any, Callable
from functools import wraps
import asyncio

from .debug_logger import get_debug_logger

logger = logging.getLogger(__name__)


class ServiceDebugger:
    """Debugger para servicios"""
    
    def __init__(self):
        self.debug_logger = get_debug_logger()
        self.service_calls: list = []
    
    def debug_service_call(
        self,
        service_name: str,
        method_name: str
    ):
        """
        Decorator para debuggear llamadas a servicios.
        
        Example:
            @service_debugger.debug_service_call("ProjectService", "create_project")
            async def create_project(...):
                pass
        """
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                self.debug_logger.debug(
                    f"Service call started: {service_name}.{method_name}",
                    service=service_name,
                    method=method_name,
                    args=str(args)[:200],
                    kwargs=str(kwargs)[:200]
                )
                
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.debug_logger.debug(
                        f"Service call completed: {service_name}.{method_name} ({duration:.4f}s)",
                        service=service_name,
                        method=method_name,
                        duration=duration,
                        success=True
                    )
                    
                    self.service_calls.append({
                        "service": service_name,
                        "method": method_name,
                        "duration": duration,
                        "success": True,
                        "timestamp": time.time()
                    })
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    
                    self.debug_logger.error(
                        f"Service call failed: {service_name}.{method_name} ({duration:.4f}s)",
                        service=service_name,
                        method=method_name,
                        duration=duration,
                        success=False,
                        error=str(e)
                    )
                    
                    self.service_calls.append({
                        "service": service_name,
                        "method": method_name,
                        "duration": duration,
                        "success": False,
                        "error": str(e),
                        "timestamp": time.time()
                    })
                    
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                self.debug_logger.debug(
                    f"Service call started: {service_name}.{method_name}",
                    service=service_name,
                    method=method_name
                )
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.debug_logger.debug(
                        f"Service call completed: {service_name}.{method_name} ({duration:.4f}s)",
                        service=service_name,
                        method=method_name,
                        duration=duration,
                        success=True
                    )
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.debug_logger.error(
                        f"Service call failed: {service_name}.{method_name}",
                        service=service_name,
                        method=method_name,
                        duration=duration,
                        success=False,
                        error=str(e)
                    )
                    raise
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def get_service_calls(self, limit: int = 100) -> list:
        """Obtiene llamadas recientes a servicios"""
        return self.service_calls[-limit:]
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de servicios"""
        stats = {}
        for call in self.service_calls:
            key = f"{call['service']}.{call['method']}"
            if key not in stats:
                stats[key] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "total_duration": 0,
                    "avg_duration": 0
                }
            
            stats[key]["total_calls"] += 1
            if call["success"]:
                stats[key]["successful_calls"] += 1
            else:
                stats[key]["failed_calls"] += 1
            stats[key]["total_duration"] += call["duration"]
        
        # Calcular promedios
        for key in stats:
            stats[key]["avg_duration"] = (
                stats[key]["total_duration"] / stats[key]["total_calls"]
            )
        
        return stats


# Instancia global
_service_debugger: Optional[ServiceDebugger] = None


def get_service_debugger() -> ServiceDebugger:
    """Obtiene instancia de service debugger"""
    global _service_debugger
    if _service_debugger is None:
        _service_debugger = ServiceDebugger()
    return _service_debugger















