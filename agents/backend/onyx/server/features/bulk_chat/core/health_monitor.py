"""
Health Monitor - Monitor de Salud Avanzado
===========================================

Sistema avanzado de monitoreo de salud del sistema.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("psutil not available, system metrics will be limited")

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Estado de salud."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class ComponentHealth:
    """Salud de un componente."""
    name: str
    status: HealthStatus
    message: str
    last_check: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """Monitor de salud avanzado."""
    
    def __init__(self):
        self.components: Dict[str, ComponentHealth] = {}
        self._lock = asyncio.Lock()
        self.start_time = datetime.now()
    
    async def check_component(
        self,
        name: str,
        check_func: callable,
        *args,
        **kwargs
    ) -> ComponentHealth:
        """
        Verificar salud de un componente.
        
        Args:
            name: Nombre del componente
            check_func: Función de verificación (async)
            *args, **kwargs: Argumentos para la función
        
        Returns:
            ComponentHealth
        """
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func(*args, **kwargs)
            else:
                result = check_func(*args, **kwargs)
            
            status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            
            health = ComponentHealth(
                name=name,
                status=status,
                message="Component is healthy" if result else "Component check failed",
                last_check=datetime.now(),
            )
        except Exception as e:
            health = ComponentHealth(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Error: {str(e)}",
                last_check=datetime.now(),
            )
        
        async with self._lock:
            self.components[name] = health
        
        return health
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Obtener salud completa del sistema."""
        # Verificar recursos del sistema
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
        else:
            # Valores por defecto si psutil no está disponible
            cpu_percent = 0
            memory = type('obj', (object,), {'percent': 0, 'available': 0})()
            disk = type('obj', (object,), {'percent': 0, 'free': 0})()
        
        # Verificar componentes
        async with self._lock:
            components_status = {
                name: {
                    "status": comp.status.value,
                    "message": comp.message,
                    "last_check": comp.last_check.isoformat(),
                }
                for name, comp in self.components.items()
            }
        
        # Determinar estado general
        overall_status = HealthStatus.HEALTHY
        
        if cpu_percent > 90 or memory.percent > 90:
            overall_status = HealthStatus.CRITICAL
        elif cpu_percent > 70 or memory.percent > 70:
            overall_status = HealthStatus.DEGRADED
        
        # Verificar componentes
        for comp in self.components.values():
            if comp.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif comp.status == HealthStatus.UNHEALTHY and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "status": overall_status.value,
            "uptime_seconds": uptime,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024),
            },
            "components": components_status,
            "timestamp": datetime.now().isoformat(),
        }
    
    async def register_component(
        self,
        name: str,
        check_func: callable,
        interval: float = 60.0,
    ):
        """Registrar componente para monitoreo periódico."""
        async def monitor_loop():
            while True:
                try:
                    await self.check_component(name, check_func)
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error monitoring component {name}: {e}")
                    await asyncio.sleep(interval)
        
        asyncio.create_task(monitor_loop())
        logger.info(f"Registered health check for component: {name}")

