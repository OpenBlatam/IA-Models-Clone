"""
Resource Manager - Gestor de Recursos
======================================

Gestión eficiente de recursos del sistema para optimizar memoria y CPU.
"""

import asyncio
import logging
import gc
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Gestor de recursos del sistema.
    
    Monitorea y optimiza el uso de memoria, CPU y otros recursos.
    """
    
    def __init__(self, cleanup_interval: float = 300.0):
        """
        Inicializar gestor de recursos.
        
        Args:
            cleanup_interval: Intervalo en segundos para limpieza automática
        """
        self.cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None
        self._resource_usage: Dict[str, List[float]] = defaultdict(list)
        self._last_cleanup = datetime.now()
        
    async def start(self) -> None:
        """Iniciar gestión automática de recursos"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("🔄 Resource manager started")
    
    async def stop(self) -> None:
        """Detener gestión automática de recursos"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("🛑 Resource manager stopped")
    
    async def _cleanup_loop(self) -> None:
        """Loop de limpieza automática"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def cleanup(self) -> None:
        """
        Limpiar recursos del sistema.
        
        Incluye:
        - Recolección de basura de Python
        - Limpieza de cachés expirados
        - Optimización de memoria
        """
        try:
            # Forzar recolección de basura
            collected = gc.collect()
            
            # Limpiar métricas antiguas
            cutoff_time = datetime.now() - timedelta(hours=24)
            for resource_type in list(self._resource_usage.keys()):
                # Mantener solo las últimas 1000 mediciones
                if len(self._resource_usage[resource_type]) > 1000:
                    self._resource_usage[resource_type] = self._resource_usage[resource_type][-1000:]
            
            self._last_cleanup = datetime.now()
            
            if collected > 0:
                logger.debug(f"🧹 Cleaned up {collected} objects")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def record_resource_usage(self, resource_type: str, value: float) -> None:
        """
        Registrar uso de un recurso.
        
        Args:
            resource_type: Tipo de recurso (memory, cpu, etc.)
            value: Valor del uso
        """
        self._resource_usage[resource_type].append(value)
        
        # Mantener solo las últimas 1000 mediciones
        if len(self._resource_usage[resource_type]) > 1000:
            self._resource_usage[resource_type].pop(0)
    
    def get_resource_stats(self, resource_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estadísticas de uso de recursos.
        
        Args:
            resource_type: Tipo de recurso específico (opcional)
            
        Returns:
            Diccionario con estadísticas
        """
        if resource_type:
            if resource_type not in self._resource_usage:
                return {}
            
            values = self._resource_usage[resource_type]
            if not values:
                return {}
            
            return {
                "type": resource_type,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1] if values else None
            }
        else:
            return {
                resource_type: self.get_resource_stats(resource_type)
                for resource_type in self._resource_usage.keys()
            }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Obtener información de memoria del sistema.
        
        Returns:
            Diccionario con información de memoria
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent(),
                "available": psutil.virtual_memory().available,
                "total": psutil.virtual_memory().total
            }
        except ImportError:
            return {"message": "psutil not available"}
        except Exception as e:
            logger.debug(f"Error getting memory info: {e}")
            return {"error": str(e)}
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """
        Obtener información de CPU.
        
        Returns:
            Diccionario con información de CPU
        """
        try:
            import psutil
            process = psutil.Process()
            
            return {
                "percent": process.cpu_percent(interval=0.1),
                "count": psutil.cpu_count(),
                "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except ImportError:
            return {"message": "psutil not available"}
        except Exception as e:
            logger.debug(f"Error getting CPU info: {e}")
            return {"error": str(e)}
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Obtener información completa del sistema.
        
        Returns:
            Diccionario con información del sistema
        """
        return {
            "memory": self.get_memory_info(),
            "cpu": self.get_cpu_info(),
            "last_cleanup": self._last_cleanup.isoformat(),
            "resource_stats": self.get_resource_stats()
        }

