"""
Resource Management
===================
Utilidades para gestión de recursos.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio

from .logger import get_logger

logger = get_logger(__name__)


class ResourcePool:
    """Pool de recursos para reutilización."""
    
    def __init__(self, max_size: int = 10):
        """
        Inicializar pool de recursos.
        
        Args:
            max_size: Tamaño máximo del pool
        """
        self.max_size = max_size
        self.resources: List[Any] = []
        self.in_use: set = set()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> Any:
        """
        Adquirir recurso del pool.
        
        Returns:
            Recurso disponible
        """
        async with self._lock:
            # Buscar recurso disponible
            for resource in self.resources:
                if resource not in self.in_use:
                    self.in_use.add(resource)
                    return resource
            
            # Crear nuevo recurso si hay espacio
            if len(self.resources) < self.max_size:
                resource = await self._create_resource()
                self.resources.append(resource)
                self.in_use.add(resource)
                return resource
            
            # Esperar hasta que haya un recurso disponible
            while True:
                await asyncio.sleep(0.1)
                for resource in self.resources:
                    if resource not in self.in_use:
                        self.in_use.add(resource)
                        return resource
    
    async def release(self, resource: Any):
        """
        Liberar recurso al pool.
        
        Args:
            resource: Recurso a liberar
        """
        async with self._lock:
            if resource in self.in_use:
                self.in_use.remove(resource)
                await self._cleanup_resource(resource)
    
    async def _create_resource(self) -> Any:
        """Crear nuevo recurso (sobrescribir en subclases)."""
        raise NotImplementedError
    
    async def _cleanup_resource(self, resource: Any):
        """Limpiar recurso (sobrescribir en subclases)."""
        pass
    
    @asynccontextmanager
    async def get_resource(self):
        """Context manager para obtener recurso."""
        resource = await self.acquire()
        try:
            yield resource
        finally:
            await self.release(resource)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del pool."""
        return {
            "total_resources": len(self.resources),
            "in_use": len(self.in_use),
            "available": len(self.resources) - len(self.in_use),
            "max_size": self.max_size,
            "usage_percent": (len(self.in_use) / self.max_size * 100) if self.max_size > 0 else 0
        }


class ResourceMonitor:
    """Monitor de recursos del sistema."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.alerts: List[Dict[str, Any]] = []
    
    def record_metric(self, name: str, value: float):
        """
        Registrar métrica de recurso.
        
        Args:
            name: Nombre de la métrica
            value: Valor
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        
        # Mantener solo últimos 1000 valores
        if len(self.metrics[name]) > 1000:
            self.metrics[name].pop(0)
    
    def check_threshold(
        self,
        metric_name: str,
        threshold: float,
        operator: str = ">"
    ) -> bool:
        """
        Verificar si métrica excede threshold.
        
        Args:
            metric_name: Nombre de la métrica
            threshold: Valor threshold
            operator: Operador (>, <, >=, <=)
            
        Returns:
            True si excede threshold
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return False
        
        current_value = self.metrics[metric_name][-1]
        
        if operator == ">":
            return current_value > threshold
        elif operator == "<":
            return current_value < threshold
        elif operator == ">=":
            return current_value >= threshold
        elif operator == "<=":
            return current_value <= threshold
        
        return False
    
    def get_metric_summary(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Obtener resumen de métrica."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        values = self.metrics[metric_name]
        
        return {
            "current": values[-1] if values else None,
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }

