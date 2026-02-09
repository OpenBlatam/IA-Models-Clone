"""
Resource Manager
================

Gestor de recursos del sistema.
"""

import time
import psutil
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Gestor de recursos del sistema.
    
    Monitorea y gestiona recursos (CPU, memoria, etc.).
    """
    
    def __init__(self):
        """Inicializar gestor de recursos."""
        self.process = psutil.Process()
        self.start_time = time.time()
    
    def get_cpu_usage(self) -> float:
        """
        Obtener uso de CPU.
        
        Returns:
            Porcentaje de CPU usado
        """
        try:
            return self.process.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Obtener uso de memoria.
        
        Returns:
            Diccionario con información de memoria
        """
        try:
            memory_info = self.process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return {
                "rss": memory_info.rss,  # Resident Set Size (bytes)
                "vms": memory_info.vms,  # Virtual Memory Size (bytes)
                "percent": self.process.memory_percent(),
                "system_total": system_memory.total,
                "system_available": system_memory.available,
                "system_percent": system_memory.percent
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {}
    
    def get_disk_usage(self, path: str = ".") -> Dict[str, Any]:
        """
        Obtener uso de disco.
        
        Args:
            path: Ruta a verificar
            
        Returns:
            Diccionario con información de disco
        """
        try:
            disk_usage = psutil.disk_usage(path)
            return {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent": disk_usage.percent
            }
        except Exception as e:
            logger.error(f"Error getting disk usage: {e}")
            return {}
    
    def get_network_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de red.
        
        Returns:
            Diccionario con estadísticas de red
        """
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except Exception as e:
            logger.error(f"Error getting network stats: {e}")
            return {}
    
    def get_system_resources(self) -> Dict[str, Any]:
        """
        Obtener todos los recursos del sistema.
        
        Returns:
            Diccionario completo de recursos
        """
        return {
            "cpu": {
                "usage_percent": self.get_cpu_usage(),
                "count": psutil.cpu_count()
            },
            "memory": self.get_memory_usage(),
            "disk": self.get_disk_usage(),
            "network": self.get_network_stats(),
            "uptime": time.time() - self.start_time
        }
    
    def check_resource_limits(
        self,
        max_cpu_percent: float = 90.0,
        max_memory_percent: float = 90.0,
        max_disk_percent: float = 90.0
    ) -> Dict[str, bool]:
        """
        Verificar límites de recursos.
        
        Args:
            max_cpu_percent: Máximo de CPU permitido
            max_memory_percent: Máximo de memoria permitido
            max_disk_percent: Máximo de disco permitido
            
        Returns:
            Diccionario de {resource: within_limit}
        """
        cpu_usage = self.get_cpu_usage()
        memory = self.get_memory_usage()
        disk = self.get_disk_usage()
        
        return {
            "cpu": cpu_usage < max_cpu_percent,
            "memory": memory.get("percent", 0) < max_memory_percent,
            "disk": disk.get("percent", 0) < max_disk_percent
        }


# Instancia global
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Obtener instancia global del gestor de recursos."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager






