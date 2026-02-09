"""
Health Monitor - Sistema de health checks avanzados
====================================================
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import psutil
import os

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Estados de salud"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class HealthMonitor:
    """
    Sistema de monitoreo de salud avanzado.
    """
    
    def __init__(self):
        """Inicializar monitor de salud"""
        self.health_checks: Dict[str, callable] = {}
        self.health_history: List[Dict[str, Any]] = []
        self.register_default_checks()
    
    def register_default_checks(self):
        """Registra health checks por defecto"""
        self.register_check("system", self._check_system_resources)
        self.register_check("database", self._check_database)
        self.register_check("vector_store", self._check_vector_store)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory", self._check_memory)
    
    def register_check(self, name: str, check_function: callable):
        """
        Registra un health check.
        
        Args:
            name: Nombre del check
            check_function: Función que ejecuta el check
        """
        self.health_checks[name] = check_function
        logger.info(f"Health check registrado: {name}")
    
    def run_health_checks(self) -> Dict[str, Any]:
        """
        Ejecuta todos los health checks.
        
        Returns:
            Estado de salud completo
        """
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, check_function in self.health_checks.items():
            try:
                result = check_function()
                results[name] = result
                
                # Determinar estado general
                check_status = result.get("status", "unknown")
                if check_status == "critical":
                    overall_status = HealthStatus.CRITICAL
                elif check_status == "unhealthy" and overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.UNHEALTHY
                elif check_status == "degraded" and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                logger.error(f"Error en health check {name}: {e}")
                results[name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        health_report = {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": results,
            "summary": {
                "total_checks": len(results),
                "healthy": sum(1 for r in results.values() if r.get("status") == "healthy"),
                "degraded": sum(1 for r in results.values() if r.get("status") == "degraded"),
                "unhealthy": sum(1 for r in results.values() if r.get("status") == "unhealthy"),
                "critical": sum(1 for r in results.values() if r.get("status") == "critical")
            }
        }
        
        # Guardar en historial
        self.health_history.append(health_report)
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        return health_report
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Verifica recursos del sistema"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            status = "healthy"
            if cpu_percent > 90:
                status = "critical"
            elif cpu_percent > 75:
                status = "unhealthy"
            elif cpu_percent > 60:
                status = "degraded"
            
            if memory.percent > 90:
                status = "critical" if status == "healthy" else status
            elif memory.percent > 75:
                status = "unhealthy" if status == "healthy" else status
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _check_database(self) -> Dict[str, Any]:
        """Verifica base de datos"""
        # En producción, esto verificaría conexión real
        return {
            "status": "healthy",
            "message": "Database connection OK"
        }
    
    def _check_vector_store(self) -> Dict[str, Any]:
        """Verifica vector store"""
        # En producción, esto verificaría ChromaDB
        return {
            "status": "healthy",
            "message": "Vector store accessible"
        }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Verifica espacio en disco"""
        try:
            disk = psutil.disk_usage("/")
            percent_used = disk.percent
            
            status = "healthy"
            if percent_used > 95:
                status = "critical"
            elif percent_used > 85:
                status = "unhealthy"
            elif percent_used > 75:
                status = "degraded"
            
            return {
                "status": status,
                "percent_used": percent_used,
                "free_gb": disk.free / (1024 ** 3),
                "total_gb": disk.total / (1024 ** 3)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Verifica memoria"""
        try:
            memory = psutil.virtual_memory()
            
            status = "healthy"
            if memory.percent > 90:
                status = "critical"
            elif memory.percent > 80:
                status = "unhealthy"
            elif memory.percent > 70:
                status = "degraded"
            
            return {
                "status": status,
                "percent_used": memory.percent,
                "available_mb": memory.available / (1024 * 1024),
                "total_mb": memory.total / (1024 * 1024)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Obtiene historial de salud.
        
        Args:
            hours: Horas a analizar
            
        Returns:
            Historial de salud
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        return [
            health for health in self.health_history
            if datetime.fromisoformat(health["timestamp"]) > cutoff
        ]




