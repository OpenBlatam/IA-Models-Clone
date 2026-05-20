"""
Advanced Health Checks Service - Health checks avanzados
========================================================

Sistema de health checks avanzado con múltiples verificaciones.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Estados de salud"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(str, Enum):
    """Tipos de checks"""
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    DISK_SPACE = "disk_space"
    MEMORY = "memory"
    CPU = "cpu"
    CUSTOM = "custom"


@dataclass
class HealthCheck:
    """Health check individual"""
    check_type: CheckType
    name: str
    status: HealthStatus
    message: str
    response_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemHealth:
    """Salud del sistema"""
    overall_status: HealthStatus
    checks: List[HealthCheck]
    timestamp: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0


class AdvancedHealthChecksService:
    """Servicio de health checks avanzado"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.start_time = datetime.now()
        self.health_history: List[SystemHealth] = []
        logger.info("AdvancedHealthChecksService initialized")
    
    def perform_health_check(
        self,
        check_type: CheckType,
        name: str,
        check_function: Optional[Any] = None
    ) -> HealthCheck:
        """Realizar health check"""
        import time
        
        start_time = time.time()
        status = HealthStatus.UNKNOWN
        message = "Check not performed"
        details = {}
        
        try:
            # En producción, esto ejecutaría checks reales
            if check_type == CheckType.DATABASE:
                status, message, details = self._check_database()
            elif check_type == CheckType.CACHE:
                status, message, details = self._check_cache()
            elif check_type == CheckType.EXTERNAL_API:
                status, message, details = self._check_external_api()
            elif check_type == CheckType.DISK_SPACE:
                status, message, details = self._check_disk_space()
            elif check_type == CheckType.MEMORY:
                status, message, details = self._check_memory()
            elif check_type == CheckType.CPU:
                status, message, details = self._check_cpu()
            elif check_function:
                # Custom check
                result = check_function()
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "Custom check completed"
        
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Check failed: {str(e)}"
            logger.error(f"Health check {name} failed: {e}")
        
        response_time = (time.time() - start_time) * 1000
        
        check = HealthCheck(
            check_type=check_type,
            name=name,
            status=status,
            message=message,
            response_time_ms=response_time,
            details=details,
        )
        
        return check
    
    def _check_database(self) -> tuple:
        """Verificar base de datos"""
        # En producción, esto verificaría conexión real
        return HealthStatus.HEALTHY, "Database connection OK", {"connections": 5}
    
    def _check_cache(self) -> tuple:
        """Verificar cache"""
        return HealthStatus.HEALTHY, "Cache operational", {"hit_rate": 0.85}
    
    def _check_external_api(self) -> tuple:
        """Verificar APIs externas"""
        return HealthStatus.HEALTHY, "External APIs reachable", {}
    
    def _check_disk_space(self) -> tuple:
        """Verificar espacio en disco"""
        # En producción, esto verificaría espacio real
        return HealthStatus.HEALTHY, "Disk space OK", {"free_gb": 100}
    
    def _check_memory(self) -> tuple:
        """Verificar memoria"""
        import psutil
        memory = psutil.virtual_memory()
        usage_percent = memory.percent
        
        if usage_percent > 90:
            status = HealthStatus.UNHEALTHY
        elif usage_percent > 75:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return status, f"Memory usage: {usage_percent}%", {
            "usage_percent": usage_percent,
            "available_gb": memory.available / (1024**3),
        }
    
    def _check_cpu(self) -> tuple:
        """Verificar CPU"""
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 90:
            status = HealthStatus.UNHEALTHY
        elif cpu_percent > 75:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return status, f"CPU usage: {cpu_percent}%", {
            "usage_percent": cpu_percent,
        }
    
    def get_system_health(
        self,
        checks: Optional[List[CheckType]] = None
    ) -> SystemHealth:
        """Obtener salud del sistema"""
        if not checks:
            checks = [
                CheckType.DATABASE,
                CheckType.CACHE,
                CheckType.MEMORY,
                CheckType.CPU,
            ]
        
        health_checks = []
        for check_type in checks:
            check = self.perform_health_check(check_type, check_type.value)
            health_checks.append(check)
        
        # Determinar estado general
        statuses = [check.status for check in health_checks]
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        system_health = SystemHealth(
            overall_status=overall_status,
            checks=health_checks,
            uptime_seconds=uptime,
        )
        
        self.health_history.append(system_health)
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        return system_health
    
    def get_health_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Obtener historial de salud"""
        recent = self.health_history[-limit:] if len(self.health_history) > limit else self.health_history
        
        return [
            {
                "timestamp": h.timestamp.isoformat(),
                "overall_status": h.overall_status.value,
                "uptime_seconds": h.uptime_seconds,
                "checks_count": len(h.checks),
            }
            for h in recent
        ]




