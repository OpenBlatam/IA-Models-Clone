"""
MCP Health Checks - Health checks avanzados
============================================
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from .exceptions import MCPError

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Estados de health"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheck(BaseModel):
    """Resultado de un health check individual"""
    name: str = Field(..., description="Nombre del check")
    status: HealthStatus = Field(..., description="Estado del check")
    message: Optional[str] = Field(None, description="Mensaje descriptivo")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = Field(None, description="Duración en milisegundos")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata adicional")


class HealthReport(BaseModel):
    """Reporte completo de health"""
    status: HealthStatus = Field(..., description="Estado general")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="Versión del servidor")
    checks: List[HealthCheck] = Field(default_factory=list, description="Checks individuales")
    uptime_seconds: Optional[float] = Field(None, description="Tiempo de actividad en segundos")


class HealthChecker:
    """
    Gestor de health checks
    
    Permite registrar y ejecutar múltiples health checks.
    """
    
    def __init__(self, server_start_time: Optional[datetime] = None):
        """
        Args:
            server_start_time: Tiempo de inicio del servidor (para calcular uptime)
        """
        self.server_start_time = server_start_time or datetime.utcnow()
        self._checks: Dict[str, callable] = {}
    
    def register_check(self, name: str, check_func: callable):
        """
        Registra un health check
        
        Args:
            name: Nombre del check
            check_func: Función que ejecuta el check (debe retornar HealthCheck o dict)
        """
        self._checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> HealthCheck:
        """
        Ejecuta un health check específico
        
        Args:
            name: Nombre del check
            
        Returns:
            HealthCheck con resultado
        """
        check_func = self._checks.get(name)
        if not check_func:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check {name} not found",
            )
        
        start_time = datetime.utcnow()
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Convertir resultado a HealthCheck si es necesario
            if isinstance(result, dict):
                result = HealthCheck(**result)
            elif not isinstance(result, HealthCheck):
                result = HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message=str(result),
                )
            
            result.duration_ms = duration
            return result
            
        except Exception as e:
            logger.error(f"Health check {name} failed: {e}")
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=duration,
            )
    
    async def run_all_checks(self) -> HealthReport:
        """
        Ejecuta todos los health checks
        
        Returns:
            HealthReport con todos los resultados
        """
        checks = []
        
        for name in self._checks.keys():
            check = await self.run_check(name)
            checks.append(check)
        
        # Determinar estado general
        status = self._determine_overall_status(checks)
        
        # Calcular uptime
        uptime = (datetime.utcnow() - self.server_start_time).total_seconds()
        
        return HealthReport(
            status=status,
            version="1.3.0",  # TODO: obtener de __version__
            checks=checks,
            uptime_seconds=uptime,
        )
    
    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """
        Determina estado general basado en checks individuales
        
        Args:
            checks: Lista de checks
            
        Returns:
            Estado general
        """
        if not checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in checks]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN


# Health checks comunes

async def check_database_health(connection_string: Optional[str] = None) -> HealthCheck:
    """Health check para base de datos"""
    if not connection_string:
        return HealthCheck(
            name="database",
            status=HealthStatus.UNKNOWN,
            message="Database connection string not configured",
        )
    
    try:
        # Intentar conexión (implementar según driver)
        return HealthCheck(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connection OK",
        )
    except Exception as e:
        return HealthCheck(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {e}",
        )


async def check_cache_health(cache: Optional[Any] = None) -> HealthCheck:
    """Health check para cache"""
    if not cache:
        return HealthCheck(
            name="cache",
            status=HealthStatus.UNKNOWN,
            message="Cache not configured",
        )
    
    try:
        # Verificar cache
        stats = cache.get_stats() if hasattr(cache, "get_stats") else {}
        return HealthCheck(
            name="cache",
            status=HealthStatus.HEALTHY,
            message="Cache OK",
            metadata=stats,
        )
    except Exception as e:
        return HealthCheck(
            name="cache",
            status=HealthStatus.UNHEALTHY,
            message=f"Cache check failed: {e}",
        )


async def check_memory_health(threshold_mb: int = 1024) -> HealthCheck:
    """Health check para memoria"""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        status = HealthStatus.HEALTHY
        if memory_mb > threshold_mb:
            status = HealthStatus.DEGRADED
        
        return HealthCheck(
            name="memory",
            status=status,
            message=f"Memory usage: {memory_mb:.2f}MB",
            metadata={"memory_mb": memory_mb, "threshold_mb": threshold_mb},
        )
    except ImportError:
        return HealthCheck(
            name="memory",
            status=HealthStatus.UNKNOWN,
            message="psutil not available",
        )

