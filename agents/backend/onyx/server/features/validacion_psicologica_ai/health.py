"""
Health Checks y Monitoring para Validación Psicológica AI
==========================================================
Sistema de health checks y monitoreo del sistema
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import structlog
import asyncio

from .service import PsychologicalValidationService
from .utils import metrics
from .config import config

logger = structlog.get_logger()


class HealthStatus(str, Enum):
    """Estado de salud"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth:
    """Salud de un componente"""
    
    def __init__(
        self,
        name: str,
        status: HealthStatus,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.status = status
        self.message = message
        self.details = details or {}
        self.checked_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat()
        }


class HealthChecker:
    """Verificador de salud del sistema"""
    
    def __init__(self, service: PsychologicalValidationService):
        """
        Inicializar verificador
        
        Args:
            service: Servicio de validación
        """
        self.service = service
        self._last_check: Optional[datetime] = None
        logger.info("HealthChecker initialized")
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Verificar salud del sistema
        
        Returns:
            Estado de salud completo
        """
        components = []
        
        # Verificar servicio
        service_health = await self._check_service()
        components.append(service_health)
        
        # Verificar métricas
        metrics_health = self._check_metrics()
        components.append(metrics_health)
        
        # Verificar configuración
        config_health = self._check_config()
        components.append(config_health)
        
        # Determinar estado general
        statuses = [c.status for c in components]
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        self._last_check = datetime.utcnow()
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "components": [c.to_dict() for c in components],
            "uptime_seconds": (
                (datetime.utcnow() - metrics._start_time).total_seconds()
                if hasattr(metrics, '_start_time') else 0
            )
        }
    
    async def _check_service(self) -> ComponentHealth:
        """Verificar servicio"""
        try:
            # Verificar que el servicio puede crear una validación de prueba
            # (simplificado - en producción hacer checks más robustos)
            if hasattr(self.service, '_validations'):
                return ComponentHealth(
                    name="service",
                    status=HealthStatus.HEALTHY,
                    message="Service is operational",
                    details={"validations_count": len(self.service._validations)}
                )
            else:
                return ComponentHealth(
                    name="service",
                    status=HealthStatus.UNHEALTHY,
                    message="Service structure not found"
                )
        except Exception as e:
            return ComponentHealth(
                name="service",
                status=HealthStatus.UNHEALTHY,
                message=f"Service check failed: {str(e)}"
            )
    
    def _check_metrics(self) -> ComponentHealth:
        """Verificar métricas"""
        try:
            all_metrics = metrics.get_all()
            
            # Verificar tasas de éxito
            success_rate = all_metrics.get("success_rate", 0.0)
            
            if success_rate >= 0.9:
                status = HealthStatus.HEALTHY
                message = "Metrics indicate healthy system"
            elif success_rate >= 0.7:
                status = HealthStatus.DEGRADED
                message = "Metrics indicate degraded performance"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Metrics indicate unhealthy system"
            
            return ComponentHealth(
                name="metrics",
                status=status,
                message=message,
                details={
                    "success_rate": success_rate,
                    "total_validations": all_metrics.get("validations_created", 0),
                    "completed": all_metrics.get("validations_completed", 0)
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="metrics",
                status=HealthStatus.UNKNOWN,
                message=f"Metrics check failed: {str(e)}"
            )
    
    def _check_config(self) -> ComponentHealth:
        """Verificar configuración"""
        try:
            issues = []
            
            # Verificar configuraciones críticas
            if not config.encryption_key or config.encryption_key == "default-key-change-in-production":
                issues.append("Default encryption key in use")
            
            if config.debug:
                issues.append("Debug mode enabled")
            
            if issues:
                return ComponentHealth(
                    name="configuration",
                    status=HealthStatus.DEGRADED,
                    message="Configuration issues detected",
                    details={"issues": issues}
                )
            else:
                return ComponentHealth(
                    name="configuration",
                    status=HealthStatus.HEALTHY,
                    message="Configuration is valid"
                )
        except Exception as e:
            return ComponentHealth(
                name="configuration",
                status=HealthStatus.UNKNOWN,
                message=f"Configuration check failed: {str(e)}"
            )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de salud
        
        Returns:
            Resumen
        """
        return {
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "status": "operational" if self._last_check else "unknown"
        }


# Instancia global del verificador de salud
health_checker = None  # Se inicializa con el servicio




