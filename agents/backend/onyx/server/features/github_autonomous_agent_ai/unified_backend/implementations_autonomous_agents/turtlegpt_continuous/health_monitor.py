"""
Health Monitor Module
====================

Monitorea la salud del agente y sus componentes.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Estado de salud."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Salud de un componente."""
    name: str
    status: HealthStatus
    last_check: datetime
    details: Dict[str, Any]
    error: Optional[str] = None


class HealthMonitor:
    """
    Monitor de salud del agente.
    
    Verifica el estado de todos los componentes y proporciona
    información sobre la salud general del sistema.
    """
    
    def __init__(self):
        """Inicializar monitor de salud."""
        self.component_health: Dict[str, ComponentHealth] = {}
        self.health_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        self._health_checks: Dict[str, Callable[[], Dict[str, Any]]] = {}
    
    def register_component(
        self,
        name: str,
        health_check: Callable[[], Dict[str, Any]]
    ):
        """
        Registrar un componente para monitoreo.
        
        Args:
            name: Nombre del componente
            health_check: Función que retorna información de salud
        """
        self.component_health[name] = ComponentHealth(
            name=name,
            status=HealthStatus.UNKNOWN,
            last_check=datetime.now(),
            details={}
        )
        self._health_checks[name] = health_check
        logger.debug(f"Registered health check for component: {name}")
    
    def check_component(self, name: str) -> ComponentHealth:
        """
        Verificar salud de un componente.
        
        Args:
            name: Nombre del componente
            
        Returns:
            ComponentHealth del componente
        """
        if name not in self.component_health:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                last_check=datetime.now(),
                details={"error": "Component not registered"}
            )
        
        try:
            health_check = self._health_checks.get(name)
            if health_check:
                details = health_check()
                status = self._determine_status(details)
            else:
                details = {}
                status = HealthStatus.UNKNOWN
            
            self.component_health[name] = ComponentHealth(
                name=name,
                status=status,
                last_check=datetime.now(),
                details=details
            )
            
            return self.component_health[name]
            
        except Exception as e:
            logger.error(f"Error checking health of {name}: {e}")
            self.component_health[name] = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.now(),
                details={},
                error=str(e)
            )
            return self.component_health[name]
    
    def check_all_components(self) -> Dict[str, ComponentHealth]:
        """
        Verificar salud de todos los componentes.
        
        Returns:
            Dict con salud de todos los componentes
        """
        results = {}
        for name in self.component_health.keys():
            results[name] = self.check_component(name)
        
        # Guardar en historial
        self._save_health_snapshot(results)
        
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """
        Obtener estado de salud general.
        
        Returns:
            HealthStatus general
        """
        if not self.component_health:
            return HealthStatus.UNKNOWN
        
        statuses = [comp.status for comp in self.component_health.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Obtener reporte completo de salud.
        
        Returns:
            Dict con información detallada de salud
        """
        self.check_all_components()
        
        return {
            "overall_status": self.get_overall_health().value,
            "components": {
                name: {
                    "status": comp.status.value,
                    "last_check": comp.last_check.isoformat(),
                    "details": comp.details,
                    "error": comp.error
                }
                for name, comp in self.component_health.items()
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _determine_status(self, details: Dict[str, Any]) -> HealthStatus:
        """
        Determinar estado basado en detalles.
        
        Args:
            details: Detalles del componente
            
        Returns:
            HealthStatus determinado
        """
        if details.get("error"):
            return HealthStatus.UNHEALTHY
        
        # Verificar métricas si están disponibles
        if "metrics" in details:
            metrics = details["metrics"]
            if metrics.get("error_rate", 0) > 0.5:
                return HealthStatus.UNHEALTHY
            elif metrics.get("error_rate", 0) > 0.2:
                return HealthStatus.DEGRADED
        
        # Verificar si está activo
        if "is_active" in details and not details["is_active"]:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def _save_health_snapshot(self, components: Dict[str, ComponentHealth]):
        """Guardar snapshot de salud en historial."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": self.get_overall_health().value,
            "components": {
                name: {
                    "status": comp.status.value,
                    "details": comp.details
                }
                for name, comp in components.items()
            }
        }
        
        self.health_history.append(snapshot)
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
