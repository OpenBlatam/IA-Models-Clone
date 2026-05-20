"""
Health Check - Verificación de Salud del Sistema
=================================================

Sistema robusto de health checks para monitoreo del estado del agente.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Estados de salud del sistema"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth:
    """Salud de un componente individual"""
    
    def __init__(self, name: str):
        self.name = name
        self.status = HealthStatus.UNKNOWN
        self.last_check = None
        self.message = ""
        self.details: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "message": self.message,
            "details": self.details
        }


class HealthChecker:
    """
    Verificador de salud del sistema.
    """
    
    def __init__(self, agent=None):
        self.agent = agent
        self.components: Dict[str, ComponentHealth] = {}
        self.overall_status = HealthStatus.UNKNOWN
        self.last_full_check = None
        self.check_history: List[Dict[str, Any]] = []
        self.max_history = 100
    
    def register_component(self, name: str) -> ComponentHealth:
        """
        Registrar un componente para verificación de salud.
        
        Args:
            name: Nombre del componente
            
        Returns:
            ComponentHealth para el componente
        """
        component = ComponentHealth(name)
        self.components[name] = component
        return component
    
    async def check_component(self, name: str, check_func: Callable) -> ComponentHealth:
        """
        Verificar salud de un componente.
        
        Args:
            name: Nombre del componente
            check_func: Función async que retorna (status, message, details)
            
        Returns:
            ComponentHealth actualizado
        """
        component = self.components.get(name)
        if not component:
            component = self.register_component(name)
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                status, message, details = await check_func()
            else:
                status, message, details = check_func()
            
            component.status = status
            component.message = message
            component.details = details or {}
            component.last_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Error checking component {name}: {e}")
            component.status = HealthStatus.UNHEALTHY
            component.message = f"Check failed: {str(e)}"
            component.last_check = datetime.now()
        
        return component
    
    async def check_all(self) -> Dict[str, Any]:
        """
        Verificar salud de todos los componentes y calcular estado general.
        
        Returns:
            Diccionario con estado de salud completo
        """
        check_start = time.time()
        
        if self.agent:
            await self._check_agent()
        
        await self._check_system()
        
        self._calculate_overall_status()
        self.last_full_check = datetime.now()
        
        check_duration = time.time() - check_start
        
        result = {
            "status": self.overall_status.value,
            "timestamp": self.last_full_check.isoformat(),
            "check_duration_ms": round(check_duration * 1000, 2),
            "components": {
                name: comp.to_dict()
                for name, comp in self.components.items()
            }
        }
        
        self.check_history.append(result)
        if len(self.check_history) > self.max_history:
            self.check_history.pop(0)
        
        return result
    
    async def _check_agent(self) -> None:
        """Verificar salud del agente"""
        if not self.agent:
            return
        
        try:
            status = await self.agent.get_status()
            
            is_healthy = (
                status.get("running", False) and
                status.get("tasks_failed", 0) < status.get("tasks_total", 0) * 0.5
            )
            
            component = self.components.get("agent") or self.register_component("agent")
            component.status = HealthStatus.HEALTHY if is_healthy else HealthStatus.DEGRADED
            component.message = f"Agent is {status.get('status', 'unknown')}"
            component.details = {
                "running": status.get("running", False),
                "tasks_total": status.get("tasks_total", 0),
                "tasks_pending": status.get("tasks_pending", 0),
                "tasks_running": status.get("tasks_running", 0),
                "tasks_completed": status.get("tasks_completed", 0),
                "tasks_failed": status.get("tasks_failed", 0)
            }
            component.last_check = datetime.now()
            
        except Exception as e:
            component = self.components.get("agent") or self.register_component("agent")
            component.status = HealthStatus.UNHEALTHY
            component.message = f"Agent check failed: {str(e)}"
            component.last_check = datetime.now()
    
    async def _check_system(self) -> None:
        """Verificar salud del sistema"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            component = self.components.get("system") or self.register_component("system")
            
            is_healthy = (
                cpu_percent < 90 and
                memory.percent < 90 and
                disk.percent < 90
            )
            
            component.status = HealthStatus.HEALTHY if is_healthy else HealthStatus.DEGRADED
            component.message = "System resources within normal limits" if is_healthy else "System resources under stress"
            component.details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
            component.last_check = datetime.now()
            
        except ImportError:
            component = self.components.get("system") or self.register_component("system")
            component.status = HealthStatus.UNKNOWN
            component.message = "psutil not available, system check skipped"
            component.last_check = datetime.now()
        except Exception as e:
            component = self.components.get("system") or self.register_component("system")
            component.status = HealthStatus.UNHEALTHY
            component.message = f"System check failed: {str(e)}"
            component.last_check = datetime.now()
    
    def _calculate_overall_status(self) -> None:
        """Calcular estado general basado en componentes"""
        if not self.components:
            self.overall_status = HealthStatus.UNKNOWN
            return
        
        statuses = [comp.status for comp in self.components.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            self.overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            self.overall_status = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            self.overall_status = HealthStatus.HEALTHY
        else:
            self.overall_status = HealthStatus.UNKNOWN
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado de salud actual sin ejecutar checks.
        
        Returns:
            Estado de salud actual
        """
        return {
            "status": self.overall_status.value,
            "last_check": self.last_full_check.isoformat() if self.last_full_check else None,
            "components": {
                name: comp.to_dict()
                for name, comp in self.components.items()
            }
        }
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener historial de checks de salud.
        
        Args:
            limit: Número máximo de entradas a retornar
            
        Returns:
            Lista de checks recientes
        """
        return self.check_history[-limit:]
