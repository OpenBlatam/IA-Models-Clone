"""
MCP Dependency Health Checks - Health checks con dependencias
==============================================================
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from .health import HealthCheck, HealthStatus

logger = logging.getLogger(__name__)


class DependencyType(str, Enum):
    """Tipos de dependencias"""
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    MESSAGE_QUEUE = "message_queue"
    STORAGE = "storage"


class DependencyHealthCheck(BaseModel):
    """Health check de dependencia"""
    dependency_id: str = Field(..., description="ID de la dependencia")
    dependency_type: DependencyType = Field(..., description="Tipo de dependencia")
    name: str = Field(..., description="Nombre de la dependencia")
    check_function: Any = Field(..., description="Función de check (callable)")
    timeout: float = Field(default=5.0, description="Timeout en segundos")
    critical: bool = Field(default=True, description="Si es crítica")
    last_check: Optional[datetime] = None
    last_status: Optional[HealthStatus] = None


class DependencyHealthChecker:
    """
    Health checker de dependencias
    
    Verifica salud de dependencias externas.
    """
    
    def __init__(self):
        self._dependencies: Dict[str, DependencyHealthCheck] = {}
    
    def register_dependency(
        self,
        dependency_id: str,
        dependency_type: DependencyType,
        name: str,
        check_function: Callable,
        timeout: float = 5.0,
        critical: bool = True,
    ):
        """
        Registra una dependencia
        
        Args:
            dependency_id: ID de la dependencia
            dependency_type: Tipo de dependencia
            name: Nombre de la dependencia
            check_function: Función que verifica salud
            timeout: Timeout en segundos
            critical: Si es crítica
        """
        dependency = DependencyHealthCheck(
            dependency_id=dependency_id,
            dependency_type=dependency_type,
            name=name,
            check_function=check_function,
            timeout=timeout,
            critical=critical,
        )
        
        self._dependencies[dependency_id] = dependency
        logger.info(f"Registered dependency health check: {name}")
    
    async def check_dependency(self, dependency_id: str) -> HealthCheck:
        """
        Verifica salud de una dependencia
        
        Args:
            dependency_id: ID de la dependencia
            
        Returns:
            HealthCheck
        """
        dependency = self._dependencies.get(dependency_id)
        
        if not dependency:
            return HealthCheck(
                name=f"Unknown dependency: {dependency_id}",
                status=HealthStatus.UNKNOWN,
                message="Dependency not registered",
            )
        
        try:
            # Ejecutar check con timeout
            if asyncio.iscoroutinefunction(dependency.check_function):
                result = await asyncio.wait_for(
                    dependency.check_function(),
                    timeout=dependency.timeout,
                )
            else:
                result = dependency.check_function()
            
            # Actualizar estado
            dependency.last_check = datetime.utcnow()
            dependency.last_status = HealthStatus.HEALTHY
            
            return HealthCheck(
                name=dependency.name,
                status=HealthStatus.HEALTHY,
                message="Dependency is healthy",
                metadata={"dependency_type": dependency.dependency_type.value},
            )
            
        except asyncio.TimeoutError:
            dependency.last_status = HealthStatus.UNHEALTHY
            return HealthCheck(
                name=dependency.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {dependency.timeout}s",
                metadata={"dependency_type": dependency.dependency_type.value},
            )
            
        except Exception as e:
            dependency.last_status = HealthStatus.UNHEALTHY
            return HealthCheck(
                name=dependency.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                metadata={"dependency_type": dependency.dependency_type.value},
            )
    
    async def check_all_dependencies(self) -> List[HealthCheck]:
        """
        Verifica todas las dependencias
        
        Returns:
            Lista de health checks
        """
        tasks = [
            self.check_dependency(dep_id)
            for dep_id in self._dependencies.keys()
        ]
        
        return await asyncio.gather(*tasks)
    
    async def check_critical_dependencies(self) -> List[HealthCheck]:
        """
        Verifica solo dependencias críticas
        
        Returns:
            Lista de health checks
        """
        critical_deps = [
            dep_id for dep_id, dep in self._dependencies.items()
            if dep.critical
        ]
        
        tasks = [self.check_dependency(dep_id) for dep_id in critical_deps]
        return await asyncio.gather(*tasks)
    
    def get_dependency_status(self, dependency_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene estado de una dependencia
        
        Args:
            dependency_id: ID de la dependencia
            
        Returns:
            Diccionario con estado o None
        """
        dependency = self._dependencies.get(dependency_id)
        
        if not dependency:
            return None
        
        return {
            "dependency_id": dependency_id,
            "name": dependency.name,
            "type": dependency.dependency_type.value,
            "critical": dependency.critical,
            "last_check": dependency.last_check.isoformat() if dependency.last_check else None,
            "last_status": dependency.last_status.value if dependency.last_status else None,
        }

