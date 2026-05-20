"""
Sistema de health checks avanzado
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time
import asyncio


class HealthStatus(str, Enum):
    """Estado de salud"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check"""
    name: str
    status: HealthStatus
    message: str
    response_time: Optional[float] = None
    last_check: str = None
    details: Optional[Dict] = None
    
    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "response_time": self.response_time,
            "last_check": self.last_check,
            "details": self.details or {}
        }


class HealthMonitor:
    """Monitor de salud del sistema"""
    
    def __init__(self):
        """Inicializa el monitor"""
        self.checks: Dict[str, HealthCheck] = {}
        self.check_functions: Dict[str, callable] = {}
    
    def register_check(self, name: str, check_function: callable):
        """
        Registra un health check
        
        Args:
            name: Nombre del check
            check_function: Función que realiza el check
        """
        self.check_functions[name] = check_function
    
    async def run_check(self, name: str) -> HealthCheck:
        """
        Ejecuta un health check
        
        Args:
            name: Nombre del check
            
        Returns:
            HealthCheck
        """
        if name not in self.check_functions:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Check no registrado"
            )
        
        check_function = self.check_functions[name]
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(check_function):
                result = await check_function()
            else:
                result = check_function()
            
            response_time = time.time() - start_time
            
            if isinstance(result, dict):
                status = HealthStatus(result.get("status", "unknown"))
                message = result.get("message", "OK")
                details = result.get("details", {})
            else:
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "OK" if result else "Failed"
                details = {}
            
            health_check = HealthCheck(
                name=name,
                status=status,
                message=message,
                response_time=response_time,
                details=details
            )
            
            self.checks[name] = health_check
            return health_check
        
        except Exception as e:
            response_time = time.time() - start_time
            health_check = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}",
                response_time=response_time
            )
            self.checks[name] = health_check
            return health_check
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Ejecuta todos los checks"""
        results = {}
        
        for name in self.check_functions.keys():
            results[name] = await self.run_check(name)
        
        return results
    
    def get_overall_health(self) -> Dict:
        """Obtiene salud general del sistema"""
        if not self.checks:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No checks executed"
            }
        
        statuses = [check.status for check in self.checks.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED
        
        return {
            "status": overall_status.value,
            "checks": {name: check.to_dict() for name, check in self.checks.items()},
            "summary": {
                "total": len(self.checks),
                "healthy": sum(1 for s in statuses if s == HealthStatus.HEALTHY),
                "degraded": sum(1 for s in statuses if s == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for s in statuses if s == HealthStatus.UNHEALTHY)
            }
        }

