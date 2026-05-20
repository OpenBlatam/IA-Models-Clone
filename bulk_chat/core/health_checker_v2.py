"""
Health Checker V2 - Verificador de Salud Avanzado
==================================================

Sistema avanzado de health checks con dependencias, timeouts y auto-recovery.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Estado de salud."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """Tipo de check."""
    HTTP = "http"
    DATABASE = "database"
    CACHE = "cache"
    DISK = "disk"
    MEMORY = "memory"
    CPU = "cpu"
    CUSTOM = "custom"


@dataclass
class HealthCheck:
    """Health check."""
    check_id: str
    name: str
    check_type: CheckType
    handler: Callable
    timeout: float = 5.0
    interval: float = 30.0
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Resultado de health check."""
    check_id: str
    status: HealthStatus
    response_time: float
    timestamp: datetime
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class HealthCheckerV2:
    """Verificador de salud avanzado."""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.check_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.check_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._checking_active = False
    
    def register_check(
        self,
        check_id: str,
        name: str,
        check_type: CheckType,
        handler: Callable,
        timeout: float = 5.0,
        interval: float = 30.0,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar health check."""
        check = HealthCheck(
            check_id=check_id,
            name=name,
            check_type=check_type,
            handler=handler,
            timeout=timeout,
            interval=interval,
            dependencies=dependencies or [],
            metadata=metadata or {},
        )
        
        async def save_check():
            async with self._lock:
                self.checks[check_id] = check
        
        asyncio.create_task(save_check())
        
        logger.info(f"Registered health check: {check_id} - {name}")
        return check_id
    
    def start_checking(self):
        """Iniciar verificaciones."""
        if self._checking_active:
            return
        
        self._checking_active = True
        
        async def start_checks():
            for check_id, check in self.checks.items():
                if check.enabled:
                    task = asyncio.create_task(self._run_check_loop(check))
                    self.check_tasks[check_id] = task
        
        asyncio.create_task(start_checks())
        
        logger.info("Health checking started")
    
    def stop_checking(self):
        """Detener verificaciones."""
        self._checking_active = False
        
        for task in self.check_tasks.values():
            task.cancel()
        
        self.check_tasks.clear()
        logger.info("Health checking stopped")
    
    async def _run_check_loop(self, check: HealthCheck):
        """Loop de verificación."""
        while self._checking_active:
            try:
                await self._execute_check(check)
                await asyncio.sleep(check.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in check loop for {check.check_id}: {e}")
                await asyncio.sleep(check.interval)
    
    async def _execute_check(self, check: HealthCheck):
        """Ejecutar health check."""
        # Verificar dependencias
        for dep_id in check.dependencies:
            dep_result = self.results.get(dep_id)
            if not dep_result or dep_result.status == HealthStatus.UNHEALTHY:
                result = HealthCheckResult(
                    check_id=check.check_id,
                    status=HealthStatus.UNHEALTHY,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    message=f"Dependency {dep_id} is unhealthy",
                )
                
                async with self._lock:
                    self.results[check.check_id] = result
                    self.check_history[check.check_id].append(result)
                
                return
        
        # Ejecutar check
        start_time = datetime.now()
        status = HealthStatus.UNHEALTHY
        message = None
        
        try:
            if asyncio.iscoroutinefunction(check.handler):
                result = await asyncio.wait_for(
                    check.handler(),
                    timeout=check.timeout
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(check.handler),
                    timeout=check.timeout
                )
            
            # Interpretar resultado
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            elif isinstance(result, dict):
                status_str = result.get("status", "unhealthy")
                status = HealthStatus(status_str) if status_str in [s.value for s in HealthStatus] else HealthStatus.UNHEALTHY
                message = result.get("message")
            else:
                status = HealthStatus.HEALTHY
        
        except asyncio.TimeoutError:
            status = HealthStatus.UNHEALTHY
            message = f"Check timed out after {check.timeout}s"
        
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Check failed: {str(e)}"
        
        response_time = (datetime.now() - start_time).total_seconds()
        
        result = HealthCheckResult(
            check_id=check.check_id,
            status=status,
            response_time=response_time,
            timestamp=datetime.now(),
            message=message,
        )
        
        async with self._lock:
            self.results[check.check_id] = result
            self.check_history[check.check_id].append(result)
    
    async def run_check(self, check_id: str) -> Optional[Dict[str, Any]]:
        """Ejecutar check manualmente."""
        check = self.checks.get(check_id)
        if not check:
            return None
        
        await self._execute_check(check)
        result = self.results.get(check_id)
        
        if not result:
            return None
        
        return {
            "check_id": result.check_id,
            "status": result.status.value,
            "response_time": result.response_time,
            "timestamp": result.timestamp.isoformat(),
            "message": result.message,
        }
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Obtener salud general."""
        if not self.results:
            return {"status": "unknown", "checks": {}}
        
        # Contar estados
        status_counts = defaultdict(int)
        for result in self.results.values():
            status_counts[result.status.value] += 1
        
        # Determinar estado general
        if status_counts[HealthStatus.UNHEALTHY.value] > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED.value] > 0:
            overall_status = HealthStatus.DEGRADED
        elif status_counts[HealthStatus.HEALTHY.value] > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return {
            "status": overall_status.value,
            "status_counts": dict(status_counts),
            "total_checks": len(self.results),
            "checks": {
                check_id: {
                    "status": result.status.value,
                    "response_time": result.response_time,
                    "timestamp": result.timestamp.isoformat(),
                }
                for check_id, result in self.results.items()
            },
        }
    
    def get_check_history(self, check_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de checks."""
        history = self.check_history.get(check_id, deque())
        
        return [
            {
                "status": r.status.value,
                "response_time": r.response_time,
                "timestamp": r.timestamp.isoformat(),
                "message": r.message,
            }
            for r in list(history)[-limit:]
        ]
    
    def get_health_checker_v2_summary(self) -> Dict[str, Any]:
        """Obtener resumen del verificador."""
        by_status: Dict[str, int] = defaultdict(int)
        
        for result in self.results.values():
            by_status[result.status.value] += 1
        
        return {
            "checking_active": self._checking_active,
            "total_checks": len(self.checks),
            "active_checks": len(self.check_tasks),
            "checks_by_status": dict(by_status),
        }


