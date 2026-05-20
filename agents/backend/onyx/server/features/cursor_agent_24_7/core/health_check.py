"""
Health Check - Sistema de verificación de salud
===============================================

Verifica el estado de salud del agente y sus componentes.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Estados de salud"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Resultado de un health check"""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """Verificador de salud del sistema"""
    
    def __init__(self, agent):
        self.agent = agent
        self.checks: List[HealthCheck] = []
    
    async def check_all(self) -> Dict[str, Any]:
        """Ejecutar todos los health checks"""
        self.checks.clear()
        
        # Ejecutar checks
        await self._check_agent_status()
        await self._check_task_queue()
        await self._check_storage()
        await self._check_memory()
        await self._check_file_watcher()
        
        # Determinar estado general
        overall_status = self._determine_overall_status()
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "timestamp": check.timestamp.isoformat(),
                    "details": check.details
                }
                for check in self.checks
            ]
        }
    
    async def _check_agent_status(self):
        """Verificar estado del agente"""
        try:
            if self.agent.status.value == "running":
                status = HealthStatus.HEALTHY
                message = "Agent is running"
            elif self.agent.status.value == "paused":
                status = HealthStatus.DEGRADED
                message = "Agent is paused"
            elif self.agent.status.value == "error":
                status = HealthStatus.UNHEALTHY
                message = "Agent is in error state"
            else:
                status = HealthStatus.UNKNOWN
                message = f"Agent status: {self.agent.status.value}"
            
            self.checks.append(HealthCheck(
                name="agent_status",
                status=status,
                message=message,
                details={"status": self.agent.status.value}
            ))
        except Exception as e:
            self.checks.append(HealthCheck(
                name="agent_status",
                status=HealthStatus.UNHEALTHY,
                message=f"Error checking agent status: {e}",
                details={"error": str(e)}
            ))
    
    async def _check_task_queue(self):
        """Verificar cola de tareas"""
        try:
            queue_size = self.agent.task_queue.qsize()
            total_tasks = len(self.agent.tasks)
            pending = sum(1 for t in self.agent.tasks.values() if t.status == "pending")
            
            if queue_size > 100:
                status = HealthStatus.DEGRADED
                message = f"Task queue is large: {queue_size} tasks"
            elif queue_size > 0:
                status = HealthStatus.HEALTHY
                message = f"Task queue has {queue_size} tasks"
            else:
                status = HealthStatus.HEALTHY
                message = "Task queue is empty"
            
            self.checks.append(HealthCheck(
                name="task_queue",
                status=status,
                message=message,
                details={
                    "queue_size": queue_size,
                    "total_tasks": total_tasks,
                    "pending_tasks": pending
                }
            ))
        except Exception as e:
            self.checks.append(HealthCheck(
                name="task_queue",
                status=HealthStatus.UNHEALTHY,
                message=f"Error checking task queue: {e}",
                details={"error": str(e)}
            ))
    
    async def _check_storage(self):
        """Verificar almacenamiento"""
        try:
            from pathlib import Path
            
            storage_path = Path(self.agent.config.storage_path)
            storage_dir = storage_path.parent
            
            if not storage_dir.exists():
                status = HealthStatus.UNHEALTHY
                message = "Storage directory does not exist"
            else:
                # Verificar espacio en disco
                try:
                    import shutil
                    total, used, free = shutil.disk_usage(storage_dir)
                    free_gb = free / (1024**3)
                    
                    if free_gb < 0.1:  # Menos de 100MB
                        status = HealthStatus.UNHEALTHY
                        message = f"Low disk space: {free_gb:.2f} GB free"
                    elif free_gb < 1.0:  # Menos de 1GB
                        status = HealthStatus.DEGRADED
                        message = f"Disk space getting low: {free_gb:.2f} GB free"
                    else:
                        status = HealthStatus.HEALTHY
                        message = f"Storage OK: {free_gb:.2f} GB free"
                    
                    self.checks.append(HealthCheck(
                        name="storage",
                        status=status,
                        message=message,
                        details={
                            "free_gb": round(free_gb, 2),
                            "used_gb": round(used / (1024**3), 2),
                            "total_gb": round(total / (1024**3), 2)
                        }
                    ))
                except Exception as e:
                    self.checks.append(HealthCheck(
                        name="storage",
                        status=HealthStatus.UNKNOWN,
                        message=f"Could not check disk space: {e}",
                        details={"error": str(e)}
                    ))
        except Exception as e:
            self.checks.append(HealthCheck(
                name="storage",
                status=HealthStatus.UNHEALTHY,
                message=f"Error checking storage: {e}",
                details={"error": str(e)}
            ))
    
    async def _check_memory(self):
        """Verificar uso de memoria"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            if memory_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High memory usage: {memory_percent:.1f}%"
            elif memory_percent > 75:
                status = HealthStatus.DEGRADED
                message = f"Memory usage getting high: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage OK: {memory_percent:.1f}%"
            
            self.checks.append(HealthCheck(
                name="memory",
                status=status,
                message=message,
                details={
                    "memory_mb": round(memory_info.rss / (1024**2), 2),
                    "memory_percent": round(memory_percent, 2)
                }
            ))
        except ImportError:
            self.checks.append(HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="psutil not available for memory check",
                details={}
            ))
        except Exception as e:
            self.checks.append(HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking memory: {e}",
                details={"error": str(e)}
            ))
    
    async def _check_file_watcher(self):
        """Verificar file watcher"""
        try:
            if hasattr(self.agent, 'file_watcher') and self.agent.file_watcher:
                if self.agent.file_watcher.running:
                    status = HealthStatus.HEALTHY
                    message = "File watcher is running"
                else:
                    status = HealthStatus.DEGRADED
                    message = "File watcher is not running"
            else:
                status = HealthStatus.UNKNOWN
                message = "File watcher not initialized"
            
            self.checks.append(HealthCheck(
                name="file_watcher",
                status=status,
                message=message,
                details={}
            ))
        except Exception as e:
            self.checks.append(HealthCheck(
                name="file_watcher",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking file watcher: {e}",
                details={"error": str(e)}
            ))
    
    def _determine_overall_status(self) -> HealthStatus:
        """Determinar estado general basado en todos los checks"""
        if not self.checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in self.checks]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN



