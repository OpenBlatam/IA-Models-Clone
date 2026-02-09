"""
Cache Warming System
====================

Sistema de precalentamiento de cache.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CacheWarmingTask:
    """Tarea de precalentamiento."""
    task_id: str
    name: str
    warmup_func: Callable
    priority: int = 5  # 1-10
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheWarmer:
    """
    Precalentador de cache.
    
    Precalienta cache con datos frecuentemente accedidos.
    """
    
    def __init__(self):
        """Inicializar precalentador de cache."""
        self.tasks: Dict[str, CacheWarmingTask] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def register_task(
        self,
        task_id: str,
        name: str,
        warmup_func: Callable,
        priority: int = 5,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CacheWarmingTask:
        """
        Registrar tarea de precalentamiento.
        
        Args:
            task_id: ID único de la tarea
            name: Nombre
            warmup_func: Función de precalentamiento
            priority: Prioridad (1-10)
            enabled: Si está habilitada
            metadata: Metadata adicional
            
        Returns:
            Tarea registrada
        """
        task = CacheWarmingTask(
            task_id=task_id,
            name=name,
            warmup_func=warmup_func,
            priority=priority,
            enabled=enabled,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        logger.info(f"Registered cache warming task: {name} ({task_id})")
        
        return task
    
    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        Ejecutar tarea de precalentamiento.
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            Resultado de la ejecución
        """
        if task_id not in self.tasks:
            return {"error": "Task not found"}
        
        task = self.tasks[task_id]
        
        if not task.enabled:
            return {"error": "Task is disabled"}
        
        start_time = datetime.now()
        
        try:
            if asyncio.iscoroutinefunction(task.warmup_func):
                result = await task.warmup_func()
            else:
                result = task.warmup_func()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            task.last_run = end_time.isoformat()
            
            execution_record = {
                "task_id": task_id,
                "name": task.name,
                "success": True,
                "duration": duration,
                "timestamp": end_time.isoformat(),
                "result": result
            }
            
            self.execution_history.append(execution_record)
            if len(self.execution_history) > self.max_history:
                self.execution_history = self.execution_history[-self.max_history:]
            
            logger.info(f"Cache warming task executed: {task.name} ({duration:.2f}s)")
            
            return execution_record
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            execution_record = {
                "task_id": task_id,
                "name": task.name,
                "success": False,
                "duration": duration,
                "timestamp": end_time.isoformat(),
                "error": str(e)
            }
            
            self.execution_history.append(execution_record)
            if len(self.execution_history) > self.max_history:
                self.execution_history = self.execution_history[-self.max_history:]
            
            logger.error(f"Error executing cache warming task {task.name}: {e}", exc_info=True)
            
            return execution_record
    
    async def warmup_all(self) -> Dict[str, Any]:
        """
        Ejecutar todas las tareas habilitadas.
        
        Returns:
            Resumen de ejecución
        """
        enabled_tasks = [
            task for task in self.tasks.values()
            if task.enabled
        ]
        
        # Ordenar por prioridad (mayor primero)
        enabled_tasks.sort(key=lambda x: x.priority, reverse=True)
        
        results = []
        for task in enabled_tasks:
            result = await self.execute_task(task.task_id)
            results.append(result)
        
        success_count = sum(1 for r in results if r.get("success", False))
        
        return {
            "total_tasks": len(enabled_tasks),
            "success_count": success_count,
            "failure_count": len(enabled_tasks) - success_count,
            "results": results
        }
    
    def get_task(self, task_id: str) -> Optional[CacheWarmingTask]:
        """Obtener tarea por ID."""
        return self.tasks.get(task_id)
    
    def list_tasks(self) -> List[CacheWarmingTask]:
        """Listar todas las tareas."""
        return list(self.tasks.values())
    
    def get_execution_history(
        self,
        task_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtener historial de ejecución.
        
        Args:
            task_id: Filtrar por tarea
            limit: Límite de resultados
            
        Returns:
            Lista de ejecuciones
        """
        history = self.execution_history
        
        if task_id:
            history = [h for h in history if h["task_id"] == task_id]
        
        return history[-limit:]


# Instancia global
_cache_warmer: Optional[CacheWarmer] = None


def get_cache_warmer() -> CacheWarmer:
    """Obtener instancia global del precalentador de cache."""
    global _cache_warmer
    if _cache_warmer is None:
        _cache_warmer = CacheWarmer()
    return _cache_warmer






