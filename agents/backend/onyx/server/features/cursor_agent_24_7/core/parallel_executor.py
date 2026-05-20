"""
Parallel Executor
=================

Sistema para ejecutar comandos en paralelo cuando no tienen dependencias,
siguiendo las mejores prácticas de Devin de eficiencia.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ParallelTask:
    """Tarea para ejecución paralela"""
    id: str
    name: str
    task: Callable[..., Awaitable[Any]]
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "name": self.name,
            "dependencies": self.dependencies,
            "status": self.status,
            "execution_time": self.execution_time,
            "error": self.error
        }


class ParallelExecutor:
    """
    Ejecutor paralelo de tareas.
    
    Ejecuta tareas en paralelo cuando no tienen dependencias,
    siguiendo las mejores prácticas de Devin.
    """
    
    def __init__(self, max_concurrent: int = 5) -> None:
        """
        Inicializar ejecutor paralelo.
        
        Args:
            max_concurrent: Máximo de tareas concurrentes (default: 5).
        """
        self.max_concurrent = max_concurrent
        self.tasks: Dict[str, ParallelTask] = {}
        logger.info(f"⚡ Parallel executor initialized (max_concurrent={max_concurrent})")
    
    def add_task(
        self,
        task_id: str,
        name: str,
        task: Callable[..., Awaitable[Any]],
        dependencies: Optional[List[str]] = None,
        *args,
        **kwargs
    ) -> ParallelTask:
        """
        Agregar tarea para ejecución.
        
        Args:
            task_id: ID único de la tarea.
            name: Nombre descriptivo de la tarea.
            task: Función async a ejecutar.
            dependencies: Lista de IDs de tareas de las que depende.
            *args: Argumentos posicionales para la tarea.
            **kwargs: Argumentos nombrados para la tarea.
        
        Returns:
            Tarea creada.
        """
        parallel_task = ParallelTask(
            id=task_id,
            name=name,
            task=task,
            args=args,
            kwargs=kwargs,
            dependencies=dependencies or []
        )
        self.tasks[task_id] = parallel_task
        return parallel_task
    
    async def execute_all(self) -> Dict[str, Any]:
        """
        Ejecutar todas las tareas respetando dependencias.
        
        Returns:
            Resultados de ejecución.
        """
        results = {
            "success": True,
            "total_tasks": len(self.tasks),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "task_results": {}
        }
        
        completed: set[str] = set()
        running: Dict[str, asyncio.Task] = {}
        
        while len(completed) < len(self.tasks):
            ready_tasks = [
                task_id
                for task_id, task in self.tasks.items()
                if task_id not in completed
                and task_id not in running
                and all(dep in completed for dep in task.dependencies)
            ]
            
            if not ready_tasks and not running:
                results["success"] = False
                results["error"] = "Circular dependency or deadlock detected"
                break
            
            for task_id in ready_tasks[:self.max_concurrent - len(running)]:
                task = self.tasks[task_id]
                task.status = "running"
                
                async def run_task(t: ParallelTask) -> None:
                    start_time = datetime.now()
                    try:
                        t.result = await t.task(*t.args, **t.kwargs)
                        t.status = "completed"
                        t.execution_time = (datetime.now() - start_time).total_seconds()
                        results["completed_tasks"] += 1
                    except Exception as e:
                        t.status = "failed"
                        t.error = str(e)
                        t.execution_time = (datetime.now() - start_time).total_seconds()
                        results["failed_tasks"] += 1
                        logger.error(f"Task {t.id} failed: {e}", exc_info=True)
                
                running[task_id] = asyncio.create_task(run_task(task))
            
            if running:
                done, pending = await asyncio.wait(
                    running.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for done_task in done:
                    for task_id, running_task in list(running.items()):
                        if running_task == done_task:
                            completed.add(task_id)
                            del running[task_id]
                            results["task_results"][task_id] = self.tasks[task_id].to_dict()
                            break
            
            await asyncio.sleep(0.1)
        
        if results["failed_tasks"] > 0:
            results["success"] = False
        
        return results
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de una tarea"""
        if task_id not in self.tasks:
            return None
        return self.tasks[task_id].to_dict()
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Obtener todas las tareas"""
        return [task.to_dict() for task in self.tasks.values()]

