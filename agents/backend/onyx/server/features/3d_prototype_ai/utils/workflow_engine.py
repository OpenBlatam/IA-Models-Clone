"""
Workflow Engine - Sistema de workflow engine
=============================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Estados de tarea"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowTask:
    """Tarea en un workflow"""
    id: str
    name: str
    action: Callable
    dependencies: List[str]
    condition: Optional[Callable] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None


class WorkflowEngine:
    """Motor de workflows"""
    
    def __init__(self):
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.executions: Dict[str, Dict[str, Any]] = {}
    
    def register_workflow(self, workflow_id: str, name: str, 
                         tasks: List[WorkflowTask], description: str = ""):
        """Registra un workflow"""
        self.workflows[workflow_id] = {
            "id": workflow_id,
            "name": name,
            "description": description,
            "tasks": {task.id: task for task in tasks},
            "task_order": [task.id for task in tasks],
            "created_at": datetime.now().isoformat()
        }
        logger.info(f"Workflow registrado: {workflow_id}")
    
    async def execute_workflow(self, workflow_id: str, 
                              initial_data: Optional[Dict[str, Any]] = None) -> str:
        """Ejecuta un workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} no encontrado")
        
        execution_id = str(uuid4())
        context = {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "data": initial_data or {},
            "task_results": {},
            "task_status": {},
            "started_at": datetime.now(),
            "completed_at": None,
            "status": "running"
        }
        
        self.executions[execution_id] = context
        
        # Ejecutar workflow
        try:
            await self._execute_tasks(workflow, context)
            context["status"] = "completed"
        except Exception as e:
            context["status"] = "failed"
            context["error"] = str(e)
            logger.error(f"Workflow {workflow_id} falló: {e}")
        finally:
            context["completed_at"] = datetime.now()
        
        return execution_id
    
    async def _execute_tasks(self, workflow: Dict[str, Any], context: Dict[str, Any]):
        """Ejecuta las tareas de un workflow"""
        tasks = workflow["tasks"]
        task_order = workflow["task_order"]
        
        for task_id in task_order:
            task = tasks[task_id]
            
            # Verificar dependencias
            if not self._check_dependencies(task, context):
                logger.warning(f"Tarea {task_id} saltada por dependencias no cumplidas")
                context["task_status"][task_id] = TaskStatus.SKIPPED.value
                continue
            
            # Verificar condición
            if task.condition and not task.condition(context):
                logger.info(f"Tarea {task_id} saltada por condición")
                context["task_status"][task_id] = TaskStatus.SKIPPED.value
                continue
            
            # Ejecutar tarea
            context["task_status"][task_id] = TaskStatus.RUNNING.value
            
            try:
                if asyncio.iscoroutinefunction(task.action):
                    result = await task.action(context)
                else:
                    result = task.action(context)
                
                context["task_results"][task_id] = result
                context["task_status"][task_id] = TaskStatus.COMPLETED.value
                
            except Exception as e:
                # Reintentar si es posible
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    logger.warning(f"Reintentando tarea {task_id} ({task.retry_count}/{task.max_retries})")
                    await asyncio.sleep(1)  # Esperar antes de reintentar
                    continue
                
                context["task_status"][task_id] = TaskStatus.FAILED.value
                context["task_results"][task_id] = {"error": str(e)}
                raise
    
    def _check_dependencies(self, task: WorkflowTask, context: Dict[str, Any]) -> bool:
        """Verifica dependencias de una tarea"""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            dep_status = context["task_status"].get(dep_id)
            if dep_status != TaskStatus.COMPLETED.value:
                return False
        
        return True
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de una ejecución"""
        execution = self.executions.get(execution_id)
        if not execution:
            return None
        
        return {
            "execution_id": execution_id,
            "workflow_id": execution["workflow_id"],
            "status": execution["status"],
            "started_at": execution["started_at"].isoformat(),
            "completed_at": execution["completed_at"].isoformat() if execution["completed_at"] else None,
            "task_status": execution["task_status"],
            "error": execution.get("error")
        }




