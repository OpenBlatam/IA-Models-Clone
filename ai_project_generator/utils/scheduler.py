"""
Scheduler - Sistema de Tareas Programadas
==========================================

Gestiona tareas programadas y ejecución automática.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Sistema de tareas programadas"""

    def __init__(self):
        """Inicializa el scheduler"""
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_history: List[Dict[str, Any]] = []

    def schedule_task(
        self,
        task_id: str,
        task_func: Callable,
        schedule_type: str,
        schedule_value: Any,
        enabled: bool = True,
    ) -> str:
        """
        Programa una tarea.

        Args:
            task_id: ID de la tarea
            task_func: Función a ejecutar
            schedule_type: Tipo de schedule (interval, cron, once)
            schedule_value: Valor del schedule
            enabled: Si está habilitada

        Returns:
            ID de la tarea programada
        """
        task_info = {
            "id": task_id,
            "func": task_func,
            "schedule_type": schedule_type,
            "schedule_value": schedule_value,
            "enabled": enabled,
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "next_run": self._calculate_next_run(schedule_type, schedule_value),
            "run_count": 0,
            "success_count": 0,
            "failure_count": 0,
        }

        self.tasks[task_id] = task_info

        if enabled:
            asyncio.create_task(self._run_scheduled_task(task_id))

        logger.info(f"Tarea programada: {task_id}")
        return task_id

    def _calculate_next_run(
        self,
        schedule_type: str,
        schedule_value: Any,
    ) -> Optional[str]:
        """Calcula la próxima ejecución"""
        if schedule_type == "interval":
            # schedule_value en segundos
            next_run = datetime.now() + timedelta(seconds=schedule_value)
            return next_run.isoformat()
        elif schedule_type == "once":
            # schedule_value es datetime
            if isinstance(schedule_value, str):
                return schedule_value
            return schedule_value.isoformat() if hasattr(schedule_value, 'isoformat') else None
        return None

    async def _run_scheduled_task(self, task_id: str):
        """Ejecuta una tarea programada"""
        task = self.tasks.get(task_id)
        if not task or not task["enabled"]:
            return

        while task["enabled"]:
            try:
                next_run = datetime.fromisoformat(task["next_run"])
                now = datetime.now()

                if now >= next_run:
                    # Ejecutar tarea
                    start_time = datetime.now()
                    try:
                        if asyncio.iscoroutinefunction(task["func"]):
                            await task["func"]()
                        else:
                            task["func"]()

                        task["success_count"] += 1
                        status = "success"
                    except Exception as e:
                        task["failure_count"] += 1
                        status = "failed"
                        logger.error(f"Error ejecutando tarea {task_id}: {e}")

                    task["last_run"] = datetime.now().isoformat()
                    task["run_count"] += 1

                    # Registrar en historial
                    self.task_history.append({
                        "task_id": task_id,
                        "status": status,
                        "started_at": start_time.isoformat(),
                        "completed_at": task["last_run"],
                    })

                    # Calcular próxima ejecución
                    if task["schedule_type"] == "interval":
                        task["next_run"] = self._calculate_next_run(
                            task["schedule_type"],
                            task["schedule_value"]
                        )
                    elif task["schedule_type"] == "once":
                        task["enabled"] = False
                        break

                # Esperar un poco antes de verificar de nuevo
                await asyncio.sleep(60)  # Verificar cada minuto

            except Exception as e:
                logger.error(f"Error en scheduler para tarea {task_id}: {e}")
                await asyncio.sleep(60)

    def enable_task(self, task_id: str) -> bool:
        """Habilita una tarea"""
        if task_id in self.tasks:
            self.tasks[task_id]["enabled"] = True
            if task_id not in self.running_tasks:
                self.running_tasks[task_id] = asyncio.create_task(
                    self._run_scheduled_task(task_id)
                )
            return True
        return False

    def disable_task(self, task_id: str) -> bool:
        """Deshabilita una tarea"""
        if task_id in self.tasks:
            self.tasks[task_id]["enabled"] = False
            if task_id in self.running_tasks:
                self.running_tasks[task_id].cancel()
                del self.running_tasks[task_id]
            return True
        return False

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de una tarea"""
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[Dict[str, Any]]:
        """Lista todas las tareas"""
        return [
            {
                "id": task["id"],
                "schedule_type": task["schedule_type"],
                "enabled": task["enabled"],
                "last_run": task["last_run"],
                "next_run": task["next_run"],
                "run_count": task["run_count"],
                "success_count": task["success_count"],
                "failure_count": task["failure_count"],
            }
            for task in self.tasks.values()
        ]


