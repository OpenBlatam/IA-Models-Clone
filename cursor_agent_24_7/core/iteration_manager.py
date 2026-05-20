"""
Iteration Manager
=================

Sistema que gestiona iteraciones sobre cambios hasta que sean correctos,
siguiendo las mejores prácticas de Devin de iterar hasta que funcione.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Iteration:
    """Iteración de cambios"""
    iteration_number: int
    description: str
    changes_made: List[str] = field(default_factory=list)
    tests_run: bool = False
    tests_passed: Optional[bool] = None
    lint_run: bool = False
    lint_passed: Optional[bool] = None
    issues_found: List[str] = field(default_factory=list)
    status: str = "in_progress"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "iteration_number": self.iteration_number,
            "description": self.description,
            "changes_made": self.changes_made,
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "lint_run": self.lint_run,
            "lint_passed": self.lint_passed,
            "issues_found": self.issues_found,
            "status": self.status,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class IterationTask:
    """Tarea con iteraciones"""
    task_id: str
    description: str
    iterations: List[Iteration] = field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 10
    status: str = "in_progress"
    completed: bool = False
    
    def start_iteration(self, description: str) -> Iteration:
        """Iniciar nueva iteración"""
        self.current_iteration += 1
        iteration = Iteration(
            iteration_number=self.current_iteration,
            description=description
        )
        self.iterations.append(iteration)
        return iteration
    
    def get_current_iteration(self) -> Optional[Iteration]:
        """Obtener iteración actual"""
        if self.iterations:
            return self.iterations[-1]
        return None
    
    def mark_completed(self) -> None:
        """Marcar tarea como completada"""
        self.status = "completed"
        self.completed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "iterations": [it.to_dict() for it in self.iterations],
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "status": self.status,
            "completed": self.completed
        }


class IterationManager:
    """
    Gestor de iteraciones.
    
    Gestiona iteraciones sobre cambios hasta que sean correctos,
    siguiendo las mejores prácticas de Devin.
    """
    
    def __init__(self) -> None:
        """Inicializar gestor de iteraciones"""
        self.tasks: Dict[str, IterationTask] = {}
        logger.info("🔄 Iteration manager initialized")
    
    def create_task(
        self,
        task_id: str,
        description: str,
        max_iterations: int = 10
    ) -> IterationTask:
        """
        Crear tarea con iteraciones.
        
        Args:
            task_id: ID único de la tarea.
            description: Descripción de la tarea.
            max_iterations: Máximo de iteraciones permitidas.
        
        Returns:
            Tarea creada.
        """
        task = IterationTask(
            task_id=task_id,
            description=description,
            max_iterations=max_iterations
        )
        self.tasks[task_id] = task
        return task
    
    def start_iteration(
        self,
        task_id: str,
        description: str
    ) -> Optional[Iteration]:
        """
        Iniciar nueva iteración.
        
        Args:
            task_id: ID de la tarea.
            description: Descripción de la iteración.
        
        Returns:
            Iteración creada o None si la tarea no existe.
        """
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        if task.current_iteration >= task.max_iterations:
            logger.warning(f"Max iterations reached for task {task_id}")
            return None
        
        return task.start_iteration(description)
    
    def record_change(
        self,
        task_id: str,
        change_description: str
    ) -> bool:
        """
        Registrar cambio en iteración actual.
        
        Args:
            task_id: ID de la tarea.
            change_description: Descripción del cambio.
        
        Returns:
            True si se registró exitosamente.
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        current = task.get_current_iteration()
        
        if current:
            current.changes_made.append(change_description)
            return True
        
        return False
    
    async def verify_iteration(
        self,
        task_id: str,
        run_tests: bool = True,
        run_lint: bool = True
    ) -> Dict[str, Any]:
        """
        Verificar iteración actual.
        
        Args:
            task_id: ID de la tarea.
            run_tests: Si ejecutar tests (default: True).
            run_lint: Si ejecutar linting (default: True).
        
        Returns:
            Resultado de la verificación.
        """
        if task_id not in self.tasks:
            return {
                "success": False,
                "error": f"Task {task_id} not found"
            }
        
        task = self.tasks[task_id]
        current = task.get_current_iteration()
        
        if not current:
            return {
                "success": False,
                "error": "No current iteration"
            }
        
        results = {
            "success": False,
            "iteration": current.iteration_number,
            "tests_passed": None,
            "lint_passed": None,
            "issues": []
        }
        
        if run_tests:
            try:
                from .test_runner import TestRunner
                test_runner = TestRunner()
                test_results = await test_runner.run_tests()
                current.tests_run = True
                if test_results:
                    current.tests_passed = all(r.success for r in test_results)
                    results["tests_passed"] = current.tests_passed
                    if not current.tests_passed:
                        results["issues"].append("Tests failed")
                        current.issues_found.append("Tests failed")
            except Exception as e:
                logger.warning(f"Could not run tests: {e}")
        
        if run_lint:
            try:
                from .test_runner import TestRunner
                test_runner = TestRunner()
                lint_result = await test_runner.run_lint()
                current.lint_run = True
                current.lint_passed = lint_result.success
                results["lint_passed"] = lint_result.success
                if not lint_result.success:
                    results["issues"].append(f"Lint found {len(lint_result.errors)} errors")
                    current.issues_found.append("Lint errors")
            except Exception as e:
                logger.warning(f"Could not run lint: {e}")
        
        if (results["tests_passed"] is None or results["tests_passed"]) and \
           (results["lint_passed"] is None or results["lint_passed"]) and \
           len(results["issues"]) == 0:
            current.status = "completed"
            results["success"] = True
            task.mark_completed()
        else:
            current.status = "needs_fix"
            results["success"] = False
        
        return results
    
    def get_task(self, task_id: str) -> Optional[IterationTask]:
        """Obtener tarea"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Obtener todas las tareas"""
        return [task.to_dict() for task in self.tasks.values()]

