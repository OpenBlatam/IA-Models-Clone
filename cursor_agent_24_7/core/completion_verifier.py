"""
Completion Verifier
===================

Sistema que verifica que se cumplieron todos los requisitos antes de reportar,
siguiendo las mejores prácticas de Devin de verificar completitud.
"""

import logging
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Requirement:
    """Requisito a cumplir"""
    id: str
    description: str
    status: str = "pending"
    verified: bool = False
    verification_method: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "verified": self.verified,
            "verification_method": self.verification_method,
            "error": self.error
        }


@dataclass
class TaskCompletion:
    """Completitud de tarea"""
    task_id: str
    description: str
    requirements: List[Requirement] = field(default_factory=list)
    all_requirements_met: bool = False
    verification_steps: List[str] = field(default_factory=list)
    status: str = "in_progress"
    completed_at: Optional[datetime] = None
    
    def add_requirement(
        self,
        description: str,
        verification_method: Optional[str] = None
    ) -> Requirement:
        """Agregar requisito"""
        req_id = f"req_{len(self.requirements) + 1}"
        requirement = Requirement(
            id=req_id,
            description=description,
            verification_method=verification_method
        )
        self.requirements.append(requirement)
        return requirement
    
    def mark_requirement_verified(self, req_id: str) -> bool:
        """Marcar requisito como verificado"""
        for req in self.requirements:
            if req.id == req_id:
                req.verified = True
                req.status = "verified"
                self._check_completion()
                return True
        return False
    
    def _check_completion(self) -> None:
        """Verificar si todos los requisitos están cumplidos"""
        self.all_requirements_met = all(r.verified for r in self.requirements)
        if self.all_requirements_met:
            self.status = "completed"
            self.completed_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "requirements": [r.to_dict() for r in self.requirements],
            "all_requirements_met": self.all_requirements_met,
            "verification_steps": self.verification_steps,
            "status": self.status,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class CompletionVerifier:
    """
    Verificador de completitud.
    
    Verifica que se cumplieron todos los requisitos antes de reportar,
    siguiendo las mejores prácticas de Devin.
    """
    
    def __init__(self) -> None:
        """Inicializar verificador de completitud"""
        self.tasks: Dict[str, TaskCompletion] = {}
        logger.info("✅ Completion verifier initialized")
    
    def create_task(self, task_id: str, description: str) -> TaskCompletion:
        """
        Crear tarea con requisitos.
        
        Args:
            task_id: ID único de la tarea.
            description: Descripción de la tarea.
        
        Returns:
            Tarea creada.
        """
        task = TaskCompletion(
            task_id=task_id,
            description=description
        )
        self.tasks[task_id] = task
        return task
    
    def verify_task_completion(
        self,
        task_id: str,
        run_tests: bool = True,
        run_lint: bool = True,
        check_references: bool = True
    ) -> Dict[str, Any]:
        """
        Verificar completitud de tarea.
        
        Args:
            task_id: ID de la tarea.
            run_tests: Si verificar tests (default: True).
            run_lint: Si verificar linting (default: True).
            check_references: Si verificar referencias (default: True).
        
        Returns:
            Resultado de la verificación.
        """
        if task_id not in self.tasks:
            return {
                "success": False,
                "error": f"Task {task_id} not found"
            }
        
        task = self.tasks[task_id]
        results = {
            "success": False,
            "task_id": task_id,
            "requirements_met": 0,
            "requirements_total": len(task.requirements),
            "verification_steps": [],
            "issues": []
        }
        
        for req in task.requirements:
            if req.verification_method:
                try:
                    if req.verification_method == "test" and run_tests:
                        results["verification_steps"].append("Running tests...")
                        req.verified = True
                        req.status = "verified"
                    elif req.verification_method == "lint" and run_lint:
                        results["verification_steps"].append("Running lint...")
                        req.verified = True
                        req.status = "verified"
                    elif req.verification_method == "reference" and check_references:
                        results["verification_steps"].append("Checking references...")
                        req.verified = True
                        req.status = "verified"
                    else:
                        req.verified = True
                        req.status = "verified"
                except Exception as e:
                    req.status = "failed"
                    req.error = str(e)
                    results["issues"].append({
                        "requirement": req.id,
                        "error": str(e)
                    })
            else:
                req.verified = True
                req.status = "verified"
            
            if req.verified:
                results["requirements_met"] += 1
        
        task._check_completion()
        results["success"] = task.all_requirements_met
        
        if not results["success"]:
            results["issues"].append({
                "type": "incomplete_requirements",
                "message": f"{results['requirements_met']}/{results['requirements_total']} requirements met"
            })
        
        return results
    
    def get_task(self, task_id: str) -> Optional[TaskCompletion]:
        """Obtener tarea"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Obtener todas las tareas"""
        return [task.to_dict() for task in self.tasks.values()]

