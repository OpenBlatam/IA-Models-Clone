"""
Sistema de Few-Shot Learning
================================

Sistema para aprendizaje con pocos ejemplos.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FewShotMethod(Enum):
    """Método de few-shot learning"""
    PROMPT_BASED = "prompt_based"
    METRIC_LEARNING = "metric_learning"
    META_LEARNING = "meta_learning"
    PROMPT_TUNING = "prompt_tuning"


@dataclass
class FewShotTask:
    """Tarea de few-shot learning"""
    task_id: str
    task_name: str
    examples: List[Dict[str, Any]]  # Pocos ejemplos
    method: FewShotMethod
    status: str
    created_at: str


class FewShotLearning:
    """
    Sistema de Few-Shot Learning
    
    Proporciona:
    - Aprendizaje con pocos ejemplos
    - Múltiples métodos (prompt-based, metric learning, meta-learning)
    - Adaptación rápida a nuevas tareas
    - Transfer de conocimiento
    - Evaluación de few-shot performance
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.tasks: Dict[str, FewShotTask] = {}
        self.learning_history: List[Dict[str, Any]] = {}
        logger.info("FewShotLearning inicializado")
    
    def create_few_shot_task(
        self,
        task_name: str,
        examples: List[Dict[str, Any]],
        method: FewShotMethod = FewShotMethod.PROMPT_BASED
    ) -> FewShotTask:
        """
        Crear tarea de few-shot learning
        
        Args:
            task_name: Nombre de la tarea
            examples: Pocos ejemplos (1-5 típicamente)
            method: Método de aprendizaje
        
        Returns:
            Tarea creada
        """
        task_id = f"fewshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = FewShotTask(
            task_id=task_id,
            task_name=task_name,
            examples=examples,
            method=method,
            status="created",
            created_at=datetime.now().isoformat()
        )
        
        self.tasks[task_id] = task
        
        logger.info(f"Tarea few-shot creada: {task_id} con {len(examples)} ejemplos")
        
        return task
    
    def learn_from_few_examples(
        self,
        task_id: str,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Aprender de pocos ejemplos
        
        Args:
            task_id: ID de la tarea
            model_id: ID del modelo (opcional)
        
        Returns:
            Resultados del aprendizaje
        """
        if task_id not in self.tasks:
            raise ValueError(f"Tarea no encontrada: {task_id}")
        
        task = self.tasks[task_id]
        task.status = "learning"
        
        # Simulación de aprendizaje few-shot
        # En producción, usaría modelos como GPT-3, T5, etc.
        learning_result = {
            "task_id": task_id,
            "task_name": task.task_name,
            "method": task.method.value,
            "num_examples": len(task.examples),
            "accuracy": 0.75,  # Buen rendimiento con pocos ejemplos
            "learning_time_seconds": 5.0,
            "timestamp": datetime.now().isoformat()
        }
        
        task.status = "completed"
        self.learning_history[task_id] = learning_result
        
        logger.info(f"Few-shot learning completado: {task_id}")
        
        return learning_result
    
    def evaluate_few_shot_performance(
        self,
        task_id: str,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluar rendimiento few-shot
        
        Args:
            task_id: ID de la tarea
            test_data: Datos de prueba
        
        Returns:
            Métricas de rendimiento
        """
        if task_id not in self.tasks:
            raise ValueError(f"Tarea no encontrada: {task_id}")
        
        task = self.tasks[task_id]
        
        # Simulación de evaluación
        performance = {
            "task_id": task_id,
            "task_name": task.task_name,
            "num_examples_used": len(task.examples),
            "num_test_samples": len(test_data),
            "accuracy": 0.75,
            "precision": 0.72,
            "recall": 0.78,
            "f1_score": 0.75,
            "efficiency": "high"  # Alta eficiencia con pocos ejemplos
        }
        
        logger.info(f"Evaluación few-shot completada: {task_id}")
        
        return performance


# Instancia global
_few_shot_learning: Optional[FewShotLearning] = None


def get_few_shot_learning() -> FewShotLearning:
    """Obtener instancia global del sistema"""
    global _few_shot_learning
    if _few_shot_learning is None:
        _few_shot_learning = FewShotLearning()
    return _few_shot_learning



