"""
Sistema de Multi-Task Learning
================================

Sistema para aprendizaje multi-tarea.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MultiTaskMethod(Enum):
    """Método de multi-task learning"""
    HARD_PARAMETER_SHARING = "hard_parameter_sharing"
    SOFT_PARAMETER_SHARING = "soft_parameter_sharing"
    TASK_EMBEDDING = "task_embedding"
    ADAPTIVE_SHARING = "adaptive_sharing"


@dataclass
class Task:
    """Tarea de aprendizaje"""
    task_id: str
    task_name: str
    task_type: str
    data: List[Dict[str, Any]]
    priority: int


class MultiTaskLearning:
    """
    Sistema de Multi-Task Learning
    
    Proporciona:
    - Aprendizaje simultáneo de múltiples tareas
    - Compartir representaciones entre tareas
    - Múltiples métodos de sharing
    - Transfer entre tareas
    - Optimización multi-objetivo
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.tasks: Dict[str, Task] = {}
        self.models: Dict[str, Dict[str, Any]] = {}
        logger.info("MultiTaskLearning inicializado")
    
    def add_task(
        self,
        task_name: str,
        task_type: str,
        data: List[Dict[str, Any]],
        priority: int = 5
    ) -> Task:
        """
        Agregar tarea
        
        Args:
            task_name: Nombre de la tarea
            task_type: Tipo de tarea
            data: Datos de la tarea
            priority: Prioridad
        
        Returns:
            Tarea creada
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = Task(
            task_id=task_id,
            task_name=task_name,
            task_type=task_type,
            data=data,
            priority=priority
        )
        
        self.tasks[task_id] = task
        
        logger.info(f"Tarea agregada: {task_id} - {task_name}")
        
        return task
    
    def train_multi_task(
        self,
        task_ids: List[str],
        method: MultiTaskMethod = MultiTaskMethod.HARD_PARAMETER_SHARING,
        epochs: int = 10
    ) -> Dict[str, Any]:
        """
        Entrenar modelo multi-tarea
        
        Args:
            task_ids: IDs de tareas
            method: Método de multi-task learning
            epochs: Número de épocas
        
        Returns:
            Resultados del entrenamiento
        """
        model_id = f"multitask_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Verificar que todas las tareas existan
        for task_id in task_ids:
            if task_id not in self.tasks:
                raise ValueError(f"Tarea no encontrada: {task_id}")
        
        # Simulación de entrenamiento multi-tarea
        training_result = {
            "model_id": model_id,
            "task_ids": task_ids,
            "method": method.value,
            "epochs": epochs,
            "task_accuracy": {
                task_id: 0.85 for task_id in task_ids
            },
            "shared_parameters": 1000000,
            "task_specific_parameters": {
                task_id: 100000 for task_id in task_ids
            },
            "timestamp": datetime.now().isoformat()
        }
        
        self.models[model_id] = training_result
        
        logger.info(f"Modelo multi-tarea entrenado: {model_id}")
        
        return training_result
    
    def evaluate_task(
        self,
        model_id: str,
        task_id: str,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluar tarea específica
        
        Args:
            model_id: ID del modelo
            task_id: ID de la tarea
            test_data: Datos de prueba
        
        Returns:
            Métricas de evaluación
        """
        if model_id not in self.models:
            raise ValueError(f"Modelo no encontrado: {model_id}")
        
        if task_id not in self.tasks:
            raise ValueError(f"Tarea no encontrada: {task_id}")
        
        # Simulación de evaluación
        evaluation = {
            "model_id": model_id,
            "task_id": task_id,
            "task_name": self.tasks[task_id].task_name,
            "test_samples": len(test_data),
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        }
        
        logger.info(f"Evaluación completada: {task_id}")
        
        return evaluation


# Instancia global
_multi_task_learning: Optional[MultiTaskLearning] = None


def get_multi_task_learning() -> MultiTaskLearning:
    """Obtener instancia global del sistema"""
    global _multi_task_learning
    if _multi_task_learning is None:
        _multi_task_learning = MultiTaskLearning()
    return _multi_task_learning


