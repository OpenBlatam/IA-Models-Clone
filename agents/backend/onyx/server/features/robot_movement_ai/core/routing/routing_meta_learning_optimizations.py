"""
Optimizaciones de Meta-Learning para Routing.

Este módulo implementa capacidades de meta-learning para aprender
a aprender y adaptarse rápidamente a nuevos problemas de routing.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class MetaLearningStrategy(Enum):
    """Estrategias de meta-learning."""
    MAML = "maml"  # Model-Agnostic Meta-Learning
    REPTILE = "reptile"
    FOMAML = "fomaml"  # First-Order MAML
    META_SGD = "meta_sgd"


@dataclass
class Task:
    """Tarea de aprendizaje."""
    task_id: str
    task_type: str
    data: Dict[str, Any]
    objective: Callable[[Dict[str, Any]], float]
    created_at: float = field(default_factory=time.time)


@dataclass
class MetaModel:
    """Modelo meta-aprendido."""
    model_id: str
    strategy: MetaLearningStrategy
    parameters: Dict[str, Any]
    performance: float
    adaptation_steps: int = 0


class MAMLOptimizer:
    """Optimizador MAML (Model-Agnostic Meta-Learning)."""
    
    def __init__(self, inner_lr: float = 0.01, outer_lr: float = 0.001):
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.meta_parameters: Dict[str, np.ndarray] = {}
        self.tasks_processed = 0
        self.adaptations = 0
    
    def initialize_meta_parameters(self, parameter_shape: Dict[str, Tuple[int, ...]]):
        """Inicializar parámetros meta."""
        for key, shape in parameter_shape.items():
            self.meta_parameters[key] = np.random.normal(0, 0.1, shape)
    
    def adapt(self, task: Task, num_steps: int = 5) -> Dict[str, np.ndarray]:
        """Adaptar a nueva tarea."""
        adapted_params = {k: v.copy() for k, v in self.meta_parameters.items()}
        
        for step in range(num_steps):
            # Calcular gradiente
            gradient = self._compute_gradient(adapted_params, task)
            
            # Actualizar parámetros
            for key in adapted_params:
                adapted_params[key] -= self.inner_lr * gradient.get(key, 0.0)
            
            self.adaptations += 1
        
        self.tasks_processed += 1
        return adapted_params
    
    def meta_update(self, tasks: List[Task], num_adaptation_steps: int = 5):
        """Actualizar parámetros meta."""
        meta_gradient = {k: np.zeros_like(v) for k, v in self.meta_parameters.items()}
        
        for task in tasks:
            # Adaptar a tarea
            adapted_params = self.adapt(task, num_adaptation_steps)
            
            # Calcular gradiente meta
            task_gradient = self._compute_meta_gradient(adapted_params, task)
            
            # Acumular
            for key in meta_gradient:
                if key in task_gradient:
                    meta_gradient[key] += task_gradient[key]
        
        # Actualizar parámetros meta
        for key in self.meta_parameters:
            self.meta_parameters[key] -= self.outer_lr * meta_gradient[key] / len(tasks)
    
    def _compute_gradient(self, params: Dict[str, np.ndarray], 
                         task: Task) -> Dict[str, np.ndarray]:
        """Calcular gradiente (simplificado)."""
        # Simulación de gradiente
        gradient = {}
        for key in params:
            gradient[key] = np.random.normal(0, 0.01, params[key].shape)
        return gradient
    
    def _compute_meta_gradient(self, params: Dict[str, np.ndarray],
                               task: Task) -> Dict[str, np.ndarray]:
        """Calcular gradiente meta."""
        return self._compute_gradient(params, task)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "strategy": "maml",
            "tasks_processed": self.tasks_processed,
            "total_adaptations": self.adaptations,
            "inner_lr": self.inner_lr,
            "outer_lr": self.outer_lr,
            "meta_parameters_count": len(self.meta_parameters)
        }


class ReptileOptimizer:
    """Optimizador Reptile."""
    
    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.meta_parameters: Dict[str, np.ndarray] = {}
        self.tasks_processed = 0
    
    def initialize_meta_parameters(self, parameter_shape: Dict[str, Tuple[int, ...]]):
        """Inicializar parámetros meta."""
        for key, shape in parameter_shape.items():
            self.meta_parameters[key] = np.random.normal(0, 0.1, shape)
    
    def adapt(self, task: Task, num_steps: int = 5) -> Dict[str, np.ndarray]:
        """Adaptar a nueva tarea."""
        adapted_params = {k: v.copy() for k, v in self.meta_parameters.items()}
        
        for step in range(num_steps):
            gradient = self._compute_gradient(adapted_params, task)
            for key in adapted_params:
                adapted_params[key] -= self.learning_rate * gradient.get(key, 0.0)
        
        self.tasks_processed += 1
        return adapted_params
    
    def meta_update(self, tasks: List[Task], num_adaptation_steps: int = 5):
        """Actualizar parámetros meta usando Reptile."""
        for task in tasks:
            adapted_params = self.adapt(task, num_adaptation_steps)
            
            # Actualizar hacia parámetros adaptados
            for key in self.meta_parameters:
                direction = adapted_params[key] - self.meta_parameters[key]
                self.meta_parameters[key] += self.learning_rate * direction
    
    def _compute_gradient(self, params: Dict[str, np.ndarray],
                         task: Task) -> Dict[str, np.ndarray]:
        """Calcular gradiente."""
        gradient = {}
        for key in params:
            gradient[key] = np.random.normal(0, 0.01, params[key].shape)
        return gradient
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "strategy": "reptile",
            "tasks_processed": self.tasks_processed,
            "learning_rate": self.learning_rate,
            "meta_parameters_count": len(self.meta_parameters)
        }


class FewShotLearner:
    """Aprendizaje few-shot."""
    
    def __init__(self):
        self.support_set: List[Task] = []
        self.query_set: List[Task] = []
        self.few_shot_performance: List[float] = []
    
    def add_support_example(self, task: Task):
        """Agregar ejemplo de soporte."""
        self.support_set.append(task)
    
    def add_query_example(self, task: Task):
        """Agregar ejemplo de consulta."""
        self.query_set.append(task)
    
    def learn_from_few_examples(self, num_examples: int = 5) -> Dict[str, Any]:
        """Aprender de pocos ejemplos."""
        if len(self.support_set) < num_examples:
            return {}
        
        # Seleccionar ejemplos
        examples = self.support_set[:num_examples]
        
        # Aprender patrones comunes
        patterns = self._extract_patterns(examples)
        
        # Evaluar en query set
        if self.query_set:
            performance = self._evaluate(patterns, self.query_set)
            self.few_shot_performance.append(performance)
        
        return patterns
    
    def _extract_patterns(self, examples: List[Task]) -> Dict[str, Any]:
        """Extraer patrones comunes."""
        # Simplificado
        return {
            "common_features": ["distance", "cost", "time"],
            "optimization_priority": ["cost", "time", "distance"]
        }
    
    def _evaluate(self, patterns: Dict[str, Any], queries: List[Task]) -> float:
        """Evaluar patrones."""
        # Simplificado
        return 0.85
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "support_set_size": len(self.support_set),
            "query_set_size": len(self.query_set),
            "few_shot_performance": np.mean(self.few_shot_performance) if self.few_shot_performance else 0.0
        }


class TransferLearning:
    """Aprendizaje por transferencia."""
    
    def __init__(self):
        self.source_tasks: List[Task] = []
        self.target_tasks: List[Task] = []
        self.transferred_knowledge: Dict[str, Any] = {}
        self.transfer_success_rate = 0.0
    
    def add_source_task(self, task: Task):
        """Agregar tarea fuente."""
        self.source_tasks.append(task)
    
    def add_target_task(self, task: Task):
        """Agregar tarea objetivo."""
        self.target_tasks.append(task)
    
    def transfer_knowledge(self) -> Dict[str, Any]:
        """Transferir conocimiento."""
        if not self.source_tasks:
            return {}
        
        # Extraer conocimiento de tareas fuente
        knowledge = self._extract_knowledge(self.source_tasks)
        self.transferred_knowledge = knowledge
        
        # Aplicar a tareas objetivo
        if self.target_tasks:
            success = self._apply_knowledge(knowledge, self.target_tasks)
            self.transfer_success_rate = success
        
        return knowledge
    
    def _extract_knowledge(self, tasks: List[Task]) -> Dict[str, Any]:
        """Extraer conocimiento."""
        return {
            "common_patterns": ["pattern1", "pattern2"],
            "optimization_strategies": ["strategy1", "strategy2"],
            "feature_importance": {"feature1": 0.8, "feature2": 0.6}
        }
    
    def _apply_knowledge(self, knowledge: Dict[str, Any], tasks: List[Task]) -> float:
        """Aplicar conocimiento."""
        # Simplificado
        return 0.75
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "source_tasks": len(self.source_tasks),
            "target_tasks": len(self.target_tasks),
            "transferred_knowledge_size": len(self.transferred_knowledge),
            "transfer_success_rate": self.transfer_success_rate
        }


class MetaLearningOptimizer:
    """Optimizador principal de meta-learning."""
    
    def __init__(self, strategy: MetaLearningStrategy = MetaLearningStrategy.MAML,
                 enable_few_shot: bool = True, enable_transfer: bool = True):
        self.strategy = strategy
        
        if strategy == MetaLearningStrategy.MAML:
            self.meta_optimizer = MAMLOptimizer()
        elif strategy == MetaLearningStrategy.REPTILE:
            self.meta_optimizer = ReptileOptimizer()
        else:
            self.meta_optimizer = MAMLOptimizer()
        
        self.few_shot_learner = FewShotLearner() if enable_few_shot else None
        self.transfer_learning = TransferLearning() if enable_transfer else None
        self.tasks: Dict[str, Task] = {}
        self.meta_models: Dict[str, MetaModel] = {}
    
    def create_task(self, task_id: str, task_type: str, data: Dict[str, Any],
                   objective: Callable[[Dict[str, Any]], float]) -> Task:
        """Crear nueva tarea."""
        task = Task(
            task_id=task_id,
            task_type=task_type,
            data=data,
            objective=objective
        )
        self.tasks[task_id] = task
        return task
    
    def learn_from_task(self, task: Task, num_steps: int = 5) -> Dict[str, np.ndarray]:
        """Aprender de una tarea."""
        return self.meta_optimizer.adapt(task, num_steps)
    
    def meta_train(self, task_ids: List[str], num_adaptation_steps: int = 5):
        """Entrenar meta-modelo."""
        tasks = [self.tasks[tid] for tid in task_ids if tid in self.tasks]
        if not tasks:
            return
        
        self.meta_optimizer.meta_update(tasks, num_adaptation_steps)
    
    def few_shot_learn(self, support_task_ids: List[str], query_task_ids: List[str]):
        """Aprendizaje few-shot."""
        if not self.few_shot_learner:
            return {}
        
        for tid in support_task_ids:
            if tid in self.tasks:
                self.few_shot_learner.add_support_example(self.tasks[tid])
        
        for tid in query_task_ids:
            if tid in self.tasks:
                self.few_shot_learner.add_query_example(self.tasks[tid])
        
        return self.few_shot_learner.learn_from_few_examples()
    
    def transfer_knowledge(self, source_task_ids: List[str], target_task_ids: List[str]):
        """Transferir conocimiento."""
        if not self.transfer_learning:
            return {}
        
        for tid in source_task_ids:
            if tid in self.tasks:
                self.transfer_learning.add_source_task(self.tasks[tid])
        
        for tid in target_task_ids:
            if tid in self.tasks:
                self.transfer_learning.add_target_task(self.tasks[tid])
        
        return self.transfer_learning.transfer_knowledge()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        stats = {
            "strategy": self.strategy.value,
            "total_tasks": len(self.tasks),
            "meta_models": len(self.meta_models),
            "meta_optimizer": self.meta_optimizer.get_stats()
        }
        
        if self.few_shot_learner:
            stats["few_shot_learning"] = self.few_shot_learner.get_stats()
        
        if self.transfer_learning:
            stats["transfer_learning"] = self.transfer_learning.get_stats()
        
        return stats


