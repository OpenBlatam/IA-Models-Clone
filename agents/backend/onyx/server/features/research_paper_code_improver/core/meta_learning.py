"""
Meta Learning Framework - Framework de meta aprendizaje
========================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MetaLearningMethod(Enum):
    """Métodos de meta aprendizaje"""
    MAML = "maml"  # Model-Agnostic Meta-Learning
    REPTILE = "reptile"
    FOMAML = "fomaml"  # First-Order MAML


@dataclass
class Task:
    """Tarea en meta aprendizaje"""
    support_set: Any  # Few-shot support examples
    query_set: Any  # Query examples for evaluation
    task_id: int = 0


class MetaLearner:
    """Meta aprendiz"""
    
    def __init__(self, method: MetaLearningMethod = MetaLearningMethod.MAML):
        self.method = method
        self.tasks: List[Task] = []
    
    def maml_step(
        self,
        model: nn.Module,
        task: Task,
        inner_lr: float = 0.01,
        num_inner_steps: int = 1,
        device: str = "cuda"
    ) -> torch.Tensor:
        """Paso de MAML"""
        device = torch.device(device)
        model = model.to(device)
        
        # Copiar parámetros para inner loop
        fast_weights = {name: param.clone() for name, param in model.named_parameters()}
        
        # Inner loop: adaptación rápida
        for _ in range(num_inner_steps):
            # Forward en support set
            if isinstance(task.support_set, dict):
                inputs = task.support_set.get("input_ids") or task.support_set.get("inputs")
                labels = task.support_set.get("labels") or task.support_set.get("targets")
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(**{**task.support_set, **{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                                         for k, v in task.support_set.items()}})
            else:
                inputs, labels = task.support_set
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = F.cross_entropy(logits, labels)
            
            # Actualizar fast weights
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            fast_weights = {
                name: fast_weights[name] - inner_lr * grad
                for (name, _), grad in zip(model.named_parameters(), grads)
            }
        
        # Outer loop: evaluar en query set
        query_loss = 0.0
        if isinstance(task.query_set, dict):
            inputs = task.query_set.get("input_ids") or task.query_set.get("inputs")
            labels = task.query_set.get("labels") or task.query_set.get("targets")
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward con fast weights
            # (Simplificado - en producción se necesitaría implementar forward con fast_weights)
            outputs = model(**task.query_set)
        else:
            inputs, labels = task.query_set
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        query_loss = F.cross_entropy(logits, labels)
        
        return query_loss
    
    def add_task(self, support_set: Any, query_set: Any, task_id: int = 0):
        """Agrega una tarea"""
        task = Task(
            support_set=support_set,
            query_set=query_set,
            task_id=task_id
        )
        self.tasks.append(task)
        logger.info(f"Tarea agregada: {task_id}")




