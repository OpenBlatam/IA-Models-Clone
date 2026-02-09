"""
Continual Learning Manager - Gestor de aprendizaje continuo
============================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ContinualLearningMethod(Enum):
    """Métodos de aprendizaje continuo"""
    EWC = "ewc"  # Elastic Weight Consolidation
    REPLAY = "replay"
    REGULARIZATION = "regularization"
    ISOLATION = "isolation"


@dataclass
class Task:
    """Tarea en aprendizaje continuo"""
    task_id: int
    task_name: str
    data_loader: Any
    importance_weights: Dict[str, float] = field(default_factory=dict)


class ContinualLearningManager:
    """Gestor de aprendizaje continuo"""
    
    def __init__(self, method: ContinualLearningMethod = ContinualLearningMethod.EWC):
        self.method = method
        self.tasks: List[Task] = []
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
    
    def add_task(self, task_id: int, task_name: str, data_loader: Any):
        """Agrega una tarea"""
        task = Task(
            task_id=task_id,
            task_name=task_name,
            data_loader=data_loader
        )
        self.tasks.append(task)
        logger.info(f"Tarea agregada: {task_name} (ID: {task_id})")
    
    def compute_fisher_information(
        self,
        model: nn.Module,
        task: Task,
        device: str = "cuda"
    ):
        """Calcula información de Fisher para EWC"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        fisher = {}
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param.data)
        
        # Calcular gradientes
        for batch in task.data_loader:
            if isinstance(batch, dict):
                inputs = batch.get("input_ids") or batch.get("inputs")
                labels = batch.get("labels") or batch.get("targets")
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(**batch)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = torch.nn.functional.cross_entropy(logits, labels)
            model.zero_grad()
            loss.backward()
            
            # Acumular gradientes al cuadrado
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Normalizar
        num_samples = len(task.data_loader)
        for name in fisher:
            fisher[name] /= num_samples
        
        self.fisher_information = fisher
        logger.info("Información de Fisher calculada")
    
    def ewc_loss(
        self,
        model: nn.Module,
        lambda_ewc: float = 0.4
    ) -> torch.Tensor:
        """Pérdida EWC"""
        if not self.fisher_information or not self.optimal_params:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal) ** 2).sum()
        
        return lambda_ewc * loss
    
    def save_optimal_params(self, model: nn.Module):
        """Guarda parámetros óptimos"""
        self.optimal_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        logger.info("Parámetros óptimos guardados")
    
    def get_tasks(self) -> List[Task]:
        """Obtiene lista de tareas"""
        return self.tasks




