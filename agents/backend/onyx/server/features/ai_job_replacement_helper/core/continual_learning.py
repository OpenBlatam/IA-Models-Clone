"""
Continual Learning Service - Aprendizaje continuo
==================================================

Sistema para aprendizaje continuo sin olvido catastrófico.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ContinualLearningMethod(str):
    """Métodos de aprendizaje continuo"""
    EWC = "ewc"  # Elastic Weight Consolidation
    REPLAY = "replay"  # Experience Replay
    REGULARIZATION = "regularization"  # Regularization-based


@dataclass
class ContinualLearningConfig:
    """Configuración de aprendizaje continuo"""
    method: ContinualLearningMethod = ContinualLearningMethod.EWC
    ewc_lambda: float = 0.4  # Para EWC
    replay_buffer_size: int = 1000  # Para Replay
    importance_weight: float = 1.0  # Para Regularization


class ContinualLearningService:
    """Servicio de aprendizaje continuo"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.task_weights: Dict[str, Dict[str, torch.Tensor]] = {}
        self.replay_buffer: List[Any] = []
        logger.info("ContinualLearningService initialized")
    
    def compute_fisher_information(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Calcular matriz de información de Fisher (para EWC).
        
        Args:
            model: Modelo
            dataloader: DataLoader con datos de la tarea anterior
            device: Dispositivo
        
        Returns:
            Diccionario con información de Fisher por parámetro
        """
        fisher = {}
        model.train()
        
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param.data)
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs = batch.to(device)
                targets = None
            
            model.zero_grad()
            outputs = model(inputs)
            
            if targets is not None:
                loss = nn.functional.cross_entropy(outputs, targets)
            else:
                loss = outputs.mean()
            
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Normalizar
        num_samples = len(dataloader)
        for name in fisher:
            fisher[name] /= num_samples
        
        return fisher
    
    def apply_ewc_loss(
        self,
        model: nn.Module,
        current_loss: torch.Tensor,
        fisher: Dict[str, torch.Tensor],
        previous_params: Dict[str, torch.Tensor],
        lambda_ewc: float
    ) -> torch.Tensor:
        """
        Aplicar pérdida EWC para prevenir olvido.
        
        Args:
            model: Modelo
            current_loss: Pérdida actual
            fisher: Matriz de Fisher
            previous_params: Parámetros de la tarea anterior
            lambda_ewc: Peso de EWC
        
        Returns:
            Pérdida total con regularización EWC
        """
        ewc_loss = 0.0
        
        for name, param in model.named_parameters():
            if name in fisher and name in previous_params:
                ewc_loss += (fisher[name] * (param - previous_params[name]) ** 2).sum()
        
        total_loss = current_loss + lambda_ewc * ewc_loss
        return total_loss
    
    def store_task_weights(
        self,
        task_name: str,
        model: nn.Module,
        fisher: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """Almacenar pesos de una tarea"""
        task_data = {
            "params": {name: param.data.clone() for name, param in model.named_parameters()}
        }
        
        if fisher:
            task_data["fisher"] = fisher
        
        self.task_weights[task_name] = task_data
        logger.info(f"Task weights stored for {task_name}")
    
    def get_task_weights(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Obtener pesos de una tarea"""
        return self.task_weights.get(task_name)
    
    def add_to_replay_buffer(
        self,
        samples: List[Any],
        max_size: int = 1000
    ) -> None:
        """Agregar muestras al buffer de replay"""
        self.replay_buffer.extend(samples)
        
        # Mantener tamaño máximo
        if len(self.replay_buffer) > max_size:
            self.replay_buffer = self.replay_buffer[-max_size:]
        
        logger.info(f"Replay buffer size: {len(self.replay_buffer)}")




