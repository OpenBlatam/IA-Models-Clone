"""
Continual Learning Utils - Utilidades de Aprendizaje Continuo
==============================================================

Utilidades para continual learning y prevención de catastrophic forgetting.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)


class EWC(nn.Module):
    """
    Elastic Weight Consolidation (EWC).
    
    Paper: https://arxiv.org/abs/1612.00796
    """
    
    def __init__(
        self,
        model: nn.Module,
        importance: float = 1.0
    ):
        """
        Inicializar EWC.
        
        Args:
            model: Modelo
            importance: Factor de importancia
        """
        super().__init__()
        self.model = model
        self.importance = importance
        self.fisher_info: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
    
    def compute_fisher_information(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None
    ):
        """
        Calcular Fisher Information Matrix.
        
        Args:
            dataloader: DataLoader de tarea anterior
            num_samples: Número de muestras (opcional)
        """
        self.model.eval()
        fisher = {}
        
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param.data)
        
        sample_count = 0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            
            sample_count += len(inputs)
            if num_samples and sample_count >= num_samples:
                break
        
        # Normalizar
        for name in fisher:
            fisher[name] /= sample_count
        
        self.fisher_info = fisher
    
    def save_optimal_params(self):
        """Guardar parámetros óptimos de tarea anterior."""
        self.optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
    
    def ewc_loss(self) -> torch.Tensor:
        """
        Calcular loss de EWC.
        
        Returns:
            Loss de EWC
        """
        loss = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_info and name in self.optimal_params:
                fisher = self.fisher_info[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal) ** 2).sum()
        
        return self.importance * loss


class ReplayBuffer:
    """
    Buffer de replay para continual learning.
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Inicializar buffer.
        
        Args:
            capacity: Capacidad del buffer
        """
        self.capacity = capacity
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
    
    def add(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Agregar muestras al buffer.
        
        Args:
            inputs: Inputs
            targets: Targets
        """
        for i in range(len(inputs)):
            if len(self.buffer) >= self.capacity:
                # Reemplazar aleatoriamente
                idx = np.random.randint(0, len(self.buffer))
                self.buffer[idx] = (inputs[i].clone(), targets[i].clone())
            else:
                self.buffer.append((inputs[i].clone(), targets[i].clone()))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Muestrear del buffer.
        
        Args:
            batch_size: Tamaño de batch
            
        Returns:
            Tupla (inputs, targets)
        """
        if len(self.buffer) == 0:
            return None, None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        inputs = torch.stack([self.buffer[i][0] for i in indices])
        targets = torch.stack([self.buffer[i][1] for i in indices])
        
        return inputs, targets
    
    def __len__(self) -> int:
        """Tamaño del buffer."""
        return len(self.buffer)


class ContinualLearningTrainer:
    """
    Trainer para continual learning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        ewc: Optional[EWC] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        replay_weight: float = 0.5
    ):
        """
        Inicializar trainer.
        
        Args:
            model: Modelo
            ewc: EWC (opcional)
            replay_buffer: Buffer de replay (opcional)
            replay_weight: Peso de replay loss
        """
        self.model = model
        self.ewc = ewc
        self.replay_buffer = replay_buffer
        self.replay_weight = replay_weight
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Realizar paso de entrenamiento.
        
        Args:
            inputs: Inputs de nueva tarea
            targets: Targets de nueva tarea
            optimizer: Optimizador
            
        Returns:
            Métricas
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Loss de nueva tarea
        outputs = self.model(inputs)
        task_loss = F.cross_entropy(outputs, targets)
        total_loss = task_loss
        
        # Loss de EWC
        ewc_loss_val = 0.0
        if self.ewc is not None:
            ewc_loss_val = self.ewc.ewc_loss()
            total_loss += ewc_loss_val
        
        # Loss de replay
        replay_loss_val = 0.0
        if self.replay_buffer is not None and len(self.replay_buffer) > 0:
            replay_inputs, replay_targets = self.replay_buffer.sample(len(inputs))
            if replay_inputs is not None:
                replay_outputs = self.model(replay_inputs)
                replay_loss_val = F.cross_entropy(replay_outputs, replay_targets)
                total_loss += self.replay_weight * replay_loss_val
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'task_loss': task_loss.item(),
            'ewc_loss': ewc_loss_val.item() if isinstance(ewc_loss_val, torch.Tensor) else 0.0,
            'replay_loss': replay_loss_val.item() if isinstance(replay_loss_val, torch.Tensor) else 0.0,
            'total_loss': total_loss.item()
        }




