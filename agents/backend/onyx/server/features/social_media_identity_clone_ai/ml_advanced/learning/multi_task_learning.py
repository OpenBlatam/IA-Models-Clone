"""
Multi-Task Learning
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MultiTaskHead(nn.Module):
    """Cabeza para multi-task learning"""
    
    def __init__(self, shared_dim: int, task_dims: Dict[str, int], dropout: float = 0.1):
        super().__init__()
        self.shared_dim = shared_dim
        self.task_dims = task_dims
        self.heads = nn.ModuleDict()
        
        for task_name, output_dim in task_dims.items():
            self.heads[task_name] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(shared_dim, shared_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(shared_dim // 2, output_dim)
            )
    
    def forward(self, shared_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass para todas las tareas"""
        outputs = {}
        for task_name, head in self.heads.items():
            outputs[task_name] = head(shared_features)
        return outputs


class MultiTaskModel(nn.Module):
    """Modelo para multi-task learning"""
    
    def __init__(
        self,
        backbone: nn.Module,
        task_configs: Dict[str, Dict[str, Any]]
    ):
        super().__init__()
        self.backbone = backbone
        self.task_configs = task_configs
        
        # Obtener dimensión compartida
        if hasattr(backbone, 'config'):
            shared_dim = backbone.config.hidden_size
        else:
            shared_dim = 768  # Default
        
        # Crear heads para cada tarea
        task_dims = {task: config['output_dim'] for task, config in task_configs.items()}
        self.multi_task_head = MultiTaskHead(shared_dim, task_dims)
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        task_name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Backbone compartido
        backbone_outputs = self.backbone(**inputs)
        
        # Obtener features compartidas
        if hasattr(backbone_outputs, 'last_hidden_state'):
            shared_features = backbone_outputs.last_hidden_state[:, 0, :]  # CLS token
        elif hasattr(backbone_outputs, 'pooler_output'):
            shared_features = backbone_outputs.pooler_output
        else:
            shared_features = backbone_outputs[0][:, 0, :]
        
        # Multi-task heads
        if task_name:
            # Tarea específica
            output = self.multi_task_head.heads[task_name](shared_features)
            return {task_name: output}
        else:
            # Todas las tareas
            return self.multi_task_head(shared_features)


class MultiTaskTrainer:
    """Trainer para multi-task learning"""
    
    def __init__(
        self,
        model: MultiTaskModel,
        task_weights: Optional[Dict[str, float]] = None
    ):
        self.model = model
        self.task_weights = task_weights or {task: 1.0 for task in model.task_configs.keys()}
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calcula loss combinado"""
        total_loss = 0.0
        
        for task_name in predictions.keys():
            pred = predictions[task_name]
            target = targets[task_name]
            weight = self.task_weights.get(task_name, 1.0)
            
            # Loss según tipo de tarea
            task_config = self.model.task_configs[task_name]
            task_type = task_config.get('type', 'classification')
            
            if task_type == 'classification':
                criterion = nn.CrossEntropyLoss()
                loss = criterion(pred, target)
            elif task_type == 'regression':
                criterion = nn.MSELoss()
                loss = criterion(pred, target)
            else:
                criterion = nn.MSELoss()
                loss = criterion(pred, target)
            
            total_loss += weight * loss
        
        return total_loss




