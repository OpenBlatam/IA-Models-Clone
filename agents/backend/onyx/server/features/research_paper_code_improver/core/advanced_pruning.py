"""
Advanced Model Pruning - Poda avanzada de modelos
=================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PruningMethod(Enum):
    """Métodos de poda"""
    MAGNITUDE = "magnitude"
    GRADIENT = "gradient"
    ACTIVATION = "activation"
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"


@dataclass
class PruningConfig:
    """Configuración de poda"""
    method: PruningMethod = PruningMethod.MAGNITUDE
    sparsity: float = 0.5  # 50% de conexiones a eliminar
    structured: bool = False
    iterative: bool = True
    num_iterations: int = 10


class AdvancedPruner:
    """Poda avanzada de modelos"""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.pruning_history: List[Dict[str, Any]] = []
    
    def prune_model(
        self,
        model: nn.Module,
        example_input: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """Poda un modelo"""
        if self.config.iterative:
            return self._iterative_prune(model, example_input)
        else:
            return self._one_shot_prune(model, example_input)
    
    def _iterative_prune(
        self,
        model: nn.Module,
        example_input: Optional[torch.Tensor]
    ) -> nn.Module:
        """Poda iterativa"""
        total_sparsity = self.config.sparsity
        sparsity_per_iteration = total_sparsity / self.config.num_iterations
        
        for iteration in range(self.config.num_iterations):
            current_sparsity = sparsity_per_iteration * (iteration + 1)
            
            if self.config.method == PruningMethod.MAGNITUDE:
                self._magnitude_prune(model, current_sparsity)
            elif self.config.method == PruningMethod.GRADIENT:
                if example_input is not None:
                    self._gradient_prune(model, example_input, current_sparsity)
                else:
                    self._magnitude_prune(model, current_sparsity)
            
            # Fine-tune (simplificado - en producción se haría entrenamiento)
            logger.info(f"Iteración {iteration + 1}/{self.config.num_iterations}, sparsity: {current_sparsity:.2%}")
        
        return model
    
    def _one_shot_prune(
        self,
        model: nn.Module,
        example_input: Optional[torch.Tensor]
    ) -> nn.Module:
        """Poda one-shot"""
        if self.config.method == PruningMethod.MAGNITUDE:
            self._magnitude_prune(model, self.config.sparsity)
        elif self.config.method == PruningMethod.GRADIENT:
            if example_input is not None:
                self._gradient_prune(model, example_input, self.config.sparsity)
            else:
                self._magnitude_prune(model, self.config.sparsity)
        
        return model
    
    def _magnitude_prune(self, model: nn.Module, sparsity: float):
        """Poda por magnitud"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                
                # Calcular threshold
                threshold = torch.quantile(torch.abs(weight), sparsity)
                
                # Crear mask
                mask = torch.abs(weight) > threshold
                
                # Aplicar mask
                module.weight.data = weight * mask.float()
    
    def _gradient_prune(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        sparsity: float
    ):
        """Poda basada en gradientes"""
        model.train()
        
        # Forward y backward
        output = model(example_input)
        loss = output.sum()
        loss.backward()
        
        # Calcular importancia basada en gradientes
        importances = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.weight.grad is not None:
                importance = torch.abs(module.weight * module.weight.grad)
                importances[name] = importance
        
        # Prune basado en importancia
        all_importances = torch.cat([imp.flatten() for imp in importances.values()])
        threshold = torch.quantile(all_importances, sparsity)
        
        for name, module in model.named_modules():
            if name in importances:
                mask = importances[name] > threshold
                module.weight.data = module.weight.data * mask.float()
        
        model.zero_grad()
    
    def get_pruning_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Obtiene estadísticas de poda"""
        total_params = 0
        pruned_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                total_params += weight.numel()
                pruned_params += (weight == 0).sum().item()
        
        sparsity = pruned_params / total_params if total_params > 0 else 0
        
        return {
            "total_parameters": total_params,
            "pruned_parameters": pruned_params,
            "sparsity": sparsity,
            "remaining_parameters": total_params - pruned_params
        }




