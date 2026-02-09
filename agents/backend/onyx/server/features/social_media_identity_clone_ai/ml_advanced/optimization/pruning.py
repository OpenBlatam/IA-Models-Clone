"""
Model pruning para reducir tamaño y acelerar inferencia
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ModelPruner:
    """Pruner de modelos"""
    
    def __init__(self):
        pass
    
    def prune_structured(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.3,
        method: str = "l1_unstructured"
    ) -> nn.Module:
        """
        Pruning estructurado
        
        Args:
            model: Modelo a podar
            pruning_ratio: Ratio de pruning (0.0 - 1.0)
            method: Método de pruning
            
        Returns:
            Modelo podado
        """
        try:
            parameters_to_prune = []
            
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    parameters_to_prune.append((module, 'weight'))
            
            if method == "l1_unstructured":
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_ratio
                )
            elif method == "ln_structured":
                prune.ln_structured(
                    parameters_to_prune[0][0],
                    name="weight",
                    amount=pruning_ratio,
                    n=2,
                    dim=0
                )
            
            logger.info(f"Modelo podado con ratio {pruning_ratio}")
            return model
            
        except Exception as e:
            logger.error(f"Error en pruning: {e}")
            return model
    
    def remove_pruning(self, model: nn.Module):
        """Remueve pruning (hace permanente)"""
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.remove(module, 'weight')
    
    def get_sparsity(self, model: nn.Module) -> Dict[str, float]:
        """Calcula sparsity del modelo"""
        total_params = 0
        pruned_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight'):
                    weight = module.weight
                    total_params += weight.numel()
                    pruned_params += (weight == 0).sum().item()
        
        sparsity = pruned_params / total_params if total_params > 0 else 0.0
        
        return {
            "total_params": total_params,
            "pruned_params": pruned_params,
            "sparsity": sparsity
        }




