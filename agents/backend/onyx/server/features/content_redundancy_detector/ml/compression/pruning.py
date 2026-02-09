"""
Model Pruning
Pruning utilities for model compression
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Any, Optional, List, Callable
import logging

logger = logging.getLogger(__name__)


class PruningStrategy:
    """Pruning strategy configuration"""
    
    UNSTRUCTURED = "unstructured"
    STRUCTURED = "structured"
    GLOBAL = "global"
    LOCAL = "local"


class ModelPruner:
    """
    Model pruning utilities
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize pruner
        
        Args:
            model: Model to prune
        """
        self.model = model
        self.pruning_info = {}
    
    def prune_unstructured(
        self,
        module: nn.Module,
        name: str,
        amount: float = 0.2,
    ) -> None:
        """
        Apply unstructured pruning
        
        Args:
            module: Module to prune
            name: Parameter name
            amount: Pruning amount (0-1)
        """
        prune.l1_unstructured(module, name=name, amount=amount)
        prune.remove(module, name)
        logger.info(f"Pruned {name} by {amount*100:.1f}%")
    
    def prune_structured(
        self,
        module: nn.Module,
        name: str,
        amount: float = 0.2,
        dim: int = 0,
    ) -> None:
        """
        Apply structured pruning
        
        Args:
            module: Module to prune
            name: Parameter name
            amount: Pruning amount (0-1)
            dim: Dimension to prune
        """
        prune.ln_structured(module, name=name, amount=amount, n=2, dim=dim)
        prune.remove(module, name)
        logger.info(f"Structured pruning of {name} by {amount*100:.1f}%")
    
    def prune_global(
        self,
        parameters: List[tuple],
        amount: float = 0.2,
    ) -> None:
        """
        Apply global pruning
        
        Args:
            parameters: List of (module, name) tuples
            amount: Pruning amount (0-1)
        """
        prune.global_unstructured(
            parameters,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        logger.info(f"Global pruning by {amount*100:.1f}%")
    
    def prune_magnitude_based(
        self,
        module: nn.Module,
        name: str,
        amount: float = 0.2,
    ) -> None:
        """
        Magnitude-based pruning
        
        Args:
            module: Module to prune
            name: Parameter name
            amount: Pruning amount (0-1)
        """
        self.prune_unstructured(module, name, amount)
    
    def get_pruning_statistics(self) -> Dict[str, Any]:
        """
        Get pruning statistics
        
        Returns:
            Dictionary with pruning stats
        """
        total_params = 0
        pruned_params = 0
        
        for module in self.model.modules():
            for name, param in module.named_parameters():
                total_params += param.numel()
                if hasattr(module, name + '_mask'):
                    mask = getattr(module, name + '_mask')
                    pruned_params += (mask == 0).sum().item()
        
        return {
            'total_parameters': total_params,
            'pruned_parameters': pruned_params,
            'remaining_parameters': total_params - pruned_params,
            'pruning_ratio': pruned_params / total_params if total_params > 0 else 0,
        }
    
    def apply_pruning(
        self,
        strategy: str = PruningStrategy.UNSTRUCTURED,
        amount: float = 0.2,
        modules_to_prune: Optional[List[str]] = None,
    ) -> None:
        """
        Apply pruning to model
        
        Args:
            strategy: Pruning strategy
            amount: Pruning amount
            modules_to_prune: List of module names to prune (None = all)
        """
        if modules_to_prune is None:
            # Prune all Conv2d and Linear layers
            modules_to_prune = [
                name for name, module in self.model.named_modules()
                if isinstance(module, (nn.Conv2d, nn.Linear))
            ]
        
        for name, module in self.model.named_modules():
            if name in modules_to_prune:
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if strategy == PruningStrategy.UNSTRUCTURED:
                        self.prune_unstructured(module, 'weight', amount)
                    elif strategy == PruningStrategy.STRUCTURED:
                        self.prune_structured(module, 'weight', amount)



