"""
Model Pruning
=============

Pruning de modelos para reducir tamaño y acelerar inferencia.
"""

import logging
import torch
import torch.nn.utils.prune as prune
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class ModelPruner:
    """Pruner de modelos."""
    
    @staticmethod
    def prune_unstructured(
        model: torch.nn.Module,
        amount: float = 0.2,
        module_type: type = torch.nn.Linear
    ) -> torch.nn.Module:
        """
        Pruning no estructurado.
        
        Args:
            model: Modelo a podar
            amount: Cantidad a podar (0-1)
            module_type: Tipo de módulo a podar
        
        Returns:
            Modelo podado
        """
        try:
            for name, module in model.named_modules():
                if isinstance(module, module_type):
                    prune.l1_unstructured(module, name="weight", amount=amount)
                    prune.remove(module, "weight")
            
            logger.info(f"Pruning no estructurado aplicado ({amount*100}%)")
            return model
        
        except Exception as e:
            logger.error(f"Error en pruning: {str(e)}")
            raise
    
    @staticmethod
    def prune_structured(
        model: torch.nn.Module,
        amount: float = 0.2,
        dim: int = 0,
        module_type: type = torch.nn.Linear
    ) -> torch.nn.Module:
        """
        Pruning estructurado.
        
        Args:
            model: Modelo a podar
            amount: Cantidad a podar (0-1)
            dim: Dimensión a podar
            module_type: Tipo de módulo
        
        Returns:
            Modelo podado
        """
        try:
            for name, module in model.named_modules():
                if isinstance(module, module_type):
                    prune.ln_structured(module, name="weight", amount=amount, n=2, dim=dim)
                    prune.remove(module, "weight")
            
            logger.info(f"Pruning estructurado aplicado ({amount*100}%)")
            return model
        
        except Exception as e:
            logger.error(f"Error en pruning estructurado: {str(e)}")
            raise
    
    @staticmethod
    def prune_magnitude(
        model: torch.nn.Module,
        amount: float = 0.2,
        module_type: type = torch.nn.Linear
    ) -> torch.nn.Module:
        """
        Pruning por magnitud.
        
        Args:
            model: Modelo a podar
            amount: Cantidad a podar
            module_type: Tipo de módulo
        
        Returns:
            Modelo podado
        """
        try:
            for name, module in model.named_modules():
                if isinstance(module, module_type):
                    prune.l1_unstructured(module, name="weight", amount=amount)
                    prune.remove(module, "weight")
            
            logger.info(f"Pruning por magnitud aplicado ({amount*100}%)")
            return model
        
        except Exception as e:
            logger.error(f"Error en pruning por magnitud: {str(e)}")
            raise
    
    @staticmethod
    def get_pruning_stats(model: torch.nn.Module) -> Dict[str, Any]:
        """
        Obtener estadísticas de pruning.
        
        Args:
            model: Modelo
        
        Returns:
            Estadísticas
        """
        try:
            total_params = 0
            pruned_params = 0
            
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    total_params += module.weight.numel()
                    pruned_params += (module.weight == 0).sum().item()
            
            return {
                "total_params": total_params,
                "pruned_params": pruned_params,
                "pruning_ratio": pruned_params / total_params if total_params > 0 else 0.0,
                "remaining_params": total_params - pruned_params
            }
        
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {str(e)}")
            return {}




