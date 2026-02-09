"""
Layer Manager
=============

Manages layer freezing/unfreezing and layer operations.
"""

import torch.nn as nn
import logging
from typing import Optional, List, Callable

logger = logging.getLogger(__name__)


class LayerManager:
    """Manages layer operations like freezing/unfreezing."""
    
    @staticmethod
    def freeze_layers(
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        freeze_all: bool = False
    ) -> None:
        """
        Freeze specified layers or all layers.
        
        Args:
            model: PyTorch model
            layer_names: List of layer names to freeze (None = all)
            freeze_all: Freeze all layers
        """
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
            logger.info("All layers frozen")
        elif layer_names:
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
            logger.info(f"Frozen layers: {layer_names}")
    
    @staticmethod
    def unfreeze_layers(
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        unfreeze_all: bool = False
    ) -> None:
        """
        Unfreeze specified layers or all layers.
        
        Args:
            model: PyTorch model
            layer_names: List of layer names to unfreeze (None = all)
            unfreeze_all: Unfreeze all layers
        """
        if unfreeze_all:
            for param in model.parameters():
                param.requires_grad = True
            logger.info("All layers unfrozen")
        elif layer_names:
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
            logger.info(f"Unfrozen layers: {layer_names}")
    
    @staticmethod
    def get_frozen_layers(model: nn.Module) -> List[str]:
        """
        Get list of frozen layer names.
        
        Args:
            model: PyTorch model
            
        Returns:
            List of frozen layer names
        """
        frozen = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                frozen.append(name)
        return frozen
    
    @staticmethod
    def apply_to_layers(
        model: nn.Module,
        func: Callable[[nn.Module, str], None],
        layer_filter: Optional[Callable[[str, nn.Module], bool]] = None
    ) -> None:
        """
        Apply function to selected layers.
        
        Args:
            model: PyTorch model
            func: Function to apply (module, name) -> None
            layer_filter: Optional filter function (name, module) -> bool
        """
        for name, module in model.named_modules():
            if layer_filter is None or layer_filter(name, module):
                func(module, name)
    
    @staticmethod
    def clone_model(model: nn.Module) -> nn.Module:
        """
        Create a deep copy of the model.
        
        Args:
            model: PyTorch model to clone
            
        Returns:
            Cloned model
        """
        import copy
        cloned = copy.deepcopy(model)
        return cloned


