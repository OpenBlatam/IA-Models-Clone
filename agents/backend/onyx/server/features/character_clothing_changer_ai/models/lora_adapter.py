"""
LoRA Adapter for Flux2 Clothing Changer
========================================

Support for Low-Rank Adaptation (LoRA) to fine-tune the model
for specific clothing styles or characters.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
from safetensors.torch import save_file, load_file
import json
import math

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        """
        Initialize LoRA layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: LoRA rank (lower = fewer parameters)
            alpha: LoRA alpha scaling factor
            dropout: Dropout rate
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        # LoRA computation: x @ A^T @ B^T * scaling
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return lora_output * self.scaling


class LoRAAdapter(nn.Module):
    """LoRA adapter for Flux2 transformer layers."""
    
    def __init__(
        self,
        target_modules: List[str],
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        """
        Initialize LoRA adapter.
        
        Args:
            target_modules: List of module names to apply LoRA to
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: Dropout rate
        """
        super().__init__()
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lora_layers: Dict[str, LoRALayer] = nn.ModuleDict()
    
    def inject_lora(
        self,
        model: nn.Module,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """
        Inject LoRA layers into target modules.
        
        Args:
            model: Model to inject LoRA into
            target_modules: Optional list of target module names
        """
        if target_modules is None:
            target_modules = self.target_modules
        
        for name, module in model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # Create LoRA layer
                    if isinstance(module, nn.Linear):
                        lora = LoRALayer(
                            module.in_features,
                            module.out_features,
                            rank=self.rank,
                            alpha=self.alpha,
                            dropout=self.dropout,
                        )
                    else:  # Conv2d
                        lora = LoRALayer(
                            module.in_channels,
                            module.out_channels,
                            rank=self.rank,
                            alpha=self.alpha,
                            dropout=self.dropout,
                        )
                    
                    # Store original weights
                    module.register_buffer('lora_enabled', torch.tensor(True))
                    
                    # Add LoRA layer
                    self.lora_layers[name] = lora
                    logger.info(f"Injected LoRA into {name}")
    
    def forward_with_lora(
        self,
        module: nn.Module,
        x: torch.Tensor,
        module_name: str,
    ) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            module: Original module
            x: Input tensor
            module_name: Name of the module
            
        Returns:
            Output tensor with LoRA adaptation
        """
        # Original output
        original_output = module(x)
        
        # LoRA adaptation if available
        if module_name in self.lora_layers:
            lora_output = self.lora_layers[module_name](x)
            return original_output + lora_output
        
        return original_output
    
    def save_lora_weights(self, path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save LoRA weights to file.
        
        Args:
            path: Path to save LoRA weights
            metadata: Optional metadata
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare weights
        weights = {}
        for name, layer in self.lora_layers.items():
            weights[f"{name}.lora_A"] = layer.lora_A
            weights[f"{name}.lora_B"] = layer.lora_B
        
        # Save safetensors
        save_file(weights, str(path))
        
        # Save metadata
        if metadata:
            metadata_path = path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"LoRA weights saved to {path}")
    
    def load_lora_weights(self, path: Path) -> None:
        """
        Load LoRA weights from file.
        
        Args:
            path: Path to LoRA weights file
        """
        path = Path(path)
        
        # Load weights
        weights = load_file(str(path))
        
        # Load into layers
        for name, layer in self.lora_layers.items():
            if f"{name}.lora_A" in weights and f"{name}.lora_B" in weights:
                layer.lora_A.data = weights[f"{name}.lora_A"]
                layer.lora_B.data = weights[f"{name}.lora_B"]
                logger.info(f"Loaded LoRA weights for {name}")
        
        logger.info(f"LoRA weights loaded from {path}")
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of trainable LoRA parameters."""
        params = []
        for layer in self.lora_layers.values():
            params.extend([layer.lora_A, layer.lora_B])
        return params
    
    def enable_lora(self) -> None:
        """Enable LoRA layers."""
        for layer in self.lora_layers.values():
            layer.train()
    
    def disable_lora(self) -> None:
        """Disable LoRA layers (set to eval mode)."""
        for layer in self.lora_layers.values():
            layer.eval()


import math

