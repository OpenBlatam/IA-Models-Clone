"""
LoRA (Low-Rank Adaptation) for Efficient Fine-tuning

Implements:
- LoRA adapters for efficient fine-tuning of transformer models
- Proper rank and alpha parameters
- Integration with existing models
- Support for multiple target modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Union
import logging
import math

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer.
    
    Implements efficient fine-tuning by adding low-rank matrices
    to existing linear layers.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False
    ):
        """
        Initialize LoRA layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: LoRA rank (r in the paper)
            alpha: LoRA alpha scaling parameter
            dropout: Dropout probability
            merge_weights: Whether to merge LoRA weights with base weights
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        
        # LoRA matrices: W = W_0 + (B @ A) * scaling
        # A: (rank, in_features)
        # B: (out_features, rank)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Initialize LoRA weights.
        
        A is initialized with Kaiming uniform, B with zeros.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LoRA layer.
        
        Args:
            x: Input tensor (batch_size, ..., in_features)
            
        Returns:
            Output tensor (batch_size, ..., out_features)
        """
        # LoRA computation: x @ A^T @ B^T * scaling
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        return lora_output
    
    def merge(self) -> torch.Tensor:
        """
        Merge LoRA weights into a single matrix.
        
        Returns:
            Merged weight matrix
        """
        return self.lora_B @ self.lora_A * self.scaling
    
    def extra_repr(self) -> str:
        """String representation."""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'rank={self.rank}, alpha={self.alpha}, scaling={self.scaling}'


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Wraps a base linear layer and adds LoRA adapters.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False
    ):
        """
        Initialize LoRA linear layer.
        
        Args:
            base_layer: Base linear layer to adapt
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: Dropout probability
            merge_weights: Whether to merge weights
        """
        super().__init__()
        
        self.base_layer = base_layer
        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            merge_weights=merge_weights
        )
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Base layer output
        base_output = self.base_layer(x)
        
        # LoRA output
        lora_output = self.lora(x)
        
        return base_output + lora_output
    
    def merge_lora_weights(self) -> None:
        """Merge LoRA weights into base layer."""
        with torch.no_grad():
            merged_weights = self.base_layer.weight + self.lora.merge()
            self.base_layer.weight.data = merged_weights


class LoRAAdapter:
    """
    LoRA adapter for adding LoRA to existing models.
    
    Supports adding LoRA to multiple target modules.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_modules: Optional[List[str]] = None,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        """
        Initialize LoRA adapter.
        
        Args:
            model: Model to adapt
            target_modules: List of module names to add LoRA to
                          (e.g., ['attention.query', 'attention.value'])
                          If None, adds to all linear layers
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: Dropout probability
        """
        self.model = model
        self.target_modules = target_modules or []
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        self.lora_layers: Dict[str, LoRALinear] = {}
        self._add_lora_layers()
    
    def _add_lora_layers(self) -> None:
        """Add LoRA layers to target modules."""
        if not self.target_modules:
            # Add to all linear layers
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    self._replace_module(name, module)
        else:
            # Add only to specified modules
            for target_name in self.target_modules:
                module = self._get_module_by_name(target_name)
                if module is not None and isinstance(module, nn.Linear):
                    self._replace_module(target_name, module)
    
    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """Get module by name."""
        parts = name.split('.')
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module
    
    def _replace_module(self, name: str, module: nn.Linear) -> None:
        """Replace module with LoRA version."""
        parts = name.split('.')
        parent = self.model
        
        # Navigate to parent module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Create LoRA linear layer
        lora_linear = LoRALinear(
            base_layer=module,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )
        
        # Replace module
        setattr(parent, parts[-1], lora_linear)
        self.lora_layers[name] = lora_linear
        
        logger.info(f"Added LoRA to module: {name}")
    
    def get_lora_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get all LoRA parameters for optimizer.
        
        Returns:
            List of LoRA parameters
        """
        parameters = []
        for lora_layer in self.lora_layers.values():
            parameters.extend(lora_layer.lora.parameters())
        return parameters
    
    def merge_weights(self) -> None:
        """Merge all LoRA weights into base layers."""
        for lora_layer in self.lora_layers.values():
            lora_layer.merge_lora_weights()
        logger.info("Merged LoRA weights into base model")
    
    def save_lora_weights(self, path: str) -> None:
        """
        Save only LoRA weights.
        
        Args:
            path: Path to save weights
        """
        state_dict = {}
        for name, lora_layer in self.lora_layers.items():
            state_dict[f"{name}.lora_A"] = lora_layer.lora.lora_A
            state_dict[f"{name}.lora_B"] = lora_layer.lora.lora_B
        
        torch.save(state_dict, path)
        logger.info(f"Saved LoRA weights to {path}")
    
    def load_lora_weights(self, path: str) -> None:
        """
        Load LoRA weights.
        
        Args:
            path: Path to load weights from
        """
        state_dict = torch.load(path, map_location='cpu')
        
        for name, lora_layer in self.lora_layers.items():
            if f"{name}.lora_A" in state_dict:
                lora_layer.lora.lora_A.data = state_dict[f"{name}.lora_A"]
            if f"{name}.lora_B" in state_dict:
                lora_layer.lora.lora_B.data = state_dict[f"{name}.lora_B"]
        
        logger.info(f"Loaded LoRA weights from {path}")


def add_lora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0
) -> LoRAAdapter:
    """
    Convenience function to add LoRA to a model.
    
    Args:
        model: Model to adapt
        target_modules: Target module names
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout probability
        
    Returns:
        LoRA adapter instance
    """
    adapter = LoRAAdapter(
        model=model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout
    )
    return adapter



