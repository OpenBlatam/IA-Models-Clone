from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import math
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
import asyncio
"""
Efficient Fine-tuning Techniques for HeyGen AI.

Implementation of efficient fine-tuning techniques like LoRA, P-tuning, and other
parameter-efficient methods following PEP 8 style guidelines.
"""


logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    rank: int = 8
    alpha: float = 16.0
    dropout_probability: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: bool = False


@dataclass
class PTuningConfig:
    """Configuration for P-tuning."""
    num_virtual_tokens: int = 20
    embedding_dimension: int = 768
    hidden_dimension: int = 512
    dropout_probability: float = 0.1


@dataclass
class AdaLoRAConfig:
    """Configuration for AdaLoRA (Adaptive LoRA)."""
    rank: int = 8
    alpha: float = 16.0
    dropout_probability: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: bool = False
    adaptive_rank: bool = True
    rank_allocation: str = "uniform"  # "uniform", "importance", "magnitude"


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout_probability: float = 0.1,
        bias: bool = False
    ):
        """Initialize LoRA layer.

        Args:
            input_dimension: Input dimension.
            output_dimension: Output dimension.
            rank: Rank of the low-rank adaptation.
            alpha: Scaling factor.
            dropout_probability: Dropout probability.
            bias: Whether to use bias.
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.rank = rank
        self.alpha = alpha
        self.scaling_factor = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, input_dimension) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(output_dimension, rank))
        
        # Optional bias
        self.bias = nn.Parameter(torch.zeros(output_dimension)) if bias else None
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_probability)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> Any:
        """Initialize LoRA weights."""
        # Initialize A with small random values
        nn.init.normal_(self.lora_A, std=0.02)
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of LoRA layer.

        Args:
            input_tensor: Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        # Apply LoRA adaptation
        lora_output = torch.matmul(input_tensor, self.lora_A.T)
        lora_output = self.dropout(lora_output)
        lora_output = torch.matmul(lora_output, self.lora_B.T)
        
        # Scale the output
        lora_output = lora_output * self.scaling_factor
        
        # Add bias if specified
        if self.bias is not None:
            lora_output = lora_output + self.bias
        
        return lora_output


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout_probability: float = 0.1,
        bias: bool = False
    ):
        """Initialize LoRA linear layer.

        Args:
            original_layer: Original linear layer to adapt.
            rank: Rank of the low-rank adaptation.
            alpha: Scaling factor.
            dropout_probability: Dropout probability.
            bias: Whether to use bias.
        """
        super().__init__()
        self.original_layer = original_layer
        self.lora_layer = LoRALayer(
            input_dimension=original_layer.in_features,
            output_dimension=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout_probability=dropout_probability,
            bias=bias
        )
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of LoRA linear layer.

        Args:
            input_tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Original layer output
        original_output = self.original_layer(input_tensor)
        
        # LoRA adaptation
        lora_output = self.lora_layer(input_tensor)
        
        # Combine outputs
        return original_output + lora_output


class PTuningEmbedding(nn.Module):
    """P-tuning virtual token embeddings."""

    def __init__(
        self,
        num_virtual_tokens: int,
        embedding_dimension: int,
        hidden_dimension: int = 512,
        dropout_probability: float = 0.1
    ):
        """Initialize P-tuning embeddings.

        Args:
            num_virtual_tokens: Number of virtual tokens.
            embedding_dimension: Dimension of the embeddings.
            hidden_dimension: Hidden dimension for the MLP.
            dropout_probability: Dropout probability.
        """
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.embedding_dimension = embedding_dimension
        
        # Virtual token embeddings
        self.virtual_embeddings = nn.Parameter(torch.randn(num_virtual_tokens, embedding_dimension))
        
        # MLP for continuous prompt optimization
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(hidden_dimension, embedding_dimension)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dimension)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> Any:
        """Initialize P-tuning weights."""
        # Initialize virtual embeddings with small random values
        nn.init.normal_(self.virtual_embeddings, std=0.02)
        
        # Initialize MLP weights
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Forward pass of P-tuning embeddings.

        Args:
            batch_size: Batch size.

        Returns:
            torch.Tensor: Virtual token embeddings of shape (batch_size, num_virtual_tokens, embedding_dim).
        """
        # Expand virtual embeddings to batch size
        virtual_embeddings = self.virtual_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply MLP transformation
        transformed_embeddings = self.mlp(virtual_embeddings)
        
        # Apply layer normalization
        normalized_embeddings = self.layer_norm(transformed_embeddings)
        
        return normalized_embeddings


class AdaLoRALayer(nn.Module):
    """AdaLoRA (Adaptive LoRA) layer with dynamic rank allocation."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout_probability: float = 0.1,
        bias: bool = False,
        adaptive_rank: bool = True,
        rank_allocation: str = "uniform"
    ):
        """Initialize AdaLoRA layer.

        Args:
            input_dimension: Input dimension.
            output_dimension: Output dimension.
            rank: Initial rank of the adaptation.
            alpha: Scaling factor.
            dropout_probability: Dropout probability.
            bias: Whether to use bias.
            adaptive_rank: Whether to use adaptive rank allocation.
            rank_allocation: Method for rank allocation.
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.rank = rank
        self.alpha = alpha
        self.adaptive_rank = adaptive_rank
        self.rank_allocation = rank_allocation
        self.scaling_factor = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, input_dimension) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(output_dimension, rank))
        
        # Rank importance scores
        if adaptive_rank:
            self.rank_importance = nn.Parameter(torch.ones(rank))
        
        # Optional bias
        self.bias = nn.Parameter(torch.zeros(output_dimension)) if bias else None
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_probability)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> Any:
        """Initialize AdaLoRA weights."""
        # Initialize A with small random values
        nn.init.normal_(self.lora_A, std=0.02)
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)
        
        if self.adaptive_rank:
            # Initialize rank importance scores
            nn.init.ones_(self.rank_importance)

    def get_effective_rank(self) -> int:
        """Get effective rank based on importance scores."""
        if not self.adaptive_rank:
            return self.rank
        
        # Calculate effective rank based on importance scores
        importance_threshold = 0.1
        effective_rank = (self.rank_importance > importance_threshold).sum().item()
        return max(1, effective_rank)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of AdaLoRA layer.

        Args:
            input_tensor: Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        # Get effective rank
        effective_rank = self.get_effective_rank()
        
        # Apply LoRA adaptation with effective rank
        lora_A_effective = self.lora_A[:effective_rank]
        lora_B_effective = self.lora_B[:, :effective_rank]
        
        lora_output = torch.matmul(input_tensor, lora_A_effective.T)
        lora_output = self.dropout(lora_output)
        lora_output = torch.matmul(lora_output, lora_B_effective.T)
        
        # Scale the output
        lora_output = lora_output * self.scaling_factor
        
        # Add bias if specified
        if self.bias is not None:
            lora_output = lora_output + self.bias
        
        return lora_output


class PrefixTuning(nn.Module):
    """Prefix tuning for efficient fine-tuning."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dimension: int,
        prefix_length: int = 20,
        dropout_probability: float = 0.1
    ):
        """Initialize prefix tuning.

        Args:
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dimension: Dimension of each attention head.
            prefix_length: Length of the prefix.
            dropout_probability: Dropout probability.
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dimension = head_dimension
        self.prefix_length = prefix_length
        
        # Prefix embeddings for each layer
        self.prefix_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(prefix_length, num_heads * head_dimension * 2))
            for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_probability)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> Any:
        """Initialize prefix tuning weights."""
        for prefix_embedding in self.prefix_embeddings:
            nn.init.normal_(prefix_embedding, std=0.02)

    def get_prefix_states(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prefix key and value states for a specific layer.

        Args:
            layer_idx: Index of the layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Key and value states.
        """
        prefix_embedding = self.prefix_embeddings[layer_idx]
        
        # Split into key and value
        key_value_dim = self.num_heads * self.head_dimension
        prefix_key = prefix_embedding[:, :key_value_dim]
        prefix_value = prefix_embedding[:, key_value_dim:]
        
        # Reshape for multi-head attention
        prefix_key = prefix_key.view(self.prefix_length, self.num_heads, self.head_dimension)
        prefix_value = prefix_value.view(self.prefix_length, self.num_heads, self.head_dimension)
        
        return prefix_key, prefix_value


class EfficientFineTuningManager:
    """Manager for efficient fine-tuning techniques."""

    def __init__(self, model: nn.Module):
        """Initialize efficient fine-tuning manager.

        Args:
            model: Base model to apply efficient fine-tuning to.
        """
        self.model = model
        self.lora_layers = {}
        self.ptuning_embeddings = None
        self.adalora_layers = {}
        self.prefix_tuning = None

    def apply_lora(
        self,
        target_modules: List[str],
        rank: int = 8,
        alpha: float = 16.0,
        dropout_probability: float = 0.1,
        bias: bool = False
    ) -> None:
        """Apply LoRA to target modules.

        Args:
            target_modules: List of module names to apply LoRA to.
            rank: Rank of the low-rank adaptation.
            alpha: Scaling factor.
            dropout_probability: Dropout probability.
            bias: Whether to use bias.
        """
        for name, module in self.model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA linear layer
                    lora_layer = LoRALinear(
                        original_layer=module,
                        rank=rank,
                        alpha=alpha,
                        dropout_probability=dropout_probability,
                        bias=bias
                    )
                    
                    # Replace the module
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent_module = self.model.get_submodule(parent_name)
                    setattr(parent_module, child_name, lora_layer)
                    
                    self.lora_layers[name] = lora_layer
                    logger.info(f"Applied LoRA to {name}")

    def apply_ptuning(
        self,
        num_virtual_tokens: int,
        embedding_dimension: int,
        hidden_dimension: int = 512,
        dropout_probability: float = 0.1
    ) -> None:
        """Apply P-tuning to the model.

        Args:
            num_virtual_tokens: Number of virtual tokens.
            embedding_dimension: Dimension of the embeddings.
            hidden_dimension: Hidden dimension for the MLP.
            dropout_probability: Dropout probability.
        """
        self.ptuning_embeddings = PTuningEmbedding(
            num_virtual_tokens=num_virtual_tokens,
            embedding_dimension=embedding_dimension,
            hidden_dimension=hidden_dimension,
            dropout_probability=dropout_probability
        )
        logger.info(f"Applied P-tuning with {num_virtual_tokens} virtual tokens")

    def apply_adalora(
        self,
        target_modules: List[str],
        rank: int = 8,
        alpha: float = 16.0,
        dropout_probability: float = 0.1,
        bias: bool = False,
        adaptive_rank: bool = True,
        rank_allocation: str = "uniform"
    ) -> None:
        """Apply AdaLoRA to target modules.

        Args:
            target_modules: List of module names to apply AdaLoRA to.
            rank: Initial rank of the adaptation.
            alpha: Scaling factor.
            dropout_probability: Dropout probability.
            bias: Whether to use bias.
            adaptive_rank: Whether to use adaptive rank allocation.
            rank_allocation: Method for rank allocation.
        """
        for name, module in self.model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Create AdaLoRA layer
                    adalora_layer = AdaLoRALayer(
                        input_dimension=module.in_features,
                        output_dimension=module.out_features,
                        rank=rank,
                        alpha=alpha,
                        dropout_probability=dropout_probability,
                        bias=bias,
                        adaptive_rank=adaptive_rank,
                        rank_allocation=rank_allocation
                    )
                    
                    # Replace the module
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent_module = self.model.get_submodule(parent_name)
                    setattr(parent_module, child_name, adalora_layer)
                    
                    self.adalora_layers[name] = adalora_layer
                    logger.info(f"Applied AdaLoRA to {name}")

    def apply_prefix_tuning(
        self,
        num_layers: int,
        num_heads: int,
        head_dimension: int,
        prefix_length: int = 20,
        dropout_probability: float = 0.1
    ) -> None:
        """Apply prefix tuning to the model.

        Args:
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dimension: Dimension of each attention head.
            prefix_length: Length of the prefix.
            dropout_probability: Dropout probability.
        """
        self.prefix_tuning = PrefixTuning(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dimension=head_dimension,
            prefix_length=prefix_length,
            dropout_probability=dropout_probability
        )
        logger.info(f"Applied prefix tuning with {prefix_length} prefix tokens")

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters from efficient fine-tuning layers.

        Returns:
            List[nn.Parameter]: List of trainable parameters.
        """
        trainable_parameters = []
        
        # LoRA parameters
        for lora_layer in self.lora_layers.values():
            trainable_parameters.extend(lora_layer.lora_layer.parameters())
        
        # P-tuning parameters
        if self.ptuning_embeddings is not None:
            trainable_parameters.extend(self.ptuning_embeddings.parameters())
        
        # AdaLoRA parameters
        for adalora_layer in self.adalora_layers.values():
            trainable_parameters.extend(adalora_layer.parameters())
        
        # Prefix tuning parameters
        if self.prefix_tuning is not None:
            trainable_parameters.extend(self.prefix_tuning.parameters())
        
        return trainable_parameters

    def count_trainable_parameters(self) -> int:
        """Count the number of trainable parameters.

        Returns:
            int: Number of trainable parameters.
        """
        trainable_parameters = self.get_trainable_parameters()
        return sum(p.numel() for p in trainable_parameters)

    def save_efficient_weights(self, filepath: str) -> None:
        """Save efficient fine-tuning weights.

        Args:
            filepath: Path to save the weights.
        """
        efficient_weights = {}
        
        # Save LoRA weights
        for name, lora_layer in self.lora_layers.items():
            efficient_weights[f"lora_{name}"] = {
                'lora_A': lora_layer.lora_layer.lora_A.data,
                'lora_B': lora_layer.lora_layer.lora_B.data,
                'bias': lora_layer.lora_layer.bias.data if lora_layer.lora_layer.bias is not None else None
            }
        
        # Save P-tuning weights
        if self.ptuning_embeddings is not None:
            efficient_weights['ptuning'] = {
                'virtual_embeddings': self.ptuning_embeddings.virtual_embeddings.data,
                'mlp_state_dict': self.ptuning_embeddings.mlp.state_dict(),
                'layer_norm_state_dict': self.ptuning_embeddings.layer_norm.state_dict()
            }
        
        # Save AdaLoRA weights
        for name, adalora_layer in self.adalora_layers.items():
            efficient_weights[f"adalora_{name}"] = {
                'lora_A': adalora_layer.lora_A.data,
                'lora_B': adalora_layer.lora_B.data,
                'rank_importance': adalora_layer.rank_importance.data if adalora_layer.adaptive_rank else None,
                'bias': adalora_layer.bias.data if adalora_layer.bias is not None else None
            }
        
        # Save prefix tuning weights
        if self.prefix_tuning is not None:
            efficient_weights['prefix_tuning'] = {
                'prefix_embeddings': [emb.data for emb in self.prefix_tuning.prefix_embeddings]
            }
        
        torch.save(efficient_weights, filepath)
        logger.info(f"Saved efficient fine-tuning weights to {filepath}")

    def load_efficient_weights(self, filepath: str) -> None:
        """Load efficient fine-tuning weights.

        Args:
            filepath: Path to load the weights from.
        """
        efficient_weights = torch.load(filepath, map_location='cpu')
        
        # Load LoRA weights
        for name, lora_layer in self.lora_layers.items():
            if f"lora_{name}" in efficient_weights:
                weights = efficient_weights[f"lora_{name}"]
                lora_layer.lora_layer.lora_A.data = weights['lora_A']
                lora_layer.lora_layer.lora_B.data = weights['lora_B']
                if weights['bias'] is not None and lora_layer.lora_layer.bias is not None:
                    lora_layer.lora_layer.bias.data = weights['bias']
        
        # Load P-tuning weights
        if self.ptuning_embeddings is not None and 'ptuning' in efficient_weights:
            weights = efficient_weights['ptuning']
            self.ptuning_embeddings.virtual_embeddings.data = weights['virtual_embeddings']
            self.ptuning_embeddings.mlp.load_state_dict(weights['mlp_state_dict'])
            self.ptuning_embeddings.layer_norm.load_state_dict(weights['layer_norm_state_dict'])
        
        # Load AdaLoRA weights
        for name, adalora_layer in self.adalora_layers.items():
            if f"adalora_{name}" in efficient_weights:
                weights = efficient_weights[f"adalora_{name}"]
                adalora_layer.lora_A.data = weights['lora_A']
                adalora_layer.lora_B.data = weights['lora_B']
                if weights['rank_importance'] is not None and adalora_layer.adaptive_rank:
                    adalora_layer.rank_importance.data = weights['rank_importance']
                if weights['bias'] is not None and adalora_layer.bias is not None:
                    adalora_layer.bias.data = weights['bias']
        
        # Load prefix tuning weights
        if self.prefix_tuning is not None and 'prefix_tuning' in efficient_weights:
            weights = efficient_weights['prefix_tuning']
            for i, emb_data in enumerate(weights['prefix_embeddings']):
                self.prefix_tuning.prefix_embeddings[i].data = emb_data
        
        logger.info(f"Loaded efficient fine-tuning weights from {filepath}")


def create_efficient_finetuning_manager(model: nn.Module) -> EfficientFineTuningManager:
    """Create efficient fine-tuning manager.

    Args:
        model: Base model to apply efficient fine-tuning to.

    Returns:
        EfficientFineTuningManager: Created manager.
    """
    return EfficientFineTuningManager(model)


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout_probability: float = 0.1,
    bias: bool = False
) -> EfficientFineTuningManager:
    """Apply LoRA to a model.

    Args:
        model: Base model.
        target_modules: List of module names to apply LoRA to.
        rank: Rank of the low-rank adaptation.
        alpha: Scaling factor.
        dropout_probability: Dropout probability.
        bias: Whether to use bias.

    Returns:
        EfficientFineTuningManager: Manager with LoRA applied.
    """
    manager = EfficientFineTuningManager(model)
    manager.apply_lora(
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout_probability=dropout_probability,
        bias=bias
    )
    return manager


def apply_ptuning_to_model(
    model: nn.Module,
    num_virtual_tokens: int,
    embedding_dimension: int,
    hidden_dimension: int = 512,
    dropout_probability: float = 0.1
) -> EfficientFineTuningManager:
    """Apply P-tuning to a model.

    Args:
        model: Base model.
        num_virtual_tokens: Number of virtual tokens.
        embedding_dimension: Dimension of the embeddings.
        hidden_dimension: Hidden dimension for the MLP.
        dropout_probability: Dropout probability.

    Returns:
        EfficientFineTuningManager: Manager with P-tuning applied.
    """
    manager = EfficientFineTuningManager(model)
    manager.apply_ptuning(
        num_virtual_tokens=num_virtual_tokens,
        embedding_dimension=embedding_dimension,
        hidden_dimension=hidden_dimension,
        dropout_probability=dropout_probability
    )
    return manager


def apply_adalora_to_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout_probability: float = 0.1,
    bias: bool = False,
    adaptive_rank: bool = True,
    rank_allocation: str = "uniform"
) -> EfficientFineTuningManager:
    """Apply AdaLoRA to a model.

    Args:
        model: Base model.
        target_modules: List of module names to apply AdaLoRA to.
        rank: Initial rank of the adaptation.
        alpha: Scaling factor.
        dropout_probability: Dropout probability.
        bias: Whether to use bias.
        adaptive_rank: Whether to use adaptive rank allocation.
        rank_allocation: Method for rank allocation.

    Returns:
        EfficientFineTuningManager: Manager with AdaLoRA applied.
    """
    manager = EfficientFineTuningManager(model)
    manager.apply_adalora(
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout_probability=dropout_probability,
        bias=bias,
        adaptive_rank=adaptive_rank,
        rank_allocation=rank_allocation
    )
    return manager


def apply_prefix_tuning_to_model(
    model: nn.Module,
    num_layers: int,
    num_heads: int,
    head_dimension: int,
    prefix_length: int = 20,
    dropout_probability: float = 0.1
) -> EfficientFineTuningManager:
    """Apply prefix tuning to a model.

    Args:
        model: Base model.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dimension: Dimension of each attention head.
        prefix_length: Length of the prefix.
        dropout_probability: Dropout probability.

    Returns:
        EfficientFineTuningManager: Manager with prefix tuning applied.
    """
    manager = EfficientFineTuningManager(model)
    manager.apply_prefix_tuning(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dimension=head_dimension,
        prefix_length=prefix_length,
        dropout_probability=dropout_probability
    )
    return manager 