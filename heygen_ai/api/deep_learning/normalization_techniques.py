from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math
import logging
from typing import Any, List, Dict, Optional
import asyncio
"""
Normalization Techniques for HeyGen AI.

Advanced normalization techniques including batch normalization, layer normalization,
group normalization, and custom normalization methods following PEP 8 style guidelines.
"""


logger = logging.getLogger(__name__)


class AdvancedBatchNorm1d(nn.Module):
    """Advanced 1D batch normalization with additional features."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        use_running_stats: bool = True
    ):
        """Initialize advanced batch normalization.

        Args:
            num_features: Number of features.
            eps: Small constant for numerical stability.
            momentum: Momentum for running statistics.
            affine: Whether to use learnable parameters.
            track_running_stats: Whether to track running statistics.
            use_running_stats: Whether to use running statistics during training.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.use_running_stats = use_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of advanced batch normalization.

        Args:
            input_tensor: Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:
                exponential_average_factor = self.momentum

        if self.training and self.use_running_stats:
            # Use batch statistics
            batch_mean = input_tensor.mean(dim=0)
            batch_var = input_tensor.var(dim=0, unbiased=False)
            
            # Update running statistics
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = exponential_average_factor * batch_mean + \
                                       (1 - exponential_average_factor) * self.running_mean
                    self.running_var = exponential_average_factor * batch_var + \
                                      (1 - exponential_average_factor) * self.running_var
            
            # Normalize using batch statistics
            normalized = (input_tensor - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # Use running statistics
            normalized = (input_tensor - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        if self.affine:
            normalized = self.weight * normalized + self.bias

        return normalized

    def extra_repr(self) -> str:
        """Extra representation string.

        Returns:
            str: Extra representation.
        """
        return f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, ' \
               f'affine={self.affine}, track_running_stats={self.track_running_stats}'


class AdvancedLayerNorm(nn.Module):
    """Advanced layer normalization with additional features."""

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        use_bias: bool = True
    ):
        """Initialize advanced layer normalization.

        Args:
            normalized_shape: Shape of features to normalize.
            eps: Small constant for numerical stability.
            elementwise_affine: Whether to use learnable parameters.
            use_bias: Whether to use bias term.
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = use_bias

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            if self.use_bias:
                self.bias = nn.Parameter(torch.zeros(normalized_shape))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of advanced layer normalization.

        Args:
            input_tensor: Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Calculate mean and variance over the last dimensions
        mean = input_tensor.mean(dim=-1, keepdim=True)
        var = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        normalized = (input_tensor - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            normalized = self.weight * normalized
            if self.use_bias:
                normalized = normalized + self.bias

        return normalized

    def extra_repr(self) -> str:
        """Extra representation string.

        Returns:
            str: Extra representation.
        """
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class GroupNormalization(nn.Module):
    """Group normalization implementation."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True
    ):
        """Initialize group normalization.

        Args:
            num_groups: Number of groups.
            num_channels: Number of channels.
            eps: Small constant for numerical stability.
            affine: Whether to use learnable parameters.
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of group normalization.

        Args:
            input_tensor: Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        batch_size, num_channels, *spatial_dims = input_tensor.shape
        
        # Reshape for group normalization
        input_reshaped = input_tensor.view(batch_size, self.num_groups, -1)
        
        # Calculate mean and variance over groups
        mean = input_reshaped.mean(dim=-1, keepdim=True)
        var = input_reshaped.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        normalized = (input_reshaped - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        normalized = normalized.view(batch_size, num_channels, *spatial_dims)

        if self.affine:
            # Apply affine transformation
            weight = self.weight.view(1, num_channels, *([1] * len(spatial_dims)))
            bias = self.bias.view(1, num_channels, *([1] * len(spatial_dims)))
            normalized = weight * normalized + bias

        return normalized

    def extra_repr(self) -> str:
        """Extra representation string.

        Returns:
            str: Extra representation.
        """
        return f'{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}'


class InstanceNormalization(nn.Module):
    """Instance normalization implementation."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = False
    ):
        """Initialize instance normalization.

        Args:
            num_features: Number of features.
            eps: Small constant for numerical stability.
            momentum: Momentum for running statistics.
            affine: Whether to use learnable parameters.
            track_running_stats: Whether to track running statistics.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of instance normalization.

        Args:
            input_tensor: Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        batch_size, num_channels, *spatial_dims = input_tensor.shape
        
        # Reshape for instance normalization
        input_reshaped = input_tensor.view(batch_size, num_channels, -1)
        
        # Calculate mean and variance over spatial dimensions
        mean = input_reshaped.mean(dim=-1, keepdim=True)
        var = input_reshaped.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        normalized = (input_reshaped - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        normalized = normalized.view(batch_size, num_channels, *spatial_dims)

        if self.affine:
            # Apply affine transformation
            weight = self.weight.view(1, num_channels, *([1] * len(spatial_dims)))
            bias = self.bias.view(1, num_channels, *([1] * len(spatial_dims)))
            normalized = weight * normalized + bias

        return normalized

    def extra_repr(self) -> str:
        """Extra representation string.

        Returns:
            str: Extra representation.
        """
        return f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, ' \
               f'affine={self.affine}, track_running_stats={self.track_running_stats}'


class AdaptiveNormalization(nn.Module):
    """Adaptive normalization that switches between different normalization methods."""

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        normalization_type: str = "layer",
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        num_groups: Optional[int] = None
    ):
        """Initialize adaptive normalization.

        Args:
            normalized_shape: Shape of features to normalize.
            normalization_type: Type of normalization ('layer', 'batch', 'group', 'instance').
            eps: Small constant for numerical stability.
            momentum: Momentum for running statistics.
            affine: Whether to use learnable parameters.
            num_groups: Number of groups for group normalization.
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.normalization_type = normalization_type
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.num_groups = num_groups

        # Create normalization layers
        if normalization_type == "layer":
            self.norm_layer = AdvancedLayerNorm(
                normalized_shape, eps, affine
            )
        elif normalization_type == "batch":
            if isinstance(normalized_shape, int):
                self.norm_layer = AdvancedBatchNorm1d(
                    normalized_shape, eps, momentum, affine
                )
            else:
                raise ValueError("Batch normalization requires integer normalized_shape")
        elif normalization_type == "group":
            if isinstance(normalized_shape, int) and num_groups is not None:
                self.norm_layer = GroupNormalization(
                    num_groups, normalized_shape, eps, affine
                )
            else:
                raise ValueError("Group normalization requires integer normalized_shape and num_groups")
        elif normalization_type == "instance":
            if isinstance(normalized_shape, int):
                self.norm_layer = InstanceNormalization(
                    normalized_shape, eps, momentum, affine
                )
            else:
                raise ValueError("Instance normalization requires integer normalized_shape")
        else:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of adaptive normalization.

        Args:
            input_tensor: Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return self.norm_layer(input_tensor)

    def extra_repr(self) -> str:
        """Extra representation string.

        Returns:
            str: Extra representation.
        """
        return f'{self.normalized_shape}, type={self.normalization_type}, eps={self.eps}, ' \
               f'momentum={self.momentum}, affine={self.affine}'


class WeightStandardization(nn.Module):
    """Weight standardization for convolutional layers."""

    def __init__(self, eps: float = 1e-5):
        """Initialize weight standardization.

        Args:
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Forward pass of weight standardization.

        Args:
            weight: Weight tensor.

        Returns:
            torch.Tensor: Standardized weight tensor.
        """
        # Calculate mean and variance over output channels
        mean = weight.mean(dim=1, keepdim=True)
        var = weight.var(dim=1, keepdim=True, unbiased=False)
        
        # Standardize weights
        standardized = (weight - mean) / torch.sqrt(var + self.eps)
        
        return standardized


class NormalizationFactory:
    """Factory for creating normalization layers."""

    @staticmethod
    def create_normalization(
        normalization_type: str,
        **kwargs
    ) -> nn.Module:
        """Create normalization layer.

        Args:
            normalization_type: Type of normalization.
            **kwargs: Additional arguments.

        Returns:
            nn.Module: Created normalization layer.

        Raises:
            ValueError: If normalization type is not supported.
        """
        if normalization_type == "batch":
            return AdvancedBatchNorm1d(**kwargs)
        elif normalization_type == "layer":
            return AdvancedLayerNorm(**kwargs)
        elif normalization_type == "group":
            return GroupNormalization(**kwargs)
        elif normalization_type == "instance":
            return InstanceNormalization(**kwargs)
        elif normalization_type == "adaptive":
            return AdaptiveNormalization(**kwargs)
        elif normalization_type == "weight_standardization":
            return WeightStandardization(**kwargs)
        else:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")

    @staticmethod
    def get_normalization_config(
        normalization_type: str,
        input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """Get configuration for normalization layer.

        Args:
            normalization_type: Type of normalization.
            input_shape: Input tensor shape.

        Returns:
            Dict[str, Any]: Normalization configuration.
        """
        config = {
            "eps": 1e-5,
            "affine": True
        }

        if normalization_type == "batch":
            config["num_features"] = input_shape[1]
            config["momentum"] = 0.1
            config["track_running_stats"] = True
        elif normalization_type == "layer":
            config["normalized_shape"] = input_shape[1:]
            config["elementwise_affine"] = True
        elif normalization_type == "group":
            config["num_channels"] = input_shape[1]
            config["num_groups"] = min(32, input_shape[1] // 4)
        elif normalization_type == "instance":
            config["num_features"] = input_shape[1]
            config["momentum"] = 0.1
            config["track_running_stats"] = False
        elif normalization_type == "adaptive":
            config["normalized_shape"] = input_shape[1:]
            config["normalization_type"] = "layer"
            config["momentum"] = 0.1

        return config


def create_normalization_layer(
    normalization_type: str,
    input_shape: Tuple[int, ...],
    **kwargs
) -> nn.Module:
    """Factory function to create normalization layer.

    Args:
        normalization_type: Type of normalization.
        input_shape: Input tensor shape.
        **kwargs: Additional arguments.

    Returns:
        nn.Module: Created normalization layer.
    """
    config = NormalizationFactory.get_normalization_config(normalization_type, input_shape)
    config.update(kwargs)
    
    return NormalizationFactory.create_normalization(normalization_type, **config) 