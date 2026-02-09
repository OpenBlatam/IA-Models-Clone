"""Base classes and utilities for model implementations."""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


@dataclass
class BaseModelArgs(ABC):
    """Base configuration for model architectures."""
    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seq_len: int = 2048
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    use_quantization: bool = False
    quantization_bits: int = 8


class RMSNorm(nn.Module):
    """RMS Normalization layer."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class BaseLinear(nn.Module):
    """Base linear layer with optional quantization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        use_quantization: bool = False,
        quantization_bits: int = 8
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        
        self.weight = Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize weight tensor."""
        if self.quantization_bits == 8:
            scale = weight.abs().max() / 127.0
            quantized = torch.round(weight / scale).clamp(-128, 127)
            return quantized * scale
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantization and self.training:
            weight = self.quantize_weight(self.weight)
        else:
            weight = self.weight
        return F.linear(x, weight, self.bias)


class SafetyLinear(BaseLinear):
    """Linear layer with constitutional AI safety filtering."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        use_quantization: bool = False,
        quantization_bits: int = 8,
        apply_safety_filter: bool = False
    ):
        super().__init__(in_features, out_features, bias, use_quantization, quantization_bits)
        self.apply_safety_filter = apply_safety_filter

    def apply_constitutional_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply constitutional AI filtering."""
        return torch.clamp(x, -10.0, 10.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x)
        if self.apply_safety_filter:
            output = self.apply_constitutional_filter(output)
        return output


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Precompute frequency tensor for rotary embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device or freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    seq_len = xq_.shape[1]
    if freqs_cis.shape[0] != seq_len:
        freqs_cis = freqs_cis[:seq_len]
    
    if len(freqs_cis.shape) == 2:
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class BaseTransformerBlock(nn.Module, ABC):
    """Base transformer block interface."""
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass through transformer block."""
        pass


class BaseTransformer(nn.Module, ABC):
    """Base transformer model interface."""
    
    @abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0
    ) -> torch.Tensor:
        """Forward pass through transformer."""
        pass

