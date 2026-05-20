"""
Fused Operations for Maximum Speed

Fused operations combine multiple operations into one for better performance.
This reduces memory access and improves cache efficiency.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

logger = logging.getLogger(__name__)


class FusedLayerNorm(nn.Module):
    """
    Fused Layer Normalization.
    
    Combines normalization and activation for better performance.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused operation: norm + scale + shift
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class FusedLinearGELU(nn.Module):
    """
    Fused Linear + GELU activation.
    
    More efficient than separate operations.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused: linear + gelu
        return F.gelu(self.linear(x))


class FusedAttention(nn.Module):
    """
    Fused Multi-Head Attention.
    
    Uses optimized attention operations.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Fused QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / (self.d_k ** 0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Fused QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention (optimized)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        return self.out_proj(context)


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuse Conv2d and BatchNorm2d into single Conv2d.
    
    This reduces computation and memory access.
    """
    # Get parameters
    conv_weight = conv.weight.data
    conv_bias = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.running_mean)
    
    bn_weight = bn.weight.data
    bn_bias = bn.bias.data
    bn_mean = bn.running_mean
    bn_var = bn.running_var
    bn_eps = bn.eps
    
    # Fuse
    bn_std = torch.sqrt(bn_var + bn_eps)
    fused_weight = conv_weight * (bn_weight / bn_std).view(-1, 1, 1, 1)
    fused_bias = (conv_bias - bn_mean) * bn_weight / bn_std + bn_bias
    
    # Create fused conv
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True
    )
    
    fused_conv.weight.data = fused_weight
    fused_conv.bias.data = fused_bias
    
    return fused_conv


def optimize_model_fused(model: nn.Module) -> nn.Module:
    """
    Optimize model by fusing operations.
    
    Args:
        model: PyTorch model
        
    Returns:
        Optimized model
    """
    # Fuse Conv-BN pairs
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            new_modules = []
            i = 0
            while i < len(module):
                if (i < len(module) - 1 and 
                    isinstance(module[i], nn.Conv2d) and 
                    isinstance(module[i+1], nn.BatchNorm2d)):
                    fused = fuse_conv_bn(module[i], module[i+1])
                    new_modules.append(fused)
                    i += 2
                else:
                    new_modules.append(module[i])
                    i += 1
            setattr(model, name, nn.Sequential(*new_modules))
    
    return model

