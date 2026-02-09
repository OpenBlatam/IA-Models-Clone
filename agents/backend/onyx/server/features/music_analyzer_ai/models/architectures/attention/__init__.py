"""
Attention Mechanisms Module

Modular attention implementations following PyTorch best practices.
"""

from .scaled_dot_product import ScaledDotProductAttention
from .multi_head import MultiHeadAttention

# Re-export for backward compatibility
__all__ = [
    "ScaledDotProductAttention",
    "MultiHeadAttention",
]



