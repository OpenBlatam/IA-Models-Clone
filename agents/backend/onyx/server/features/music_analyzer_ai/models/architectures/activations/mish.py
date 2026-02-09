"""
Mish Activation Module

Implements Mish activation function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    """Mish activation function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Mish activation."""
        return x * torch.tanh(F.softplus(x))



