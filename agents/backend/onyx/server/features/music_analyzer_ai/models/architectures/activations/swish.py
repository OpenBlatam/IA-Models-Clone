"""
Swish Activation Module

Implements Swish activation (SiLU - Sigmoid Linear Unit).
"""

import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish activation (SiLU - Sigmoid Linear Unit)."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Swish activation."""
        return x * torch.sigmoid(x)



