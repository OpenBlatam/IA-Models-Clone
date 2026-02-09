"""
GELU Activation Module

Implements Gaussian Error Linear Unit activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU activation."""
        return F.gelu(x)



