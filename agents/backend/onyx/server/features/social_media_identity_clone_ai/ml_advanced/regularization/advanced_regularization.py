"""
Técnicas avanzadas de regularización
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DropBlock(nn.Module):
    """DropBlock regularization"""
    
    def __init__(self, block_size: int = 7, drop_prob: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        
        # Calcular gamma
        gamma = self._compute_gamma(x)
        
        # Crear mask
        mask = (torch.rand(x.shape[0], *x.shape[2:], device=x.device) < gamma).float()
        mask = mask.unsqueeze(1)
        
        # Aplicar block drop
        mask = F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2
        )
        
        mask = 1 - mask
        normalize = mask.numel() / mask.sum()
        
        return x * mask * normalize
    
    def _compute_gamma(self, x: torch.Tensor) -> float:
        """Calcula gamma para DropBlock"""
        return self.drop_prob / (self.block_size ** 2)


class SpectralNormalization:
    """Spectral Normalization para estabilidad"""
    
    @staticmethod
    def apply(module: nn.Module, name: str = 'weight', n_power_iterations: int = 1):
        """Aplica spectral normalization"""
        fn = SpectralNorm(name, n_power_iterations)
        module.register_forward_pre_hook(fn)
        return module


class SpectralNorm:
    """Spectral normalization hook"""
    
    def __init__(self, name: str = 'weight', n_power_iterations: int = 1):
        self.name = name
        self.n_power_iterations = n_power_iterations
    
    def compute_weight(self, module: nn.Module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        
        height = weight.size(0)
        weight_mat = weight.view(height, -1)
        
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=1e-12)
                u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=1e-12)
        
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma
        return weight, u, v
    
    def __call__(self, module: nn.Module, inputs: Any):
        weight, u, v = self.compute_weight(module)
        setattr(module, self.name, weight)
        setattr(module, self.name + '_u', u)
        setattr(module, self.name + '_v', v)


class LabelSmoothingRegularization:
    """Label Smoothing avanzado"""
    
    def __init__(self, smoothing: float = 0.1, num_classes: int = 10):
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
    
    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class MixUp:
    """MixUp data augmentation"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> tuple:
        """Aplica MixUp"""
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1.0
        
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size).to(inputs.device)
        
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        
        return mixed_inputs, mixed_targets, lam




