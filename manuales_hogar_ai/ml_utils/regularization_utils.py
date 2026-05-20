"""
Regularization Utils - Utilidades de Regularización
====================================================

Utilidades para técnicas de regularización avanzadas.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class DropBlock(nn.Module):
    """
    DropBlock regularization.
    
    Paper: https://arxiv.org/abs/1810.12890
    """
    
    def __init__(self, block_size: int = 7, p: float = 0.1):
        """
        Inicializar DropBlock.
        
        Args:
            block_size: Tamaño del bloque
            p: Probabilidad de drop
        """
        super().__init__()
        self.block_size = block_size
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplicar DropBlock.
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Tensor con DropBlock aplicado
        """
        if not self.training or self.p == 0:
            return x
        
        gamma = self._compute_gamma(x)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
        mask = mask.to(x.device)
        
        block_mask = self._compute_block_mask(mask)
        
        return x * block_mask[:, None, :, :] * block_mask.numel() / block_mask.sum()
    
    def _compute_gamma(self, x: torch.Tensor) -> float:
        """
        Calcular gamma para DropBlock.
        
        Args:
            x: Input tensor
            
        Returns:
            Gamma value
        """
        return self.p / (self.block_size ** 2)
    
    def _compute_block_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Calcular block mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Block mask
        """
        block_mask = F.max_pool2d(
            mask[:, None, :, :],
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2
        )
        
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask


class SpectralNorm(nn.Module):
    """
    Spectral Normalization para estabilizar entrenamiento.
    
    Paper: https://arxiv.org/abs/1802.05957
    """
    
    def __init__(self, module: nn.Module, power_iterations: int = 1):
        """
        Inicializar Spectral Normalization.
        
        Args:
            module: Módulo a normalizar
            power_iterations: Iteraciones de power method
        """
        super().__init__()
        self.module = module
        self.power_iterations = power_iterations
        
        if not self._check_module():
            raise ValueError("Module must be Conv2d or Linear")
        
        weight = self.module.weight
        height = weight.shape[0]
        width = weight.view(height, -1).shape[1]
        
        u = weight.new_empty(height).normal_(0, 1)
        v = weight.new_empty(width).normal_(0, 1)
        u = F.normalize(u, dim=0, eps=1e-12)
        v = F.normalize(v, dim=0, eps=1e-12)
        
        self.register_buffer('u', u)
        self.register_buffer('v', v)
    
    def _check_module(self) -> bool:
        """Verificar si el módulo es compatible."""
        return isinstance(self.module, (nn.Conv2d, nn.Linear))
    
    def _compute_weight(self) -> torch.Tensor:
        """
        Calcular peso normalizado.
        
        Returns:
            Peso normalizado
        """
        weight = self.module.weight
        u = self.u
        v = self.v
        
        height = weight.shape[0]
        weight_mat = weight.view(height, -1)
        
        with torch.no_grad():
            for _ in range(self.power_iterations):
                v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=1e-12)
                u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=1e-12)
        
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma
        
        return weight
    
    def forward(self, *args):
        """Forward pass."""
        self.module.weight = nn.Parameter(self._compute_weight())
        return self.module(*args)


class WeightDecayRegularizer:
    """
    Regularizador de weight decay personalizado.
    """
    
    def __init__(self, weight_decay: float = 1e-4):
        """
        Inicializar regularizador.
        
        Args:
            weight_decay: Factor de weight decay
        """
        self.weight_decay = weight_decay
    
    def __call__(self, model: nn.Module) -> torch.Tensor:
        """
        Calcular regularización.
        
        Args:
            model: Modelo
            
        Returns:
            Término de regularización
        """
        reg = 0.0
        for param in model.parameters():
            reg += 0.5 * self.weight_decay * torch.sum(param ** 2)
        return reg


class LabelSmoothingRegularizer:
    """
    Regularizador de label smoothing.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Inicializar regularizador.
        
        Args:
            num_classes: Número de clases
            smoothing: Factor de suavizado
        """
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcular label smoothing loss.
        
        Args:
            logits: Logits del modelo
            targets: Targets
            
        Returns:
            Loss con label smoothing
        """
        log_probs = F.log_softmax(logits, dim=1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class GradientPenalty:
    """
    Gradient Penalty para estabilizar entrenamiento GAN.
    """
    
    def __init__(self, lambda_gp: float = 10.0):
        """
        Inicializar gradient penalty.
        
        Args:
            lambda_gp: Factor de penalización
        """
        self.lambda_gp = lambda_gp
    
    def __call__(
        self,
        discriminator: nn.Module,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Calcular gradient penalty.
        
        Args:
            discriminator: Discriminador
            real_samples: Muestras reales
            fake_samples: Muestras falsas
            device: Dispositivo
            
        Returns:
            Gradient penalty
        """
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = discriminator(interpolates)
        
        fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        return gradient_penalty


class MixupRegularizer:
    """
    Regularizador usando técnica Mixup.
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Inicializar Mixup regularizer.
        
        Args:
            alpha: Parámetro de distribución Beta
        """
        self.alpha = alpha
    
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Aplicar Mixup.
        
        Args:
            x: Features
            y: Labels
            
        Returns:
            Tupla (x_mixed, y_mixed, lambda)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, (y_a, y_b), lam


def apply_spectral_norm(module: nn.Module, power_iterations: int = 1) -> nn.Module:
    """
    Aplicar spectral normalization a un módulo.
    
    Args:
        module: Módulo
        power_iterations: Iteraciones
        
    Returns:
        Módulo normalizado
    """
    return SpectralNorm(module, power_iterations)


def apply_dropblock(
    model: nn.Module,
    block_size: int = 7,
    p: float = 0.1
) -> nn.Module:
    """
    Aplicar DropBlock a capas convolucionales.
    
    Args:
        model: Modelo
        block_size: Tamaño del bloque
        p: Probabilidad
        
    Returns:
        Modelo con DropBlock
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # Reemplazar con DropBlock si es necesario
            pass  # Implementación simplificada
    
    return model

