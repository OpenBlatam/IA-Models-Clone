"""
Diffusion Model for Trajectory Generation
=======================================

Modelo de difusión para generar trayectorias de robot suaves y naturales
usando el proceso de difusión forward y reverse.
"""

import logging
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .base_model import BaseRobotModel

logger = logging.getLogger(__name__)


class SinusoidalPositionalEmbedding(nn.Module):
    """Embedding posicional sinusoidal para timesteps de difusión."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Crear embeddings posicionales para timesteps.
        
        Args:
            time: Tensor de timesteps [batch_size]
            
        Returns:
            Embeddings [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class DiffusionUNet(nn.Module):
    """
    UNet para modelo de difusión de trayectorias.
    
    Arquitectura similar a DDPM pero adaptada para datos de trayectorias 1D/2D.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_res_blocks: int = 2,
        time_embed_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Inicializar UNet.
        
        Args:
            input_dim: Dimensión de entrada (tamaño de trayectoria)
            hidden_dim: Dimensión oculta
            num_res_blocks: Número de bloques residuales
            time_embed_dim: Dimensión de embedding temporal
            dropout: Tasa de dropout
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Embedding temporal
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Entrada
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        dim = hidden_dim
        for _ in range(3):  # 3 niveles de downsampling
            self.down_blocks.append(self._make_res_block(dim, dim, dropout))
            dim = dim * 2
        
        # Middle
        self.middle_block = self._make_res_block(dim, dim, dropout)
        
        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        for _ in range(3):
            dim = dim // 2
            self.up_blocks.append(self._make_res_block(dim * 2, dim, dropout))
        
        # Salida
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def _make_res_block(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float
    ) -> nn.Module:
        """Crear bloque residual."""
        return nn.Sequential(
            nn.GroupNorm(8, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(dropout),
            nn.GroupNorm(8, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Trayectoria ruidosa [batch_size, trajectory_length, input_dim]
            timestep: Timestep de difusión [batch_size]
            
        Returns:
            Predicción de ruido [batch_size, trajectory_length, input_dim]
        """
        # Embedding temporal
        t_emb = self.time_embed(timestep)  # [batch_size, hidden_dim]
        
        # Proyección de entrada
        h = self.input_proj(x)  # [batch_size, trajectory_length, hidden_dim]
        
        # Encoder
        skip_connections = []
        for down_block in self.down_blocks:
            h = down_block(h)
            skip_connections.append(h)
            # Downsample (simplificado - solo reducción de dimensión)
            h = F.avg_pool1d(h.transpose(1, 2), kernel_size=2).transpose(1, 2)
        
        # Middle
        h = self.middle_block(h)
        
        # Decoder
        for up_block in self.up_blocks:
            # Upsample
            h = F.interpolate(h.transpose(1, 2), scale_factor=2, mode='nearest').transpose(1, 2)
            # Skip connection
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=-1)
            h = up_block(h)
        
        # Aplicar embedding temporal
        t_emb = t_emb.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        h = h + t_emb
        
        # Salida
        output = self.output_proj(h)
        
        return output


class DiffusionTrajectoryGenerator(BaseRobotModel):
    """
    Generador de trayectorias usando modelo de difusión.
    
    Implementa el proceso de difusión forward y reverse para generar
    trayectorias suaves y naturales.
    """
    
    def __init__(
        self,
        trajectory_length: int,
        trajectory_dim: int = 3,  # x, y, z
        hidden_dim: int = 256,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        """
        Inicializar generador de difusión.
        
        Args:
            trajectory_length: Longitud de la trayectoria
            trajectory_dim: Dimensión de cada punto (3 para x,y,z)
            hidden_dim: Dimensión oculta del modelo
            num_timesteps: Número de pasos de difusión
            beta_schedule: Tipo de schedule de beta
            beta_start: Valor inicial de beta
            beta_end: Valor final de beta
        """
        input_size = trajectory_length * trajectory_dim
        output_size = trajectory_length * trajectory_dim
        
        super().__init__(input_size, output_size, name="DiffusionTrajectoryGenerator")
        
        self.trajectory_length = trajectory_length
        self.trajectory_dim = trajectory_dim
        self.num_timesteps = num_timesteps
        
        # UNet para predicción de ruido
        self.unet = DiffusionUNet(
            input_dim=trajectory_dim,
            hidden_dim=hidden_dim
        )
        
        # Schedule de beta
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            # Cosine schedule (mejor para generación)
            s = 0.008
            steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # Calcular alphas y otros valores necesarios
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: agregar ruido a la trayectoria.
        
        Args:
            x_start: Trayectoria inicial [batch_size, trajectory_length, dim]
            t: Timesteps [batch_size]
            noise: Ruido opcional
            
        Returns:
            Trayectoria ruidosa [batch_size, trajectory_length, dim]
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int
    ) -> torch.Tensor:
        """
        Reverse diffusion: remover ruido paso a paso.
        
        Args:
            x: Trayectoria ruidosa [batch_size, trajectory_length, dim]
            t: Timesteps [batch_size]
            t_index: Índice del timestep actual
            
        Returns:
            Trayectoria menos ruidosa [batch_size, trajectory_length, dim]
        """
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1)
        
        # Predecir ruido
        predicted_noise = self.unet(x, t)
        
        # Calcular media del posterior
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: predecir ruido.
        
        Args:
            x: Trayectoria ruidosa [batch_size, trajectory_length, dim]
            timestep: Timesteps [batch_size]
            
        Returns:
            Ruido predicho [batch_size, trajectory_length, dim]
        """
        return self.unet(x, timestep)
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generar nueva trayectoria mediante reverse diffusion.
        
        Args:
            batch_size: Tamaño del batch
            device: Dispositivo (CPU/GPU)
            
        Returns:
            Trayectoria generada [batch_size, trajectory_length, dim]
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Empezar con ruido puro
        shape = (batch_size, self.trajectory_length, self.trajectory_dim)
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, i)
        
        return x
    
    def compute_loss(
        self,
        x_start: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcular pérdida de entrenamiento.
        
        Args:
            x_start: Trayectorias reales [batch_size, trajectory_length, dim]
            noise: Ruido opcional
            
        Returns:
            Pérdida MSE
        """
        batch_size = x_start.size(0)
        
        # Sample timesteps aleatorios
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_start.device).long()
        
        # Agregar ruido
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predecir ruido
        predicted_noise = self.unet(x_noisy, t)
        
        # Pérdida MSE
        loss = F.mse_loss(noise, predicted_noise)
        
        return loss

