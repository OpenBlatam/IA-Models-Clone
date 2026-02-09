from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from enum import Enum
from torch.utils.data import Dataset, DataLoader
import warnings
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Diffusion Models
Comprehensive implementation of diffusion models with advanced features and best practices.
Enhanced with proper forward and reverse diffusion processes, noise schedulers, and sampling methods.
"""



class DiffusionType(Enum):
    """Types of diffusion models."""
    DDPM = "ddpm"
    DDIM = "ddim"
    DPM_SOLVER = "dpm_solver"
    EULER_ANCESTRAL = "euler_ancestral"
    EULER = "euler"
    HEUN = "heun"
    LMS = "lms"


class SchedulerType(Enum):
    """Types of noise schedulers."""
    LINEAR = "linear"
    COSINE = "cosine"
    QUADRATIC = "quadratic"
    SIGMOID = "sigmoid"
    SCALED_LINEAR = "scaled_linear"
    KARRAS = "karras"


@dataclass
class DiffusionConfig:
    """Configuration for diffusion models."""
    # Model architecture
    image_size: int = 64
    in_channels: int = 3
    out_channels: int = 3
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 8
    
    # Diffusion parameters
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    scheduler_type: SchedulerType = SchedulerType.LINEAR
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 100
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    
    # Sampling parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0  # For DDIM
    clip_denoised: bool = True
    use_clipped_model_output: bool = True
    
    # Advanced features
    use_attention: bool = True
    use_conditioning: bool = False
    use_classifier_free_guidance: bool = True
    prediction_type: str = "epsilon"  # "epsilon" or "v_prediction"
    loss_type: str = "l2"  # "l2" or "l1" or "huber"


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""
    
    def __init__(self, dim: int):
        
    """__init__ function."""
super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Forward pass for sinusoidal embedding."""
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNetBlock(nn.Module):
    """UNet block with residual connections."""
    
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, 
                 use_attention: bool = True):
        
    """__init__ function."""
super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Group normalization
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        residual = self.residual_conv(x)
        
        # First convolution
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)
        
        # Time embedding
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        x = x + time_emb
        
        # Second convolution
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)
        
        # Attention mechanism
        if self.use_attention:
            b, c, h, w = x.shape
            x_flat = x.view(b, c, h * w).transpose(1, 2)
            attn_out, _ = self.attention(x_flat, x_flat, x_flat)
            attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
            x = x + attn_out
        
        # Residual connection
        x = x + residual
        
        return x


class UNet(nn.Module):
    """UNet architecture for diffusion models."""
    
    def __init__(self, config: DiffusionConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Time embedding
        time_dim = config.hidden_size
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(config.in_channels, config.hidden_size, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        current_channels = config.hidden_size
        
        for i in range(config.num_layers):
            out_channels = current_channels * 2 if i < config.num_layers - 1 else current_channels
            self.down_blocks.append(UNetBlock(current_channels, out_channels, time_dim, config.use_attention))
            current_channels = out_channels
        
        # Middle block
        self.middle_block = UNetBlock(current_channels, current_channels, time_dim, config.use_attention)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in range(config.num_layers):
            in_channels = current_channels
            out_channels = current_channels // 2 if i < config.num_layers - 1 else config.out_channels
            self.up_blocks.append(UNetBlock(in_channels, out_channels, time_dim, config.use_attention))
            current_channels = out_channels
        
        # Final convolution
        self.final_conv = nn.Conv2d(config.out_channels, config.out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass through UNet."""
        # Time embedding
        time_emb = self.time_embedding(timesteps)
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Downsampling
        down_features = []
        for block in self.down_blocks:
            x = block(x, time_emb)
            down_features.append(x)
            x = F.avg_pool2d(x, 2)
        
        # Middle block
        x = self.middle_block(x, time_emb)
        
        # Upsampling
        for i, block in enumerate(self.up_blocks):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i < len(down_features):
                x = torch.cat([x, down_features[-(i+1)]], dim=1)
            x = block(x, time_emb)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x


class AdvancedNoiseScheduler:
    """Advanced noise scheduler for diffusion models with multiple scheduling strategies."""
    
    def __init__(self, config: DiffusionConfig):
        
    """__init__ function."""
self.config = config
        self.num_timesteps = config.num_timesteps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        
        # Create noise schedule
        self.betas = self._create_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # For DDIM
        self.eta = config.eta
        
        # For v-prediction
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1.0 - self.alphas_cumprod_prev)
    
    def _create_noise_schedule(self) -> torch.Tensor:
        """Create noise schedule based on configuration."""
        if self.config.scheduler_type == SchedulerType.LINEAR:
            return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.config.scheduler_type == SchedulerType.COSINE:
            return self._cosine_beta_schedule()
        elif self.config.scheduler_type == SchedulerType.QUADRATIC:
            return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_timesteps) ** 2
        elif self.config.scheduler_type == SchedulerType.SIGMOID:
            return torch.sigmoid(torch.linspace(-6, 6, self.num_timesteps)) * (self.beta_end - self.beta_start) + self.beta_start
        elif self.config.scheduler_type == SchedulerType.SCALED_LINEAR:
            return self._scaled_linear_beta_schedule()
        elif self.config.scheduler_type == SchedulerType.KARRAS:
            return self._karras_beta_schedule()
        else:
            return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine beta schedule."""
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _scaled_linear_beta_schedule(self) -> torch.Tensor:
        """Scaled linear beta schedule."""
        scale = 1000 / self.num_timesteps
        beta_start = self.beta_start * scale
        beta_end = self.beta_end * scale
        return torch.linspace(beta_start, beta_end, self.num_timesteps)
    
    def _karras_beta_schedule(self) -> torch.Tensor:
        """Karras beta schedule."""
        sigma_min = 0.002
        sigma_max = 80.0
        rho = 7.0
        
        ramp = torch.linspace(0, 1, self.num_timesteps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        sigmas = torch.cat([sigmas, torch.zeros(1)])
        
        alphas_cumprod = 1 / (1 + sigmas ** 2)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, x_start: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to images at given timesteps (Forward Diffusion Process)."""
        noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        noisy_images = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_images, noise
    
    def remove_noise(self, x_t: torch.Tensor, noise_pred: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Remove predicted noise from noisy images (Reverse Diffusion Process)."""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        x_start = (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
        return x_start
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, 
             eta: float = None) -> torch.Tensor:
        """Single denoising step (Reverse Diffusion Process)."""
        if eta is None:
            eta = self.eta
        
        # 1. compute previous: (x_t -> x_t-1)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod_prev[timestep] if timestep > 0 else torch.tensor(1.0)
        
        variance = 0
        if eta > 0:
            noise = torch.randn_like(model_output)
            variance = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5 * eta * noise
        
        # "pred_original_sample" -> "x_0"
        pred_original_sample = (sample - ((1 - alpha_prod_t) ** 0.5) * model_output) / alpha_prod_t ** 0.5
        
        # "pred_sample_direction" -> "direction pointing to x_t"
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        
        # x_t-1 = (pred_original_sample * alpha_prod_t_prev ** 0.5) + pred_sample_direction + variance
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction + variance
        
        return prev_sample
    
    def get_velocity(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Get velocity for v-prediction."""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        velocity = sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_start
        return velocity


class DiffusionModel(nn.Module):
    """Complete diffusion model implementation with enhanced forward and reverse processes."""
    
    def __init__(self, config: DiffusionConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # UNet backbone
        self.unet = UNet(config)
        
        # Noise scheduler
        self.scheduler = AdvancedNoiseScheduler(config)
        
        # Classifier-free guidance (if enabled)
        self.use_classifier_free_guidance = config.use_classifier_free_guidance
        if self.use_classifier_free_guidance:
            self.null_embedding = nn.Parameter(torch.randn(1, config.hidden_size))
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, 
                conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for training."""
        # Add noise (Forward Diffusion Process)
        noisy_x, noise = self.scheduler.add_noise(x, timesteps)
        
        # Predict noise or velocity
        if self.config.prediction_type == "epsilon":
            # Predict noise
            model_output = self.unet(noisy_x, timesteps)
        elif self.config.prediction_type == "v_prediction":
            # Predict velocity
            velocity = self.scheduler.get_velocity(x, noise, timesteps)
            model_output = self.unet(noisy_x, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
        
        return model_output
    
    def sample(self, batch_size: int = 1, conditioning: Optional[torch.Tensor] = None,
               guidance_scale: float = 7.5, num_inference_steps: int = 50,
               eta: float = None, use_ddim: bool = False) -> torch.Tensor:
        """Generate samples using the diffusion model (Reverse Diffusion Process)."""
        device = next(self.parameters()).device
        
        # Initialize with random noise
        x = torch.randn(batch_size, self.config.in_channels, self.config.image_size, 
                       self.config.image_size, device=device)
        
        # Set timesteps
        if use_ddim:
            timesteps = self._get_ddim_timesteps(num_inference_steps)
        else:
            timesteps = torch.linspace(0, self.config.num_timesteps - 1, num_inference_steps, 
                                      dtype=torch.long, device=device).flip(0)
        
        # Sampling loop
        for i, t in enumerate(timesteps):
            # Create timestep tensor
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise/velocity
            if self.use_classifier_free_guidance and conditioning is not None:
                # Classifier-free guidance
                noise_pred_uncond = self.unet(x, timestep)
                noise_pred_cond = self.unet(x, timestep)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.unet(x, timestep)
            
            # Denoising step
            if use_ddim:
                x = self._ddim_step(x, noise_pred, timestep, eta)
            else:
                x = self._ddpm_step(x, noise_pred, timestep)
        
        return x
    
    def _ddpm_step(self, x: torch.Tensor, noise_pred: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """DDPM denoising step."""
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        alpha_prod_t_prev = self.scheduler.alphas_cumprod_prev[timestep].view(-1, 1, 1, 1)
        
        # Predict x_0
        pred_original_sample = (x - ((1 - alpha_prod_t) ** 0.5) * noise_pred) / alpha_prod_t ** 0.5
        
        # Clip if needed
        if self.config.clip_denoised:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Predict mean
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
        
        # Add noise
        noise = torch.randn_like(x) if timestep[0] > 0 else torch.zeros_like(x)
        pred_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction + \
                     ((1 - alpha_prod_t_prev) ** 0.5) * noise
        
        return pred_sample
    
    def _ddim_step(self, x: torch.Tensor, noise_pred: torch.Tensor, timestep: torch.Tensor, 
                   eta: float = None) -> torch.Tensor:
        """DDIM denoising step."""
        if eta is None:
            eta = self.config.eta
        
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        alpha_prod_t_prev = self.scheduler.alphas_cumprod_prev[timestep].view(-1, 1, 1, 1)
        
        # Predict x_0
        pred_original_sample = (x - ((1 - alpha_prod_t) ** 0.5) * noise_pred) / alpha_prod_t ** 0.5
        
        # Clip if needed
        if self.config.clip_denoised:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Predict direction
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
        
        # Add variance if eta > 0
        variance = 0
        if eta > 0:
            noise = torch.randn_like(x)
            variance = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5 * eta * noise
        
        pred_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction + variance
        
        return pred_sample
    
    def _get_ddim_timesteps(self, num_inference_steps: int) -> torch.Tensor:
        """Get timesteps for DDIM sampling."""
        step_ratio = self.config.num_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).flip(0)
        return timesteps
    
    def _add_noise_for_sampling(self, x: torch.Tensor, current_t: torch.Tensor, 
                                next_t: torch.Tensor) -> torch.Tensor:
        """Add noise during sampling process."""
        alpha_cumprod_t = self.scheduler.alphas_cumprod[current_t].view(-1, 1, 1, 1)
        alpha_cumprod_next = self.scheduler.alphas_cumprod[next_t].view(-1, 1, 1, 1)
        
        # Calculate noise to add
        noise = torch.randn_like(x)
        x = torch.sqrt(alpha_cumprod_next) * x + torch.sqrt(1 - alpha_cumprod_next) * noise
        
        return x


class DiffusionTrainer:
    """Trainer for diffusion models with enhanced training process."""
    
    def __init__(self, model: DiffusionModel, config: DiffusionConfig):
        
    """__init__ function."""
self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Loss function
        if config.loss_type == "l2":
            self.loss_fn = nn.MSELoss()
        elif config.loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif config.loss_type == "huber":
            self.loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")
        
        # Training state
        self.epoch = 0
        self.step = 0
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> Any:
        """Setup logging for training."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('diffusion_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def training_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Perform training step with enhanced forward and reverse processes."""
        batch = batch.to(self.device)
        batch_size = batch.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.config.num_timesteps, (batch_size,), 
                                 device=self.device, dtype=torch.long)
        
        # Forward Diffusion Process: Add noise
        noisy_images, noise = self.model.scheduler.add_noise(batch, timesteps)
        
        # Predict noise/velocity
        if self.config.prediction_type == "epsilon":
            # Predict noise
            model_output = self.model.unet(noisy_images, timesteps)
            target = noise
        elif self.config.prediction_type == "v_prediction":
            # Predict velocity
            velocity = self.model.scheduler.get_velocity(batch, noise, timesteps)
            model_output = self.model.unet(noisy_images, timesteps)
            target = velocity
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
        
        # Calculate loss
        loss = self.loss_fn(model_output, target)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        self.optimizer.step()
        
        self.step += 1
        
        return {
            'loss': loss.item(),
            'step': self.step
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            step_results = self.training_step(batch)
            epoch_loss += step_results['loss']
            num_batches += 1
            
            # Logging
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {self.epoch + 1}, Step {self.step}, "
                    f"Loss: {step_results['loss']:.6f}"
                )
        
        return {
            'epoch_loss': epoch_loss / num_batches,
            'num_batches': num_batches
        }
    
    def train(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Complete training loop."""
        self.logger.info(f"Starting diffusion training for {self.config.num_epochs} epochs")
        
        training_history = []
        
        for epoch in range(self.config.num_epochs):
            # Training
            epoch_results = self.train_epoch(dataloader)
            training_history.append(epoch_results)
            
            # Logging
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"Loss: {epoch_results['epoch_loss']:.6f}"
            )
            
            # Generate sample images periodically
            if (epoch + 1) % 10 == 0:
                self._generate_samples(epoch + 1)
            
            self.epoch += 1
        
        return {
            'training_history': training_history,
            'final_loss': training_history[-1]['epoch_loss'] if training_history else None
        }
    
    def _generate_samples(self, epoch: int, num_samples: int = 4):
        """Generate sample images during training."""
        self.model.eval()
        
        with torch.no_grad():
            # Test different sampling methods
            samples_ddpm = self.model.sample(
                batch_size=num_samples,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                use_ddim=False
            )
            
            samples_ddim = self.model.sample(
                batch_size=num_samples,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                use_ddim=True
            )
            
            # Save samples (simplified - in practice, you'd save as images)
            self.logger.info(f"Generated {num_samples} samples at epoch {epoch}")
            self.logger.info(f"DDPM samples shape: {samples_ddpm.shape}")
            self.logger.info(f"DDIM samples shape: {samples_ddim.shape}")
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epoch': self.epoch,
            'step': self.step
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.step = checkpoint.get('step', 0)
        self.logger.info(f"Model loaded from {path}")


class DiffusionAnalyzer:
    """Analyzer for diffusion models with enhanced analysis capabilities."""
    
    def __init__(self) -> Any:
        self.analysis_results = {}
    
    def analyze_model(self, model: DiffusionModel) -> Dict[str, Any]:
        """Analyze diffusion model properties."""
        analysis = {
            'model_class': model.__class__.__name__,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
            'config': model.config.__dict__,
            'scheduler_type': model.config.scheduler_type.value,
            'prediction_type': model.config.prediction_type,
            'loss_type': model.config.loss_type
        }
        
        return analysis
    
    def benchmark_sampling(self, model: DiffusionModel, num_samples: int = 4, 
                          num_inference_steps: int = 50) -> Dict[str, Any]:
        """Benchmark sampling performance with different methods."""
        device = next(model.parameters()).device
        
        # Warmup
        with torch.no_grad():
            _ = model.sample(batch_size=1, num_inference_steps=10)
        
        # Benchmark DDPM
        ddpm_times = []
        ddpm_memory = []
        
        for _ in range(5):
            start_time = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                samples = model.sample(
                    batch_size=num_samples,
                    num_inference_steps=num_inference_steps,
                    use_ddim=False
                )
            
            end_time = time.time()
            ddpm_times.append(end_time - start_time)
            
            if torch.cuda.is_available():
                ddpm_memory.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        
        # Benchmark DDIM
        ddim_times = []
        ddim_memory = []
        
        for _ in range(5):
            start_time = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                samples = model.sample(
                    batch_size=num_samples,
                    num_inference_steps=num_inference_steps,
                    use_ddim=True
                )
            
            end_time = time.time()
            ddim_times.append(end_time - start_time)
            
            if torch.cuda.is_available():
                ddim_memory.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        
        return {
            'ddpm_sampling': {
                'time': {
                    'mean': np.mean(ddpm_times),
                    'std': np.std(ddpm_times),
                    'min': np.min(ddpm_times),
                    'max': np.max(ddpm_times)
                },
                'memory_usage_mb': {
                    'mean': np.mean(ddpm_memory) if ddpm_memory else 0,
                    'max': np.max(ddpm_memory) if ddpm_memory else 0
                },
                'samples_per_second': num_samples / np.mean(ddpm_times)
            },
            'ddim_sampling': {
                'time': {
                    'mean': np.mean(ddim_times),
                    'std': np.std(ddim_times),
                    'min': np.min(ddim_times),
                    'max': np.max(ddim_times)
                },
                'memory_usage_mb': {
                    'mean': np.mean(ddim_memory) if ddim_memory else 0,
                    'max': np.max(ddim_memory) if ddim_memory else 0
                },
                'samples_per_second': num_samples / np.mean(ddim_times)
            }
        }


def demonstrate_diffusion_models():
    """Demonstrate diffusion models capabilities with enhanced features."""
    print("Enhanced Diffusion Models Demonstration")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        DiffusionConfig(
            image_size=32,
            hidden_size=64,
            num_layers=3,
            num_timesteps=100,
            num_inference_steps=20,
            scheduler_type=SchedulerType.LINEAR,
            prediction_type="epsilon"
        ),
        DiffusionConfig(
            image_size=64,
            hidden_size=128,
            num_layers=4,
            num_timesteps=200,
            num_inference_steps=30,
            scheduler_type=SchedulerType.COSINE,
            prediction_type="v_prediction"
        ),
        DiffusionConfig(
            image_size=32,
            hidden_size=64,
            num_layers=3,
            num_timesteps=100,
            num_inference_steps=20,
            scheduler_type=SchedulerType.KARRAS,
            prediction_type="epsilon"
        )
    ]
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\nTesting diffusion model {i+1}:")
        print(f"  Scheduler: {config.scheduler_type.value}")
        print(f"  Prediction type: {config.prediction_type}")
        
        try:
            # Create model
            model = DiffusionModel(config)
            
            # Analyze model
            analyzer = DiffusionAnalyzer()
            model_analysis = analyzer.analyze_model(model)
            print(f"  Model size: {model_analysis['total_parameters']:,} parameters")
            print(f"  Model size: {model_analysis['model_size_mb']:.2f} MB")
            
            # Benchmark sampling
            benchmark_results = analyzer.benchmark_sampling(model, num_samples=2, num_inference_steps=20)
            print(f"  DDPM sampling time: {benchmark_results['ddpm_sampling']['time']['mean']:.4f}s")
            print(f"  DDIM sampling time: {benchmark_results['ddim_sampling']['time']['mean']:.4f}s")
            print(f"  DDPM samples per second: {benchmark_results['ddpm_sampling']['samples_per_second']:.2f}")
            print(f"  DDIM samples per second: {benchmark_results['ddim_sampling']['samples_per_second']:.2f}")
            
            # Test sampling
            with torch.no_grad():
                samples_ddpm = model.sample(batch_size=2, num_inference_steps=10, use_ddim=False)
                samples_ddim = model.sample(batch_size=2, num_inference_steps=10, use_ddim=True)
                print(f"  DDPM samples shape: {samples_ddpm.shape}")
                print(f"  DDIM samples shape: {samples_ddim.shape}")
            
            results[f"model_{i}"] = {
                'config': config,
                'model_analysis': model_analysis,
                'benchmark_results': benchmark_results,
                'success': True
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[f"model_{i}"] = {
                'config': config,
                'error': str(e),
                'success': False
            }
    
    return results


if __name__ == "__main__":
    # Demonstrate diffusion models
    results = demonstrate_diffusion_models()
    print("\nEnhanced diffusion models demonstration completed!") 