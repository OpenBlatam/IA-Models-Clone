"""
Enhanced Diffusion Models System with Advanced Noise Schedulers and Sampling Methods
Integrates with Diffusers library for production-ready diffusion model implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import json
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Enhanced imports for advanced diffusion capabilities
try:
    from diffusers import (
        DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler,
        HeunDiscreteScheduler, DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler, UniPCMultistepScheduler,
        StableDiffusionPipeline, StableDiffusionXLPipeline,
        DiffusionPipeline, AutoencoderKL, UNet2DConditionModel
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: Diffusers library not available. Using custom implementations.")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers library not available.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiffusionConfig:
    """Configuration for diffusion models"""
    # Model architecture
    in_channels: int = 3
    out_channels: int = 3
    model_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16, 8)
    dropout: float = 0.1
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    conv_resample: bool = True
    num_heads: int = 8
    use_spatial_transformer: bool = True
    transformer_depth: int = 1
    context_dim: Optional[int] = None
    n_embed: Optional[int] = None
    legacy: bool = False
    
    # Diffusion process
    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_diffusion_timesteps: int = 1000
    schedule_type: str = "linear"  # linear, cosine, quadratic, sigmoid, exponential
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 100
    gradient_clip_val: float = 1.0
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    # Sampling
    sampling_method: str = "ddpm"  # ddpm, ddim, ancestral, euler, heun, dpm_solver
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    classifier_free_guidance: bool = True
    
    # Advanced features
    use_ema: bool = True
    ema_decay: float = 0.9999
    use_gradient_checkpointing: bool = False
    use_xformers: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DiffusionConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DiffusionConfig':
        """Load config from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, yaml_path: str):
        """Save config to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

class NoiseScheduler(ABC):
    """Abstract base class for noise schedulers"""
    
    @abstractmethod
    def get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get beta value at timestep t"""
        pass
    
    @abstractmethod
    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha value at timestep t"""
        pass
    
    @abstractmethod
    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Get sigma value at timestep t"""
        pass

class LinearNoiseScheduler(NoiseScheduler):
    """Linear noise schedule"""
    
    def __init__(self, beta_start: float, beta_end: float, num_timesteps: int):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt(self.betas)
    
    def get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.betas[t]
    
    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.alphas[t]
    
    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigmas[t]

class CosineNoiseScheduler(NoiseScheduler):
    """Cosine noise schedule"""
    
    def __init__(self, beta_start: float, beta_end: float, num_timesteps: int):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps
        
        # Cosine schedule
        t = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
        alpha_cumprod = torch.cos((t + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
        
        self.betas = 1 - alpha_cumprod[1:] / alpha_cumprod[:-1]
        self.betas = torch.clamp(self.betas, 0.0001, 0.9999)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt(self.betas)
    
    def get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.betas[t]
    
    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.alphas[t]
    
    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigmas[t]

class QuadraticNoiseScheduler(NoiseScheduler):
    """Quadratic noise schedule"""
    
    def __init__(self, beta_start: float, beta_end: float, num_timesteps: int):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps
        
        # Quadratic schedule
        t = torch.arange(num_timesteps, dtype=torch.float32) / num_timesteps
        self.betas = beta_start + (beta_end - beta_start) * (t ** 2)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt(self.betas)
    
    def get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.betas[t]
    
    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.alphas[t]
    
    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigmas[t]

class SigmoidNoiseScheduler(NoiseScheduler):
    """Sigmoid noise schedule"""
    
    def __init__(self, beta_start: float, beta_end: float, num_timesteps: int):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps
        
        # Sigmoid schedule
        t = torch.arange(num_timesteps, dtype=torch.float32) / num_timesteps
        self.betas = beta_start + (beta_end - beta_start) * torch.sigmoid(10 * (t - 0.5))
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt(self.betas)
    
    def get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.betas[t]
    
    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.alphas[t]
    
    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigmas[t]

class ExponentialNoiseScheduler(NoiseScheduler):
    """Exponential noise schedule"""
    
    def __init__(self, beta_start: float, beta_end: float, num_timesteps: int):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps
        
        # Exponential schedule
        t = torch.arange(num_timesteps, dtype=torch.float32) / num_timesteps
        self.betas = beta_start * (beta_end / beta_start) ** t
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt(self.betas)
    
    def get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.betas[t]
    
    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.alphas[t]
    
    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigmas[t]

class NoiseSchedulerFactory:
    """Factory for creating noise schedulers"""
    
    @staticmethod
    def create_scheduler(schedule_type: str, **kwargs) -> NoiseScheduler:
        """Create a noise scheduler based on type"""
        schedulers = {
            "linear": LinearNoiseScheduler,
            "cosine": CosineNoiseScheduler,
            "quadratic": QuadraticNoiseScheduler,
            "sigmoid": SigmoidNoiseScheduler,
            "exponential": ExponentialNoiseScheduler
        }
        
        if schedule_type not in schedulers:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        return schedulers[schedule_type](**kwargs)

class SamplingMethod(ABC):
    """Abstract base class for sampling methods"""
    
    @abstractmethod
    def sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, 
               scheduler: NoiseScheduler, **kwargs) -> torch.Tensor:
        """Sample from the model"""
        pass

class DDPMSampling(SamplingMethod):
    """DDPM sampling method"""
    
    def sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor,
               scheduler: NoiseScheduler, **kwargs) -> torch.Tensor:
        """DDPM sampling step"""
        # Predict noise
        predicted_noise = model(x_t, t)
        
        # Get scheduler values
        alpha_t = scheduler.get_alpha_t(t)
        beta_t = scheduler.get_beta_t(t)
        sigma_t = scheduler.get_sigma_t(t)
        
        # DDPM update rule
        x_prev = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_t)) * predicted_noise)
        
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise
        
        return x_prev

class DDIMSampling(SamplingMethod):
    """DDIM sampling method"""
    
    def __init__(self, eta: float = 0.0):
        self.eta = eta  # 0 = deterministic, 1 = stochastic
    
    def sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor,
               scheduler: NoiseScheduler, **kwargs) -> torch.Tensor:
        """DDIM sampling step"""
        # Predict noise
        predicted_noise = model(x_t, t)
        
        # Get scheduler values
        alpha_t = scheduler.get_alpha_t(t)
        alpha_t_prev = scheduler.get_alpha_t(t - 1) if t[0] > 0 else torch.ones_like(alpha_t)
        
        # DDIM update rule
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        c1 = self.eta * torch.sqrt((1 - alpha_t / alpha_t_prev) * (1 - alpha_t_prev) / (1 - alpha_t))
        c2 = torch.sqrt(1 - alpha_t_prev - c1 ** 2)
        
        x_prev = torch.sqrt(alpha_t_prev) * x_0_pred + c1 * torch.randn_like(x_t) + c2 * predicted_noise
        
        return x_prev

class AncestralSampling(SamplingMethod):
    """Ancestral sampling method"""
    
    def sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor,
               scheduler: NoiseScheduler, **kwargs) -> torch.Tensor:
        """Ancestral sampling step"""
        # Predict noise
        predicted_noise = model(x_t, t)
        
        # Get scheduler values
        alpha_t = scheduler.get_alpha_t(t)
        beta_t = scheduler.get_beta_t(t)
        
        # Ancestral update rule
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Add noise for stochasticity
        noise = torch.randn_like(x_t)
        x_prev = torch.sqrt(alpha_t) * x_0_pred + torch.sqrt(1 - alpha_t) * noise
        
        return x_prev

class EulerSampling(SamplingMethod):
    """Euler sampling method"""
    
    def sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor,
               scheduler: NoiseScheduler, **kwargs) -> torch.Tensor:
        """Euler sampling step"""
        # Predict noise
        predicted_noise = model(x_t, t)
        
        # Get scheduler values
        beta_t = scheduler.get_beta_t(t)
        
        # Euler update rule
        x_prev = x_t - 0.5 * beta_t * predicted_noise
        
        # Add noise for stochasticity
        noise = torch.randn_like(x_t)
        x_prev = x_prev + torch.sqrt(beta_t) * noise
        
        return x_prev

class HeunSampling(SamplingMethod):
    """Heun sampling method (2nd order)"""
    
    def sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor,
               scheduler: NoiseScheduler, **kwargs) -> torch.Tensor:
        """Heun sampling step"""
        # Predict noise
        predicted_noise = model(x_t, t)
        
        # Get scheduler values
        beta_t = scheduler.get_beta_t(t)
        
        # First step (Euler)
        x_euler = x_t - 0.5 * beta_t * predicted_noise
        
        # Second step (Heun)
        noise_euler = model(x_euler, t - 1) if t[0] > 0 else predicted_noise
        x_prev = x_t - 0.5 * beta_t * (predicted_noise + noise_euler)
        
        # Add noise for stochasticity
        noise = torch.randn_like(x_t)
        x_prev = x_prev + torch.sqrt(beta_t) * noise
        
        return x_prev

class SamplingMethodFactory:
    """Factory for creating sampling methods"""
    
    @staticmethod
    def create_sampler(method: str, **kwargs) -> SamplingMethod:
        """Create a sampling method based on type"""
        samplers = {
            "ddpm": DDPMSampling,
            "ddim": DDIMSampling,
            "ancestral": AncestralSampling,
            "euler": EulerSampling,
            "heun": HeunSampling
        }
        
        if method not in samplers:
            raise ValueError(f"Unknown sampling method: {method}")
        
        return samplers[method](**kwargs)

class DiffusionUNet(nn.Module):
    """Enhanced UNet for diffusion models with advanced features"""
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(config.model_channels, config.model_channels * 4),
            nn.SiLU(),
            nn.Linear(config.model_channels * 4, config.model_channels)
        )
        
        # Input projection
        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(
            nn.Conv2d(config.in_channels, config.model_channels, kernel_size=3, padding=1)
        )
        
        # Downsampling blocks
        input_block_chans = [config.model_channels]
        ch = config.model_channels
        ds = 1
        
        for level, mult in enumerate(config.channel_mult):
            for _ in range(config.num_res_blocks):
                layers = [
                    ResBlock(ch, config.model_channels, dropout=config.dropout, time_emb_dim=config.model_channels)
                ]
                ch = config.model_channels * mult
                
                if ds in config.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=config.num_heads))
                
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != len(config.channel_mult) - 1:
                self.input_blocks.append(
                    nn.ModuleList([Downsample(ch)])
                )
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle blocks
        self.middle_block = nn.ModuleList([
            ResBlock(ch, config.model_channels, dropout=config.dropout, time_emb_dim=config.model_channels),
            AttentionBlock(ch, num_heads=config.num_heads),
            ResBlock(ch, config.model_channels, dropout=config.dropout, time_emb_dim=config.model_channels)
        ])
        
        # Upsampling blocks
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(config.channel_mult))[::-1]:
            for i in range(config.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, config.model_channels * mult, dropout=config.dropout, time_emb_dim=config.model_channels)
                ]
                ch = config.model_channels * mult
                
                if ds in config.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=config.num_heads))
                
                if level and i == config.num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                
                self.output_blocks.append(nn.ModuleList(layers))
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, config.out_channels, kernel_size=3, padding=1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Time embedding
        t_emb = timestep_embedding(timesteps, self.config.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # Input processing
        h = x
        hs = []
        
        # Downsampling
        for module in self.input_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle
        for module in self.middle_block:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
        
        # Upsampling
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
        
        # Output
        return self.out(h)

class ResBlock(nn.Module):
    """Residual block with time embedding"""
    
    def __init__(self, channels: int, out_channels: int, dropout: float, time_emb_dim: int):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.dropout = dropout
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if self.channels != self.out_channels:
            self.shortcut = nn.Conv2d(channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """Attention block with multi-head self-attention"""
    
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        scale = 1 / math.sqrt(math.sqrt(C))
        
        attn = torch.einsum("bchw,bcij->bhwij", q * scale, k * scale)
        attn = torch.softmax(attn, dim=-1)
        
        h = torch.einsum("bhwij,bcij->bchw", attn, v)
        h = self.proj(h)
        
        return x + h

class Downsample(nn.Module):
    """Downsampling layer"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    """Upsampling layer"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DiffusionTrainer:
    """Enhanced trainer for diffusion models with advanced features"""
    
    def __init__(self, model: nn.Module, config: DiffusionConfig, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize components
        self.scheduler = NoiseSchedulerFactory.create_scheduler(
            config.schedule_type,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            num_timesteps=config.num_diffusion_timesteps
        )
        
        self.sampler = SamplingMethodFactory.create_sampler(config.sampling_method)
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # EMA
        if config.use_ema:
            self.ema_model = self._create_ema_model()
        
        # Gradient checkpointing
        if config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # XFormers optimization
        if config.use_xformers and hasattr(torch.backends, 'xformers'):
            try:
                torch.backends.xformers.enable()
                logger.info("XFormers optimization enabled")
            except Exception as e:
                logger.warning(f"Failed to enable XFormers: {e}")
    
    def _create_ema_model(self) -> nn.Module:
        """Create EMA model"""
        ema_model = type(self.model)(self.config)
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    def _update_ema(self):
        """Update EMA model"""
        if hasattr(self, 'ema_model'):
            with torch.no_grad():
                for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                    ema_param.data.mul_(self.config.ema_decay).add_(
                        param.data, alpha=1 - self.config.ema_decay
                    )
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        batch = batch.to(self.device)
        batch_size = batch.shape[0]
        
        # Sample timesteps
        t = torch.randint(0, self.config.num_diffusion_timesteps, (batch_size,), device=self.device)
        
        # Add noise
        noise = torch.randn_like(batch)
        alpha_t = self.scheduler.get_alpha_t(t)
        x_t = torch.sqrt(alpha_t)[:, None, None, None] * batch + torch.sqrt(1 - alpha_t)[:, None, None, None] * noise
        
        # Predict noise
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            predicted_noise = self.model(x_t, t)
            loss = F.mse_loss(predicted_noise, noise)
        
        # Backward pass
        if self.config.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_val > 0:
            if self.config.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
        
        # Optimizer step
        if self.config.use_mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Update EMA
        self._update_ema()
        
        return {"loss": loss.item()}
    
    def sample(self, batch_size: int = 1, num_steps: Optional[int] = None) -> torch.Tensor:
        """Generate samples using the trained model"""
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        
        # Use EMA model if available
        model = self.ema_model if hasattr(self, 'ema_model') else self.model
        model.eval()
        
        with torch.no_grad():
            # Start from random noise
            x = torch.randn(batch_size, self.config.in_channels, 64, 64, device=self.device)
            
            # Sampling loop
            for i in tqdm(range(num_steps), desc="Sampling"):
                t = torch.full((batch_size,), num_steps - i - 1, device=self.device, dtype=torch.long)
                x = self.sampler.sample(model, x, t, self.scheduler)
            
            return x
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "config": self.config,
            "scheduler": self.scheduler,
            "sampler": self.sampler
        }
        
        if hasattr(self, 'ema_model'):
            checkpoint["ema_model_state_dict"] = self.ema_model.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        
        if "ema_model_state_dict" in checkpoint and hasattr(self, 'ema_model'):
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        
        logger.info(f"Checkpoint loaded from {path}")

class DiffusionPipeline:
    """Production-ready diffusion pipeline with advanced features"""
    
    def __init__(self, config: DiffusionConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Initialize model
        self.model = DiffusionUNet(config).to(device)
        
        # Initialize trainer
        self.trainer = DiffusionTrainer(self.model, config, device)
        
        # Diffusers integration
        if DIFFUSERS_AVAILABLE:
            self._setup_diffusers_pipeline()
    
    def _setup_diffusers_pipeline(self):
        """Setup Diffusers pipeline if available"""
        try:
            # Create scheduler
            if self.config.schedule_type == "linear":
                self.diffusers_scheduler = DDPMScheduler(
                    num_train_timesteps=self.config.num_diffusion_timesteps,
                    beta_start=self.config.beta_start,
                    beta_end=self.config.beta_end
                )
            elif self.config.schedule_type == "cosine":
                self.diffusers_scheduler = DDPMScheduler(
                    num_train_timesteps=self.config.num_diffusion_timesteps,
                    beta_schedule="cosine"
                )
            
            # Create pipeline
            self.diffusers_pipeline = DiffusionPipeline(
                unet=self.model,
                scheduler=self.diffusers_scheduler
            ).to(self.device)
            
            logger.info("Diffusers pipeline initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Diffusers pipeline: {e}")
    
    def train(self, dataloader: DataLoader, num_epochs: Optional[int] = None) -> List[Dict[str, float]]:
        """Train the diffusion model"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        self.model.train()
        training_history = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                loss_dict = self.trainer.train_step(batch)
                epoch_losses.append(loss_dict["loss"])
            
            # Update learning rate
            self.trainer.lr_scheduler.step()
            
            # Log epoch results
            avg_loss = np.mean(epoch_losses)
            training_history.append({"epoch": epoch + 1, "loss": avg_loss})
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
                self.trainer.save_checkpoint(checkpoint_path)
        
        return training_history
    
    def generate(self, prompt: Optional[str] = None, batch_size: int = 1, 
                 num_steps: Optional[int] = None) -> torch.Tensor:
        """Generate samples"""
        return self.trainer.sample(batch_size, num_steps)
    
    def save_model(self, path: str):
        """Save the complete model"""
        self.trainer.save_checkpoint(path)
    
    def load_model(self, path: str):
        """Load the complete model"""
        self.trainer.load_checkpoint(path)

def create_diffusion_pipeline(config_path: str, device: str = "cuda") -> DiffusionPipeline:
    """Create a diffusion pipeline from config file"""
    config = DiffusionConfig.from_yaml(config_path)
    return DiffusionPipeline(config, device)

def main():
    """Main function for testing the diffusion system"""
    # Create configuration
    config = DiffusionConfig(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        num_heads=8,
        schedule_type="cosine",
        sampling_method="ddim",
        learning_rate=1e-4,
        batch_size=16,
        num_epochs=100,
        use_mixed_precision=True,
        use_ema=True
    )
    
    # Save config
    config.save_yaml("diffusion_config.yaml")
    
    # Create pipeline
    pipeline = DiffusionPipeline(config)
    
    # Create dummy dataset
    dummy_data = torch.randn(100, 3, 64, 64)
    dataloader = DataLoader(dummy_data, batch_size=16, shuffle=True)
    
    # Train model
    print("Starting training...")
    history = pipeline.train(dataloader, num_epochs=5)
    
    # Generate samples
    print("Generating samples...")
    samples = pipeline.generate(batch_size=4, num_steps=20)
    
    print(f"Generated {len(samples)} samples with shape {samples.shape}")
    print("Training history:", history)

if __name__ == "__main__":
    main()



