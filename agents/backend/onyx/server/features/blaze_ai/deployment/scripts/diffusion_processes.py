#!/usr/bin/env python3
"""
Forward and Reverse Diffusion Processes for Blaze AI
Understanding and implementing the mathematical foundations of diffusion models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from tqdm import tqdm
import warnings
import math

# Diffusers imports
try:
    from diffusers import (
        DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler,
        DPMSolverMultistepScheduler, UNet2DConditionModel
    )
    from diffusers.utils import randn_tensor
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    warnings.warn("Diffusers library not available. Install with: pip install diffusers")

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DiffusionConfig:
    """Configuration for diffusion processes"""
    # Diffusion parameters
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # Model parameters
    image_size: int = 32
    in_channels: int = 3
    hidden_size: int = 128
    num_blocks: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 100
    
    # Sampling parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5


class DiffusionMathematics:
    """Mathematical foundations of diffusion processes"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.betas = self._create_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for efficiency
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # For reverse process
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas_cumprod) / (1.0 - self.alphas_cumprod)
        )
    
    def _create_beta_schedule(self) -> torch.Tensor:
        """Create beta schedule for diffusion process"""
        # Default to linear schedule
        return torch.linspace(self.config.beta_start, self.config.beta_end, self.config.num_timesteps)
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine beta schedule as proposed in Improved DDPM"""
        steps = self.config.num_timesteps + 1
        x = torch.linspace(0, self.config.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.config.num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _sigmoid_beta_schedule(self) -> torch.Tensor:
        """Sigmoid beta schedule"""
        steps = self.config.num_timesteps + 1
        x = torch.linspace(-6, 6, steps)
        alphas_cumprod = torch.sigmoid(x)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_start: Original image [B, C, H, W]
            t: Timestep [B]
            noise: Optional noise tensor
            
        Returns:
            x_t: Noisy image at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get alpha values for timestep t
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Forward process: x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the posterior mean and variance of q(x_{t-1} | x_t, x_0)
        
        Args:
            x_start: Original image
            x_t: Noisy image at timestep t
            t: Timestep
            
        Returns:
            posterior_mean: Mean of posterior distribution
            posterior_variance: Variance of posterior distribution
            posterior_log_variance_clipped: Log variance (clipped)
        """
        posterior_mean = (
            self.posterior_mean_coef1[t].view(-1, 1, 1, 1) * x_start +
            self.posterior_mean_coef2[t].view(-1, 1, 1, 1) * x_t
        )
        
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1)
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, 
                       clip_denoised: bool = True, return_codebook_ids: bool = False,
                       quantize_denoised: bool = False) -> Dict[str, torch.Tensor]:
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of x_0
        
        Args:
            model: The model to predict noise
            x: Input tensor [B, C, H, W]
            t: Timestep [B]
            clip_denoised: Whether to clip denoised output
            return_codebook_ids: Whether to return codebook IDs
            quantize_denoised: Whether to quantize denoised output
            
        Returns:
            Dictionary containing predicted mean, variance, and x_0
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)
        
        # Predict noise using the model
        model_output = model(x, t)
        
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None
        
        # Extract predicted noise
        if model_output.shape[1] == C:
            pred_noise = model_output
        else:
            pred_noise = model_output[:, :C]
        
        # Predict x_0 from x_t and predicted noise
        # x_0 = (x_t - sqrt(1 - α_t) * ε_pred) / sqrt(α_t)
        pred_x_start = self._predict_xstart_from_eps(x, t, pred_noise)
        
        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, -1, 1)
        
        # Compute posterior mean and variance
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            x_start=pred_x_start, x_t=x, t=t
        )
        
        if not return_codebook_ids:
            posterior_log_variance = posterior_log_variance.expand(model_mean.shape)
        
        return {
            "mean": model_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_x_start": pred_x_start,
            "extra": extra
        }
    
    def _predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise"""
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps
    
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, 
                clip_denoised: bool = True, return_codebook_ids: bool = False,
                quantize_denoised: bool = False, temperature: float = 1.0,
                noise_dropout: float = 0.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample from p(x_{t-1} | x_t) using the reverse process
        
        Args:
            model: The model to predict noise
            x: Input tensor [B, C, H, W]
            t: Timestep [B]
            clip_denoised: Whether to clip denoised output
            return_codebook_ids: Whether to return codebook IDs
            quantize_denoised: Whether to quantize denoised output
            temperature: Temperature for sampling
            noise_dropout: Dropout for noise
            
        Returns:
            Sampled x_{t-1} or tuple with codebook IDs
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)
        
        # Get model predictions
        model_output = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised, return_codebook_ids=return_codebook_ids,
            quantize_denoised=quantize_denoised
        )
        
        if return_codebook_ids:
            model_mean, model_log_variance, pred_x_start, extra = (
                model_output["mean"], model_output["log_variance"], 
                model_output["pred_x_start"], model_output["extra"]
            )
        else:
            model_mean, model_log_variance, pred_x_start = (
                model_output["mean"], model_output["log_variance"], 
                model_output["pred_x_start"]
            )
            extra = None
        
        # Sample from the posterior
        noise = torch.randn_like(x)
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        
        # Add temperature scaling
        noise = noise * temperature
        
        # Sample: x_{t-1} = μ + σ * ε
        # where μ is the predicted mean and σ is sqrt(variance)
        model_std = torch.exp(0.5 * model_log_variance)
        sample = model_mean + model_std * noise
        
        if return_codebook_ids:
            return sample, extra
        else:
            return sample
    
    def p_sample_loop(self, model: nn.Module, shape: Tuple[int, ...], 
                     noise: Optional[torch.Tensor] = None, clip_denoised: bool = True,
                     return_codebook_ids: bool = False, quantize_denoised: bool = False,
                     temperature: float = 1.0, noise_dropout: float = 0.0,
                     progress: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate samples by sampling from the model
        
        Args:
            model: The model to sample from
            shape: Shape of samples to generate
            noise: Initial noise
            clip_denoised: Whether to clip denoised output
            return_codebook_ids: Whether to return codebook IDs
            quantize_denoised: Whether to quantize denoised output
            temperature: Temperature for sampling
            noise_dropout: Dropout for noise
            progress: Whether to show progress bar
            
        Returns:
            Generated samples or tuple with codebook IDs
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model, shape, noise=noise, clip_denoised=clip_denoised,
            return_codebook_ids=return_codebook_ids, quantize_denoised=quantize_denoised,
            temperature=temperature, noise_dropout=noise_dropout, progress=progress
        ):
            final = sample
        
        if return_codebook_ids:
            return final[0], final[1]
        else:
            return final
    
    def p_sample_loop_progressive(self, model: nn.Module, shape: Tuple[int, ...],
                                noise: Optional[torch.Tensor] = None, clip_denoised: bool = True,
                                return_codebook_ids: bool = False, quantize_denoised: bool = False,
                                temperature: float = 1.0, noise_dropout: float = 0.0,
                                progress: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate samples progressively, yielding intermediate results
        
        Args:
            model: The model to sample from
            shape: Shape of samples to generate
            noise: Initial noise
            clip_denoised: Whether to clip denoised output
            return_codebook_ids: Whether to return codebook IDs
            quantize_denoised: Whether to quantize denoised output
            temperature: Temperature for sampling
            noise_dropout: Dropout for noise
            progress: Whether to show progress bar
            
        Yields:
            Intermediate samples
        """
        if noise is None:
            noise = torch.randn(*shape, device=next(model.parameters()).device)
        
        x = noise
        
        timesteps = list(range(self.config.num_timesteps))[::-1]
        if progress:
            timesteps = tqdm(timesteps, desc="Sampling")
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=x.device, dtype=torch.long)
            
            x = self.p_sample(
                model, x, t_batch, clip_denoised=clip_denoised,
                return_codebook_ids=return_codebook_ids, quantize_denoised=quantize_denoised,
                temperature=temperature, noise_dropout=noise_dropout
            )
            
            if return_codebook_ids:
                yield x
            else:
                yield x


class SimpleUNet(nn.Module):
    """Simple UNet for diffusion model"""
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        
        # Time embedding
        time_dim = config.hidden_size * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(config.hidden_size),
            nn.Linear(config.hidden_size, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(config.in_channels, config.hidden_size, kernel_size=3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        in_channels = config.hidden_size
        
        for i in range(config.num_blocks):
            out_channels = in_channels * 2
            self.down_blocks.append(nn.ModuleList([
                ResNetBlock(in_channels, in_channels, time_dim, config.dropout),
                ResNetBlock(in_channels, in_channels, time_dim, config.dropout),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            ]))
            in_channels = out_channels
        
        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            ResNetBlock(in_channels, in_channels, time_dim, config.dropout),
            ResNetBlock(in_channels, in_channels, time_dim, config.dropout)
        ])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        for i in range(config.num_blocks):
            out_channels = in_channels // 2
            self.up_blocks.append(nn.ModuleList([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                ResNetBlock(in_channels, in_channels, time_dim, config.dropout),
                ResNetBlock(out_channels, out_channels, time_dim, config.dropout)
            ]))
            in_channels = out_channels
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, config.in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_mlp(timesteps)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Downsampling
        down_features = []
        for down_block in self.down_blocks:
            # ResNet blocks
            x = down_block[0](x, time_emb)
            x = down_block[1](x, time_emb)
            
            # Store features for skip connections
            down_features.append(x)
            
            # Downsample
            x = down_block[2](x)
        
        # Middle blocks
        for middle_block in self.middle_blocks:
            x = middle_block(x, time_emb)
        
        # Upsampling with skip connections
        for i, up_block in enumerate(self.up_blocks):
            # Upsample
            x = up_block[0](x)
            
            # Concatenate with skip connection
            skip_feature = down_features[-(i + 1)]
            x = torch.cat([x, skip_feature], dim=1)
            
            # ResNet blocks
            x = up_block[1](x, time_emb)
            x = up_block[2](x, time_emb)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timesteps"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResNetBlock(nn.Module):
    """ResNet block with time embedding"""
    
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
        # Convolutional layers
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        
        # First block
        x = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        x = x + time_emb.unsqueeze(-1).unsqueeze(-1)
        
        # Second block
        x = self.block2(x)
        
        # Residual connection
        x = x + residual
        
        return x


class DiffusionTrainer:
    """Trainer for diffusion models"""
    
    def __init__(self, model: nn.Module, diffusion: DiffusionMathematics, config: DiffusionConfig):
        self.model = model
        self.diffusion = diffusion
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
    
    def train_step(self, x_start: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.config.num_timesteps, (batch_size,), device=x_start.device)
        
        # Add noise to images (forward process)
        noise = torch.randn_like(x_start)
        x_noisy, noise_target = self.diffusion.q_sample(x_start, t, noise)
        
        # Predict noise (reverse process)
        noise_pred = self.model(x_noisy, t)
        
        # Compute loss
        loss = self.criterion(noise_pred, noise_target)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x_start = batch[0]
            else:
                x_start = batch
            
            # Move to device
            if hasattr(self, 'device'):
                x_start = x_start.to(self.device)
            
            # Training step
            step_metrics = self.train_step(x_start)
            total_loss += step_metrics["loss"]
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {"train_loss": avg_loss}
    
    def sample(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """Generate samples using the trained model"""
        self.model.eval()
        
        with torch.no_grad():
            # Generate noise
            shape = (batch_size, self.config.in_channels, self.config.image_size, self.config.image_size)
            samples = self.diffusion.p_sample_loop(
                self.model, shape, progress=True
            )
        
        return samples


class DiffusersIntegration:
    """Integration with Hugging Face Diffusers library"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.schedulers = {}
        
        if DIFFUSERS_AVAILABLE:
            self._setup_schedulers()
    
    def _setup_schedulers(self):
        """Setup different diffusion schedulers"""
        # DDPM Scheduler
        self.schedulers["ddpm"] = DDPMScheduler(
            num_train_timesteps=self.config.num_timesteps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            beta_schedule="linear"
        )
        
        # DDIM Scheduler
        self.schedulers["ddim"] = DDIMScheduler(
            num_train_timesteps=self.config.num_timesteps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            beta_schedule="linear",
            clip_sample=False,
            set_alpha_to_one=False
        )
        
        # Euler Scheduler
        self.schedulers["euler"] = EulerDiscreteScheduler(
            num_train_timesteps=self.config.num_timesteps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            beta_schedule="linear"
        )
        
        # DPM-Solver Scheduler
        self.schedulers["dpm"] = DPMSolverMultistepScheduler(
            num_train_timesteps=self.config.num_timesteps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            beta_schedule="linear",
            algorithm_type="dpmsolver++",
            solver_type="midpoint"
        )
    
    def demonstrate_forward_process(self, x_start: torch.Tensor, num_steps: int = 10) -> List[torch.Tensor]:
        """Demonstrate forward diffusion process using Diffusers"""
        if not DIFFUSERS_AVAILABLE:
            logger.warning("Diffusers library not available")
            return []
        
        logger.info("Demonstrating forward diffusion process with Diffusers...")
        
        # Use DDPM scheduler for demonstration
        scheduler = self.schedulers["ddpm"]
        
        # Add noise step by step
        x = x_start
        noisy_images = [x.clone()]
        
        for i in range(num_steps):
            # Get noise for current timestep
            noise = torch.randn_like(x)
            timesteps = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            
            # Add noise using scheduler
            x = scheduler.add_noise(x, noise, timesteps)
            noisy_images.append(x.clone())
            
            logger.info(f"Step {i+1}: Added noise to image")
        
        return noisy_images
    
    def demonstrate_reverse_process(self, x_noisy: torch.Tensor, num_steps: int = 10) -> List[torch.Tensor]:
        """Demonstrate reverse diffusion process using Diffusers"""
        if not DIFFUSERS_AVAILABLE:
            logger.warning("Diffusers library not available")
            return []
        
        logger.info("Demonstrating reverse diffusion process with Diffusers...")
        
        # Use DDIM scheduler for demonstration (faster than DDPM)
        scheduler = self.schedulers["ddim"]
        
        # Set timesteps
        scheduler.set_timesteps(num_steps)
        
        # Reverse process
        x = x_noisy
        denoised_images = [x.clone()]
        
        for i, t in enumerate(scheduler.timesteps):
            # Predict noise (in practice, this would come from a model)
            # For demonstration, we'll use random noise
            noise_pred = torch.randn_like(x)
            
            # Denoise step
            x = scheduler.step(noise_pred, t, x).prev_sample
            denoised_images.append(x.clone())
            
            logger.info(f"Step {i+1}: Denoised image")
        
        return denoised_images


class DiffusionVisualization:
    """Visualization tools for diffusion processes"""
    
    @staticmethod
    def plot_noise_schedule(diffusion: DiffusionMathematics, save_path: Optional[str] = None):
        """Plot the noise schedule (betas, alphas, etc.)"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Beta schedule
        axes[0, 0].plot(diffusion.betas.cpu().numpy())
        axes[0, 0].set_title("Beta Schedule")
        axes[0, 0].set_xlabel("Timestep")
        axes[0, 0].set_ylabel("Beta")
        axes[0, 0].grid(True)
        
        # Alpha schedule
        axes[0, 1].plot(diffusion.alphas.cpu().numpy())
        axes[0, 1].set_title("Alpha Schedule")
        axes[0, 1].set_xlabel("Timestep")
        axes[0, 1].set_ylabel("Alpha")
        axes[0, 1].grid(True)
        
        # Cumulative alpha
        axes[1, 0].plot(diffusion.alphas_cumprod.cpu().numpy())
        axes[1, 0].set_title("Cumulative Alpha")
        axes[1, 0].set_xlabel("Timestep")
        axes[1, 0].set_ylabel("Cumulative Alpha")
        axes[1, 0].grid(True)
        
        # Posterior variance
        axes[1, 1].plot(diffusion.posterior_variance.cpu().numpy())
        axes[1, 1].set_title("Posterior Variance")
        axes[1, 1].set_xlabel("Timestep")
        axes[1, 1].set_ylabel("Variance")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_diffusion_process(images: List[torch.Tensor], title: str, save_path: Optional[str] = None):
        """Plot a sequence of images showing diffusion process"""
        if not images:
            return
        
        # Convert to numpy and normalize
        images_np = []
        for img in images:
            img_np = img.detach().cpu().numpy()
            if img_np.shape[0] == 1:  # Single image
                img_np = img_np[0]
            else:  # Batch of images
                img_np = img_np[0]  # Take first image
            
            # Normalize to [0, 1]
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            images_np.append(img_np)
        
        # Create grid
        n_images = len(images_np)
        cols = min(5, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, img in enumerate(images_np):
            row = i // cols
            col = i % cols
            
            if img.shape[0] == 3:  # RGB
                axes[row, col].imshow(np.transpose(img, (1, 2, 0)))
            else:  # Grayscale
                axes[row, col].imshow(img[0], cmap='gray')
            
            axes[row, col].set_title(f"Step {i}")
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(n_images, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def demonstrate_diffusion_processes():
    """Demonstrate forward and reverse diffusion processes"""
    logger.info("Starting Diffusion Processes Demonstration...")
    
    # Configuration
    config = DiffusionConfig(
        num_timesteps=100,
        image_size=32,
        in_channels=3,
        hidden_size=64,
        num_blocks=3
    )
    
    # Create diffusion mathematics
    logger.info("Creating diffusion mathematics...")
    diffusion = DiffusionMathematics(config)
    
    # Create model
    logger.info("Creating UNet model...")
    model = SimpleUNet(config)
    
    # Create trainer
    logger.info("Creating diffusion trainer...")
    trainer = DiffusionTrainer(model, diffusion, config)
    
    # Create diffusers integration
    logger.info("Setting up Diffusers integration...")
    diffusers_integration = DiffusersIntegration(config)
    
    # Create visualization tools
    logger.info("Setting up visualization tools...")
    viz = DiffusionVisualization()
    
    # Demonstrate noise schedule
    logger.info("Plotting noise schedule...")
    viz.plot_noise_schedule(diffusion, save_path="./noise_schedule.png")
    
    # Create sample data
    logger.info("Creating sample data...")
    batch_size = 4
    x_start = torch.randn(batch_size, config.in_channels, config.image_size, config.image_size)
    
    # Demonstrate forward process (custom implementation)
    logger.info("Demonstrating forward diffusion process (custom)...")
    t = torch.randint(0, config.num_timesteps, (batch_size,))
    x_t, noise = diffusion.q_sample(x_start, t)
    
    logger.info(f"Original image shape: {x_start.shape}")
    logger.info(f"Noisy image shape: {x_t.shape}")
    logger.info(f"Added noise shape: {noise.shape}")
    
    # Demonstrate forward process (Diffusers)
    logger.info("Demonstrating forward diffusion process (Diffusers)...")
    noisy_images_diffusers = diffusers_integration.demonstrate_forward_process(x_start, num_steps=10)
    
    # Visualize forward process
    if noisy_images_diffusers:
        viz.plot_diffusion_process(
            noisy_images_diffusers, 
            "Forward Diffusion Process (Diffusers)",
            save_path="./forward_diffusion_diffusers.png"
        )
    
    # Demonstrate reverse process (Diffusers)
    logger.info("Demonstrating reverse diffusion process (Diffusers)...")
    denoised_images_diffusers = diffusers_integration.demonstrate_reverse_process(
        noisy_images_diffusers[-1] if noisy_images_diffusers else x_t, 
        num_steps=10
    )
    
    # Visualize reverse process
    if denoised_images_diffusers:
        viz.plot_diffusion_process(
            denoised_images_diffusers, 
            "Reverse Diffusion Process (Diffusers)",
            save_path="./reverse_diffusion_diffusers.png"
        )
    
    # Demonstrate sampling (custom implementation)
    logger.info("Demonstrating sampling with custom implementation...")
    try:
        samples = diffusion.p_sample_loop(
            model, 
            (2, config.in_channels, config.image_size, config.image_size),
            progress=True
        )
        logger.info(f"Generated samples shape: {samples.shape}")
        
        # Visualize samples
        viz.plot_diffusion_process(
            [samples], 
            "Generated Samples (Custom)",
            save_path="./generated_samples_custom.png"
        )
    except Exception as e:
        logger.warning(f"Custom sampling failed: {e}")
    
    # Summary
    logger.info("Diffusion Processes Summary:")
    logger.info(f"✓ Forward process (custom): {x_start.shape} -> {x_t.shape}")
    logger.info(f"✓ Forward process (Diffusers): {len(noisy_images_diffusers)} steps")
    logger.info(f"✓ Reverse process (Diffusers): {len(denoised_images_diffusers)} steps")
    logger.info(f"✓ Noise schedule visualization created")
    logger.info(f"✓ Custom sampling: {'✓' if 'samples' in locals() else '✗'}")
    
    logger.info("Diffusion Processes demonstration completed successfully!")


def main():
    """Main execution function"""
    logger.info("Starting Diffusion Processes Implementation...")
    
    # Demonstrate diffusion processes
    demonstrate_diffusion_processes()
    
    logger.info("All demonstrations completed!")


if __name__ == "__main__":
    main()
