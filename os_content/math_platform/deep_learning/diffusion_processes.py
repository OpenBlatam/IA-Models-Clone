from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import gc
from pathlib import Path
from tqdm import tqdm
import warnings
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Forward and Reverse Diffusion Processes
Production-ready implementation of diffusion processes with proper PyTorch implementations.
"""


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class DiffusionConfig:
    """Configuration for diffusion processes."""
    # Diffusion parameters
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule: str = "linear"  # linear, cosine, sigmoid, cosine_beta
    
    # Model parameters
    image_size: int = 256
    in_channels: int = 3
    model_channels: int = 128
    out_channels: int = 3
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Mixed precision and optimization
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    
    # Generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    clip_denoised: bool = True
    
    # Output configuration
    output_dir: str = "./diffusion_outputs"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


class DiffusionScheduler:
    """Advanced diffusion scheduler with multiple noise schedules."""
    
    def __init__(self, config: DiffusionConfig):
        
    """__init__ function."""
self.config = config
        self.num_timesteps = config.num_timesteps
        
        # Setup noise schedule
        self.betas = self._get_beta_schedule()
        
        # Precompute values for efficiency
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Variance for reverse process
        self.variance = self._get_variance()
        
        # Log SNR for classifier-free guidance
        self.log_snr = torch.log(self.alphas_cumprod / (1 - self.alphas_cumprod))
        
        logger.info(f"Diffusion scheduler initialized with {self.num_timesteps} timesteps")
        logger.info(f"Schedule: {config.schedule}")
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """Get beta schedule based on configuration."""
        if self.config.schedule == "linear":
            return torch.linspace(self.config.beta_start, self.config.beta_end, self.num_timesteps)
        elif self.config.schedule == "cosine":
            return self._cosine_beta_schedule()
        elif self.config.schedule == "sigmoid":
            return self._sigmoid_beta_schedule()
        elif self.config.schedule == "cosine_beta":
            return self._cosine_beta_schedule_v2()
        else:
            raise ValueError(f"Unknown schedule: {self.config.schedule}")
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine beta schedule as described in 'Improved Denoising Diffusion Probabilistic Models'."""
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def _sigmoid_beta_schedule(self) -> torch.Tensor:
        """Sigmoid beta schedule."""
        betas = torch.sigmoid(torch.linspace(-6, 6, self.num_timesteps))
        betas = betas * (self.config.beta_end - self.config.beta_start) + self.config.beta_start
        return betas
    
    def _cosine_beta_schedule_v2(self) -> torch.Tensor:
        """Cosine beta schedule v2 with better properties."""
        max_beta = 0.999
        alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi * 0.5) ** 2
        betas = []
        for i in range(self.num_timesteps):
            t1 = i / self.num_timesteps
            t2 = (i + 1) / self.num_timesteps
            beta = min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta)
            betas.append(beta)
        return torch.tensor(betas)
    
    def _get_variance(self) -> torch.Tensor:
        """Get variance for reverse process."""
        # DDPM variance
        variance = (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * self.betas
        
        # Clip variance for numerical stability
        variance = torch.clamp(variance, min=0.0, max=1.0)
        
        return variance
    
    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process: add noise to samples."""
        # Get values for the given timesteps
        sqrt_alpha_cumprod = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        # Reshape for broadcasting
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1, 1)
        
        # Generate noise
        noise = torch.randn_like(original_samples)
        
        # Add noise according to forward process
        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        
        return noisy_samples, noise
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, 
             eta: float = 0.0, use_clipped_model_output: bool = True) -> torch.Tensor:
        """Single reverse diffusion step."""
        # Get values for this timestep
        alpha_cumprod = self.alphas_cumprod[timestep]
        alpha_cumprod_prev = self.alphas_cumprod_prev[timestep]
        beta = self.betas[timestep]
        variance = self.variance[timestep]
        
        # Clip model output if requested
        if use_clipped_model_output:
            model_output = torch.clamp(model_output, -1, 1)
        
        # Predict original sample
        pred_original_sample = (sample - beta.sqrt() * model_output) / alpha_cumprod.sqrt()
        
        # Clip predicted original sample
        if self.config.clip_denoised:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Predict previous sample mean
        pred_sample_direction = (1 - alpha_cumprod_prev).sqrt() * model_output
        
        # Add noise for stochastic sampling
        if eta > 0:
            noise = torch.randn_like(sample)
            pred_prev_sample = alpha_cumprod_prev.sqrt() * pred_original_sample + pred_sample_direction + eta * variance.sqrt() * noise
        else:
            pred_prev_sample = alpha_cumprod_prev.sqrt() * pred_original_sample + pred_sample_direction
        
        return pred_prev_sample
    
    def step_ddim(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, 
                  eta: float = 0.0) -> torch.Tensor:
        """DDIM reverse diffusion step."""
        # Get values for this timestep
        alpha_cumprod = self.alphas_cumprod[timestep]
        alpha_cumprod_prev = self.alphas_cumprod_prev[timestep]
        
        # Predict original sample
        pred_original_sample = (sample - (1 - alpha_cumprod).sqrt() * model_output) / alpha_cumprod.sqrt()
        
        # Clip predicted original sample
        if self.config.clip_denoised:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Predict previous sample
        pred_sample_direction = (1 - alpha_cumprod_prev).sqrt() * model_output
        
        # Add noise for stochastic sampling
        if eta > 0:
            noise = torch.randn_like(sample)
            pred_prev_sample = alpha_cumprod_prev.sqrt() * pred_original_sample + pred_sample_direction + eta * (1 - alpha_cumprod_prev).sqrt() * noise
        else:
            pred_prev_sample = alpha_cumprod_prev.sqrt() * pred_original_sample + pred_sample_direction
        
        return pred_prev_sample
    
    def get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """Get sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ForwardDiffusionProcess:
    """Forward diffusion process implementation."""
    
    def __init__(self, scheduler: DiffusionScheduler):
        
    """__init__ function."""
self.scheduler = scheduler
    
    def __call__(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion process."""
        return self.scheduler.add_noise(original_samples, timesteps)
    
    def get_training_target(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Get training target for the model (noise prediction)."""
        noisy_samples, noise = self.scheduler.add_noise(original_samples, timesteps)
        return noise
    
    def get_conditioning(self, timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """Get timestep conditioning for the model."""
        return self.scheduler.get_timestep_embedding(timesteps, embedding_dim)


class ReverseDiffusionProcess:
    """Reverse diffusion process implementation."""
    
    def __init__(self, scheduler: DiffusionScheduler, use_ddim: bool = False):
        
    """__init__ function."""
self.scheduler = scheduler
        self.use_ddim = use_ddim
    
    def __call__(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, 
                 eta: float = 0.0) -> torch.Tensor:
        """Apply reverse diffusion step."""
        if self.use_ddim:
            return self.scheduler.step_ddim(model_output, timestep, sample, eta)
        else:
            return self.scheduler.step(model_output, timestep, sample, eta)
    
    def sample(self, model: nn.Module, shape: Tuple[int, ...], num_inference_steps: int,
               guidance_scale: float = 1.0, eta: float = 0.0, 
               timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from the reverse diffusion process."""
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Set timesteps
        if timesteps is None:
            timesteps = torch.linspace(self.scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        # Reverse diffusion loop
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
                # Prepare timestep
                timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                # Get model prediction
                model_output = model(x, timestep)
                
                # Apply classifier-free guidance if scale > 1
                if guidance_scale > 1.0:
                    # Unconditional prediction (you'd need to implement this)
                    uncond_output = model_output  # Placeholder
                    model_output = uncond_output + guidance_scale * (model_output - uncond_output)
                
                # Apply reverse diffusion step
                x = self(model_output, t.item(), x, eta)
        
        return x
    
    def sample_with_classifier_free_guidance(self, model: nn.Module, shape: Tuple[int, ...], 
                                           text_embeddings: torch.Tensor, uncond_embeddings: torch.Tensor,
                                           num_inference_steps: int, guidance_scale: float = 7.5,
                                           eta: float = 0.0) -> torch.Tensor:
        """Sample with classifier-free guidance."""
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Set timesteps
        timesteps = torch.linspace(self.scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        # Reverse diffusion loop
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc="Sampling with CFG")):
                # Prepare timestep
                timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                # Get conditional prediction
                cond_output = model(x, timestep, text_embeddings)
                
                # Get unconditional prediction
                uncond_output = model(x, timestep, uncond_embeddings)
                
                # Apply classifier-free guidance
                model_output = uncond_output + guidance_scale * (cond_output - uncond_output)
                
                # Apply reverse diffusion step
                x = self(model_output, t.item(), x, eta)
        
        return x


class DiffusionTrainingSystem:
    """Complete diffusion training system with forward and reverse processes."""
    
    def __init__(self, config: DiffusionConfig, model: nn.Module):
        
    """__init__ function."""
self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.scheduler = DiffusionScheduler(config)
        self.forward_process = ForwardDiffusionProcess(self.scheduler)
        self.reverse_process = ReverseDiffusionProcess(self.scheduler)
        
        # Model
        self.model = model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.1
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
        # Enable gradient checkpointing
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info(f"Diffusion training system initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step with forward diffusion."""
        batch = batch.to(self.device)
        batch_size = batch.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.config.num_timesteps, (batch_size,), device=self.device)
        
        # Get training target (noise)
        target = self.forward_process.get_training_target(batch, timesteps)
        
        # Forward pass
        with torch.cuda.amp.autocast() if self.config.fp16 else torch.no_grad():
            noise_pred = self.model(batch, timesteps)
            loss = F.mse_loss(noise_pred, target, reduction='mean')
        
        # Backward pass
        if self.config.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def sample(self, shape: Tuple[int, ...], num_inference_steps: Optional[int] = None,
               guidance_scale: Optional[float] = None, eta: Optional[float] = None) -> torch.Tensor:
        """Sample from the trained diffusion model."""
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale or self.config.guidance_scale
        eta = eta or self.config.eta
        
        return self.reverse_process.sample(
            self.model, shape, num_inference_steps, guidance_scale, eta
        )
    
    def save_model(self, path: str):
        """Save the trained model."""
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': self.config.__dict__
        }, os.path.join(path, 'diffusion_model.pth'))
        
        logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(os.path.join(path, 'diffusion_model.pth'), map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Model loaded from: {path}")


def create_diffusion_training_system(config: DiffusionConfig, model: nn.Module) -> DiffusionTrainingSystem:
    """Create a diffusion training system."""
    return DiffusionTrainingSystem(config, model)


# Example usage
if __name__ == "__main__":
    # Create a simple UNet model for demonstration
    class SimpleUNet(nn.Module):
        def __init__(self, config: DiffusionConfig):
            
    """__init__ function."""
super().__init__()
            self.config = config
            
            # Simple UNet architecture
            self.conv1 = nn.Conv2d(config.in_channels, config.model_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(config.model_channels, config.out_channels, 3, padding=1)
            
            # Timestep embedding
            self.time_embed = nn.Sequential(
                nn.Linear(128, config.model_channels),
                nn.SiLU(),
                nn.Linear(config.model_channels, config.model_channels)
            )
        
        def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            # Timestep embedding
            time_emb = self.time_embed(timesteps.float())
            time_emb = time_emb.view(-1, 1, 1, 1)
            
            # Simple forward pass
            x = F.relu(self.conv1(x))
            x = x + time_emb
            x = self.conv2(x)
            
            return x
    
    # Create configuration
    config = DiffusionConfig(
        num_timesteps=1000,
        schedule="cosine",
        batch_size=2,
        num_epochs=10
    )
    
    # Create model
    model = SimpleUNet(config)
    
    # Create training system
    training_system = create_diffusion_training_system(config, model)
    
    # Sample training data
    sample_batch = torch.randn(2, 3, 256, 256)
    
    # Training step
    loss_info = training_system.train_step(sample_batch)
    print(f"Training loss: {loss_info['loss']:.4f}")
    
    # Sample from model
    generated_samples = training_system.sample((1, 3, 256, 256), num_inference_steps=50)
    print(f"Generated samples shape: {generated_samples.shape}")
    
    # Save model
    training_system.save_model("./diffusion_checkpoint") 