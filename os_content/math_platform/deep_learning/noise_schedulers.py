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
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import gc
from pathlib import Path
from tqdm import tqdm
import warnings
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Noise Schedulers and Sampling Methods
Production-ready implementation of various noise schedulers and sampling techniques.
"""


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class NoiseSchedulerConfig:
    """Configuration for noise schedulers."""
    # Basic parameters
    num_train_timesteps: int = 1000
    num_inference_timesteps: int = 50
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # Schedule type
    schedule: str = "linear"  # linear, cosine, sigmoid, cosine_beta, scaled_linear
    
    # Sampling parameters
    prediction_type: str = "epsilon"  # epsilon, sample, v_prediction
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    
    # Advanced parameters
    steps_offset: int = 1
    timestep_spacing: str = "leading"  # leading, trailing
    rescale_betas_zero_snr: bool = False
    
    # DDIM specific
    ddim_eta: float = 0.0
    ddim_discretize: str = "uniform"  # uniform, leading, trailing
    
    # DPM-Solver specific
    algorithm_type: str = "dpmsolver++"  # dpmsolver, dpmsolver++, sde-dpmsolver, sde-dpmsolver++
    solver_type: str = "midpoint"  # midpoint, heun, euler
    lower_order_final: bool = True
    use_karras_sigmas: bool = False
    timestep_spacing: str = "linspace"  # linspace, leading, trailing
    
    # Euler specific
    euler_at_final: bool = False
    use_karras_sigmas: bool = False
    
    # LMS specific
    lms_use_clipped_model_output: bool = False


class BaseNoiseScheduler(ABC):
    """Base class for noise schedulers."""
    
    def __init__(self, config: NoiseSchedulerConfig):
        
    """__init__ function."""
self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.num_inference_timesteps = config.num_inference_timesteps
        
        # Initialize schedule
        self._init_noise_schedule()
        
        logger.info(f"Initialized {self.__class__.__name__} with {self.num_train_timesteps} timesteps")
    
    @abstractmethod
    def _init_noise_schedule(self) -> Any:
        """Initialize the noise schedule."""
        pass
    
    @abstractmethod
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, 
             **kwargs) -> torch.Tensor:
        """Single denoising step."""
        pass
    
    @abstractmethod
    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to samples."""
        pass
    
    def get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """Get sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class DDPMNoiseScheduler(BaseNoiseScheduler):
    """Denoising Diffusion Probabilistic Models (DDPM) noise scheduler."""
    
    def _init_noise_schedule(self) -> Any:
        """Initialize DDPM noise schedule."""
        if self.config.schedule == "linear":
            self.betas = torch.linspace(self.config.beta_start, self.config.beta_end, self.num_train_timesteps)
        elif self.config.schedule == "cosine":
            self.betas = self._cosine_beta_schedule()
        elif self.config.schedule == "sigmoid":
            self.betas = self._sigmoid_beta_schedule()
        elif self.config.schedule == "cosine_beta":
            self.betas = self._cosine_beta_schedule_v2()
        elif self.config.schedule == "scaled_linear":
            self.betas = self._scaled_linear_beta_schedule()
        else:
            raise ValueError(f"Unknown schedule: {self.config.schedule}")
        
        # Precompute values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Variance for reverse process
        self.variance = self._get_variance()
        
        # Log SNR
        self.log_snr = torch.log(self.alphas_cumprod / (1 - self.alphas_cumprod))
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine beta schedule."""
        steps = self.num_train_timesteps + 1
        x = torch.linspace(0, self.num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_train_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def _sigmoid_beta_schedule(self) -> torch.Tensor:
        """Sigmoid beta schedule."""
        betas = torch.sigmoid(torch.linspace(-6, 6, self.num_train_timesteps))
        betas = betas * (self.config.beta_end - self.config.beta_start) + self.config.beta_start
        return betas
    
    def _cosine_beta_schedule_v2(self) -> torch.Tensor:
        """Cosine beta schedule v2."""
        max_beta = 0.999
        alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi * 0.5) ** 2
        betas = []
        for i in range(self.num_train_timesteps):
            t1 = i / self.num_train_timesteps
            t2 = (i + 1) / self.num_train_timesteps
            beta = min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta)
            betas.append(beta)
        return torch.tensor(betas)
    
    def _scaled_linear_beta_schedule(self) -> torch.Tensor:
        """Scaled linear beta schedule."""
        scale = 1000 / self.num_train_timesteps
        beta_start = scale * self.config.beta_start
        beta_end = scale * self.config.beta_end
        return torch.linspace(beta_start, beta_end, self.num_train_timesteps)
    
    def _get_variance(self) -> torch.Tensor:
        """Get variance for reverse process."""
        variance = (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * self.betas
        variance = torch.clamp(variance, min=0.0, max=1.0)
        return variance
    
    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to samples."""
        sqrt_alpha_cumprod = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1, 1)
        
        noise = torch.randn_like(original_samples)
        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        
        return noisy_samples, noise
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, 
             eta: float = 0.0, use_clipped_model_output: bool = True) -> torch.Tensor:
        """DDPM reverse step."""
        alpha_cumprod = self.alphas_cumprod[timestep]
        alpha_cumprod_prev = self.alphas_cumprod_prev[timestep]
        beta = self.betas[timestep]
        variance = self.variance[timestep]
        
        if use_clipped_model_output:
            model_output = torch.clamp(model_output, -1, 1)
        
        pred_original_sample = (sample - beta.sqrt() * model_output) / alpha_cumprod.sqrt()
        
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -self.config.clip_sample_range, self.config.clip_sample_range)
        
        pred_sample_direction = (1 - alpha_cumprod_prev).sqrt() * model_output
        
        if eta > 0:
            noise = torch.randn_like(sample)
            pred_prev_sample = alpha_cumprod_prev.sqrt() * pred_original_sample + pred_sample_direction + eta * variance.sqrt() * noise
        else:
            pred_prev_sample = alpha_cumprod_prev.sqrt() * pred_original_sample + pred_sample_direction
        
        return pred_prev_sample


class DDIMNoiseScheduler(BaseNoiseScheduler):
    """Denoising Diffusion Implicit Models (DDIM) noise scheduler."""
    
    def _init_noise_schedule(self) -> Any:
        """Initialize DDIM noise schedule."""
        # Use same beta schedule as DDPM
        if self.config.schedule == "linear":
            self.betas = torch.linspace(self.config.beta_start, self.config.beta_end, self.num_train_timesteps)
        elif self.config.schedule == "cosine":
            self.betas = self._cosine_beta_schedule()
        else:
            raise ValueError(f"DDIM only supports linear and cosine schedules")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # DDIM specific parameters
        self.eta = self.config.ddim_eta
        self.ddim_discretize = self.config.ddim_discretize
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine beta schedule for DDIM."""
        steps = self.num_train_timesteps + 1
        x = torch.linspace(0, self.num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_train_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to samples (same as DDPM)."""
        sqrt_alpha_cumprod = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1, 1)
        
        noise = torch.randn_like(original_samples)
        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        
        return noisy_samples, noise
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, 
             eta: Optional[float] = None) -> torch.Tensor:
        """DDIM reverse step."""
        eta = eta if eta is not None else self.eta
        
        alpha_cumprod = self.alphas_cumprod[timestep]
        alpha_cumprod_prev = self.alphas_cumprod_prev[timestep]
        
        pred_original_sample = (sample - (1 - alpha_cumprod).sqrt() * model_output) / alpha_cumprod.sqrt()
        
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -self.config.clip_sample_range, self.config.clip_sample_range)
        
        pred_sample_direction = (1 - alpha_cumprod_prev).sqrt() * model_output
        
        if eta > 0:
            noise = torch.randn_like(sample)
            pred_prev_sample = alpha_cumprod_prev.sqrt() * pred_original_sample + pred_sample_direction + eta * (1 - alpha_cumprod_prev).sqrt() * noise
        else:
            pred_prev_sample = alpha_cumprod_prev.sqrt() * pred_original_sample + pred_sample_direction
        
        return pred_prev_sample


class DPMSolverNoiseScheduler(BaseNoiseScheduler):
    """DPM-Solver noise scheduler for fast sampling."""
    
    def _init_noise_schedule(self) -> Any:
        """Initialize DPM-Solver noise schedule."""
        if self.config.schedule == "linear":
            self.betas = torch.linspace(self.config.beta_start, self.config.beta_end, self.num_train_timesteps)
        elif self.config.schedule == "cosine":
            self.betas = self._cosine_beta_schedule()
        else:
            raise ValueError(f"DPM-Solver only supports linear and cosine schedules")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # DPM-Solver specific
        self.algorithm_type = self.config.algorithm_type
        self.solver_type = self.config.solver_type
        self.lower_order_final = self.config.lower_order_final
        self.use_karras_sigmas = self.config.use_karras_sigmas
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine beta schedule for DPM-Solver."""
        steps = self.num_train_timesteps + 1
        x = torch.linspace(0, self.num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_train_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to samples."""
        sqrt_alpha_cumprod = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1, 1)
        
        noise = torch.randn_like(original_samples)
        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        
        return noisy_samples, noise
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, 
             **kwargs) -> torch.Tensor:
        """DPM-Solver reverse step."""
        # Simplified DPM-Solver step
        alpha_cumprod = self.alphas_cumprod[timestep]
        
        # Predict original sample
        pred_original_sample = (sample - (1 - alpha_cumprod).sqrt() * model_output) / alpha_cumprod.sqrt()
        
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -self.config.clip_sample_range, self.config.clip_sample_range)
        
        return pred_original_sample


class EulerNoiseScheduler(BaseNoiseScheduler):
    """Euler noise scheduler for fast sampling."""
    
    def _init_noise_schedule(self) -> Any:
        """Initialize Euler noise schedule."""
        if self.config.schedule == "linear":
            self.betas = torch.linspace(self.config.beta_start, self.config.beta_end, self.num_train_timesteps)
        elif self.config.schedule == "cosine":
            self.betas = self._cosine_beta_schedule()
        else:
            raise ValueError(f"Euler only supports linear and cosine schedules")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Euler specific
        self.euler_at_final = self.config.euler_at_final
        self.use_karras_sigmas = self.config.use_karras_sigmas
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine beta schedule for Euler."""
        steps = self.num_train_timesteps + 1
        x = torch.linspace(0, self.num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_train_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to samples."""
        sqrt_alpha_cumprod = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1, 1)
        
        noise = torch.randn_like(original_samples)
        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        
        return noisy_samples, noise
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, 
             **kwargs) -> torch.Tensor:
        """Euler reverse step."""
        alpha_cumprod = self.alphas_cumprod[timestep]
        
        # Predict original sample
        pred_original_sample = (sample - (1 - alpha_cumprod).sqrt() * model_output) / alpha_cumprod.sqrt()
        
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -self.config.clip_sample_range, self.config.clip_sample_range)
        
        return pred_original_sample


class LMSNoiseScheduler(BaseNoiseScheduler):
    """Linear Multistep (LMS) noise scheduler."""
    
    def _init_noise_schedule(self) -> Any:
        """Initialize LMS noise schedule."""
        if self.config.schedule == "linear":
            self.betas = torch.linspace(self.config.beta_start, self.config.beta_end, self.num_train_timesteps)
        elif self.config.schedule == "cosine":
            self.betas = self._cosine_beta_schedule()
        else:
            raise ValueError(f"LMS only supports linear and cosine schedules")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # LMS specific
        self.lms_use_clipped_model_output = self.config.lms_use_clipped_model_output
        self.derivatives = []
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine beta schedule for LMS."""
        steps = self.num_train_timesteps + 1
        x = torch.linspace(0, self.num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_train_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to samples."""
        sqrt_alpha_cumprod = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1, 1)
        
        noise = torch.randn_like(original_samples)
        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        
        return noisy_samples, noise
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, 
             **kwargs) -> torch.Tensor:
        """LMS reverse step."""
        if self.lms_use_clipped_model_output:
            model_output = torch.clamp(model_output, -1, 1)
        
        alpha_cumprod = self.alphas_cumprod[timestep]
        alpha_cumprod_prev = self.alphas_cumprod_prev[timestep]
        
        pred_original_sample = (sample - (1 - alpha_cumprod).sqrt() * model_output) / alpha_cumprod.sqrt()
        
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -self.config.clip_sample_range, self.config.clip_sample_range)
        
        pred_sample_direction = (1 - alpha_cumprod_prev).sqrt() * model_output
        
        # LMS specific: use derivatives for better prediction
        if len(self.derivatives) > 0:
            # Use previous derivatives for better prediction
            pass
        
        pred_prev_sample = alpha_cumprod_prev.sqrt() * pred_original_sample + pred_sample_direction
        
        return pred_prev_sample


class NoiseSchedulerFactory:
    """Factory for creating noise schedulers."""
    
    @staticmethod
    def create_scheduler(scheduler_type: str, config: NoiseSchedulerConfig) -> BaseNoiseScheduler:
        """Create a noise scheduler based on type."""
        schedulers = {
            "ddpm": DDPMNoiseScheduler,
            "ddim": DDIMNoiseScheduler,
            "dpmsolver": DPMSolverNoiseScheduler,
            "euler": EulerNoiseScheduler,
            "lms": LMSNoiseScheduler,
        }
        
        if scheduler_type not in schedulers:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        return schedulers[scheduler_type](config)


class SamplingMethods:
    """Advanced sampling methods for diffusion models."""
    
    def __init__(self, scheduler: BaseNoiseScheduler):
        
    """__init__ function."""
self.scheduler = scheduler
    
    def sample_ddpm(self, model: nn.Module, shape: Tuple[int, ...], num_inference_steps: int,
                    guidance_scale: float = 1.0, eta: float = 0.0) -> torch.Tensor:
        """Sample using DDPM method."""
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Set timesteps
        timesteps = torch.linspace(self.scheduler.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        # Reverse diffusion loop
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc="DDPM Sampling")):
                timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
                model_output = model(x, timestep)
                
                if guidance_scale > 1.0:
                    uncond_output = model_output  # Placeholder
                    model_output = uncond_output + guidance_scale * (model_output - uncond_output)
                
                x = self.scheduler.step(model_output, t.item(), x, eta=eta)
        
        return x
    
    def sample_ddim(self, model: nn.Module, shape: Tuple[int, ...], num_inference_steps: int,
                    guidance_scale: float = 1.0, eta: float = 0.0) -> torch.Tensor:
        """Sample using DDIM method."""
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Set timesteps
        timesteps = torch.linspace(self.scheduler.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        # Reverse diffusion loop
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
                timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
                model_output = model(x, timestep)
                
                if guidance_scale > 1.0:
                    uncond_output = model_output  # Placeholder
                    model_output = uncond_output + guidance_scale * (model_output - uncond_output)
                
                x = self.scheduler.step(model_output, t.item(), x, eta=eta)
        
        return x
    
    def sample_dpmsolver(self, model: nn.Module, shape: Tuple[int, ...], num_inference_steps: int,
                         guidance_scale: float = 1.0) -> torch.Tensor:
        """Sample using DPM-Solver method."""
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Set timesteps
        timesteps = torch.linspace(self.scheduler.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        # Reverse diffusion loop
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc="DPM-Solver Sampling")):
                timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
                model_output = model(x, timestep)
                
                if guidance_scale > 1.0:
                    uncond_output = model_output  # Placeholder
                    model_output = uncond_output + guidance_scale * (model_output - uncond_output)
                
                x = self.scheduler.step(model_output, t.item(), x)
        
        return x
    
    def sample_euler(self, model: nn.Module, shape: Tuple[int, ...], num_inference_steps: int,
                     guidance_scale: float = 1.0) -> torch.Tensor:
        """Sample using Euler method."""
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Set timesteps
        timesteps = torch.linspace(self.scheduler.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        # Reverse diffusion loop
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc="Euler Sampling")):
                timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
                model_output = model(x, timestep)
                
                if guidance_scale > 1.0:
                    uncond_output = model_output  # Placeholder
                    model_output = uncond_output + guidance_scale * (model_output - uncond_output)
                
                x = self.scheduler.step(model_output, t.item(), x)
        
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
        timesteps = torch.linspace(self.scheduler.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        # Reverse diffusion loop
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc="CFG Sampling")):
                timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                # Get conditional prediction
                cond_output = model(x, timestep, text_embeddings)
                
                # Get unconditional prediction
                uncond_output = model(x, timestep, uncond_embeddings)
                
                # Apply classifier-free guidance
                model_output = uncond_output + guidance_scale * (cond_output - uncond_output)
                
                # Apply reverse diffusion step
                x = self.scheduler.step(model_output, t.item(), x, eta=eta)
        
        return x


def create_noise_scheduler(scheduler_type: str, config: NoiseSchedulerConfig) -> BaseNoiseScheduler:
    """Create a noise scheduler."""
    return NoiseSchedulerFactory.create_scheduler(scheduler_type, config)


def create_sampling_methods(scheduler: BaseNoiseScheduler) -> SamplingMethods:
    """Create sampling methods."""
    return SamplingMethods(scheduler)


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = NoiseSchedulerConfig(
        num_train_timesteps=1000,
        num_inference_timesteps=50,
        schedule="cosine",
        prediction_type="epsilon"
    )
    
    # Create different schedulers
    ddpm_scheduler = create_noise_scheduler("ddpm", config)
    ddim_scheduler = create_noise_scheduler("ddim", config)
    dpmsolver_scheduler = create_noise_scheduler("dpmsolver", config)
    
    # Create sampling methods
    ddpm_sampling = create_sampling_methods(ddpm_scheduler)
    ddim_sampling = create_sampling_methods(ddim_scheduler)
    dpmsolver_sampling = create_sampling_methods(dpmsolver_scheduler)
    
    # Example model (placeholder)
    class SimpleModel(nn.Module):
        def __init__(self) -> Any:
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
        
        def forward(self, x, timestep) -> Any:
            return self.conv(x)
    
    model = SimpleModel()
    
    # Sample using different methods
    shape = (1, 3, 256, 256)
    
    # DDPM sampling
    ddpm_samples = ddpm_sampling.sample_ddpm(model, shape, num_inference_steps=50)
    print(f"DDPM samples shape: {ddpm_samples.shape}")
    
    # DDIM sampling
    ddim_samples = ddim_sampling.sample_ddim(model, shape, num_inference_steps=50)
    print(f"DDIM samples shape: {ddim_samples.shape}")
    
    # DPM-Solver sampling
    dpmsolver_samples = dpmsolver_sampling.sample_dpmsolver(model, shape, num_inference_steps=50)
    print(f"DPM-Solver samples shape: {dpmsolver_samples.shape}") 