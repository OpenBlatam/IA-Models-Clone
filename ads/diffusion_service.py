from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import (
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import json
import os
from datetime import datetime
import asyncio
from functools import lru_cache
try:
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    aioredis = None  # type: ignore[assignment]
import hashlib
import base64
from io import BytesIO
import requests
from dataclasses import dataclass
import logging
import math
from enum import Enum
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.optimized_config import settings
from onyx.server.features.ads.tokenization_service import TokenizationService
from onyx.server.features.ads.training_logger import TrainingLogger, TrainingPhase, AsyncTrainingLogger
from typing import Any, List, Dict, Optional
"""
Advanced diffusion models service for ads generation and image manipulation.
Supports text-to-image, image-to-image, inpainting, and style transfer.
Enhanced with comprehensive noise schedulers and sampling methods.
"""
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DDIMScheduler,
    PNDMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    ControlNetModel,
    UniPCMultistepScheduler,
    LCMScheduler,
    TCDScheduler,
    DDPMWuerstchenScheduler,
    WuerstchenCombinedScheduler,
    DiffusionPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    DDPMScheduler,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    DPMSolverSDEScheduler,
    VQDiffusionScheduler,
    ScoreSdeVeScheduler,
    ScoreSdeVpScheduler,
    IPNDMScheduler,
    KarrasVeScheduler,
    DDPMWuerstchenScheduler,
    WuerstchenCombinedScheduler
)


logger = setup_logger()

class NoiseScheduleType(Enum):
    """Types of noise schedules for diffusion models."""
    LINEAR = "linear"
    SCALED_LINEAR = "scaled_linear"
    COSINE = "cosine"
    SQUAREDCOS_CAP_V2 = "squaredcos_cap_v2"
    SIGMA = "sigma"
    KARRAS = "karras"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"

class SamplingMethod(Enum):
    """Sampling methods for diffusion models."""
    DDIM = "ddim"
    EULER = "euler"
    EULER_ANCESTRAL = "euler_ancestral"
    HEUN = "heun"
    DPM_SOLVER = "dpm_solver"
    DPM_SOLVER_PP = "dpm_solver_pp"
    DPM_SOLVER_SDE = "dpm_solver_sde"
    UNIPC = "unipc"
    LCM = "lcm"
    TCD = "tcd"
    PNDM = "pndm"
    LMS = "lms"
    KDPM2 = "kdpm2"
    KDPM2_ANCESTRAL = "kdpm2_ancestral"

@dataclass
class NoiseScheduleConfig:
    """Configuration for noise scheduling."""
    schedule_type: NoiseScheduleType = NoiseScheduleType.LINEAR
    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_train_timesteps: int = 1000
    num_inference_timesteps: int = 50
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    prediction_type: str = "epsilon"
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    sample_max_value: float = 1.0
    timestep_spacing: str = "leading"
    steps_offset: int = 0
    use_karras_sigmas: bool = False
    sigma_min: float = 0.1
    sigma_max: float = 80.0
    rho: float = 7.0

@dataclass
class SamplingConfig:
    """Configuration for sampling methods."""
    method: SamplingMethod = SamplingMethod.DDIM
    num_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    strength: float = 0.8
    seed: Optional[int] = None
    use_classifier_free_guidance: bool = True
    use_negative_prompt: bool = True
    add_noise: bool = True
    return_intermediates: bool = False
    callback_steps: int = 1
    callback: Optional[Callable] = None

@dataclass
class DiffusionConfig:
    """Configuration for diffusion models."""
    model_name: str = "runwayml/stable-diffusion-v1-5"
    scheduler_type: str = "DDIM"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    strength: float = 0.8
    eta: float = 0.0
    seed: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    noise_schedule: NoiseScheduleConfig = None
    sampling: SamplingConfig = None

@dataclass
class GenerationParams:
    """Parameters for image generation."""
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_images: int = 1
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    seed: Optional[int] = None
    style_preset: Optional[str] = None
    aspect_ratio: Optional[str] = None
    noise_schedule: Optional[NoiseScheduleConfig] = None
    sampling: Optional[SamplingConfig] = None

@dataclass
class DiffusionProcessConfig:
    """Configuration for diffusion process analysis."""
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    prediction_type: str = "epsilon"
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    sample_max_value: float = 1.0
    timestep_spacing: str = "leading"
    steps_offset: int = 0

class AdvancedNoiseScheduler:
    """Advanced noise scheduler with multiple scheduling strategies."""
    
    def __init__(self, config: NoiseScheduleConfig):
        """Initialize the advanced noise scheduler."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance
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
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
        # Signal-to-noise ratio
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)
        
        # Karras sigmas for advanced sampling
        if self.config.use_karras_sigmas:
            self.karras_sigmas = self._get_karras_sigmas()
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """Generate beta schedule based on configuration."""
        if self.config.schedule_type == NoiseScheduleType.LINEAR:
            return torch.linspace(self.config.beta_start, self.config.beta_end, self.config.num_train_timesteps)
        elif self.config.schedule_type == NoiseScheduleType.SCALED_LINEAR:
            return torch.linspace(self.config.beta_start, self.config.beta_end, self.config.num_train_timesteps) ** 0.5
        elif self.config.schedule_type == NoiseScheduleType.COSINE:
            return self._cosine_beta_schedule()
        elif self.config.schedule_type == NoiseScheduleType.SQUAREDCOS_CAP_V2:
            return self._betas_for_alpha_bar(self.config.num_train_timesteps, alpha_transform_type="cosine")
        elif self.config.schedule_type == NoiseScheduleType.SIGMA:
            return self._sigma_beta_schedule()
        elif self.config.schedule_type == NoiseScheduleType.KARRAS:
            return self._karras_beta_schedule()
        elif self.config.schedule_type == NoiseScheduleType.EXPONENTIAL:
            return self._exponential_beta_schedule()
        elif self.config.schedule_type == NoiseScheduleType.POLYNOMIAL:
            return self._polynomial_beta_schedule()
        else:
            raise ValueError(f"Unknown noise schedule type: {self.config.schedule_type}")
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Generate cosine beta schedule."""
        def alpha_bar_fn(t) -> Any:
            return math.cos((t + 0.008) / 1.008 * math.pi * 0.5) ** 2
        
        betas = []
        for i in range(self.config.num_train_timesteps):
            t1 = i / self.config.num_train_timesteps
            t2 = (i + 1) / self.config.num_train_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), 0.999))
        
        return torch.tensor(betas)
    
    def _sigma_beta_schedule(self) -> torch.Tensor:
        """Generate sigma-based beta schedule."""
        sigmas = torch.linspace(self.config.sigma_min, self.config.sigma_max, self.config.num_train_timesteps)
        alphas = 1.0 / (1.0 + sigmas ** 2)
        betas = 1.0 - alphas
        return betas
    
    def _karras_beta_schedule(self) -> torch.Tensor:
        """Generate Karras beta schedule."""
        sigmas = self._get_karras_sigmas()
        alphas = 1.0 / (1.0 + sigmas ** 2)
        betas = 1.0 - alphas
        return betas
    
    def _exponential_beta_schedule(self) -> torch.Tensor:
        """Generate exponential beta schedule."""
        return torch.exp(torch.linspace(
            math.log(self.config.beta_start),
            math.log(self.config.beta_end),
            self.config.num_train_timesteps
        ))
    
    def _polynomial_beta_schedule(self) -> torch.Tensor:
        """Generate polynomial beta schedule."""
        t = torch.linspace(0, 1, self.config.num_train_timesteps)
        return self.config.beta_start + (self.config.beta_end - self.config.beta_start) * (t ** 2)
    
    def _get_karras_sigmas(self) -> torch.Tensor:
        """Generate Karras sigmas for advanced sampling."""
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max
        rho = self.config.rho
        num_steps = self.config.num_inference_timesteps
        
        # Karras sigmas formula
        ramp = torch.linspace(0, 1, num_steps)
        sigmas = sigma_min ** (1 - ramp) * sigma_max ** ramp
        sigmas = torch.cat([sigmas, torch.tensor([0.0])])
        
        return sigmas
    
    def _betas_for_alpha_bar(self, num_diffusion_timesteps: int, alpha_transform_type: str = "cosine") -> torch.Tensor:
        """Generate betas for alpha bar schedule."""
        if alpha_transform_type == "cosine":
            def alpha_bar_fn(t) -> Any:
                return math.cos((t + 0.008) / 1.008 * math.pi * 0.5) ** 2
        else:
            raise ValueError(f"Unknown alpha transform type: {alpha_transform_type}")
        
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), 0.999))
        
        return torch.tensor(betas)
    
    def get_timesteps(self, num_inference_steps: int) -> torch.Tensor:
        """Get timesteps for inference."""
        if self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // num_inference_steps
            timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round()
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps // num_inference_steps
            timesteps = torch.flip(torch.arange(0, num_inference_steps) * step_ratio, (0,)).round()
        else:
            timesteps = torch.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps).round()
        
        timesteps = timesteps + self.config.steps_offset
        return timesteps.long()
    
    def get_noise_schedule_info(self) -> Dict[str, Any]:
        """Get information about the noise schedule."""
        return {
            "schedule_type": self.config.schedule_type.value,
            "beta_start": self.config.beta_start,
            "beta_end": self.config.beta_end,
            "num_train_timesteps": self.config.num_train_timesteps,
            "num_inference_timesteps": self.config.num_inference_timesteps,
            "alphas_cumprod_range": (self.alphas_cumprod.min().item(), self.alphas_cumprod.max().item()),
            "snr_range": (self.snr.min().item(), self.snr.max().item()),
            "use_karras_sigmas": self.config.use_karras_sigmas
        }

class AdvancedSamplingMethod:
    """Advanced sampling methods for diffusion models."""
    
    def __init__(self, config: SamplingConfig, noise_scheduler: AdvancedNoiseScheduler):
        """Initialize the advanced sampling method."""
        self.config = config
        self.noise_scheduler = noise_scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def sample(self, model, x_t: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample using the configured method."""
        if self.config.method == SamplingMethod.DDIM:
            return self._ddim_sample(model, x_t, t, **kwargs)
        elif self.config.method == SamplingMethod.EULER:
            return self._euler_sample(model, x_t, t, **kwargs)
        elif self.config.method == SamplingMethod.EULER_ANCESTRAL:
            return self._euler_ancestral_sample(model, x_t, t, **kwargs)
        elif self.config.method == SamplingMethod.HEUN:
            return self._heun_sample(model, x_t, t, **kwargs)
        elif self.config.method == SamplingMethod.DPM_SOLVER:
            return self._dpm_solver_sample(model, x_t, t, **kwargs)
        elif self.config.method == SamplingMethod.DPM_SOLVER_PP:
            return self._dpm_solver_pp_sample(model, x_t, t, **kwargs)
        elif self.config.method == SamplingMethod.UNIPC:
            return self._unipc_sample(model, x_t, t, **kwargs)
        else:
            raise ValueError(f"Unknown sampling method: {self.config.method}")
    
    def _ddim_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """DDIM sampling method."""
        # Get predicted noise
        noise_pred = model(x_t, t, **kwargs)
        
        # DDIM update rule
        alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]
        alpha_cumprod_prev = self.noise_scheduler.alphas_cumprod_prev[t]
        
        # Predicted x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_cumprod_prev) * noise_pred
        
        # DDIM step
        x_prev = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt
        
        return x_prev
    
    def _euler_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Euler sampling method."""
        # Get predicted noise
        noise_pred = model(x_t, t, **kwargs)
        
        # Euler step
        alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]
        beta_t = self.noise_scheduler.betas[t]
        
        # Predicted x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        
        # Euler update
        x_prev = pred_x0 + torch.sqrt(beta_t) * noise_pred
        
        return x_prev
    
    def _euler_ancestral_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Euler ancestral sampling method."""
        # Get predicted noise
        noise_pred = model(x_t, t, **kwargs)
        
        # Euler ancestral step
        alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]
        beta_t = self.noise_scheduler.betas[t]
        
        # Predicted x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        
        # Add ancestral noise
        noise = torch.randn_like(x_t) if self.config.add_noise else 0
        
        # Euler ancestral update
        x_prev = pred_x0 + torch.sqrt(beta_t) * (noise_pred + noise)
        
        return x_prev
    
    def _heun_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Heun sampling method (2nd order Runge-Kutta)."""
        # First step (Euler)
        noise_pred_1 = model(x_t, t, **kwargs)
        x_mid = self._euler_sample(model, x_t, t, noise_pred=noise_pred_1, **kwargs)
        
        # Second step (Heun)
        noise_pred_2 = model(x_mid, t - 1, **kwargs)
        
        # Heun update
        noise_pred = (noise_pred_1 + noise_pred_2) / 2
        x_prev = self._euler_sample(model, x_t, t, noise_pred=noise_pred, **kwargs)
        
        return x_prev
    
    def _dpm_solver_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """DPM-Solver sampling method."""
        # DPM-Solver step
        alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]
        alpha_cumprod_prev = self.noise_scheduler.alphas_cumprod_prev[t]
        
        # Get predicted noise
        noise_pred = model(x_t, t, **kwargs)
        
        # DPM-Solver update rule
        h = torch.log(alpha_cumprod_t / alpha_cumprod_prev)
        x_prev = x_t - h * noise_pred
        
        return x_prev
    
    def _dpm_solver_pp_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """DPM-Solver++ sampling method."""
        # DPM-Solver++ step
        alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]
        alpha_cumprod_prev = self.noise_scheduler.alphas_cumprod_prev[t]
        
        # Get predicted noise
        noise_pred = model(x_t, t, **kwargs)
        
        # DPM-Solver++ update rule
        h = torch.log(alpha_cumprod_t / alpha_cumprod_prev)
        x_prev = x_t - h * noise_pred + 0.5 * h ** 2 * noise_pred
        
        return x_prev
    
    def _unipc_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """UniPC sampling method."""
        # UniPC step
        alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]
        alpha_cumprod_prev = self.noise_scheduler.alphas_cumprod_prev[t]
        
        # Get predicted noise
        noise_pred = model(x_t, t, **kwargs)
        
        # UniPC update rule
        h = torch.log(alpha_cumprod_t / alpha_cumprod_prev)
        x_prev = x_t - h * noise_pred + (h ** 2 / 6) * noise_pred
        
        return x_prev

class DiffusionProcessAnalyzer:
    """Analyzes and implements forward and reverse diffusion processes."""
    
    def __init__(self, config: DiffusionProcessConfig):
        """Initialize the diffusion process analyzer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance
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
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """Generate beta schedule for diffusion process."""
        if self.config.beta_schedule == "linear":
            return torch.linspace(self.config.beta_start, self.config.beta_end, self.config.num_timesteps)
        elif self.config.beta_schedule == "scaled_linear":
            return torch.linspace(self.config.beta_start, self.config.beta_end, self.config.num_timesteps) ** 0.5
        elif self.config.beta_schedule == "squaredcos_cap_v2":
            return self._betas_for_alpha_bar(self.config.num_timesteps, alpha_transform_type="cosine")
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")
    
    def _betas_for_alpha_bar(self, num_diffusion_timesteps: int, alpha_transform_type: str = "cosine") -> torch.Tensor:
        """Generate betas for alpha bar schedule."""
        if alpha_transform_type == "cosine":
            def alpha_bar_fn(t) -> Any:
                return math.cos((t + 0.008) / 1.008 * math.pi * 0.5) ** 2
        else:
            raise ValueError(f"Unknown alpha transform type: {alpha_transform_type}")
        
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), 0.999))
        
        return torch.tensor(betas)
    
    def forward_diffusion_step(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of forward diffusion process.
        
        Args:
            x_start: Original image tensor [B, C, H, W]
            t: Timestep tensor [B]
            noise: Optional noise tensor, if None will be sampled
            
        Returns:
            x_t: Noisy image at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get alpha_cumprod for timestep t
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Forward diffusion equation: x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        x_t = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        
        return x_t, noise
    
    def reverse_diffusion_step(self, x_t: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        """
        Perform one step of reverse diffusion process (denoising).
        
        Args:
            x_t: Noisy image at timestep t [B, C, H, W]
            t: Timestep tensor [B]
            predicted_noise: Predicted noise from the model
            eta: Controls the amount of noise to add (0 = deterministic, 1 = stochastic)
            
        Returns:
            x_prev: Denoised image at timestep t-1
        """
        # Get coefficients for timestep t
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        
        # Calculate predicted x_0
        sqrt_recip_alpha_cumprod_t = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recipm1_alpha_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # x_0 = (x_t - sqrt(1 - α_t) * ε_pred) / sqrt(α_t)
        predicted_x_0 = sqrt_recip_alpha_cumprod_t * x_t - sqrt_recipm1_alpha_cumprod_t * predicted_noise
        
        if self.config.clip_sample:
            predicted_x_0 = torch.clamp(predicted_x_0, -self.config.clip_sample_range, self.config.clip_sample_range)
        
        # Calculate mean of q(x_{t-1} | x_t, x_0)
        if self.config.prediction_type == "epsilon":
            # Using epsilon prediction
            pred_original_sample = predicted_x_0
            pred_epsilon = predicted_noise
        elif self.config.prediction_type == "sample":
            # Using sample prediction
            pred_original_sample = predicted_noise
            pred_epsilon = (x_t - self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * pred_original_sample) / self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
        
        # Calculate mean
        pred_sample_direction = (1 - alpha_cumprod_prev_t).sqrt() * pred_epsilon
        prev_sample = alpha_cumprod_prev_t.sqrt() * pred_original_sample + pred_sample_direction
        
        # Add noise if eta > 0 (stochastic sampling)
        if eta > 0:
            noise = torch.randn_like(x_t)
            variance = (1 - alpha_cumprod_prev_t) * (1 - alpha_cumprod_t) / (1 - alpha_cumprod_t)
            variance = torch.clamp(variance, min=0.001)
            prev_sample = prev_sample + eta * variance.sqrt() * noise
        
        return prev_sample
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.config.num_timesteps, (batch_size,), device=self.device)
    
    def get_noise_schedule(self) -> Dict[str, torch.Tensor]:
        """Get the complete noise schedule."""
        return {
            "betas": self.betas,
            "alphas": self.alphas,
            "alphas_cumprod": self.alphas_cumprod,
            "sqrt_alphas_cumprod": self.sqrt_alphas_cumprod,
            "sqrt_one_minus_alphas_cumprod": self.sqrt_one_minus_alphas_cumprod,
            "posterior_variance": self.posterior_variance,
            "posterior_log_variance_clipped": self.posterior_log_variance_clipped
        }
    
    def analyze_diffusion_process(self, x_start: torch.Tensor, num_steps: int = 10) -> Dict[str, torch.Tensor]:
        """
        Analyze the forward diffusion process step by step.
        
        Args:
            x_start: Original image tensor
            num_steps: Number of steps to analyze
            
        Returns:
            Dictionary containing intermediate states and statistics
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample timesteps
        timesteps = torch.linspace(0, self.config.num_timesteps - 1, num_steps, dtype=torch.long, device=device)
        
        # Initialize storage
        x_t_list = []
        noise_list = []
        alpha_cumprod_list = []
        signal_to_noise_list = []
        
        x_t = x_start
        
        for i, t in enumerate(timesteps):
            # Repeat t for batch size
            t_batch = t.repeat(batch_size)
            
            # Forward diffusion step
            x_t, noise = self.forward_diffusion_step(x_t, t_batch)
            
            # Store results
            x_t_list.append(x_t.clone())
            noise_list.append(noise.clone())
            alpha_cumprod_list.append(self.alphas_cumprod[t].item())
            
            # Calculate signal-to-noise ratio
            signal_power = torch.mean(x_start ** 2)
            noise_power = torch.mean(noise ** 2)
            snr = 10 * torch.log10(signal_power / noise_power)
            signal_to_noise_list.append(snr.item())
        
        return {
            "x_t_states": torch.stack(x_t_list),
            "noise_states": torch.stack(noise_list),
            "alpha_cumprod": torch.tensor(alpha_cumprod_list),
            "signal_to_noise_ratio": torch.tensor(signal_to_noise_list),
            "timesteps": timesteps
        }
    
    def visualize_diffusion_process(self, x_start: torch.Tensor, save_path: Optional[str] = None) -> List[Image.Image]:
        """
        Create a visualization of the forward diffusion process.
        
        Args:
            x_start: Original image tensor
            save_path: Optional path to save the visualization
            
        Returns:
            List of PIL images showing the diffusion process
        """
        # Analyze the process
        analysis = self.analyze_diffusion_process(x_start, num_steps=20)
        
        # Convert tensors to images
        images = []
        for i in range(analysis["x_t_states"].shape[0]):
            # Convert tensor to PIL image
            img_tensor = analysis["x_t_states"][i]
            img_tensor = (img_tensor + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            # Convert to PIL
            img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            if img_array.shape[0] == 1:  # Grayscale
                img = Image.fromarray(img_array[0], mode='L')
            else:  # RGB
                img = Image.fromarray(np.transpose(img_array, (1, 2, 0)))
            
            images.append(img)
        
        if save_path:
            # Create a grid of images
            grid_size = int(math.ceil(math.sqrt(len(images))))
            grid_img = Image.new('RGB', (grid_size * images[0].width, grid_size * images[0].height))
            
            for i, img in enumerate(images):
                row = i // grid_size
                col = i % grid_size
                grid_img.paste(img, (col * img.width, row * img.height))
            
            grid_img.save(save_path)
        
        return images
    
    def calculate_diffusion_loss(self, x_start: torch.Tensor, predicted_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Calculate the diffusion loss for training.
        
        Args:
            x_start: Original image tensor
            predicted_noise: Noise predicted by the model
            t: Timestep tensor
            
        Returns:
            Loss tensor
        """
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion to get x_t
        x_t, _ = self.forward_diffusion_step(x_start, t, noise)
        
        # Calculate loss (MSE between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        
        return loss

class DiffusionSchedulerFactory:
    """Factory for creating diffusion schedulers with advanced options."""
    
    @staticmethod
    def create_scheduler(scheduler_type: str, **kwargs):
        """Create a scheduler based on type with optimized defaults."""
        schedulers = {
            "DDIM": DDIMScheduler,
            "PNDM": PNDMScheduler,
            "Euler": EulerDiscreteScheduler,
            "DPM++": DPMSolverMultistepScheduler,
            "DPM++_SDE": DPMSolverSinglestepScheduler,
            "Heun": HeunDiscreteScheduler,
            "KDPM2": KDPM2DiscreteScheduler,
            "KDPM2_Ancestral": KDPM2AncestralDiscreteScheduler,
            "UniPC": UniPCMultistepScheduler,
            "LCM": LCMScheduler,
            "TCD": TCDScheduler,
            "Euler_Ancestral": EulerAncestralDiscreteScheduler,
            "LMS": LMSDiscreteScheduler,
            "DPM_SDE": DPMSolverSDEScheduler,
            "VQ": VQDiffusionScheduler,
            "ScoreSdeVe": ScoreSdeVeScheduler,
            "ScoreSdeVp": ScoreSdeVpScheduler,
            "IPNDM": IPNDMScheduler,
            "KarrasVe": KarrasVeScheduler,
            "Wuerstchen": DDPMWuerstchenScheduler,
            "WuerstchenCombined": WuerstchenCombinedScheduler
        }
        
        scheduler_class = schedulers.get(scheduler_type, DDIMScheduler)
        
        # Set optimized defaults based on scheduler type
        if scheduler_type == "LCM":
            kwargs.setdefault("beta_start", 0.00085)
            kwargs.setdefault("beta_end", 0.012)
            kwargs.setdefault("beta_schedule", "scaled_linear")
        elif scheduler_type == "TCD":
            kwargs.setdefault("beta_start", 0.00085)
            kwargs.setdefault("beta_end", 0.012)
            kwargs.setdefault("beta_schedule", "scaled_linear")
        elif scheduler_type in ["DPM++", "DPM++_SDE"]:
            kwargs.setdefault("algorithm_type", "dpmsolver++")
            kwargs.setdefault("solver_type", "midpoint")
        elif scheduler_type == "KarrasVe":
            kwargs.setdefault("sigma_min", 0.1)
            kwargs.setdefault("sigma_max", 80.0)
            kwargs.setdefault("rho", 7.0)
        elif scheduler_type == "ScoreSdeVe":
            kwargs.setdefault("sigma_min", 0.01)
            kwargs.setdefault("sigma_max", 50.0)
        elif scheduler_type == "ScoreSdeVp":
            kwargs.setdefault("beta_min", 0.1)
            kwargs.setdefault("beta_max", 20.0)
        
        return scheduler_class(**kwargs)
    
    @staticmethod
    def create_advanced_noise_scheduler(config: NoiseScheduleConfig) -> AdvancedNoiseScheduler:
        """Create an advanced noise scheduler with custom configuration."""
        return AdvancedNoiseScheduler(config)
    
    @staticmethod
    def create_advanced_sampling_method(config: SamplingConfig, noise_scheduler: AdvancedNoiseScheduler) -> AdvancedSamplingMethod:
        """Create an advanced sampling method with custom configuration."""
        return AdvancedSamplingMethod(config, noise_scheduler)
    
    @staticmethod
    def get_optimal_scheduler_for_task(task: str, quality: str = "balanced"):
        """Get optimal scheduler for specific task and quality level."""
        if quality == "fast":
            if task == "text2img":
                return "LCM"
            elif task == "img2img":
                return "TCD"
            else:
                return "UniPC"
        elif quality == "high":
            if task == "text2img":
                return "DPM++"
            elif task == "img2img":
                return "Heun"
            else:
                return "DDIM"
        else:  # balanced
            if task == "text2img":
                return "Euler"
            elif task == "img2img":
                return "DPM++"
            else:
                return "PNDM"
    
    @staticmethod
    def get_optimal_noise_schedule_for_task(task: str, quality: str = "balanced") -> NoiseScheduleConfig:
        """Get optimal noise schedule configuration for specific task and quality level."""
        if quality == "fast":
            return NoiseScheduleConfig(
                schedule_type=NoiseScheduleType.COSINE,
                num_inference_timesteps=20,
                use_karras_sigmas=True
            )
        elif quality == "high":
            return NoiseScheduleConfig(
                schedule_type=NoiseScheduleType.KARRAS,
                num_inference_timesteps=100,
                use_karras_sigmas=True,
                sigma_min=0.1,
                sigma_max=80.0,
                rho=7.0
            )
        else:  # balanced
            return NoiseScheduleConfig(
                schedule_type=NoiseScheduleType.SCALED_LINEAR,
                num_inference_timesteps=50
            )
    
    @staticmethod
    def get_optimal_sampling_config_for_task(task: str, quality: str = "balanced") -> SamplingConfig:
        """Get optimal sampling configuration for specific task and quality level."""
        if quality == "fast":
            if task == "text2img":
                return SamplingConfig(
                    method=SamplingMethod.LCM,
                    num_steps=4,
                    guidance_scale=1.5
                )
            elif task == "img2img":
                return SamplingConfig(
                    method=SamplingMethod.TCD,
                    num_steps=4,
                    guidance_scale=1.5
                )
            else:
                return SamplingConfig(
                    method=SamplingMethod.UNIPC,
                    num_steps=10
                )
        elif quality == "high":
            if task == "text2img":
                return SamplingConfig(
                    method=SamplingMethod.DPM_SOLVER_PP,
                    num_steps=100,
                    guidance_scale=7.5
                )
            elif task == "img2img":
                return SamplingConfig(
                    method=SamplingMethod.HEUN,
                    num_steps=50,
                    guidance_scale=7.5
                )
            else:
                return SamplingConfig(
                    method=SamplingMethod.DDIM,
                    num_steps=50,
                    guidance_scale=7.5
                )
        else:  # balanced
            if task == "text2img":
                return SamplingConfig(
                    method=SamplingMethod.EULER,
                    num_steps=30,
                    guidance_scale=7.5
                )
            elif task == "img2img":
                return SamplingConfig(
                    method=SamplingMethod.DPM_SOLVER,
                    num_steps=30,
                    guidance_scale=7.5
                )
            else:
                return SamplingConfig(
                    method=SamplingMethod.PNDM,
                    num_steps=30,
                    guidance_scale=7.5
                )

class DiffusionModelManager:
    """Manages multiple diffusion models and pipelines."""
    
    def __init__(self) -> Any:
        """Initialize the diffusion model manager."""
        self._models = {}
        self._pipelines = {}
        self._redis_client = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenization_service = TokenizationService()
        
    @property
    async def redis_client(self) -> Any:
        """Lazy initialization of Redis client."""
        if self._redis_client is None:
            self._redis_client = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis_client
    
    def _get_model_key(self, model_name: str, task: str) -> str:
        """Generate cache key for model."""
        return f"diffusion:{model_name}:{task}"
    
    @lru_cache(maxsize=5)
    def get_text_to_image_pipeline(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """Get or create text-to-image pipeline with advanced optimizations."""
        if model_name not in self._pipelines:
            logger.info(f"Loading text-to-image pipeline: {model_name}")
            
            # Use AutoPipeline for automatic model detection
            if "xl" in model_name.lower():
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    variant="fp16" if torch.cuda.is_available() else None
                )
            else:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    variant="fp16" if torch.cuda.is_available() else None
                )
            
            if torch.cuda.is_available():
                pipeline = pipeline.to(self.device)
                
                # Enable advanced optimizations
                pipeline.enable_attention_slicing()
                pipeline.enable_vae_slicing()
                pipeline.enable_model_cpu_offload()
                
                # Enable memory efficient attention if available
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except:
                    logger.warning("xformers not available, using standard attention")
                
                # Enable sequential CPU offload for large models
                if "xl" in model_name.lower():
                    pipeline.enable_sequential_cpu_offload()
            
            self._pipelines[model_name] = pipeline
            
        return self._pipelines[model_name]
    
    @lru_cache(maxsize=5)
    def get_image_to_image_pipeline(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """Get or create image-to-image pipeline with advanced optimizations."""
        pipeline_key = f"{model_name}_img2img"
        
        if pipeline_key not in self._pipelines:
            logger.info(f"Loading image-to-image pipeline: {model_name}")
            
            # Use appropriate pipeline based on model
            if "xl" in model_name.lower():
                pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    variant="fp16" if torch.cuda.is_available() else None
                )
            else:
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    variant="fp16" if torch.cuda.is_available() else None
                )
            
            if torch.cuda.is_available():
                pipeline = pipeline.to(self.device)
                
                # Enable advanced optimizations
                pipeline.enable_attention_slicing()
                pipeline.enable_vae_slicing()
                pipeline.enable_model_cpu_offload()
                
                # Enable memory efficient attention if available
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except:
                    logger.warning("xformers not available, using standard attention")
                
                # Enable sequential CPU offload for large models
                if "xl" in model_name.lower():
                    pipeline.enable_sequential_cpu_offload()
            
            self._pipelines[pipeline_key] = pipeline
            
        return self._pipelines[pipeline_key]
    
    @lru_cache(maxsize=5)
    def get_inpaint_pipeline(self, model_name: str = "runwayml/stable-diffusion-inpainting"):
        """Get or create inpainting pipeline."""
        pipeline_key = f"{model_name}_inpaint"
        
        if pipeline_key not in self._pipelines:
            logger.info(f"Loading inpainting pipeline: {model_name}")
            
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            if torch.cuda.is_available():
                pipeline = pipeline.to(self.device)
                pipeline.enable_attention_slicing()
                pipeline.enable_vae_slicing()
            
            self._pipelines[pipeline_key] = pipeline
            
        return self._pipelines[pipeline_key]
    
    def get_controlnet_pipeline(self, controlnet_type: str = "canny"):
        """Get or create ControlNet pipeline with advanced optimizations."""
        pipeline_key = f"controlnet_{controlnet_type}"
        
        if pipeline_key not in self._pipelines:
            logger.info(f"Loading ControlNet pipeline: {controlnet_type}")
            
            # Load ControlNet model
            controlnet = ControlNetModel.from_pretrained(
                f"lllyasviel/sd-controlnet-{controlnet_type}",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Create pipeline
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                variant="fp16" if torch.cuda.is_available() else None
            )
            
            if torch.cuda.is_available():
                pipeline = pipeline.to(self.device)
                
                # Enable advanced optimizations
                pipeline.enable_attention_slicing()
                pipeline.enable_vae_slicing()
                pipeline.enable_model_cpu_offload()
                
                # Enable memory efficient attention if available
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except:
                    logger.warning("xformers not available, using standard attention")
            
            self._pipelines[pipeline_key] = pipeline
            
        return self._pipelines[pipeline_key]
    
    def get_lcm_pipeline(self, model_name: str = "SimianLuo/LCM_Dreamshaper_v7"):
        """Get or create LCM (Latent Consistency Model) pipeline for fast generation."""
        pipeline_key = f"lcm_{model_name}"
        
        if pipeline_key not in self._pipelines:
            logger.info(f"Loading LCM pipeline: {model_name}")
            
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                variant="fp16" if torch.cuda.is_available() else None
            )
            
            # Set LCM scheduler
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
            
            if torch.cuda.is_available():
                pipeline = pipeline.to(self.device)
                pipeline.enable_attention_slicing()
                pipeline.enable_vae_slicing()
                pipeline.enable_model_cpu_offload()
                
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except:
                    logger.warning("xformers not available, using standard attention")
            
            self._pipelines[pipeline_key] = pipeline
            
        return self._pipelines[pipeline_key]
    
    def get_tcd_pipeline(self, model_name: str = "h1t/TCD-SD15"):
        """Get or create TCD (Trajectory Consistency Distillation) pipeline for fast generation."""
        pipeline_key = f"tcd_{model_name}"
        
        if pipeline_key not in self._pipelines:
            logger.info(f"Loading TCD pipeline: {model_name}")
            
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                variant="fp16" if torch.cuda.is_available() else None
            )
            
            # Set TCD scheduler
            pipeline.scheduler = TCDScheduler.from_config(pipeline.scheduler.config)
            
            if torch.cuda.is_available():
                pipeline = pipeline.to(self.device)
                pipeline.enable_attention_slicing()
                pipeline.enable_vae_slicing()
                pipeline.enable_model_cpu_offload()
                
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except:
                    logger.warning("xformers not available, using standard attention")
            
            self._pipelines[pipeline_key] = pipeline
            
        return self._pipelines[pipeline_key]

class ImageProcessor:
    """Handles image processing and manipulation."""
    
    def __init__(self) -> Any:
        """Initialize image processor."""
        self.supported_formats = ['png', 'jpg', 'jpeg', 'webp']
        
    def load_image(self, image_path: str) -> Image.Image:
        """Load image from path."""
        try:
            image = Image.open(image_path)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def load_image_from_url(self, url: str) -> Image.Image:
        """Load image from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image from URL {url}: {e}")
            raise
    
    def load_image_from_base64(self, base64_string: str) -> Image.Image:
        """Load image from base64 string."""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image from base64: {e}")
            raise
    
    def resize_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        """Resize image to specified dimensions."""
        return image.resize((width, height), Image.Resampling.LANCZOS)
    
    def create_mask(self, image: Image.Image, mask_type: str = "center") -> Image.Image:
        """Create mask for inpainting."""
        width, height = image.size
        
        if mask_type == "center":
            # Create circular mask in center
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            
            draw.ellipse(
                [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                fill=255
            )
        elif mask_type == "random":
            # Create random rectangular mask
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            
            mask_width = width // 3
            mask_height = height // 3
            x = np.random.randint(0, width - mask_width)
            y = np.random.randint(0, height - mask_height)
            
            draw.rectangle([x, y, x + mask_width, y + mask_height], fill=255)
        else:
            # Full mask
            mask = Image.new('L', (width, height), 255)
        
        return mask
    
    def apply_canny_edge_detection(self, image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
        """Apply Canny edge detection for ControlNet."""
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply Canny edge detection
        edges = cv2.Canny(image_cv, low_threshold, high_threshold)
        
        # Convert back to PIL
        edges_pil = Image.fromarray(edges)
        
        return edges_pil
    
    def apply_depth_estimation(self, image: Image.Image) -> Image.Image:
        """Apply depth estimation for ControlNet."""
        # This is a simplified depth estimation
        # In production, you might want to use a proper depth estimation model
        
        # Convert to grayscale
        gray = image.convert('L')
        
        # Apply Gaussian blur to simulate depth
        gray_array = np.array(gray)
        blurred = cv2.GaussianBlur(gray_array, (15, 15), 0)
        
        # Convert back to PIL
        depth_pil = Image.fromarray(blurred)
        
        return depth_pil
    
    def save_image(self, image: Image.Image, path: str, format: str = "PNG") -> str:
        """Save image to file."""
        try:
            image.save(path, format=format)
            return path
        except Exception as e:
            logger.error(f"Error saving image to {path}: {e}")
            raise
    
    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"

class DiffusionService:
    """Main diffusion service for ads generation."""
    
    def __init__(self) -> Any:
        """Initialize diffusion service."""
        self.model_manager = DiffusionModelManager()
        self.image_processor = ImageProcessor()
        self._redis_client = None
        self.training_logger = None
        
    @property
    async def redis_client(self) -> Any:
        """Lazy initialization of Redis client."""
        if self._redis_client is None:
            self._redis_client = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis_client
    
    async def generate_text_to_image(
        self,
        params: GenerationParams,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        user_id: Optional[int] = None
    ) -> List[Image.Image]:
        """Generate images from text prompt."""
        # Initialize training logger if user_id provided
        if user_id and not self.training_logger:
            self.training_logger = AsyncTrainingLogger(
                user_id=user_id,
                model_name=model_name,
                log_dir=f"logs/diffusion/user_{user_id}",
                enable_autograd_debug=True  # Enable PyTorch debugging for diffusion
            )
        
        try:
            if self.training_logger:
                self.training_logger.log_info("Starting text-to-image generation", TrainingPhase.INFERENCE)
                self.training_logger.log_info(f"Prompt: {params.prompt[:100]}...")
                self.training_logger.log_info(f"Parameters: {params.width}x{params.height}, {params.num_images} images, {params.num_inference_steps} steps")
            
            # Check cache
            cache_key = self._get_cache_key(params, model_name, "text2img")
            redis = await self.redis_client
            cached_result = await redis.get(cache_key)
            
            if cached_result:
                if self.training_logger:
                    self.training_logger.log_info(f"Cache hit for text-to-image generation: {cache_key}")
                logger.info(f"Cache hit for text-to-image generation: {cache_key}")
                return self._decode_cached_images(cached_result)
            
            if self.training_logger:
                self.training_logger.log_info("Loading text-to-image pipeline")
            
            # Get pipeline
            pipeline = self.model_manager.get_text_to_image_pipeline(model_name)
            
            if self.training_logger:
                self.training_logger.log_info("Pipeline loaded successfully")
            
            # Set seed if provided
            if params.seed is not None:
                torch.manual_seed(params.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(params.seed)
                if self.training_logger:
                    self.training_logger.log_info(f"Seed set to: {params.seed}")
            
            # Generate images
            if self.training_logger:
                self.training_logger.log_info("Starting image generation...")
            
            start_time = time.time()
            
            # Debug pipeline tensors before generation
            if self.training_logger:
                # Check pipeline components for anomalies
                if hasattr(pipeline, 'unet') and hasattr(pipeline.unet, 'parameters'):
                    for name, param in pipeline.unet.named_parameters():
                        if param.requires_grad:
                            await self.training_logger.check_tensor_anomalies_async(param, f"unet_{name}")
            
            with torch.no_grad():
                result = pipeline(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    width=params.width,
                    height=params.height,
                    num_images_per_prompt=params.num_images,
                    guidance_scale=params.guidance_scale,
                    num_inference_steps=params.num_inference_steps,
                    eta=0.0
                )
            
            generation_time = time.time() - start_time
            images = result.images
            
            if self.training_logger:
                self.training_logger.log_info(f"Image generation completed in {generation_time:.2f} seconds")
                self.training_logger.log_info(f"Generated {len(images)} images successfully")
            
            # Cache result
            await redis.setex(
                cache_key,
                3600,  # 1 hour
                self._encode_images_for_cache(images)
            )
            
            if self.training_logger:
                self.training_logger.log_info("Results cached successfully")
            
            logger.info(f"Generated {len(images)} images from text prompt")
            return images
            
        except Exception as e:
            if self.training_logger:
                self.training_logger.log_error(e, TrainingPhase.INFERENCE, {
                    "model_name": model_name,
                    "prompt": params.prompt,
                    "width": params.width,
                    "height": params.height,
                    "num_images": params.num_images,
                    "guidance_scale": params.guidance_scale,
                    "num_inference_steps": params.num_inference_steps
                })
            logger.exception("Error in text-to-image generation")
            raise
    
    async def generate_image_to_image(
        self,
        init_image: Union[str, Image.Image],
        params: GenerationParams,
        model_name: str = "runwayml/stable-diffusion-v1-5"
    ) -> List[Image.Image]:
        """Generate images from initial image."""
        try:
            # Load image if needed
            if isinstance(init_image, str):
                if init_image.startswith('http'):
                    image = self.image_processor.load_image_from_url(init_image)
                elif init_image.startswith('data:image'):
                    image = self.image_processor.load_image_from_base64(init_image)
                else:
                    image = self.image_processor.load_image(init_image)
            else:
                image = init_image
            
            # Resize image if needed
            if image.size != (params.width, params.height):
                image = self.image_processor.resize_image(image, params.width, params.height)
            
            # Check cache
            cache_key = self._get_cache_key(params, model_name, "img2img", image)
            redis = await self.redis_client
            cached_result = await redis.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for image-to-image generation: {cache_key}")
                return self._decode_cached_images(cached_result)
            
            # Get pipeline
            pipeline = self.model_manager.get_image_to_image_pipeline(model_name)
            
            # Set seed if provided
            if params.seed is not None:
                torch.manual_seed(params.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(params.seed)
            
            # Generate images
            with torch.no_grad():
                result = pipeline(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    image=image,
                    strength=0.8,
                    guidance_scale=params.guidance_scale,
                    num_inference_steps=params.num_inference_steps,
                    num_images_per_prompt=params.num_images
                )
            
            images = result.images
            
            # Cache result
            await redis.setex(
                cache_key,
                3600,  # 1 hour
                self._encode_images_for_cache(images)
            )
            
            logger.info(f"Generated {len(images)} images from initial image")
            return images
            
        except Exception as e:
            logger.exception("Error in image-to-image generation")
            raise
    
    async def inpaint_image(
        self,
        image: Union[str, Image.Image],
        mask: Union[str, Image.Image, str],
        params: GenerationParams,
        model_name: str = "runwayml/stable-diffusion-inpainting"
    ) -> List[Image.Image]:
        """Inpaint image using mask."""
        try:
            # Load image if needed
            if isinstance(image, str):
                if image.startswith('http'):
                    image = self.image_processor.load_image_from_url(image)
                elif image.startswith('data:image'):
                    image = self.image_processor.load_image_from_base64(image)
                else:
                    image = self.image_processor.load_image(image)
            
            # Load or create mask
            if isinstance(mask, str):
                if mask in ["center", "random", "full"]:
                    mask = self.image_processor.create_mask(image, mask)
                elif mask.startswith('http'):
                    mask = self.image_processor.load_image_from_url(mask)
                elif mask.startswith('data:image'):
                    mask = self.image_processor.load_image_from_base64(mask)
                else:
                    mask = self.image_processor.load_image(mask)
            
            # Ensure mask is grayscale
            if mask.mode != 'L':
                mask = mask.convert('L')
            
            # Resize if needed
            if image.size != (params.width, params.height):
                image = self.image_processor.resize_image(image, params.width, params.height)
                mask = self.image_processor.resize_image(mask, params.width, params.height)
            
            # Check cache
            cache_key = self._get_cache_key(params, model_name, "inpaint", image, mask)
            redis = await self.redis_client
            cached_result = await redis.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for inpainting: {cache_key}")
                return self._decode_cached_images(cached_result)
            
            # Get pipeline
            pipeline = self.model_manager.get_inpaint_pipeline(model_name)
            
            # Set seed if provided
            if params.seed is not None:
                torch.manual_seed(params.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(params.seed)
            
            # Generate images
            with torch.no_grad():
                result = pipeline(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    image=image,
                    mask_image=mask,
                    guidance_scale=params.guidance_scale,
                    num_inference_steps=params.num_inference_steps,
                    num_images_per_prompt=params.num_images
                )
            
            images = result.images
            
            # Cache result
            await redis.setex(
                cache_key,
                3600,  # 1 hour
                self._encode_images_for_cache(images)
            )
            
            logger.info(f"Inpainted {len(images)} images")
            return images
            
        except Exception as e:
            logger.exception("Error in image inpainting")
            raise
    
    async def generate_with_controlnet(
        self,
        control_image: Union[str, Image.Image],
        params: GenerationParams,
        controlnet_type: str = "canny"
    ) -> List[Image.Image]:
        """Generate images using ControlNet."""
        try:
            # Load control image if needed
            if isinstance(control_image, str):
                if control_image.startswith('http'):
                    image = self.image_processor.load_image_from_url(control_image)
                elif control_image.startswith('data:image'):
                    image = self.image_processor.load_image_from_base64(control_image)
                else:
                    image = self.image_processor.load_image(control_image)
            else:
                image = control_image
            
            # Apply control processing
            if controlnet_type == "canny":
                control_image = self.image_processor.apply_canny_edge_detection(image)
            elif controlnet_type == "depth":
                control_image = self.image_processor.apply_depth_estimation(image)
            else:
                control_image = image
            
            # Resize if needed
            if control_image.size != (params.width, params.height):
                control_image = self.image_processor.resize_image(control_image, params.width, params.height)
            
            # Check cache
            cache_key = self._get_cache_key(params, "controlnet", controlnet_type, control_image)
            redis = await self.redis_client
            cached_result = await redis.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for ControlNet generation: {cache_key}")
                return self._decode_cached_images(cached_result)
            
            # Get pipeline
            pipeline = self.model_manager.get_controlnet_pipeline(controlnet_type)
            
            # Set seed if provided
            if params.seed is not None:
                torch.manual_seed(params.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(params.seed)
            
            # Generate images
            with torch.no_grad():
                result = pipeline(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    image=control_image,
                    width=params.width,
                    height=params.height,
                    num_images_per_prompt=params.num_images,
                    guidance_scale=params.guidance_scale,
                    num_inference_steps=params.num_inference_steps
                )
            
            images = result.images
            
            # Cache result
            await redis.setex(
                cache_key,
                3600,  # 1 hour
                self._encode_images_for_cache(images)
            )
            
            logger.info(f"Generated {len(images)} images with ControlNet")
            return images
            
        except Exception as e:
            logger.exception("Error in ControlNet generation")
            raise
    
    async def generate_with_lcm(
        self,
        params: GenerationParams,
        model_name: str = "SimianLuo/LCM_Dreamshaper_v7"
    ) -> List[Image.Image]:
        """Generate images using LCM (Latent Consistency Model) for fast generation."""
        try:
            # Check cache
            cache_key = self._get_cache_key(params, model_name, "lcm")
            redis = await self.redis_client
            cached_result = await redis.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for LCM generation: {cache_key}")
                return self._decode_cached_images(cached_result)
            
            # Get LCM pipeline
            pipeline = self.model_manager.get_lcm_pipeline(model_name)
            
            # Set seed if provided
            if params.seed is not None:
                torch.manual_seed(params.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(params.seed)
            
            # Generate images with LCM (typically 4-8 steps)
            with torch.no_grad():
                result = pipeline(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    width=params.width,
                    height=params.height,
                    num_images_per_prompt=params.num_images,
                    guidance_scale=params.guidance_scale,
                    num_inference_steps=min(params.num_inference_steps, 8),  # LCM is fast
                    eta=0.0
                )
            
            images = result.images
            
            # Cache result
            await redis.setex(
                cache_key,
                3600,  # 1 hour
                self._encode_images_for_cache(images)
            )
            
            logger.info(f"Generated {len(images)} images with LCM")
            return images
            
        except Exception as e:
            logger.exception("Error in LCM generation")
            raise
    
    async def generate_with_tcd(
        self,
        params: GenerationParams,
        model_name: str = "h1t/TCD-SD15"
    ) -> List[Image.Image]:
        """Generate images using TCD (Trajectory Consistency Distillation) for fast generation."""
        try:
            # Check cache
            cache_key = self._get_cache_key(params, model_name, "tcd")
            redis = await self.redis_client
            cached_result = await redis.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for TCD generation: {cache_key}")
                return self._decode_cached_images(cached_result)
            
            # Get TCD pipeline
            pipeline = self.model_manager.get_tcd_pipeline(model_name)
            
            # Set seed if provided
            if params.seed is not None:
                torch.manual_seed(params.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(params.seed)
            
            # Generate images with TCD (typically 1-4 steps)
            with torch.no_grad():
                result = pipeline(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    width=params.width,
                    height=params.height,
                    num_images_per_prompt=params.num_images,
                    guidance_scale=params.guidance_scale,
                    num_inference_steps=min(params.num_inference_steps, 4),  # TCD is very fast
                    eta=0.0
                )
            
            images = result.images
            
            # Cache result
            await redis.setex(
                cache_key,
                3600,  # 1 hour
                self._encode_images_for_cache(images)
            )
            
            logger.info(f"Generated {len(images)} images with TCD")
            return images
            
        except Exception as e:
            logger.exception("Error in TCD generation")
            raise
    
    async def generate_with_custom_scheduler(
        self,
        params: GenerationParams,
        scheduler_type: str,
        model_name: str = "runwayml/stable-diffusion-v1-5"
    ) -> List[Image.Image]:
        """Generate images with custom scheduler for specific use cases."""
        try:
            # Check cache
            cache_key = self._get_cache_key(params, model_name, f"scheduler_{scheduler_type}")
            redis = await self.redis_client
            cached_result = await redis.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for custom scheduler generation: {cache_key}")
                return self._decode_cached_images(cached_result)
            
            # Get pipeline
            pipeline = self.model_manager.get_text_to_image_pipeline(model_name)
            
            # Create and set custom scheduler
            scheduler = DiffusionSchedulerFactory.create_scheduler(scheduler_type)
            pipeline.scheduler = scheduler
            
            # Set seed if provided
            if params.seed is not None:
                torch.manual_seed(params.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(params.seed)
            
            # Generate images
            with torch.no_grad():
                result = pipeline(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    width=params.width,
                    height=params.height,
                    num_images_per_prompt=params.num_images,
                    guidance_scale=params.guidance_scale,
                    num_inference_steps=params.num_inference_steps,
                    eta=0.0
                )
            
            images = result.images
            
            # Cache result
            await redis.setex(
                cache_key,
                3600,  # 1 hour
                self._encode_images_for_cache(images)
            )
            
            logger.info(f"Generated {len(images)} images with {scheduler_type} scheduler")
            return images
            
        except Exception as e:
            logger.exception("Error in custom scheduler generation")
            raise
    
    async def generate_with_advanced_options(
        self,
        params: GenerationParams,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        scheduler_type: str = None,
        use_lora: bool = False,
        lora_path: str = None,
        use_textual_inversion: bool = False,
        textual_inversion_path: str = None
    ) -> List[Image.Image]:
        """Generate images with advanced Diffusers options."""
        try:
            # Check cache
            cache_key = self._get_cache_key(params, model_name, "advanced")
            redis = await self.redis_client
            cached_result = await redis.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for advanced generation: {cache_key}")
                return self._decode_cached_images(cached_result)
            
            # Get pipeline
            pipeline = self.model_manager.get_text_to_image_pipeline(model_name)
            
            # Apply custom scheduler if specified
            if scheduler_type:
                scheduler = DiffusionSchedulerFactory.create_scheduler(scheduler_type)
                pipeline.scheduler = scheduler
            
            # Load LoRA if specified
            if use_lora and lora_path:
                pipeline.load_lora_weights(lora_path)
                logger.info(f"Loaded LoRA from {lora_path}")
            
            # Load Textual Inversion if specified
            if use_textual_inversion and textual_inversion_path:
                pipeline.load_textual_inversion(textual_inversion_path)
                logger.info(f"Loaded Textual Inversion from {textual_inversion_path}")
            
            # Set seed if provided
            if params.seed is not None:
                torch.manual_seed(params.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(params.seed)
            
            # Generate images
            with torch.no_grad():
                result = pipeline(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    width=params.width,
                    height=params.height,
                    num_images_per_prompt=params.num_images,
                    guidance_scale=params.guidance_scale,
                    num_inference_steps=params.num_inference_steps,
                    eta=0.0
                )
            
            images = result.images
            
            # Cache result
            await redis.setex(
                cache_key,
                3600,  # 1 hour
                self._encode_images_for_cache(images)
            )
            
            logger.info(f"Generated {len(images)} images with advanced options")
            return images
            
        except Exception as e:
            logger.exception("Error in advanced generation")
            raise
    
    def _get_cache_key(self, params: GenerationParams, model_name: str, task: str, *args) -> str:
        """Generate cache key for generation."""
        key_data = {
            'prompt': params.prompt,
            'negative_prompt': params.negative_prompt,
            'width': params.width,
            'height': params.height,
            'guidance_scale': params.guidance_scale,
            'num_inference_steps': params.num_inference_steps,
            'seed': params.seed,
            'model_name': model_name,
            'task': task
        }
        
        # Add additional args to key
        for i, arg in enumerate(args):
            if isinstance(arg, Image.Image):
                # Hash image data
                buffer = BytesIO()
                arg.save(buffer, format='PNG')
                key_data[f'arg_{i}'] = hashlib.md5(buffer.getvalue()).hexdigest()
            else:
                key_data[f'arg_{i}'] = str(arg)
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _encode_images_for_cache(self, images: List[Image.Image]) -> str:
        """Encode images for caching."""
        encoded_images = []
        for image in images:
            encoded = self.image_processor.image_to_base64(image)
            encoded_images.append(encoded)
        return json.dumps(encoded_images)
    
    def _decode_cached_images(self, cached_data: str) -> List[Image.Image]:
        """Decode cached images."""
        encoded_images = json.loads(cached_data)
        images = []
        for encoded in encoded_images:
            image = self.image_processor.load_image_from_base64(encoded)
            images.append(image)
        return images
    
    async def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        redis = await self.redis_client
        
        # Get cache statistics
        cache_keys = await redis.keys("diffusion:*")
        cache_hits = await redis.get("diffusion:cache_hits") or 0
        cache_misses = await redis.get("diffusion:cache_misses") or 0
        
        return {
            'total_cache_entries': len(cache_keys),
            'cache_hits': int(cache_hits),
            'cache_misses': int(cache_misses),
            'cache_hit_rate': int(cache_hits) / (int(cache_hits) + int(cache_misses)) if (int(cache_hits) + int(cache_misses)) > 0 else 0,
            'loaded_models': len(self.model_manager._pipelines),
            'device': str(self.model_manager.device)
        }
    
    async def cleanup_cache(self, max_age_hours: int = 24) -> Dict[str, int]:
        """Clean up old cache entries."""
        redis = await self.redis_client
        
        # This is a simplified cleanup - in production you might want more sophisticated logic
        deleted_count = 0
        
        # Get all diffusion cache keys
        cache_keys = await redis.keys("diffusion:*")
        
        for key in cache_keys:
            # Check if key is older than max_age_hours
            ttl = await redis.ttl(key)
            if ttl == -1 or ttl > max_age_hours * 3600:
                await redis.delete(key)
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} cache entries")
        return {'deleted_entries': deleted_count}
    
    async def close(self) -> Any:
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()
    
    async def analyze_diffusion_process(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        num_steps: int = 20,
        save_visualization: bool = False,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze the forward diffusion process for a given image.
        
        Args:
            image: Input image (path, PIL Image, or tensor)
            num_steps: Number of diffusion steps to analyze
            save_visualization: Whether to save visualization
            output_path: Path to save visualization
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Convert image to tensor
            if isinstance(image, str):
                pil_image = self.image_processor.load_image(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                pil_image = image
            
            # Convert PIL to tensor
            if not isinstance(pil_image, torch.Tensor):
                # Convert PIL to tensor
                img_array = np.array(pil_image)
                if len(img_array.shape) == 3:
                    img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
                img_tensor = torch.from_numpy(img_array).float() / 255.0
                img_tensor = img_tensor * 2 - 1  # Normalize to [-1, 1]
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            else:
                img_tensor = pil_image
            
            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img_tensor = img_tensor.to(device)
            
            # Analyze diffusion process
            analysis = self.diffusion_analyzer.analyze_diffusion_process(img_tensor, num_steps)
            
            # Create visualization if requested
            if save_visualization:
                images = self.diffusion_analyzer.visualize_diffusion_process(
                    img_tensor, 
                    output_path
                )
                analysis["visualization_images"] = len(images)
            
            # Add metadata
            analysis["metadata"] = {
                "num_steps": num_steps,
                "image_shape": list(img_tensor.shape),
                "device": str(device),
                "config": {
                    "num_timesteps": self.diffusion_analyzer.config.num_timesteps,
                    "beta_start": self.diffusion_analyzer.config.beta_start,
                    "beta_end": self.diffusion_analyzer.config.beta_end,
                    "beta_schedule": self.diffusion_analyzer.config.beta_schedule
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.exception("Error in diffusion process analysis")
            raise
    
    async def demonstrate_forward_diffusion(
        self,
        image: Union[str, Image.Image],
        timesteps: List[int] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Demonstrate forward diffusion process at specific timesteps.
        
        Args:
            image: Input image
            timesteps: List of timesteps to demonstrate (if None, uses default)
            save_path: Path to save demonstration images
            
        Returns:
            Dictionary containing demonstration results
        """
        try:
            if timesteps is None:
                timesteps = [0, 50, 100, 200, 300, 500, 700, 900, 999]
            
            # Convert image to tensor
            if isinstance(image, str):
                pil_image = self.image_processor.load_image(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                raise ValueError("Image must be a path or PIL Image")
            
            # Convert PIL to tensor
            img_array = np.array(pil_image)
            if len(img_array.shape) == 3:
                img_array = np.transpose(img_array, (2, 0, 1))
            img_tensor = torch.from_numpy(img_array).float() / 255.0
            img_tensor = img_tensor * 2 - 1
            img_tensor = img_tensor.unsqueeze(0)
            
            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img_tensor = img_tensor.to(device)
            
            # Demonstrate forward diffusion
            results = {}
            images = []
            
            for t in timesteps:
                # Create timestep tensor
                t_tensor = torch.tensor([t], device=device)
                
                # Forward diffusion step
                x_t, noise = self.diffusion_analyzer.forward_diffusion_step(
                    img_tensor, t_tensor
                )
                
                # Convert back to PIL
                x_t_denorm = (x_t + 1) / 2
                x_t_denorm = torch.clamp(x_t_denorm, 0, 1)
                x_t_array = (x_t_denorm.cpu().numpy() * 255).astype(np.uint8)
                
                if x_t_array.shape[1] == 1:  # Grayscale
                    pil_img = Image.fromarray(x_t_array[0, 0], mode='L')
                else:  # RGB
                    pil_img = Image.fromarray(np.transpose(x_t_array[0], (1, 2, 0)))
                
                images.append(pil_img)
                
                # Calculate statistics
                alpha_cumprod = self.diffusion_analyzer.alphas_cumprod[t].item()
                signal_power = torch.mean(img_tensor ** 2).item()
                noise_power = torch.mean(noise ** 2).item()
                snr = 10 * math.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
                
                results[f"t_{t}"] = {
                    "alpha_cumprod": alpha_cumprod,
                    "signal_power": signal_power,
                    "noise_power": noise_power,
                    "signal_to_noise_ratio": snr,
                    "image_shape": list(x_t.shape)
                }
            
            # Save demonstration if requested
            if save_path:
                # Create a grid of images
                grid_size = int(math.ceil(math.sqrt(len(images))))
                grid_img = Image.new('RGB', (grid_size * images[0].width, grid_size * images[0].height))
                
                for i, img in enumerate(images):
                    row = i // grid_size
                    col = i % grid_size
                    grid_img.paste(img, (col * img.width, row * img.height))
                
                grid_img.save(save_path)
            
            return {
                "timesteps": timesteps,
                "results": results,
                "images": images,
                "metadata": {
                    "total_timesteps": self.diffusion_analyzer.config.num_timesteps,
                    "beta_schedule": self.diffusion_analyzer.config.beta_schedule
                }
            }
            
        except Exception as e:
            logger.exception("Error in forward diffusion demonstration")
            raise
    
    async def demonstrate_reverse_diffusion(
        self,
        noise_image: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Demonstrate reverse diffusion process (denoising).
        
        Args:
            noise_image: Starting noise image
            num_steps: Number of denoising steps
            eta: Noise level for stochastic sampling
            save_path: Path to save demonstration images
            
        Returns:
            Dictionary containing demonstration results
        """
        try:
            device = noise_image.device
            batch_size = noise_image.shape[0]
            
            # Create timesteps (reverse order)
            timesteps = torch.linspace(
                self.diffusion_analyzer.config.num_timesteps - 1, 
                0, 
                num_steps, 
                dtype=torch.long, 
                device=device
            )
            
            # Initialize with noise
            x_t = noise_image
            results = {}
            images = []
            
            for i, t in enumerate(timesteps):
                # Create timestep tensor for batch
                t_batch = t.repeat(batch_size)
                
                # For demonstration, we'll use a simple noise prediction
                # In practice, this would come from a trained model
                predicted_noise = torch.randn_like(x_t) * 0.1  # Simplified prediction
                
                # Reverse diffusion step
                x_prev = self.diffusion_analyzer.reverse_diffusion_step(
                    x_t, t_batch, predicted_noise, eta
                )
                
                # Convert to PIL for visualization
                x_prev_denorm = (x_prev + 1) / 2
                x_prev_denorm = torch.clamp(x_prev_denorm, 0, 1)
                x_prev_array = (x_prev_denorm.cpu().numpy() * 255).astype(np.uint8)
                
                if x_prev_array.shape[1] == 1:  # Grayscale
                    pil_img = Image.fromarray(x_prev_array[0, 0], mode='L')
                else:  # RGB
                    pil_img = Image.fromarray(np.transpose(x_prev_array[0], (1, 2, 0)))
                
                images.append(pil_img)
                
                # Calculate statistics
                alpha_cumprod = self.diffusion_analyzer.alphas_cumprod[t].item()
                variance = torch.var(x_prev).item()
                
                results[f"step_{i}"] = {
                    "timestep": t.item(),
                    "alpha_cumprod": alpha_cumprod,
                    "variance": variance,
                    "eta": eta
                }
                
                # Update for next step
                x_t = x_prev
            
            # Save demonstration if requested
            if save_path:
                # Create a grid of images
                grid_size = int(math.ceil(math.sqrt(len(images))))
                grid_img = Image.new('RGB', (grid_size * images[0].width, grid_size * images[0].height))
                
                for i, img in enumerate(images):
                    row = i // grid_size
                    col = i % grid_size
                    grid_img.paste(img, (col * img.width, row * img.height))
                
                grid_img.save(save_path)
            
            return {
                "num_steps": num_steps,
                "eta": eta,
                "results": results,
                "images": images,
                "final_image": images[-1] if images else None
            }
            
        except Exception as e:
            logger.exception("Error in reverse diffusion demonstration")
            raise
    
    async def get_diffusion_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the diffusion process.
        
        Returns:
            Dictionary containing diffusion statistics
        """
        try:
            # Get noise schedule
            noise_schedule = self.diffusion_analyzer.get_noise_schedule()
            
            # Calculate statistics
            betas = noise_schedule["betas"]
            alphas = noise_schedule["alphas"]
            alphas_cumprod = noise_schedule["alphas_cumprod"]
            
            # Basic statistics
            beta_stats = {
                "mean": betas.mean().item(),
                "std": betas.std().item(),
                "min": betas.min().item(),
                "max": betas.max().item(),
                "start": betas[0].item(),
                "end": betas[-1].item()
            }
            
            alpha_stats = {
                "mean": alphas.mean().item(),
                "std": alphas.std().item(),
                "min": alphas.min().item(),
                "max": alphas.max().item()
            }
            
            # Signal-to-noise ratio over time
            snr_values = []
            for t in range(len(alphas_cumprod)):
                alpha_t = alphas_cumprod[t].item()
                snr = alpha_t / (1 - alpha_t) if alpha_t < 1 else float('inf')
                snr_values.append(snr)
            
            snr_stats = {
                "values": snr_values,
                "mean": np.mean(snr_values),
                "std": np.std(snr_values),
                "min": np.min(snr_values),
                "max": np.max(snr_values)
            }
            
            return {
                "config": {
                    "num_timesteps": self.diffusion_analyzer.config.num_timesteps,
                    "beta_start": self.diffusion_analyzer.config.beta_start,
                    "beta_end": self.diffusion_analyzer.config.beta_end,
                    "beta_schedule": self.diffusion_analyzer.config.beta_schedule,
                    "prediction_type": self.diffusion_analyzer.config.prediction_type
                },
                "beta_statistics": beta_stats,
                "alpha_statistics": alpha_stats,
                "signal_to_noise_statistics": snr_stats,
                "noise_schedule_keys": list(noise_schedule.keys())
            }
            
        except Exception as e:
            logger.exception("Error getting diffusion statistics")
            raise 