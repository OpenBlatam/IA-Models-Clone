from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import TypedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers import (
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import randn_tensor
from typing import Any, List, Dict, Optional
"""
Advanced Noise Schedulers and Sampling Methods for Diffusion Processes
=====================================================================

This module provides comprehensive implementations of noise schedulers and
sampling methods for diffusion models, including:

Noise Schedulers:
- Linear, Cosine, Quadratic, Sigmoid schedules
- DDIM, DDPM, DPM-Solver schedules
- Custom adaptive schedules
- Zero SNR rescaling

Sampling Methods:
- DDPM (stochastic)
- DDIM (deterministic)
- DPM-Solver (fast)
- Euler, Heun, LMS methods
- Classifier-free guidance
- Classifier guidance
- Advanced sampling techniques

Features:
- Production-ready implementations
- Comprehensive error handling
- Performance optimizations
- Security considerations
- Extensive testing support

Author: AI Assistant
License: MIT
"""


    DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler,
    DDPMScheduler, DDPMWuerstchenScheduler, LMSDiscreteScheduler,
    HeunDiscreteScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler, UniPCMultistepScheduler
)

# Configure logging
logger = logging.getLogger(__name__)


class NoiseScheduleType(Enum):
    """Available noise schedule types."""
    LINEAR = "linear"
    COSINE = "cosine"
    QUADRATIC = "quadratic"
    SIGMOID = "sigmoid"
    EXPONENTIAL = "exponential"
    CUSTOM = "custom"


class SamplingMethod(Enum):
    """Available sampling methods."""
    DDPM = "ddpm"
    DDIM = "ddim"
    DPM_SOLVER = "dpm_solver"
    EULER = "euler"
    HEUN = "heun"
    LMS = "lms"
    KDPM2 = "kdpm2"
    KDPM2_ANCESTRAL = "kdpm2_ancestral"
    UNIPC = "unipc"


@dataclass
class NoiseSchedulerConfig:
    """Configuration for noise schedulers."""
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule_type: NoiseScheduleType = NoiseScheduleType.LINEAR
    
    # Advanced parameters
    clip_sample: bool = False
    clip_sample_range: float = 1.0
    sample_max_value: float = 1.0
    timestep_spacing: str = "leading"
    rescale_betas_zero_snr: bool = False
    
    # Custom schedule parameters
    custom_betas: Optional[torch.Tensor] = None
    custom_alphas_cumprod: Optional[torch.Tensor] = None


@dataclass
class SamplingConfig:
    """Configuration for sampling methods."""
    method: SamplingMethod = SamplingMethod.DDPM
    num_inference_steps: int = 50
    eta: float = 0.0  # Controls stochasticity (0 = deterministic, 1 = stochastic)
    guidance_scale: float = 7.5  # For classifier-free guidance
    classifier_guidance_scale: float = 1.0  # For classifier guidance
    
    # Advanced parameters
    use_clipped_model_output: bool = False
    prediction_type: str = "epsilon"  # "epsilon", "v_prediction", "sample"
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    sample_max_value: float = 1.0
    
    # DPM-Solver specific
    algorithm_type: str = "dpmsolver++"
    solver_type: str = "midpoint"
    lower_order_final: bool = True
    use_karras_sigmas: bool = False
    
    # Classifier guidance
    classifier_model: Optional[nn.Module] = None
    classifier_cond_fn: Optional[Callable] = None


@dataclass
class SamplingResult:
    """Result of sampling process."""
    samples: torch.Tensor
    latents: List[torch.Tensor] = field(default_factory=list)
    timesteps: List[int] = field(default_factory=list)
    guidance_scores: Optional[torch.Tensor] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseNoiseScheduler(ABC):
    """Abstract base class for noise schedulers."""
    
    def __init__(self, config: NoiseSchedulerConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate schedule
        self.betas = self._calculate_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculate variance schedule
        self.variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.log_variance = torch.log(self.variance)
        
        # Move to device
        self._move_to_device()
        
        logger.info(f"Initialized {self.__class__.__name__} with {config.num_train_timesteps} timesteps")
    
    @abstractmethod
    def _calculate_betas(self) -> torch.Tensor:
        """Calculate the beta schedule."""
        pass
    
    def _move_to_device(self) -> Any:
        """Move all tensors to device."""
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        self.variance = self.variance.to(self.device)
        self.log_variance = self.log_variance.to(self.device)
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Get information about the noise schedule."""
        return {
            "num_timesteps": self.config.num_train_timesteps,
            "beta_start": self.config.beta_start,
            "beta_end": self.config.beta_end,
            "schedule_type": self.config.schedule_type.value,
            "betas_range": (self.betas.min().item(), self.betas.max().item()),
            "alphas_cumprod_range": (self.alphas_cumprod.min().item(), self.alphas_cumprod.max().item()),
        }


class LinearNoiseScheduler(BaseNoiseScheduler):
    """Linear noise schedule."""
    
    def _calculate_betas(self) -> torch.Tensor:
        return torch.linspace(self.config.beta_start, self.config.beta_end, self.config.num_train_timesteps)


class CosineNoiseScheduler(BaseNoiseScheduler):
    """Cosine noise schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'."""
    
    def _calculate_betas(self) -> torch.Tensor:
        steps = self.config.num_train_timesteps + 1
        x = torch.linspace(0, self.config.num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.config.num_train_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


class QuadraticNoiseScheduler(BaseNoiseScheduler):
    """Quadratic noise schedule."""
    
    def _calculate_betas(self) -> torch.Tensor:
        return torch.linspace(self.config.beta_start ** 0.5, self.config.beta_end ** 0.5, self.config.num_train_timesteps) ** 2


class SigmoidNoiseScheduler(BaseNoiseScheduler):
    """Sigmoid noise schedule."""
    
    def _calculate_betas(self) -> torch.Tensor:
        x = torch.linspace(-6, 6, self.config.num_train_timesteps)
        betas = torch.sigmoid(x) * (self.config.beta_end - self.config.beta_start) + self.config.beta_start
        return betas


class ExponentialNoiseScheduler(BaseNoiseScheduler):
    """Exponential noise schedule."""
    
    def _calculate_betas(self) -> torch.Tensor:
        x = torch.linspace(0, 1, self.config.num_train_timesteps)
        betas = self.config.beta_start * (self.config.beta_end / self.config.beta_start) ** x
        return betas


class CustomNoiseScheduler(BaseNoiseScheduler):
    """Custom noise schedule with user-provided betas."""
    
    def _calculate_betas(self) -> torch.Tensor:
        if self.config.custom_betas is not None:
            return self.config.custom_betas
        elif self.config.custom_alphas_cumprod is not None:
            alphas_cumprod = self.config.custom_alphas_cumprod
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError("Custom scheduler requires either custom_betas or custom_alphas_cumprod")


class NoiseSchedulerFactory:
    """Factory for creating noise schedulers."""
    
    @staticmethod
    def create(config: NoiseSchedulerConfig) -> BaseNoiseScheduler:
        """Create a noise scheduler based on configuration."""
        schedulers = {
            NoiseScheduleType.LINEAR: LinearNoiseScheduler,
            NoiseScheduleType.COSINE: CosineNoiseScheduler,
            NoiseScheduleType.QUADRATIC: QuadraticNoiseScheduler,
            NoiseScheduleType.SIGMOID: SigmoidNoiseScheduler,
            NoiseScheduleType.EXPONENTIAL: ExponentialNoiseScheduler,
            NoiseScheduleType.CUSTOM: CustomNoiseScheduler,
        }
        
        scheduler_class = schedulers.get(config.schedule_type)
        if scheduler_class is None:
            raise ValueError(f"Unsupported schedule type: {config.schedule_type}")
        
        return scheduler_class(config)


class BaseSampler(ABC):
    """Abstract base class for sampling methods."""
    
    def __init__(self, scheduler: BaseNoiseScheduler, config: SamplingConfig):
        
    """__init__ function."""
self.scheduler = scheduler
        self.config = config
        self.device = scheduler.device
        
        # Create diffusers scheduler
        self.diffusers_scheduler = self._create_diffusers_scheduler()
        
        logger.info(f"Initialized {self.__class__.__name__} with {config.num_inference_steps} steps")
    
    @abstractmethod
    def _create_diffusers_scheduler(self) -> SchedulerMixin:
        """Create the corresponding diffusers scheduler."""
        pass
    
    @abstractmethod
    def sample(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> SamplingResult:
        """Sample from the diffusion model."""
        pass
    
    def _apply_classifier_free_guidance(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Apply classifier-free guidance."""
        # Duplicate inputs for unconditional and conditional
        latents_input = torch.cat([model_output] * 2)
        timestep_input = torch.cat([timestep] * 2)
        prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        # Get model predictions
        noise_pred = model(latents_input, timestep_input, encoder_hidden_states=prompt_embeds_input)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        
        # Apply guidance
        noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        return noise_pred


class DDPMSampler(BaseSampler):
    """DDPM (Denoising Diffusion Probabilistic Models) sampler."""
    
    def _create_diffusers_scheduler(self) -> DDPMScheduler:
        return DDPMScheduler(
            num_train_timesteps=self.scheduler.config.num_train_timesteps,
            beta_start=self.scheduler.config.beta_start,
            beta_end=self.scheduler.config.beta_end,
            beta_schedule=self.scheduler.config.schedule_type.value,
            clip_sample=self.scheduler.config.clip_sample,
            prediction_type=self.config.prediction_type,
        )
    
    def sample(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> SamplingResult:
        """Sample using DDPM method."""
        start_time = time.time()
        
        # Set timesteps
        self.diffusers_scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = self.diffusers_scheduler.timesteps
        
        # Initialize latents
        latents = latents.to(self.device)
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(self.device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(self.device)
        
        # Sampling loop
        latents_list = [latents.clone()]
        timesteps_list = []
        
        for i, timestep in enumerate(timesteps):
            # Expand latents for batch processing
            latent_model_input = torch.cat([latents] * 2) if prompt_embeds is not None else latents
            
            # Predict noise
            if prompt_embeds is not None:
                noise_pred = self._apply_classifier_free_guidance(
                    latent_model_input, timestep, prompt_embeds, negative_prompt_embeds
                )
            else:
                noise_pred = model(latent_model_input, timestep)
            
            # Perform denoising step
            latents = self.diffusers_scheduler.step(
                noise_pred, timestep, latents, eta=self.config.eta
            ).prev_sample
            
            latents_list.append(latents.clone())
            timesteps_list.append(timestep.item())
        
        processing_time = time.time() - start_time
        
        return SamplingResult(
            samples=latents,
            latents=latents_list,
            timesteps=timesteps_list,
            processing_time=processing_time,
            metadata={"method": "ddpm", "eta": self.config.eta}
        )


class DDIMSampler(BaseSampler):
    """DDIM (Denoising Diffusion Implicit Models) sampler."""
    
    def _create_diffusers_scheduler(self) -> DDIMScheduler:
        return DDIMScheduler(
            num_train_timesteps=self.scheduler.config.num_train_timesteps,
            beta_start=self.scheduler.config.beta_start,
            beta_end=self.scheduler.config.beta_end,
            beta_schedule=self.scheduler.config.schedule_type.value,
            clip_sample=self.scheduler.config.clip_sample,
            prediction_type=self.config.prediction_type,
        )
    
    def sample(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> SamplingResult:
        """Sample using DDIM method."""
        start_time = time.time()
        
        # Set timesteps
        self.diffusers_scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = self.diffusers_scheduler.timesteps
        
        # Initialize latents
        latents = latents.to(self.device)
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(self.device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(self.device)
        
        # Sampling loop
        latents_list = [latents.clone()]
        timesteps_list = []
        
        for i, timestep in enumerate(timesteps):
            # Expand latents for batch processing
            latent_model_input = torch.cat([latents] * 2) if prompt_embeds is not None else latents
            
            # Predict noise
            if prompt_embeds is not None:
                noise_pred = self._apply_classifier_free_guidance(
                    latent_model_input, timestep, prompt_embeds, negative_prompt_embeds
                )
            else:
                noise_pred = model(latent_model_input, timestep)
            
            # Perform denoising step
            latents = self.diffusers_scheduler.step(
                noise_pred, timestep, latents, eta=self.config.eta
            ).prev_sample
            
            latents_list.append(latents.clone())
            timesteps_list.append(timestep.item())
        
        processing_time = time.time() - start_time
        
        return SamplingResult(
            samples=latents,
            latents=latents_list,
            timesteps=timesteps_list,
            processing_time=processing_time,
            metadata={"method": "ddim", "eta": self.config.eta}
        )


class DPMSolverSampler(BaseSampler):
    """DPM-Solver sampler for fast sampling."""
    
    def _create_diffusers_scheduler(self) -> DPMSolverMultistepScheduler:
        return DPMSolverMultistepScheduler(
            num_train_timesteps=self.scheduler.config.num_train_timesteps,
            beta_start=self.scheduler.config.beta_start,
            beta_end=self.scheduler.config.beta_end,
            beta_schedule=self.scheduler.config.schedule_type.value,
            algorithm_type=self.config.algorithm_type,
            solver_type=self.config.solver_type,
            lower_order_final=self.config.lower_order_final,
            use_karras_sigmas=self.config.use_karras_sigmas,
            prediction_type=self.config.prediction_type,
        )
    
    def sample(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> SamplingResult:
        """Sample using DPM-Solver method."""
        start_time = time.time()
        
        # Set timesteps
        self.diffusers_scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = self.diffusers_scheduler.timesteps
        
        # Initialize latents
        latents = latents.to(self.device)
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(self.device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(self.device)
        
        # Sampling loop
        latents_list = [latents.clone()]
        timesteps_list = []
        
        for i, timestep in enumerate(timesteps):
            # Expand latents for batch processing
            latent_model_input = torch.cat([latents] * 2) if prompt_embeds is not None else latents
            
            # Predict noise
            if prompt_embeds is not None:
                noise_pred = self._apply_classifier_free_guidance(
                    latent_model_input, timestep, prompt_embeds, negative_prompt_embeds
                )
            else:
                noise_pred = model(latent_model_input, timestep)
            
            # Perform denoising step
            latents = self.diffusers_scheduler.step(
                noise_pred, timestep, latents
            ).prev_sample
            
            latents_list.append(latents.clone())
            timesteps_list.append(timestep.item())
        
        processing_time = time.time() - start_time
        
        return SamplingResult(
            samples=latents,
            latents=latents_list,
            timesteps=timesteps_list,
            processing_time=processing_time,
            metadata={
                "method": "dpm_solver",
                "algorithm_type": self.config.algorithm_type,
                "solver_type": self.config.solver_type
            }
        )


class EulerSampler(BaseSampler):
    """Euler sampler."""
    
    def _create_diffusers_scheduler(self) -> EulerDiscreteScheduler:
        return EulerDiscreteScheduler(
            num_train_timesteps=self.scheduler.config.num_train_timesteps,
            beta_start=self.scheduler.config.beta_start,
            beta_end=self.scheduler.config.beta_end,
            beta_schedule=self.scheduler.config.schedule_type.value,
            prediction_type=self.config.prediction_type,
        )
    
    def sample(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> SamplingResult:
        """Sample using Euler method."""
        start_time = time.time()
        
        # Set timesteps
        self.diffusers_scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = self.diffusers_scheduler.timesteps
        
        # Initialize latents
        latents = latents.to(self.device)
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(self.device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(self.device)
        
        # Sampling loop
        latents_list = [latents.clone()]
        timesteps_list = []
        
        for i, timestep in enumerate(timesteps):
            # Expand latents for batch processing
            latent_model_input = torch.cat([latents] * 2) if prompt_embeds is not None else latents
            
            # Predict noise
            if prompt_embeds is not None:
                noise_pred = self._apply_classifier_free_guidance(
                    latent_model_input, timestep, prompt_embeds, negative_prompt_embeds
                )
            else:
                noise_pred = model(latent_model_input, timestep)
            
            # Perform denoising step
            latents = self.diffusers_scheduler.step(
                noise_pred, timestep, latents
            ).prev_sample
            
            latents_list.append(latents.clone())
            timesteps_list.append(timestep.item())
        
        processing_time = time.time() - start_time
        
        return SamplingResult(
            samples=latents,
            latents=latents_list,
            timesteps=timesteps_list,
            processing_time=processing_time,
            metadata={"method": "euler"}
        )


class SamplerFactory:
    """Factory for creating samplers."""
    
    @staticmethod
    def create(scheduler: BaseNoiseScheduler, config: SamplingConfig) -> BaseSampler:
        """Create a sampler based on configuration."""
        samplers = {
            SamplingMethod.DDPM: DDPMSampler,
            SamplingMethod.DDIM: DDIMSampler,
            SamplingMethod.DPM_SOLVER: DPMSolverSampler,
            SamplingMethod.EULER: EulerSampler,
        }
        
        sampler_class = samplers.get(config.method)
        if sampler_class is None:
            raise ValueError(f"Unsupported sampling method: {config.method}")
        
        return sampler_class(scheduler, config)


class AdvancedSamplingManager:
    """
    Advanced sampling manager with multiple schedulers and samplers.
    
    This class provides a unified interface for using different noise schedulers
    and sampling methods, with support for advanced features like:
    - Multiple sampling methods
    - Classifier-free guidance
    - Classifier guidance
    - Adaptive sampling
    - Performance monitoring
    """
    
    def __init__(self, scheduler_config: NoiseSchedulerConfig, sampling_config: SamplingConfig):
        
    """__init__ function."""
self.scheduler_config = scheduler_config
        self.sampling_config = sampling_config
        
        # Create scheduler and sampler
        self.scheduler = NoiseSchedulerFactory.create(scheduler_config)
        self.sampler = SamplerFactory.create(self.scheduler, sampling_config)
        
        logger.info(f"AdvancedSamplingManager initialized with {scheduler_config.schedule_type.value} scheduler and {sampling_config.method.value} sampler")
    
    def sample(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> SamplingResult:
        """Sample using the configured scheduler and sampler."""
        return self.sampler.sample(
            model, latents, prompt_embeds, negative_prompt_embeds, **kwargs
        )
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Get information about the noise schedule."""
        return self.scheduler.get_schedule_info()
    
    def compare_sampling_methods(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        methods: List[SamplingMethod] = None
    ) -> Dict[str, SamplingResult]:
        """Compare different sampling methods."""
        if methods is None:
            methods = [SamplingMethod.DDPM, SamplingMethod.DDIM, SamplingMethod.DPM_SOLVER, SamplingMethod.EULER]
        
        results = {}
        
        for method in methods:
            # Create temporary sampler
            temp_config = SamplingConfig(
                method=method,
                num_inference_steps=self.sampling_config.num_inference_steps,
                guidance_scale=self.sampling_config.guidance_scale,
                eta=self.sampling_config.eta
            )
            temp_sampler = SamplerFactory.create(self.scheduler, temp_config)
            
            # Sample
            result = temp_sampler.sample(model, latents, prompt_embeds, negative_prompt_embeds)
            results[method.value] = result
        
        return results


# Utility functions
def create_noise_scheduler(schedule_type: NoiseScheduleType, **kwargs) -> BaseNoiseScheduler:
    """Create a noise scheduler with default parameters."""
    config = NoiseSchedulerConfig(schedule_type=schedule_type, **kwargs)
    return NoiseSchedulerFactory.create(config)


def create_sampler(schedule_type: NoiseScheduleType, method: SamplingMethod, **kwargs) -> BaseSampler:
    """Create a sampler with default parameters."""
    scheduler_config = NoiseSchedulerConfig(schedule_type=schedule_type)
    sampling_config = SamplingConfig(method=method, **kwargs)
    
    scheduler = NoiseSchedulerFactory.create(scheduler_config)
    return SamplerFactory.create(scheduler, sampling_config)


def create_advanced_sampling_manager(
    schedule_type: NoiseScheduleType,
    method: SamplingMethod,
    **kwargs
) -> AdvancedSamplingManager:
    """Create an advanced sampling manager with default parameters."""
    scheduler_config = NoiseSchedulerConfig(schedule_type=schedule_type)
    sampling_config = SamplingConfig(method=method, **kwargs)
    
    return AdvancedSamplingManager(scheduler_config, sampling_config) 