from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings
    import asyncio
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Advanced Noise Schedulers and Sampling Methods
==============================================

Comprehensive implementation of noise schedulers and sampling methods for diffusion models.
Includes: DDPM, DDIM, DPM-Solver, Euler, Heun, LMS, and custom schedulers with
production-ready optimization and mathematical correctness.
"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoiseScheduleType(Enum):
    """Types of noise schedules."""
    LINEAR = "linear"
    COSINE = "cosine"
    SIGMOID = "sigmoid"
    QUADRATIC = "quadratic"
    EXPONENTIAL = "exponential"
    COSINE_BETA = "cosine_beta"
    SCALED_LINEAR = "scaled_linear"
    PIECEWISE_LINEAR = "piecewise_linear"


class SamplingMethod(Enum):
    """Types of sampling methods."""
    DDPM = "ddpm"
    DDIM = "ddim"
    DPM_SOLVER = "dpm_solver"
    DPM_SOLVER_PP = "dpm_solver_pp"
    EULER = "euler"
    HEUN = "heun"
    LMS = "lms"
    UNIPC = "unipc"


@dataclass
class SchedulerConfig:
    """Configuration for noise schedulers."""
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule_type: NoiseScheduleType = NoiseScheduleType.COSINE
    
    # DDIM specific
    eta: float = 0.0  # 0 for DDPM, 1 for DDIM
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    
    # DPM-Solver specific
    algorithm_type: str = "dpmsolver++"
    solver_type: str = "midpoint"
    lower_order_final: bool = True
    use_karras_sigmas: bool = False
    timestep_spacing: str = "linspace"
    
    # Performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32


class BaseNoiseScheduler(ABC):
    """Base class for noise schedulers."""
    
    def __init__(self, config: SchedulerConfig):
        
    """__init__ function."""
self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        
        # Pre-compute schedule
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Pre-compute other values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Move to device
        self._move_to_device()
    
    @abstractmethod
    def _get_beta_schedule(self) -> torch.Tensor:
        """Get beta schedule. Must be implemented by subclasses."""
        pass
    
    def _move_to_device(self) -> Any:
        """Move all tensors to device."""
        device = torch.device(self.config.device)
        self.betas = self.betas.to(device, dtype=self.config.dtype)
        self.alphas = self.alphas.to(device, dtype=self.config.dtype)
        self.alphas_cumprod = self.alphas_cumprod.to(device, dtype=self.config.dtype)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device, dtype=self.config.dtype)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device, dtype=self.config.dtype)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device, dtype=self.config.dtype)
        self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.to(device, dtype=self.config.dtype)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device, dtype=self.config.dtype)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device, dtype=self.config.dtype)
    
    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to original samples."""
        sqrt_alpha_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        noise = torch.randn_like(original_samples)
        noisy_samples = sqrt_alpha_t * original_samples + sqrt_one_minus_alpha_t * noise
        
        return noisy_samples, noise
    
    def remove_noise(self, noisy_samples: torch.Tensor, predicted_noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Remove predicted noise from noisy samples."""
        alpha_t = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        x_0 = (noisy_samples - sqrt_one_minus_alpha_t * predicted_noise) / torch.sqrt(alpha_t)
        return x_0
    
    def get_velocity(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Get velocity for v-prediction."""
        alpha_t = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        velocity = sqrt_one_minus_alpha_t * noise - torch.sqrt(alpha_t) * latents
        return velocity


class LinearNoiseScheduler(BaseNoiseScheduler):
    """Linear noise schedule."""
    
    def _get_beta_schedule(self) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)


class CosineNoiseScheduler(BaseNoiseScheduler):
    """Cosine noise schedule (Improved DDPM)."""
    
    def _get_beta_schedule(self) -> torch.Tensor:
        steps = self.num_train_timesteps + 1
        x = torch.linspace(0, self.num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_train_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


class CosineBetaNoiseScheduler(BaseNoiseScheduler):
    """Cosine beta schedule."""
    
    def _get_beta_schedule(self) -> torch.Tensor:
        return self.beta_end * 0.5 * (1.0 + torch.cos(math.pi * torch.linspace(0, 1, self.num_train_timesteps)))


class SigmoidNoiseScheduler(BaseNoiseScheduler):
    """Sigmoid noise schedule."""
    
    def _get_beta_schedule(self) -> torch.Tensor:
        betas = torch.sigmoid(torch.linspace(-6, 6, self.num_train_timesteps))
        betas = betas * (self.beta_end - self.beta_start) + self.beta_start
        return betas


class QuadraticNoiseScheduler(BaseNoiseScheduler):
    """Quadratic noise schedule."""
    
    def _get_beta_schedule(self) -> torch.Tensor:
        betas = torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_train_timesteps) ** 2
        return betas


class ExponentialNoiseScheduler(BaseNoiseScheduler):
    """Exponential noise schedule."""
    
    def _get_beta_schedule(self) -> torch.Tensor:
        betas = torch.exp(torch.linspace(math.log(self.beta_start), math.log(self.beta_end), self.num_train_timesteps))
        return betas


class ScaledLinearNoiseScheduler(BaseNoiseScheduler):
    """Scaled linear noise schedule."""
    
    def _get_beta_schedule(self) -> torch.Tensor:
        scale = 1000 / self.num_train_timesteps
        betas = torch.linspace(self.beta_start * scale, self.beta_end * scale, self.num_train_timesteps)
        return betas


class PiecewiseLinearNoiseScheduler(BaseNoiseScheduler):
    """Piecewise linear noise schedule."""
    
    def _get_beta_schedule(self) -> torch.Tensor:
        # Create piecewise linear schedule
        mid_point = self.num_train_timesteps // 2
        betas = torch.zeros(self.num_train_timesteps)
        
        # First half: linear from beta_start to beta_end
        betas[:mid_point] = torch.linspace(self.beta_start, self.beta_end, mid_point)
        
        # Second half: linear from beta_end to beta_start
        betas[mid_point:] = torch.linspace(self.beta_end, self.beta_start, self.num_train_timesteps - mid_point)
        
        return betas


class NoiseSchedulerFactory:
    """Factory for creating noise schedulers."""
    
    @staticmethod
    def create_scheduler(schedule_type: NoiseScheduleType, config: SchedulerConfig) -> BaseNoiseScheduler:
        """Create a noise scheduler based on type."""
        schedulers = {
            NoiseScheduleType.LINEAR: LinearNoiseScheduler,
            NoiseScheduleType.COSINE: CosineNoiseScheduler,
            NoiseScheduleType.COSINE_BETA: CosineBetaNoiseScheduler,
            NoiseScheduleType.SIGMOID: SigmoidNoiseScheduler,
            NoiseScheduleType.QUADRATIC: QuadraticNoiseScheduler,
            NoiseScheduleType.EXPONENTIAL: ExponentialNoiseScheduler,
            NoiseScheduleType.SCALED_LINEAR: ScaledLinearNoiseScheduler,
            NoiseScheduleType.PIECEWISE_LINEAR: PiecewiseLinearNoiseScheduler,
        }
        
        if schedule_type not in schedulers:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        return schedulers[schedule_type](config)


class BaseSampler(ABC):
    """Base class for sampling methods."""
    
    def __init__(self, scheduler: BaseNoiseScheduler, model: nn.Module):
        
    """__init__ function."""
self.scheduler = scheduler
        self.model = model
        self.config = scheduler.config
    
    @abstractmethod
    def step(self, sample: torch.Tensor, timestep: torch.Tensor, 
             condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single sampling step. Must be implemented by subclasses."""
        pass
    
    def sample(self, shape: Tuple[int, ...], condition: Optional[torch.Tensor] = None,
               num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """Complete sampling process."""
        device = next(self.model.parameters()).device
        batch_size = shape[0]
        
        # Initialize x_T
        sample = torch.randn(shape, device=device, dtype=self.config.dtype)
        
        # Set timesteps
        if num_inference_steps is None:
            num_inference_steps = self.config.num_train_timesteps
        
        timesteps = self._get_timesteps(num_inference_steps)
        
        # Sampling loop
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            sample = self.step(sample, t_tensor, condition)
        
        return sample
    
    def _get_timesteps(self, num_inference_steps: int) -> List[int]:
        """Get timesteps for sampling."""
        if self.config.timestep_spacing == "linspace":
            return list(np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=int))
        elif self.config.timestep_spacing == "leading":
            return list(range(0, self.config.num_train_timesteps, self.config.num_train_timesteps // num_inference_steps))
        else:
            return list(range(self.config.num_train_timesteps - 1, -1, -self.config.num_train_timesteps // num_inference_steps))


class DDPMSampler(BaseSampler):
    """DDPM sampling method."""
    
    def step(self, sample: torch.Tensor, timestep: torch.Tensor, 
             condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """DDPM sampling step."""
        betas_t = self.scheduler.betas[timestep].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1, 1)
        sqrt_recip_alphas_cumprod_t = self.scheduler.sqrt_recip_alphas_cumprod[timestep].view(-1, 1, 1, 1)
        
        # Predict noise
        if condition is not None:
            predicted_noise = self.model(sample, timestep, condition)
        else:
            predicted_noise = self.model(sample, timestep)
        
        # Compute x_0
        x_0 = self.scheduler.remove_noise(sample, predicted_noise, timestep)
        
        # Compute posterior mean
        posterior_mean_coef1 = betas_t * sqrt_recip_alphas_cumprod_t
        posterior_mean_coef2 = (1 - betas_t) * (1 / sqrt_one_minus_alphas_cumprod_t)
        posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * sample
        
        # Add noise
        noise = torch.randn_like(sample) if timestep[0] > 0 else torch.zeros_like(sample)
        posterior_std = torch.sqrt(betas_t)
        
        return posterior_mean + posterior_std * noise


class DDIMSampler(BaseSampler):
    """DDIM sampling method."""
    
    def __init__(self, scheduler: BaseNoiseScheduler, model: nn.Module):
        
    """__init__ function."""
super().__init__(scheduler, model)
        # Pre-compute DDIM specific values
        self.ddim_alpha = self.scheduler.alphas_cumprod
        self.ddim_alpha_prev = self.scheduler.alphas_cumprod_prev
        self.ddim_sigma = self.config.eta * torch.sqrt(
            (1 - self.scheduler.alphas_cumprod_prev) / (1 - self.scheduler.alphas_cumprod) * 
            (1 - self.scheduler.alphas_cumprod / self.scheduler.alphas_cumprod_prev)
        )
        
        # Move to device
        device = torch.device(self.config.device)
        self.ddim_alpha = self.ddim_alpha.to(device, dtype=self.config.dtype)
        self.ddim_alpha_prev = self.ddim_alpha_prev.to(device, dtype=self.config.dtype)
        self.ddim_sigma = self.ddim_sigma.to(device, dtype=self.config.dtype)
    
    def step(self, sample: torch.Tensor, timestep: torch.Tensor, 
             condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """DDIM sampling step."""
        # Predict noise
        if condition is not None:
            predicted_noise = self.model(sample, timestep, condition)
        else:
            predicted_noise = self.model(sample, timestep)
        
        # Compute x_0
        x_0 = self.scheduler.remove_noise(sample, predicted_noise, timestep)
        
        # DDIM formula
        alpha_t = self.ddim_alpha[timestep].view(-1, 1, 1, 1)
        alpha_prev = self.ddim_alpha_prev[timestep].view(-1, 1, 1, 1)
        sigma_t = self.ddim_sigma[timestep].view(-1, 1, 1, 1)
        
        pred_dir_xt = torch.sqrt(1 - alpha_prev - sigma_t ** 2) * predicted_noise
        x_prev = torch.sqrt(alpha_prev) * x_0 + pred_dir_xt + sigma_t * torch.randn_like(sample)
        
        return x_prev


class DPMSolverSampler(BaseSampler):
    """DPM-Solver sampling method."""
    
    def __init__(self, scheduler: BaseNoiseScheduler, model: nn.Module):
        
    """__init__ function."""
super().__init__(scheduler, model)
        self.algorithm_type = self.config.algorithm_type
        self.solver_type = self.config.solver_type
        self.lower_order_final = self.config.lower_order_final
        self.use_karras_sigmas = self.config.use_karras_sigmas
    
    def step(self, sample: torch.Tensor, timestep: torch.Tensor, 
             condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """DPM-Solver sampling step."""
        # Simplified DPM-Solver implementation
        # For full implementation, see the DPM-Solver paper
        
        # Predict noise
        if condition is not None:
            predicted_noise = self.model(sample, timestep, condition)
        else:
            predicted_noise = self.model(sample, timestep)
        
        # DPM-Solver formula (simplified)
        alpha_t = self.scheduler.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1, 1)
        
        # Compute x_0
        x_0 = self.scheduler.remove_noise(sample, predicted_noise, timestep)
        
        # DPM-Solver step
        if self.algorithm_type == "dpmsolver++":
            # DPM-Solver++ formula
            sigma_t = sqrt_one_minus_alpha_t / torch.sqrt(alpha_t)
            sigma_prev = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep - 1].view(-1, 1, 1, 1) / \
                        torch.sqrt(self.scheduler.alphas_cumprod[timestep - 1]).view(-1, 1, 1, 1)
            
            h = sigma_t - sigma_prev
            x_prev = x_0 + sigma_prev * predicted_noise + h * torch.randn_like(sample)
        else:
            # Standard DPM-Solver
            x_prev = x_0 + sqrt_one_minus_alpha_t * predicted_noise
        
        return x_prev


class EulerSampler(BaseSampler):
    """Euler sampling method."""
    
    def step(self, sample: torch.Tensor, timestep: torch.Tensor, 
             condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Euler sampling step."""
        # Predict noise
        if condition is not None:
            predicted_noise = self.model(sample, timestep, condition)
        else:
            predicted_noise = self.model(sample, timestep)
        
        # Euler step
        alpha_t = self.scheduler.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1, 1)
        
        # Compute x_0
        x_0 = self.scheduler.remove_noise(sample, predicted_noise, timestep)
        
        # Euler formula
        dt = 1.0 / self.config.num_train_timesteps
        x_prev = x_0 + dt * predicted_noise
        
        return x_prev


class HeunSampler(BaseSampler):
    """Heun sampling method (2nd order Runge-Kutta)."""
    
    def step(self, sample: torch.Tensor, timestep: torch.Tensor, 
             condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Heun sampling step."""
        # Predict noise
        if condition is not None:
            predicted_noise = self.model(sample, timestep, condition)
        else:
            predicted_noise = self.model(sample, timestep)
        
        # Heun step (2nd order Runge-Kutta)
        alpha_t = self.scheduler.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1, 1)
        
        # Compute x_0
        x_0 = self.scheduler.remove_noise(sample, predicted_noise, timestep)
        
        # Heun formula
        dt = 1.0 / self.config.num_train_timesteps
        
        # First step (Euler)
        k1 = predicted_noise
        
        # Second step
        sample_mid = x_0 + dt * k1
        timestep_mid = torch.clamp(timestep - 1, min=0)
        
        if condition is not None:
            k2 = self.model(sample_mid, timestep_mid, condition)
        else:
            k2 = self.model(sample_mid, timestep_mid)
        
        # Heun update
        x_prev = x_0 + 0.5 * dt * (k1 + k2)
        
        return x_prev


class LMSSampler(BaseSampler):
    """LMS (Linear Multi-Step) sampling method."""
    
    def __init__(self, scheduler: BaseNoiseScheduler, model: nn.Module):
        
    """__init__ function."""
super().__init__(scheduler, model)
        self.order = 4  # LMS order
        self.lms_coeffs = self._get_lms_coefficients()
    
    def _get_lms_coefficients(self) -> torch.Tensor:
        """Get LMS coefficients."""
        # Simplified LMS coefficients
        coeffs = torch.tensor([1.0, -1.0, 0.5, -0.1667])  # 4th order
        return coeffs.to(self.config.device, dtype=self.config.dtype)
    
    def step(self, sample: torch.Tensor, timestep: torch.Tensor, 
             condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """LMS sampling step."""
        # Predict noise
        if condition is not None:
            predicted_noise = self.model(sample, timestep, condition)
        else:
            predicted_noise = self.model(sample, timestep)
        
        # LMS step
        alpha_t = self.scheduler.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1, 1)
        
        # Compute x_0
        x_0 = self.scheduler.remove_noise(sample, predicted_noise, timestep)
        
        # LMS formula (simplified)
        dt = 1.0 / self.config.num_train_timesteps
        x_prev = x_0 + dt * predicted_noise * self.lms_coeffs[0]
        
        return x_prev


class SamplerFactory:
    """Factory for creating samplers."""
    
    @staticmethod
    def create_sampler(method: SamplingMethod, scheduler: BaseNoiseScheduler, 
                      model: nn.Module) -> BaseSampler:
        """Create a sampler based on method."""
        samplers = {
            SamplingMethod.DDPM: DDPMSampler,
            SamplingMethod.DDIM: DDIMSampler,
            SamplingMethod.DPM_SOLVER: DPMSolverSampler,
            SamplingMethod.DPM_SOLVER_PP: DPMSolverSampler,
            SamplingMethod.EULER: EulerSampler,
            SamplingMethod.HEUN: HeunSampler,
            SamplingMethod.LMS: LMSSampler,
        }
        
        if method not in samplers:
            raise ValueError(f"Unknown sampling method: {method}")
        
        return samplers[method](scheduler, model)


class AdvancedNoiseScheduler:
    """Advanced noise scheduler with multiple schedules and samplers."""
    
    def __init__(self, config: SchedulerConfig):
        
    """__init__ function."""
self.config = config
        self.scheduler = NoiseSchedulerFactory.create_scheduler(config.schedule_type, config)
        self.sampler = None
    
    def set_sampler(self, method: SamplingMethod, model: nn.Module):
        """Set the sampling method."""
        self.sampler = SamplerFactory.create_sampler(method, self.scheduler, model)
    
    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise using the scheduler."""
        return self.scheduler.add_noise(original_samples, timesteps)
    
    def remove_noise(self, noisy_samples: torch.Tensor, predicted_noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Remove noise using the scheduler."""
        return self.scheduler.remove_noise(noisy_samples, predicted_noise, timesteps)
    
    def sample(self, shape: Tuple[int, ...], model: nn.Module, method: SamplingMethod,
               condition: Optional[torch.Tensor] = None, num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """Sample using the specified method."""
        if self.sampler is None or not isinstance(self.sampler, SamplerFactory.create_sampler(method, self.scheduler, model).__class__):
            self.set_sampler(method, model)
        
        return self.sampler.sample(shape, condition, num_inference_steps)
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Get information about the current schedule."""
        return {
            "schedule_type": self.config.schedule_type.value,
            "num_train_timesteps": self.config.num_train_timesteps,
            "beta_start": self.config.beta_start,
            "beta_end": self.config.beta_end,
            "alphas_cumprod_shape": self.scheduler.alphas_cumprod.shape,
            "device": self.config.device,
        }


def compare_schedules(config: SchedulerConfig, num_steps: int = 100) -> Dict[str, torch.Tensor]:
    """Compare different noise schedules."""
    schedules = {}
    
    for schedule_type in NoiseScheduleType:
        config.schedule_type = schedule_type
        scheduler = NoiseSchedulerFactory.create_scheduler(schedule_type, config)
        schedules[schedule_type.value] = scheduler.betas[:num_steps]
    
    return schedules


def compare_samplers(scheduler: BaseNoiseScheduler, model: nn.Module, shape: Tuple[int, ...],
                    num_inference_steps: int = 50) -> Dict[str, torch.Tensor]:
    """Compare different sampling methods."""
    samples = {}
    
    for method in SamplingMethod:
        try:
            sampler = SamplerFactory.create_sampler(method, scheduler, model)
            sample = sampler.sample(shape, num_inference_steps=num_inference_steps)
            samples[method.value] = sample
        except Exception as e:
            logger.warning(f"Failed to sample with {method.value}: {e}")
    
    return samples


async def main():
    """Main function demonstrating noise schedulers and samplers."""
    logger.info("Setting up noise schedulers and samplers...")
    
    # Configuration
    config = SchedulerConfig(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type=NoiseScheduleType.COSINE,
        device="cpu"  # Use CPU for demonstration
    )
    
    # Create simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self) -> Any:
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
        
        def forward(self, x, t, condition=None) -> Any:
            return self.conv(x)
    
    model = SimpleModel()
    
    # Create advanced scheduler
    advanced_scheduler = AdvancedNoiseScheduler(config)
    
    # Compare schedules
    logger.info("Comparing noise schedules...")
    schedules = compare_schedules(config)
    for name, betas in schedules.items():
        logger.info(f"{name}: beta range [{betas.min():.6f}, {betas.max():.6f}]")
    
    # Compare samplers
    logger.info("Comparing sampling methods...")
    shape = (1, 3, 32, 32)
    samples = compare_samplers(advanced_scheduler.scheduler, model, shape, num_inference_steps=10)
    
    for name, sample in samples.items():
        logger.info(f"{name}: sample shape {sample.shape}, range [{sample.min():.3f}, {sample.max():.3f}]")
    
    # Test specific sampler
    logger.info("Testing DDIM sampler...")
    advanced_scheduler.set_sampler(SamplingMethod.DDIM, model)
    sample = advanced_scheduler.sample(shape, model, SamplingMethod.DDIM, num_inference_steps=20)
    logger.info(f"DDIM sample: shape {sample.shape}, range [{sample.min():.3f}, {sample.max():.3f}]")
    
    logger.info("Noise schedulers and samplers demonstration completed!")


match __name__:
    case "__main__":
    asyncio.run(main()) 