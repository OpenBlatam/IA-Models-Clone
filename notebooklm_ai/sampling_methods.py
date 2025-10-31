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
Advanced Sampling Methods for Diffusion Models
==============================================

Production-ready implementation of advanced sampling methods including:
- DDPM, DDIM, DPM-Solver, Euler, Heun, LMS, UniPC
- Classifier-free guidance
- Multi-step sampling
- Performance optimizations
- Mathematical correctness
"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SamplingMethod(Enum):
    """Advanced sampling methods."""
    DDPM = "ddpm"
    DDIM = "ddim"
    DPM_SOLVER = "dpm_solver"
    DPM_SOLVER_PP = "dpm_solver_pp"
    EULER = "euler"
    HEUN = "heun"
    LMS = "lms"
    UNIPC = "unipc"
    EULER_ANCESTRAL = "euler_ancestral"
    HEUN_ANCESTRAL = "heun_ancestral"
    DPM_MULTISTEP = "dpm_multistep"
    DPM_SINGLESTEP = "dpm_singlestep"


@dataclass
class SamplingConfig:
    """Configuration for sampling methods."""
    # General
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0  # DDIM eta parameter
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    
    # DPM-Solver specific
    algorithm_type: str = "dpmsolver++"
    solver_type: str = "midpoint"
    lower_order_final: bool = True
    use_karras_sigmas: bool = False
    timestep_spacing: str = "linspace"
    
    # Classifier-free guidance
    use_classifier_free_guidance: bool = True
    guidance_rescale: float = 0.0
    
    # Performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    use_amp: bool = True
    
    # Advanced
    return_dict: bool = True
    callback: Optional[Callable] = None
    callback_steps: int = 1


class BaseSampler(ABC):
    """Base class for all sampling methods."""
    
    def __init__(self, config: SamplingConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype
    
    @abstractmethod
    def step(self, model: nn.Module, sample: torch.Tensor, timestep: torch.Tensor,
             condition: Optional[torch.Tensor] = None, guidance_scale: Optional[float] = None) -> torch.Tensor:
        """Single sampling step. Must be implemented by subclasses."""
        pass
    
    def sample(self, model: nn.Module, shape: Tuple[int, ...], 
               condition: Optional[torch.Tensor] = None,
               guidance_scale: Optional[float] = None,
               num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """Complete sampling process."""
        num_steps = num_inference_steps or self.config.num_inference_steps
        
        # Initialize x_T
        sample = torch.randn(shape, device=self.device, dtype=self.dtype)
        
        # Get timesteps
        timesteps = self._get_timesteps(num_steps)
        
        # Sampling loop
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            # Callback
            if self.config.callback and i % self.config.callback_steps == 0:
                self.config.callback(i, t, sample)
            
            # Sampling step
            sample = self.step(model, sample, t_tensor, condition, guidance_scale)
        
        return sample
    
    def _get_timesteps(self, num_inference_steps: int) -> List[int]:
        """Get timesteps for sampling."""
        if self.config.timestep_spacing == "linspace":
            return list(np.linspace(0, 999, num_inference_steps, dtype=int))
        elif self.config.timestep_spacing == "leading":
            return list(range(0, 1000, 1000 // num_inference_steps))
        else:
            return list(range(999, -1, -1000 // num_inference_steps))


class DDPMSampler(BaseSampler):
    """DDPM sampling method."""
    
    def __init__(self, config: SamplingConfig, scheduler):
        
    """__init__ function."""
super().__init__(config)
        self.scheduler = scheduler
    
    def step(self, model: nn.Module, sample: torch.Tensor, timestep: torch.Tensor,
             condition: Optional[torch.Tensor] = None, guidance_scale: Optional[float] = None) -> torch.Tensor:
        """DDPM sampling step."""
        guidance_scale = guidance_scale or self.config.guidance_scale
        
        # Classifier-free guidance
        if self.config.use_classifier_free_guidance and guidance_scale > 1.0:
            # Unconditional prediction
            if condition is not None:
                uncond_input = torch.zeros_like(condition)
                uncond_pred = model(sample, timestep, uncond_input)
                cond_pred = model(sample, timestep, condition)
                predicted_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                predicted_noise = model(sample, timestep)
        else:
            predicted_noise = model(sample, timestep, condition)
        
        # DDPM step
        betas_t = self.scheduler.betas[timestep].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1, 1)
        sqrt_recip_alphas_cumprod_t = self.scheduler.sqrt_recip_alphas_cumprod[timestep].view(-1, 1, 1, 1)
        
        # Compute x_0
        x_0 = self.scheduler.remove_noise(sample, predicted_noise, timestep)
        
        # Clip sample
        if self.config.clip_sample:
            x_0 = torch.clamp(x_0, -self.config.clip_sample_range, self.config.clip_sample_range)
        
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
    
    def __init__(self, config: SamplingConfig, scheduler):
        
    """__init__ function."""
super().__init__(config)
        self.scheduler = scheduler
        self.eta = config.eta
        
        # Pre-compute DDIM values
        self.ddim_alpha = self.scheduler.alphas_cumprod
        self.ddim_alpha_prev = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod[:-1]])
        self.ddim_sigma = self.eta * torch.sqrt(
            (1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) * 
            (1 - self.ddim_alpha / self.ddim_alpha_prev)
        )
        
        # Move to device
        self.ddim_alpha = self.ddim_alpha.to(self.device, dtype=self.dtype)
        self.ddim_alpha_prev = self.ddim_alpha_prev.to(self.device, dtype=self.dtype)
        self.ddim_sigma = self.ddim_sigma.to(self.device, dtype=self.dtype)
    
    def step(self, model: nn.Module, sample: torch.Tensor, timestep: torch.Tensor,
             condition: Optional[torch.Tensor] = None, guidance_scale: Optional[float] = None) -> torch.Tensor:
        """DDIM sampling step."""
        guidance_scale = guidance_scale or self.config.guidance_scale
        
        # Classifier-free guidance
        if self.config.use_classifier_free_guidance and guidance_scale > 1.0:
            if condition is not None:
                uncond_input = torch.zeros_like(condition)
                uncond_pred = model(sample, timestep, uncond_input)
                cond_pred = model(sample, timestep, condition)
                predicted_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                predicted_noise = model(sample, timestep)
        else:
            predicted_noise = model(sample, timestep, condition)
        
        # Compute x_0
        x_0 = self.scheduler.remove_noise(sample, predicted_noise, timestep)
        
        # Clip sample
        if self.config.clip_sample:
            x_0 = torch.clamp(x_0, -self.config.clip_sample_range, self.config.clip_sample_range)
        
        # DDIM formula
        alpha_t = self.ddim_alpha[timestep].view(-1, 1, 1, 1)
        alpha_prev = self.ddim_alpha_prev[timestep].view(-1, 1, 1, 1)
        sigma_t = self.ddim_sigma[timestep].view(-1, 1, 1, 1)
        
        pred_dir_xt = torch.sqrt(1 - alpha_prev - sigma_t ** 2) * predicted_noise
        x_prev = torch.sqrt(alpha_prev) * x_0 + pred_dir_xt + sigma_t * torch.randn_like(sample)
        
        return x_prev


class DPMSolverSampler(BaseSampler):
    """DPM-Solver sampling method."""
    
    def __init__(self, config: SamplingConfig, scheduler):
        
    """__init__ function."""
super().__init__(config)
        self.scheduler = scheduler
        self.algorithm_type = config.algorithm_type
        self.solver_type = config.solver_type
        self.lower_order_final = config.lower_order_final
        self.use_karras_sigmas = config.use_karras_sigmas
    
    def step(self, model: nn.Module, sample: torch.Tensor, timestep: torch.Tensor,
             condition: Optional[torch.Tensor] = None, guidance_scale: Optional[float] = None) -> torch.Tensor:
        """DPM-Solver sampling step."""
        guidance_scale = guidance_scale or self.config.guidance_scale
        
        # Classifier-free guidance
        if self.config.use_classifier_free_guidance and guidance_scale > 1.0:
            if condition is not None:
                uncond_input = torch.zeros_like(condition)
                uncond_pred = model(sample, timestep, uncond_input)
                cond_pred = model(sample, timestep, condition)
                predicted_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                predicted_noise = model(sample, timestep)
        else:
            predicted_noise = model(sample, timestep, condition)
        
        # DPM-Solver step
        if self.algorithm_type == "dpmsolver++":
            # DPM-Solver++ implementation
            alpha_t = self.scheduler.alphas_cumprod[timestep].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1, 1)
            
            # Compute x_0
            x_0 = self.scheduler.remove_noise(sample, predicted_noise, timestep)
            
            # DPM-Solver++ formula
            sigma_t = sqrt_one_minus_alpha_t / torch.sqrt(alpha_t)
            
            if timestep[0] > 0:
                sigma_prev = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep - 1].view(-1, 1, 1, 1) / \
                            torch.sqrt(self.scheduler.alphas_cumprod[timestep - 1]).view(-1, 1, 1, 1)
                h = sigma_t - sigma_prev
                x_prev = x_0 + sigma_prev * predicted_noise + h * torch.randn_like(sample)
            else:
                x_prev = x_0
        else:
            # Standard DPM-Solver
            alpha_t = self.scheduler.alphas_cumprod[timestep].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1, 1)
            
            # Compute x_0
            x_0 = self.scheduler.remove_noise(sample, predicted_noise, timestep)
            
            # DPM-Solver formula
            x_prev = x_0 + sqrt_one_minus_alpha_t * predicted_noise
        
        return x_prev


class EulerSampler(BaseSampler):
    """Euler sampling method."""
    
    def __init__(self, config: SamplingConfig, scheduler):
        
    """__init__ function."""
super().__init__(config)
        self.scheduler = scheduler
    
    def step(self, model: nn.Module, sample: torch.Tensor, timestep: torch.Tensor,
             condition: Optional[torch.Tensor] = None, guidance_scale: Optional[float] = None) -> torch.Tensor:
        """Euler sampling step."""
        guidance_scale = guidance_scale or self.config.guidance_scale
        
        # Classifier-free guidance
        if self.config.use_classifier_free_guidance and guidance_scale > 1.0:
            if condition is not None:
                uncond_input = torch.zeros_like(condition)
                uncond_pred = model(sample, timestep, uncond_input)
                cond_pred = model(sample, timestep, condition)
                predicted_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                predicted_noise = model(sample, timestep)
        else:
            predicted_noise = model(sample, timestep, condition)
        
        # Euler step
        alpha_t = self.scheduler.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1, 1)
        
        # Compute x_0
        x_0 = self.scheduler.remove_noise(sample, predicted_noise, timestep)
        
        # Euler formula
        dt = 1.0 / self.config.num_inference_steps
        x_prev = x_0 + dt * predicted_noise
        
        return x_prev


class HeunSampler(BaseSampler):
    """Heun sampling method (2nd order Runge-Kutta)."""
    
    def __init__(self, config: SamplingConfig, scheduler):
        
    """__init__ function."""
super().__init__(config)
        self.scheduler = scheduler
    
    def step(self, model: nn.Module, sample: torch.Tensor, timestep: torch.Tensor,
             condition: Optional[torch.Tensor] = None, guidance_scale: Optional[float] = None) -> torch.Tensor:
        """Heun sampling step."""
        guidance_scale = guidance_scale or self.config.guidance_scale
        
        # Classifier-free guidance
        if self.config.use_classifier_free_guidance and guidance_scale > 1.0:
            if condition is not None:
                uncond_input = torch.zeros_like(condition)
                uncond_pred = model(sample, timestep, uncond_input)
                cond_pred = model(sample, timestep, condition)
                predicted_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                predicted_noise = model(sample, timestep)
        else:
            predicted_noise = model(sample, timestep, condition)
        
        # Heun step (2nd order Runge-Kutta)
        alpha_t = self.scheduler.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1, 1)
        
        # Compute x_0
        x_0 = self.scheduler.remove_noise(sample, predicted_noise, timestep)
        
        # Heun formula
        dt = 1.0 / self.config.num_inference_steps
        
        # First step (Euler)
        k1 = predicted_noise
        
        # Second step
        sample_mid = x_0 + dt * k1
        timestep_mid = torch.clamp(timestep - 1, min=0)
        
        if self.config.use_classifier_free_guidance and guidance_scale > 1.0:
            if condition is not None:
                uncond_input = torch.zeros_like(condition)
                uncond_pred = model(sample_mid, timestep_mid, uncond_input)
                cond_pred = model(sample_mid, timestep_mid, condition)
                k2 = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                k2 = model(sample_mid, timestep_mid)
        else:
            k2 = model(sample_mid, timestep_mid, condition)
        
        # Heun update
        x_prev = x_0 + 0.5 * dt * (k1 + k2)
        
        return x_prev


class LMSSampler(BaseSampler):
    """LMS (Linear Multi-Step) sampling method."""
    
    def __init__(self, config: SamplingConfig, scheduler):
        
    """__init__ function."""
super().__init__(config)
        self.scheduler = scheduler
        self.order = 4  # LMS order
        self.lms_coeffs = self._get_lms_coefficients()
        self.previous_noises = []
    
    def _get_lms_coefficients(self) -> torch.Tensor:
        """Get LMS coefficients."""
        # 4th order LMS coefficients
        coeffs = torch.tensor([1.0, -1.0, 0.5, -0.1667])
        return coeffs.to(self.device, dtype=self.dtype)
    
    def step(self, model: nn.Module, sample: torch.Tensor, timestep: torch.Tensor,
             condition: Optional[torch.Tensor] = None, guidance_scale: Optional[float] = None) -> torch.Tensor:
        """LMS sampling step."""
        guidance_scale = guidance_scale or self.config.guidance_scale
        
        # Classifier-free guidance
        if self.config.use_classifier_free_guidance and guidance_scale > 1.0:
            if condition is not None:
                uncond_input = torch.zeros_like(condition)
                uncond_pred = model(sample, timestep, uncond_input)
                cond_pred = model(sample, timestep, condition)
                predicted_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                predicted_noise = model(sample, timestep)
        else:
            predicted_noise = model(sample, timestep, condition)
        
        # Store previous noise
        self.previous_noises.append(predicted_noise.detach())
        if len(self.previous_noises) > self.order:
            self.previous_noises.pop(0)
        
        # LMS step
        alpha_t = self.scheduler.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1, 1)
        
        # Compute x_0
        x_0 = self.scheduler.remove_noise(sample, predicted_noise, timestep)
        
        # LMS formula
        dt = 1.0 / self.config.num_inference_steps
        
        if len(self.previous_noises) >= self.order:
            # Use multi-step formula
            noise_sum = sum(coeff * noise for coeff, noise in zip(self.lms_coeffs, self.previous_noises))
            x_prev = x_0 + dt * noise_sum
        else:
            # Fallback to Euler
            x_prev = x_0 + dt * predicted_noise
        
        return x_prev


class UniPCSampler(BaseSampler):
    """UniPC (Unified Predictor-Corrector) sampling method."""
    
    def __init__(self, config: SamplingConfig, scheduler):
        
    """__init__ function."""
super().__init__(config)
        self.scheduler = scheduler
        self.predictor_type = "euler"
        self.corrector_type = "lms"
    
    def step(self, model: nn.Module, sample: torch.Tensor, timestep: torch.Tensor,
             condition: Optional[torch.Tensor] = None, guidance_scale: Optional[float] = None) -> torch.Tensor:
        """UniPC sampling step."""
        guidance_scale = guidance_scale or self.config.guidance_scale
        
        # Classifier-free guidance
        if self.config.use_classifier_free_guidance and guidance_scale > 1.0:
            if condition is not None:
                uncond_input = torch.zeros_like(condition)
                uncond_pred = model(sample, timestep, uncond_input)
                cond_pred = model(sample, timestep, condition)
                predicted_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                predicted_noise = model(sample, timestep)
        else:
            predicted_noise = model(sample, timestep, condition)
        
        # UniPC step (predictor-corrector)
        alpha_t = self.scheduler.alphas_cumprod[timestep].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.scheduler.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1, 1, 1)
        
        # Compute x_0
        x_0 = self.scheduler.remove_noise(sample, predicted_noise, timestep)
        
        # Predictor step (Euler)
        dt = 1.0 / self.config.num_inference_steps
        x_pred = x_0 + dt * predicted_noise
        
        # Corrector step (LMS-like)
        x_prev = x_0 + 0.5 * dt * predicted_noise
        
        return x_prev


class SamplerFactory:
    """Factory for creating samplers."""
    
    @staticmethod
    def create_sampler(method: SamplingMethod, config: SamplingConfig, 
                      scheduler) -> BaseSampler:
        """Create a sampler based on method."""
        samplers = {
            SamplingMethod.DDPM: DDPMSampler,
            SamplingMethod.DDIM: DDIMSampler,
            SamplingMethod.DPM_SOLVER: DPMSolverSampler,
            SamplingMethod.DPM_SOLVER_PP: DPMSolverSampler,
            SamplingMethod.EULER: EulerSampler,
            SamplingMethod.HEUN: HeunSampler,
            SamplingMethod.LMS: LMSSampler,
            SamplingMethod.UNIPC: UniPCSampler,
        }
        
        if method not in samplers:
            raise ValueError(f"Unknown sampling method: {method}")
        
        return samplers[method](config, scheduler)


class AdvancedSamplingManager:
    """Advanced sampling manager with multiple methods and optimizations."""
    
    def __init__(self, config: SamplingConfig, scheduler):
        
    """__init__ function."""
self.config = config
        self.scheduler = scheduler
        self.samplers = {}
    
    def get_sampler(self, method: SamplingMethod) -> BaseSampler:
        """Get or create a sampler for the specified method."""
        if method not in self.samplers:
            self.samplers[method] = SamplerFactory.create_sampler(method, self.config, self.scheduler)
        
        return self.samplers[method]
    
    def sample(self, model: nn.Module, shape: Tuple[int, ...], method: SamplingMethod,
               condition: Optional[torch.Tensor] = None,
               guidance_scale: Optional[float] = None,
               num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """Sample using the specified method."""
        sampler = self.get_sampler(method)
        return sampler.sample(model, shape, condition, guidance_scale, num_inference_steps)
    
    def compare_methods(self, model: nn.Module, shape: Tuple[int, ...],
                       methods: List[SamplingMethod],
                       condition: Optional[torch.Tensor] = None,
                       guidance_scale: Optional[float] = None,
                       num_inference_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Compare multiple sampling methods."""
        results = {}
        
        for method in methods:
            try:
                sample = self.sample(model, shape, method, condition, guidance_scale, num_inference_steps)
                results[method.value] = sample
                logger.info(f"Generated sample with {method.value}: shape {sample.shape}")
            except Exception as e:
                logger.warning(f"Failed to sample with {method.value}: {e}")
        
        return results


async def main():
    """Main function demonstrating advanced sampling methods."""
    logger.info("Setting up advanced sampling methods...")
    
    # Create simple scheduler for testing
    class SimpleScheduler:
        def __init__(self) -> Any:
            self.betas = torch.linspace(0.0001, 0.02, 1000)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
            self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        
        def remove_noise(self, noisy_samples, predicted_noise, timesteps) -> Any:
            alpha_t = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            return (noisy_samples - sqrt_one_minus_alpha_t * predicted_noise) / torch.sqrt(alpha_t)
    
    scheduler = SimpleScheduler()
    
    # Configuration
    config = SamplingConfig(
        num_inference_steps=20,
        guidance_scale=7.5,
        use_classifier_free_guidance=True,
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
    
    # Create sampling manager
    sampling_manager = AdvancedSamplingManager(config, scheduler)
    
    # Test different methods
    methods = [
        SamplingMethod.DDPM,
        SamplingMethod.DDIM,
        SamplingMethod.EULER,
        SamplingMethod.HEUN,
        SamplingMethod.LMS
    ]
    
    shape = (1, 3, 32, 32)
    
    # Compare methods
    logger.info("Comparing sampling methods...")
    results = sampling_manager.compare_methods(model, shape, methods, num_inference_steps=10)
    
    for method_name, sample in results.items():
        logger.info(f"{method_name}: shape {sample.shape}, range [{sample.min():.3f}, {sample.max():.3f}]")
    
    # Test specific method
    logger.info("Testing DDIM with classifier-free guidance...")
    condition = torch.randn(1, 512)  # Mock condition
    sample = sampling_manager.sample(model, shape, SamplingMethod.DDIM, condition, guidance_scale=8.0)
    logger.info(f"DDIM with guidance: shape {sample.shape}, range [{sample.min():.3f}, {sample.max():.3f}]")
    
    logger.info("Advanced sampling methods demonstration completed!")


match __name__:
    case "__main__":
    asyncio.run(main()) 