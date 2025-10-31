"""
Advanced Noise Scheduler and Sampling Methods System
for Diffusion Models with PyTorch and Diffusers

This system implements:
- Multiple beta schedules (linear, cosine, quadratic, sigmoid)
- Advanced noise schedulers (DDPM, DDIM, PNDM, Euler, Heun)
- Sampling methods (DDPM, DDIM, ancestral, classifier-free guidance)
- Custom schedulers and samplers
- Integration with Hugging Face Diffusers library
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import math
from tqdm import tqdm
import logging
from pathlib import Path
import json
import yaml
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BetaSchedule(Enum):
    """Available beta schedules for diffusion models."""
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
    ANCESTRAL = "ancestral"
    EULER = "euler"
    HEUN = "heun"
    PNDM = "pndm"
    DPM_SOLVER = "dpm_solver"
    UNIPC = "unipc"

@dataclass
class NoiseSchedulerConfig:
    """Configuration for noise schedulers."""
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: BetaSchedule = BetaSchedule.LINEAR
    clip_sample: bool = True
    prediction_type: str = "epsilon"  # "epsilon", "v_prediction", "sample"
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    sample_max_value: float = 1.0
    timestep_spacing: str = "leading"  # "leading", "trailing"
    steps_offset: int = 1
    use_clipped_model_output: bool = True
    variance_type: str = "fixed_small"  # "fixed_small", "fixed_small_log", "learned", "learned_range"
    clip_sample_range: float = 1.0
    sample_padding_threshold: float = 0.0
    sample_padding_norm: float = 1.0

class BaseNoiseScheduler(ABC):
    """Abstract base class for noise schedulers."""
    
    def __init__(self, config: NoiseSchedulerConfig):
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # q(x_t | x_0) posterior parameters
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
    
    @abstractmethod
    def _get_beta_schedule(self) -> torch.Tensor:
        """Generate beta schedule based on configuration."""
        pass
    
    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to the original samples."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1)
        
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1)
        
        noise = torch.randn_like(original_samples)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        
        return noisy_samples, noise
    
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Get velocity from sample and noise."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1)
        
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

class LinearNoiseScheduler(BaseNoiseScheduler):
    """Linear beta schedule noise scheduler."""
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """Generate linear beta schedule."""
        return torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.num_train_timesteps,
            dtype=torch.float32
        )

class CosineNoiseScheduler(BaseNoiseScheduler):
    """Cosine beta schedule noise scheduler."""
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """Generate cosine beta schedule."""
        steps = self.config.num_train_timesteps + 1
        x = torch.linspace(0, self.config.num_train_timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / self.config.num_train_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

class QuadraticNoiseScheduler(BaseNoiseScheduler):
    """Quadratic beta schedule noise scheduler."""
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """Generate quadratic beta schedule."""
        t = torch.linspace(0, 1, self.config.num_train_timesteps, dtype=torch.float32)
        betas = self.config.beta_start + (self.config.beta_end - self.config.beta_start) * t ** 2
        return betas

class SigmoidNoiseScheduler(BaseNoiseScheduler):
    """Sigmoid beta schedule noise scheduler."""
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """Generate sigmoid beta schedule."""
        t = torch.linspace(-6, 6, self.config.num_train_timesteps, dtype=torch.float32)
        betas = self.config.beta_start + (self.config.beta_end - self.config.beta_start) * torch.sigmoid(t)
        return betas

class ExponentialNoiseScheduler(BaseNoiseScheduler):
    """Exponential beta schedule noise scheduler."""
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """Generate exponential beta schedule."""
        t = torch.linspace(0, 1, self.config.num_train_timesteps, dtype=torch.float32)
        betas = self.config.beta_start * (self.config.beta_end / self.config.beta_start) ** t
        return betas

class NoiseSchedulerFactory:
    """Factory for creating noise schedulers."""
    
    @staticmethod
    def create_scheduler(config: NoiseSchedulerConfig) -> BaseNoiseScheduler:
        """Create a noise scheduler based on configuration."""
        if config.beta_schedule == BetaSchedule.LINEAR:
            return LinearNoiseScheduler(config)
        elif config.beta_schedule == BetaSchedule.COSINE:
            return CosineNoiseScheduler(config)
        elif config.beta_schedule == BetaSchedule.QUADRATIC:
            return QuadraticNoiseScheduler(config)
        elif config.beta_schedule == BetaSchedule.SIGMOID:
            return SigmoidNoiseScheduler(config)
        elif config.beta_schedule == BetaSchedule.EXPONENTIAL:
            return ExponentialNoiseScheduler(config)
        else:
            raise ValueError(f"Unknown beta schedule: {config.beta_schedule}")

class BaseSampler(ABC):
    """Abstract base class for samplers."""
    
    def __init__(self, scheduler: BaseNoiseScheduler):
        self.scheduler = scheduler
    
    @abstractmethod
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, **kwargs) -> torch.Tensor:
        """Perform one sampling step."""
        pass

class DDPMSampler(BaseSampler):
    """DDPM (Denoising Diffusion Probabilistic Models) sampler."""
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, **kwargs) -> torch.Tensor:
        """DDPM sampling step."""
        t = timestep
        prev_t = t - 1
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # This is the formula for the mean of the posterior distribution
        # q(x_{t-1} | x_t, x_0)
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (beta_prod_t_prev / beta_prod_t) ** 0.5 * noise
        
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample

class DDIMSampler(BaseSampler):
    """DDIM (Denoising Diffusion Implicit Models) sampler."""
    
    def __init__(self, scheduler: BaseNoiseScheduler, eta: float = 0.0):
        super().__init__(scheduler)
        self.eta = eta
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, **kwargs) -> torch.Tensor:
        """DDIM sampling step."""
        t = timestep
        prev_t = t - 1
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        
        # This is the formula for the mean of the posterior distribution
        # q(x_{t-1} | x_t, x_0)
        pred_original_sample = (sample - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        pred_epsilon = model_output
        
        # This is the formula for the variance of the posterior distribution
        # q(x_{t-1} | x_t, x_0)
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * pred_epsilon
        
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        if self.eta > 0:
            # Add noise if eta > 0
            noise = torch.randn_like(model_output)
            pred_prev_sample = pred_prev_sample + self.eta * (1 - alpha_prod_t_prev) ** 0.5 * noise
        
        return pred_prev_sample

class AncestralSampler(BaseSampler):
    """Ancestral sampling method."""
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, **kwargs) -> torch.Tensor:
        """Ancestral sampling step."""
        t = timestep
        prev_t = t - 1
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        
        # This is the formula for the mean of the posterior distribution
        # q(x_{t-1} | x_t, x_0)
        pred_original_sample = (sample - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        pred_epsilon = model_output
        
        # This is the formula for the variance of the posterior distribution
        # q(x_{t-1} | x_t, x_0)
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * pred_epsilon
        
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        # Add noise for ancestral sampling
        noise = torch.randn_like(model_output)
        pred_prev_sample = pred_prev_sample + (1 - alpha_prod_t_prev) ** 0.5 * noise
        
        return pred_prev_sample

class EulerSampler(BaseSampler):
    """Euler method sampler."""
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, **kwargs) -> torch.Tensor:
        """Euler sampling step."""
        t = timestep
        prev_t = t - 1
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        
        # This is the formula for the mean of the posterior distribution
        # q(x_{t-1} | x_t, x_0)
        pred_original_sample = (sample - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        pred_epsilon = model_output
        
        # This is the formula for the variance of the posterior distribution
        # q(x_{t-1} | x_t, x_0)
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * pred_epsilon
        
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        return pred_prev_sample

class HeunSampler(BaseSampler):
    """Heun method sampler (2nd order Runge-Kutta)."""
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, **kwargs) -> torch.Tensor:
        """Heun sampling step."""
        t = timestep
        prev_t = t - 1
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        
        # This is the formula for the mean of the posterior distribution
        # q(x_{t-1} | x_t, x_0)
        pred_original_sample = (sample - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        pred_epsilon = model_output
        
        # This is the formula for the variance of the posterior distribution
        # q(x_{t-1} | x_t, x_0)
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * pred_epsilon
        
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        return pred_prev_sample

class SamplerFactory:
    """Factory for creating samplers."""
    
    @staticmethod
    def create_sampler(method: SamplingMethod, scheduler: BaseNoiseScheduler, **kwargs) -> BaseSampler:
        """Create a sampler based on method."""
        if method == SamplingMethod.DDPM:
            return DDPMSampler(scheduler)
        elif method == SamplingMethod.DDIM:
            eta = kwargs.get('eta', 0.0)
            return DDIMSampler(scheduler, eta)
        elif method == SamplingMethod.ANCESTRAL:
            return AncestralSampler(scheduler)
        elif method == SamplingMethod.EULER:
            return EulerSampler(scheduler)
        elif method == SamplingMethod.HEUN:
            return HeunSampler(scheduler)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

class DiffusionPipeline:
    """Complete diffusion pipeline with noise scheduler and sampler."""
    
    def __init__(self, config: NoiseSchedulerConfig, sampling_method: SamplingMethod = SamplingMethod.DDPM):
        self.config = config
        self.scheduler = NoiseSchedulerFactory.create_scheduler(config)
        self.sampler = SamplerFactory.create_sampler(sampling_method, self.scheduler)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initialized DiffusionPipeline with {config.beta_schedule.value} scheduler and {sampling_method.value} sampler")
    
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        classifier_free_guidance: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Generate samples using the diffusion pipeline."""
        batch_size = shape[0]
        device = self.device
        
        # Initialize sample
        sample = torch.randn(shape, device=device)
        
        # Set timesteps
        timesteps = torch.linspace(
            self.config.num_train_timesteps - 1,
            0,
            num_inference_steps,
            dtype=torch.long,
            device=device
        )
        
        # Sampling loop
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
                # Expand sample for batch processing
                sample_input = sample.expand(batch_size, *sample.shape[1:])
                
                # Model prediction
                if classifier_free_guidance:
                    # Classifier-free guidance
                    uncond_input = torch.zeros_like(sample_input)
                    model_output = model(sample_input, t, **kwargs)
                    uncond_output = model(uncond_input, t, **kwargs)
                    
                    # Apply guidance
                    model_output = uncond_output + guidance_scale * (model_output - uncond_output)
                else:
                    model_output = model(sample_input, t, **kwargs)
                
                # Sampler step
                sample = self.sampler.step(model_output, t, sample, **kwargs)
                
                # Clip sample if configured
                if self.config.clip_sample:
                    sample = torch.clamp(sample, -1, 1)
        
        return sample
    
    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to original samples."""
        return self.scheduler.add_noise(original_samples, timesteps)
    
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Get velocity from sample and noise."""
        return self.scheduler.get_velocity(sample, noise, timesteps)

class AdvancedDiffusionSystem:
    """Advanced diffusion system with multiple schedulers and samplers."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.pipelines: Dict[str, DiffusionPipeline] = {}
        self.configs: Dict[str, NoiseSchedulerConfig] = {}
        
        if config_path:
            self.load_config(config_path)
        
        logger.info("AdvancedDiffusionSystem initialized")
    
    def create_pipeline(
        self,
        name: str,
        config: NoiseSchedulerConfig,
        sampling_method: SamplingMethod = SamplingMethod.DDPM
    ) -> DiffusionPipeline:
        """Create a new diffusion pipeline."""
        pipeline = DiffusionPipeline(config, sampling_method)
        self.pipelines[name] = pipeline
        self.configs[name] = config
        
        logger.info(f"Created pipeline '{name}' with {config.beta_schedule.value} scheduler and {sampling_method.value} sampler")
        return pipeline
    
    def get_pipeline(self, name: str) -> DiffusionPipeline:
        """Get a pipeline by name."""
        if name not in self.pipelines:
            raise ValueError(f"Pipeline '{name}' not found")
        return self.pipelines[name]
    
    def list_pipelines(self) -> List[str]:
        """List all available pipelines."""
        return list(self.pipelines.keys())
    
    def compare_schedulers(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compare different schedulers on the same task."""
        results = {}
        
        for name, pipeline in self.pipelines.items():
            logger.info(f"Testing pipeline: {name}")
            
            # Create a dummy model for testing
            dummy_model = self._create_dummy_model(shape[1:])
            
            try:
                sample = pipeline.sample(
                    dummy_model,
                    shape,
                    num_inference_steps,
                    **kwargs
                )
                results[name] = sample
                logger.info(f"Pipeline {name} completed successfully")
            except Exception as e:
                logger.error(f"Pipeline {name} failed: {e}")
                results[name] = None
        
        return results
    
    def _create_dummy_model(self, shape: Tuple[int, ...]) -> nn.Module:
        """Create a dummy model for testing."""
        class DummyModel(nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.shape = shape
            
            def forward(self, x, t, **kwargs):
                # Return random noise for testing
                return torch.randn_like(x)
        
        return DummyModel(shape)
    
    def load_config(self, config_path: str):
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    config_data = json.load(f)
                else:
                    raise ValueError("Unsupported config file format")
            
            for name, config_dict in config_data.items():
                config = NoiseSchedulerConfig(**config_dict)
                self.create_pipeline(name, config)
                
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def save_config(self, config_path: str):
        """Save current configuration to file."""
        try:
            config_data = {}
            for name, config in self.configs.items():
                config_data[name] = {
                    'num_train_timesteps': config.num_train_timesteps,
                    'beta_start': config.beta_start,
                    'beta_end': config.beta_end,
                    'beta_schedule': config.beta_schedule.value,
                    'clip_sample': config.clip_sample,
                    'prediction_type': config.prediction_type,
                    'thresholding': config.thresholding,
                    'dynamic_thresholding_ratio': config.dynamic_thresholding_ratio,
                    'sample_max_value': config.sample_max_value,
                    'timestep_spacing': config.timestep_spacing,
                    'steps_offset': config.steps_offset,
                    'use_clipped_model_output': config.use_clipped_model_output,
                    'variance_type': config.variance_type,
                    'clip_sample_range': config.clip_sample_range,
                    'sample_padding_threshold': config.sample_padding_threshold,
                    'sample_padding_norm': config.sample_padding_norm
                }
            
            with open(config_path, 'w') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    yaml.dump(config_data, f, default_flow_style=False)
                elif config_path.endswith('.json'):
                    json.dump(config_data, f, indent=2)
                else:
                    raise ValueError("Unsupported config file format")
            
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

def main():
    """Main function to demonstrate the noise scheduler and sampling system."""
    # Create configurations for different schedulers
    configs = {
        'linear': NoiseSchedulerConfig(
            beta_schedule=BetaSchedule.LINEAR,
            num_train_timesteps=1000
        ),
        'cosine': NoiseSchedulerConfig(
            beta_schedule=BetaSchedule.COSINE,
            num_train_timesteps=1000
        ),
        'quadratic': NoiseSchedulerConfig(
            beta_schedule=BetaSchedule.QUADRATIC,
            num_train_timesteps=1000
        ),
        'sigmoid': NoiseSchedulerConfig(
            beta_schedule=BetaSchedule.SIGMOID,
            num_train_timesteps=1000
        )
    }
    
    # Create advanced diffusion system
    system = AdvancedDiffusionSystem()
    
    # Create pipelines with different schedulers
    for name, config in configs.items():
        system.create_pipeline(name, config, SamplingMethod.DDPM)
    
    # Test the system
    shape = (1, 3, 64, 64)  # Batch size, channels, height, width
    results = system.compare_schedulers(shape, num_inference_steps=20)
    
    print(f"Created {len(system.list_pipelines())} pipelines:")
    for name in system.list_pipelines():
        print(f"  - {name}")
    
    print("\nTest results:")
    for name, result in results.items():
        if result is not None:
            print(f"  - {name}: Success (shape: {result.shape})")
        else:
            print(f"  - {name}: Failed")

if __name__ == "__main__":
    main()


