import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
import math
from tqdm import tqdm
from dataclasses import dataclass
from enum import Enum


class BetaSchedule(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    SIGMOID = "sigmoid"
    QUADRATIC = "quadratic"


@dataclass
class DiffusionConfig:
    """Configuration for diffusion processes"""
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: BetaSchedule = BetaSchedule.LINEAR
    prediction_type: str = "epsilon"  # "epsilon", "x0", "velocity"
    loss_type: str = "mse"  # "mse", "l1", "huber"
    sample_clamp: bool = True
    sample_clamp_min: float = -1.0
    sample_clamp_max: float = 1.0


class DiffusionProcess:
    """
    Base class for diffusion processes
    """
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.num_timesteps = config.num_timesteps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.beta_schedule = config.beta_schedule
        
        # Initialize noise schedule
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Pre-compute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Pre-compute values for q_posterior
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
        """Generate beta schedule based on configuration"""
        if self.beta_schedule == BetaSchedule.LINEAR:
            return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.beta_schedule == BetaSchedule.COSINE:
            return self._cosine_beta_schedule()
        elif self.beta_schedule == BetaSchedule.SIGMOID:
            return self._sigmoid_beta_schedule()
        elif self.beta_schedule == BetaSchedule.QUADRATIC:
            return self._quadratic_beta_schedule()
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine beta schedule as proposed in Improved DDPM"""
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _sigmoid_beta_schedule(self) -> torch.Tensor:
        """Sigmoid beta schedule"""
        betas = torch.sigmoid(torch.linspace(-6, 6, self.num_timesteps))
        return self.beta_start + (self.beta_end - self.beta_start) * betas
    
    def _quadratic_beta_schedule(self) -> torch.Tensor:
        """Quadratic beta schedule"""
        return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_timesteps) ** 2


class ForwardDiffusionProcess(DiffusionProcess):
    """
    Forward diffusion process (q) that gradually adds noise to data
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__(config)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from q(x_t | x_0)
        
        Args:
            x_start: Starting point x_0
            t: Timestep
            noise: Optional noise to use (for reproducibility)
        
        Returns:
            x_t: Noisy sample at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get sqrt_alphas_cumprod and sqrt_one_minus_alphas_cumprod for timestep t
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # q(x_t | x_0) = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of q(x_{t-1} | x_t, x_0)
        
        Args:
            x_start: Starting point x_0
            x_t: Noisy sample at timestep t
            t: Timestep
        
        Returns:
            posterior_mean: Mean of q(x_{t-1} | x_t, x_0)
            posterior_variance: Variance of q(x_{t-1} | x_t, x_0)
            posterior_log_variance_clipped: Log variance (clipped)
        """
        posterior_mean_coef1_t = self.posterior_mean_coef1[t].reshape(-1, 1, 1, 1)
        posterior_mean_coef2_t = self.posterior_mean_coef2[t].reshape(-1, 1, 1, 1)
        posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
        posterior_log_variance_clipped_t = self.posterior_log_variance_clipped[t].reshape(-1, 1, 1, 1)
        
        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t
        
        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t


class ReverseDiffusionProcess(DiffusionProcess):
    """
    Reverse diffusion process (p) that gradually denoises data
    """
    
    def __init__(self, config: DiffusionConfig, model: nn.Module):
        super().__init__(config)
        self.model = model
    
    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        clip_denoised: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the mean and variance of p(x_{t-1} | x_t)
        
        Args:
            x_t: Noisy sample at timestep t
            t: Timestep
            context: Optional context for conditional generation
            clip_denoised: Whether to clip denoised values
        
        Returns:
            Dictionary containing mean, variance, and log variance
        """
        # Get model prediction
        model_output = self.model(x_t, t, context)
        
        # Extract prediction based on prediction type
        if self.config.prediction_type == "epsilon":
            pred_epsilon = model_output
            pred_x_start = self._predict_x_start_from_epsilon(x_t, t, pred_epsilon)
        elif self.config.prediction_type == "x0":
            pred_x_start = model_output
            pred_epsilon = self._predict_epsilon_from_x_start(x_t, t, pred_x_start)
        elif self.config.prediction_type == "velocity":
            pred_velocity = model_output
            pred_epsilon = self._predict_epsilon_from_velocity(x_t, t, pred_velocity)
            pred_x_start = self._predict_x_start_from_epsilon(x_t, t, pred_epsilon)
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
        
        # Clip predicted x_start if requested
        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)
        
        # Compute posterior mean and variance
        posterior_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            pred_x_start, x_t, t
        )
        
        return {
            "mean": posterior_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_x_start": pred_x_start,
            "pred_epsilon": pred_epsilon
        }
    
    def _predict_x_start_from_epsilon(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        epsilon: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from epsilon prediction"""
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * epsilon
    
    def _predict_epsilon_from_x_start(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_start: torch.Tensor
    ) -> torch.Tensor:
        """Predict epsilon from x_0 prediction"""
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return (sqrt_recip_alphas_cumprod_t * x_t - x_start) / sqrt_recipm1_alphas_cumprod_t
    
    def _predict_epsilon_from_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        velocity: torch.Tensor
    ) -> torch.Tensor:
        """Predict epsilon from velocity prediction"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return -sqrt_one_minus_alphas_cumprod_t * velocity + sqrt_alphas_cumprod_t * x_t
    
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample from p(x_{t-1} | x_t)
        
        Args:
            x_t: Noisy sample at timestep t
            t: Timestep
            context: Optional context for conditional generation
            clip_denoised: Whether to clip denoised values
            return_dict: Whether to return dictionary with additional info
        
        Returns:
            x_{t-1}: Denoised sample at timestep t-1
        """
        b, *_, device = *x_t.shape, x_t.device
        
        # Get model prediction
        out = self.p_mean_variance(x_t, t, context, clip_denoised=clip_denoised)
        
        # Sample from posterior
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        
        if return_dict:
            return {"sample": sample, **out}
        else:
            return sample
    
    def p_sample_loop(
        self,
        shape: Tuple[int, ...],
        context: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate samples by running the reverse diffusion process
        
        Args:
            shape: Shape of the samples to generate
            context: Optional context for conditional generation
            clip_denoised: Whether to clip denoised values
            return_dict: Whether to return dictionary with additional info
        
        Returns:
            Generated samples
        """
        device = next(self.model.parameters()).device
        
        # Start from pure noise
        x_t = torch.randn(shape, device=device)
        
        # Reverse diffusion loop
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc="Sampling"):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_tensor, context, clip_denoised, return_dict=False)
        
        if return_dict:
            return {"sample": x_t}
        else:
            return x_t


class DiffusionTraining:
    """
    Training class for diffusion models
    """
    
    def __init__(
        self,
        model: nn.Module,
        forward_process: ForwardDiffusionProcess,
        config: DiffusionConfig
    ):
        self.model = model
        self.forward_process = forward_process
        self.config = config
        
        # Loss function
        if config.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif config.loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif config.loss_type == "huber":
            self.loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")
    
    def compute_loss(
        self,
        x_start: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute training loss for diffusion model
        
        Args:
            x_start: Starting point x_0
            context: Optional context for conditional generation
            noise: Optional noise to use (for reproducibility)
        
        Returns:
            Loss value
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Add noise to x_start
        x_t, target_noise = self.forward_process.q_sample(x_start, t, noise)
        
        # Get model prediction
        if self.config.prediction_type == "epsilon":
            predicted_noise = self.model(x_t, t, context)
            loss = self.loss_fn(predicted_noise, target_noise)
        elif self.config.prediction_type == "x0":
            predicted_x_start = self.model(x_t, t, context)
            loss = self.loss_fn(predicted_x_start, x_start)
        elif self.config.prediction_type == "velocity":
            predicted_velocity = self.model(x_t, t, context)
            target_velocity = self._compute_velocity(x_start, x_t, t)
            loss = self.loss_fn(predicted_velocity, target_velocity)
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
        
        return loss
    
    def _compute_velocity(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute velocity target for velocity prediction"""
        sqrt_alphas_cumprod_t = self.forward_process.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.forward_process.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # Velocity = (x_t - sqrt(α_t) * x_0) / sqrt(1 - α_t)
        velocity = (x_t - sqrt_alphas_cumprod_t * x_start) / sqrt_one_minus_alphas_cumprod_t
        return velocity


class DiffusionVisualizer:
    """
    Visualization utilities for diffusion processes
    """
    
    def __init__(self, forward_process: ForwardDiffusionProcess):
        self.forward_process = forward_process
    
    def visualize_forward_process(
        self,
        x_start: torch.Tensor,
        num_steps: int = 10,
        save_path: Optional[str] = None
    ) -> List[torch.Tensor]:
        """
        Visualize the forward diffusion process
        
        Args:
            x_start: Starting point x_0
            num_steps: Number of steps to visualize
            save_path: Optional path to save visualization
        
        Returns:
            List of noisy samples at different timesteps
        """
        device = x_start.device
        samples = [x_start]
        
        # Sample timesteps
        timesteps = torch.linspace(0, self.forward_process.num_timesteps - 1, num_steps, dtype=torch.long)
        
        for t in timesteps:
            t_tensor = torch.full((x_start.shape[0],), t, device=device, dtype=torch.long)
            x_t, _ = self.forward_process.q_sample(x_start, t_tensor)
            samples.append(x_t)
        
        return samples
    
    def visualize_reverse_process(
        self,
        reverse_process: ReverseDiffusionProcess,
        shape: Tuple[int, ...],
        context: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        save_path: Optional[str] = None
    ) -> List[torch.Tensor]:
        """
        Visualize the reverse diffusion process
        
        Args:
            reverse_process: Reverse diffusion process
            shape: Shape of the samples to generate
            context: Optional context for conditional generation
            num_steps: Number of steps to visualize
            save_path: Optional path to save visualization
        
        Returns:
            List of denoised samples at different timesteps
        """
        device = next(reverse_process.model.parameters()).device
        
        # Start from pure noise
        x_t = torch.randn(shape, device=device)
        samples = [x_t]
        
        # Sample timesteps
        timesteps = torch.linspace(0, reverse_process.num_timesteps - 1, num_steps, dtype=torch.long)
        
        for t in reversed(timesteps):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x_t = reverse_process.p_sample(x_t, t_tensor, context, return_dict=False)
            samples.append(x_t)
        
        return samples


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = DiffusionConfig(
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=BetaSchedule.COSINE,
        prediction_type="epsilon",
        loss_type="mse"
    )
    
    # Create forward process
    forward_process = ForwardDiffusionProcess(config)
    
    # Create a simple UNet model (you would use your actual model here)
    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.time_embed = nn.Linear(1, 64)
            self.final = nn.Conv2d(64, 3, 3, padding=1)
        
        def forward(self, x, t, context=None):
            # Simple time embedding
            t_emb = self.time_embed(t.float().unsqueeze(-1))
            t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1)
            
            # Simple forward pass
            h = F.relu(self.conv1(x))
            h = h + t_emb
            h = F.relu(self.conv2(h))
            return self.final(h)
    
    # Create model and reverse process
    model = SimpleUNet()
    reverse_process = ReverseDiffusionProcess(config, model)
    
    # Test forward process
    x_start = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, config.num_timesteps, (2,))
    x_t, noise = forward_process.q_sample(x_start, t)
    print("Forward process test - x_t shape:", x_t.shape)
    
    # Test reverse process
    out = reverse_process.p_mean_variance(x_t, t)
    print("Reverse process test - mean shape:", out["mean"].shape)
    
    # Test training
    trainer = DiffusionTraining(model, forward_process, config)
    loss = trainer.compute_loss(x_start)
    print("Training loss:", loss.item())
    
    # Test visualization
    visualizer = DiffusionVisualizer(forward_process)
    forward_samples = visualizer.visualize_forward_process(x_start, num_steps=5)
    print("Forward visualization - number of samples:", len(forward_samples))
    
    reverse_samples = visualizer.visualize_reverse_process(
        reverse_process, (2, 3, 32, 32), num_steps=5
    )
    print("Reverse visualization - number of samples:", len(reverse_samples))


