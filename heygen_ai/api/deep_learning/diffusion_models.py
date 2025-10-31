from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Iterator
import logging
import math
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
import asyncio
"""
Diffusion Models Implementation for HeyGen AI.

Implementation of diffusion models including DDPM, DDIM, and advanced variants
following PEP 8 style guidelines and best practices.
"""


logger = logging.getLogger(__name__)


@dataclass
class DiffusionConfig:
    """Configuration for diffusion models."""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # "linear", "cosine", "sigmoid"
    prediction_type: str = "epsilon"  # "epsilon", "x0", "v"
    loss_type: str = "mse"  # "mse", "l1", "huber"
    guidance_scale: float = 7.5
    classifier_free_guidance: bool = True
    clip_denoised: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9999


class NoiseScheduler:
    """Noise scheduler for diffusion models."""

    def __init__(self, config: DiffusionConfig):
        """Initialize noise scheduler.

        Args:
            config: Diffusion configuration.
        """
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
        
        # Precompute values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # DDIM specific
        self.ddim_timesteps = self._make_ddim_timesteps()
        self.ddim_alphas = self.alphas_cumprod[self.ddim_timesteps]
        self.ddim_alphas_prev = self.alphas_cumprod_prev[self.ddim_timesteps]
        self.ddim_sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[self.ddim_timesteps]
        self.ddim_sigmas = self._ddim_sigmas()
        
        logger.info(f"Initialized noise scheduler with {self.num_timesteps} timesteps")

    def _get_beta_schedule(self) -> torch.Tensor:
        """Get beta schedule.

        Returns:
            torch.Tensor: Beta schedule.
        """
        if self.beta_schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.beta_schedule == "cosine":
            return self._cosine_beta_schedule()
        elif self.beta_schedule == "sigmoid":
            return self._sigmoid_beta_schedule()
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine beta schedule.

        Returns:
            torch.Tensor: Cosine beta schedule.
        """
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _sigmoid_beta_schedule(self) -> torch.Tensor:
        """Sigmoid beta schedule.

        Returns:
            torch.Tensor: Sigmoid beta schedule.
        """
        betas = torch.linspace(-6, 6, self.num_timesteps)
        return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start

    def _make_ddim_timesteps(self, ddim_discr_method: str = "uniform", ddim_eta: float = 0.0) -> torch.Tensor:
        """Make DDIM timesteps.

        Args:
            ddim_discr_method: DDIM discretization method.
            ddim_eta: DDIM eta parameter.

        Returns:
            torch.Tensor: DDIM timesteps.
        """
        if ddim_discr_method == "uniform":
            c = self.num_timesteps // 50
            ddim_timesteps = torch.arange(0, self.num_timesteps, c)
        elif ddim_discr_method == "quad":
            ddim_timesteps = ((torch.arange(0, self.num_timesteps) ** 2) / self.num_timesteps).long()
        else:
            raise ValueError(f"Unknown DDIM discretization method: {ddim_discr_method}")
        
        return ddim_timesteps

    def _ddim_sigmas(self) -> torch.Tensor:
        """Get DDIM sigmas.

        Returns:
            torch.Tensor: DDIM sigmas.
        """
        ddim_sigmas = self.ddim_sqrt_one_minus_alphas / self.sqrt_alphas_cumprod[self.ddim_timesteps]
        return ddim_sigmas

    def add_noise(self, x_start: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to input.

        Args:
            x_start: Input tensor.
            timesteps: Timesteps.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Noisy tensor and noise.
        """
        noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        noisy_x = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_x, noise

    def remove_noise(self, x_t: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Remove noise from input.

        Args:
            x_t: Noisy tensor.
            noise: Predicted noise.
            timesteps: Timesteps.

        Returns:
            torch.Tensor: Denoised tensor.
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        x_start = (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
        return x_start

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute q posterior mean and variance.

        Args:
            x_start: Start tensor.
            x_t: Current tensor.
            timesteps: Timesteps.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Posterior mean, variance, and log variance.
        """
        posterior_mean = (
            self.sqrt_recip_alphas_cumprod[timesteps].view(-1, 1, 1, 1) * x_t -
            self.sqrt_recipm1_alphas_cumprod[timesteps].view(-1, 1, 1, 1) * x_start
        )
        posterior_variance = self.betas[timesteps].view(-1, 1, 1, 1)
        posterior_log_variance = torch.log(posterior_variance)
        
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute p mean and variance.

        Args:
            model_output: Model output.
            x_t: Current tensor.
            timesteps: Timesteps.
            clip_denoised: Whether to clip denoised values.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Mean, variance, and log variance.
        """
        if self.config.prediction_type == "epsilon":
            pred_epsilon = model_output
            pred_x_start = self.remove_noise(x_t, pred_epsilon, timesteps)
        elif self.config.prediction_type == "x0":
            pred_x_start = model_output
            pred_epsilon = (x_t - self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1) * pred_x_start) / self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
        
        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)
        
        posterior_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            pred_x_start, x_t, timesteps
        )
        
        return posterior_mean, posterior_variance, posterior_log_variance

    def ddim_step(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """DDIM step.

        Args:
            model_output: Model output.
            x_t: Current tensor.
            timesteps: Timesteps.
            eta: DDIM eta parameter.

        Returns:
            torch.Tensor: Next tensor.
        """
        if self.config.prediction_type == "epsilon":
            pred_epsilon = model_output
            pred_x_start = self.remove_noise(x_t, pred_epsilon, timesteps)
        elif self.config.prediction_type == "x0":
            pred_x_start = model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
        
        # DDIM step
        alpha_cumprod_t = self.ddim_alphas[timesteps].view(-1, 1, 1, 1)
        alpha_cumprod_t_prev = self.ddim_alphas_prev[timesteps].view(-1, 1, 1, 1)
        sigma_t = eta * self.ddim_sigmas[timesteps].view(-1, 1, 1, 1)
        
        pred_noise = (x_t - torch.sqrt(alpha_cumprod_t) * pred_x_start) / torch.sqrt(1 - alpha_cumprod_t)
        
        x_prev = (
            torch.sqrt(alpha_cumprod_t_prev) * pred_x_start +
            torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t ** 2) * pred_noise +
            sigma_t * torch.randn_like(x_t)
        )
        
        return x_prev


class UNetBlock(nn.Module):
    """UNet block implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """Initialize UNet block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            time_emb_dim: Time embedding dimension.
            dropout: Dropout rate.
            use_attention: Whether to use attention.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.use_attention = use_attention
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
        
        # Attention
        if use_attention:
            self.attention = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True)
        else:
            self.attention = None

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.
            time_emb: Time embedding.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = self.residual_conv(x)
        
        # First convolution
        h = self.conv1(x)
        h = F.silu(h)
        
        # Time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb.view(-1, self.out_channels, 1, 1)
        
        # Second convolution
        h = self.conv2(h)
        h = self.dropout(h)
        h = F.silu(h)
        
        # Residual connection
        h = h + residual
        
        # Attention
        if self.attention is not None:
            b, c, h, w = h.shape
            h_flat = h.view(b, c, h * w).transpose(1, 2)  # (b, h*w, c)
            h_attended, _ = self.attention(h_flat, h_flat, h_flat)
            h_attended = h_attended.transpose(1, 2).view(b, c, h, w)
            h = h + h_attended
        
        return h


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings."""

    def __init__(self, dim: int):
        """Initialize sinusoidal position embeddings.

        Args:
            dim: Embedding dimension.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            time: Time tensor.

        Returns:
            torch.Tensor: Position embeddings.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNet(nn.Module):
    """UNet model for diffusion."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        conv_resample: bool = True,
        num_heads: int = 8,
        use_spatial_transformer: bool = False,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
    ):
        """Initialize UNet.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            model_channels: Model channels.
            num_res_blocks: Number of residual blocks.
            attention_resolutions: Attention resolutions.
            dropout: Dropout rate.
            channel_mult: Channel multipliers.
            conv_resample: Whether to use conv resampling.
            num_heads: Number of attention heads.
            use_spatial_transformer: Whether to use spatial transformer.
            transformer_depth: Transformer depth.
            context_dim: Context dimension.
            use_checkpoint: Whether to use checkpointing.
            use_fp16: Whether to use fp16.
            num_heads_upsample: Number of heads for upsampling.
            use_scale_shift_norm: Whether to use scale shift norm.
            resblock_updown: Whether to use resblock updown.
            use_new_attention_order: Whether to use new attention order.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.use_spatial_transformer = use_spatial_transformer
        self.transformer_depth = transformer_depth
        self.context_dim = context_dim
        self.use_checkpoint = use_checkpoint
        self.use_fp16 = use_fp16
        self.num_heads_upsample = num_heads_upsample
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input projection
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        ])
        
        # Downsampling
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    UNetBlock(
                        ch, mult * model_channels, time_embed_dim,
                        dropout, use_attention=ds in attention_resolutions
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        nn.MultiheadAttention(ch, num_heads, batch_first=True)
                    )
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)])
                )
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle block
        self.middle_block = nn.ModuleList([
            UNetBlock(ch, ch, time_embed_dim, dropout, use_attention=True),
            nn.MultiheadAttention(ch, num_heads, batch_first=True),
            UNetBlock(ch, ch, time_embed_dim, dropout, use_attention=False),
        ])
        
        # Upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    UNetBlock(
                        ch + ich, mult * model_channels, time_embed_dim,
                        dropout, use_attention=ds in attention_resolutions
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        nn.MultiheadAttention(ch, num_heads, batch_first=True)
                    )
                if level and i == num_res_blocks:
                    layers.append(
                        nn.ConvTranspose2d(ch, ch, 4, 2, 1)
                    )
                    ds //= 2
                self.output_blocks.append(nn.ModuleList(layers))
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.
            timesteps: Timesteps.
            context: Context tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Input blocks
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Conv2d):
                h = module(h)
            else:
                for layer in module:
                    if isinstance(layer, UNetBlock):
                        h = layer(h, time_emb)
                    elif isinstance(layer, nn.MultiheadAttention):
                        b, c, h_h, h_w = h.shape
                        h_flat = h.view(b, c, h_h * h_w).transpose(1, 2)
                        h_attended, _ = layer(h_flat, h_flat, h_flat)
                        h = h_attended.transpose(1, 2).view(b, c, h_h, h_w)
            hs.append(h)
        
        # Middle block
        for module in self.middle_block:
            if isinstance(module, UNetBlock):
                h = module(h, time_emb)
            elif isinstance(module, nn.MultiheadAttention):
                b, c, h_h, h_w = h.shape
                h_flat = h.view(b, c, h_h * h_w).transpose(1, 2)
                h_attended, _ = module(h_flat, h_flat, h_flat)
                h = h_attended.transpose(1, 2).view(b, c, h_h, h_w)
        
        # Output blocks
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, UNetBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, nn.MultiheadAttention):
                    b, c, h_h, h_w = h.shape
                    h_flat = h.view(b, c, h_h * h_w).transpose(1, 2)
                    h_attended, _ = layer(h_flat, h_flat, h_flat)
                    h = h_attended.transpose(1, 2).view(b, c, h_h, h_w)
                elif isinstance(layer, nn.ConvTranspose2d):
                    h = layer(h)
        
        # Output
        return self.out(h)


class DiffusionModel(ABC):
    """Abstract base class for diffusion models."""

    def __init__(self, config: DiffusionConfig):
        """Initialize diffusion model.

        Args:
            config: Diffusion configuration.
        """
        self.config = config
        self.scheduler = NoiseScheduler(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initialized diffusion model with {config.num_timesteps} timesteps")

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Get the model.

        Returns:
            nn.Module: The model.
        """
        pass

    @abstractmethod
    def training_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Training step.

        Args:
            batch: Input batch.
            optimizer: Optimizer.

        Returns:
            Dict[str, float]: Training metrics.
        """
        pass

    @abstractmethod
    def sampling_step(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """Sampling step.

        Args:
            x_t: Current tensor.
            timesteps: Timesteps.
            model_output: Model output.

        Returns:
            torch.Tensor: Next tensor.
        """
        pass

    def sample(
        self,
        batch_size: int = 1,
        image_size: int = 64,
        channels: int = 3,
        guidance_scale: Optional[float] = None
    ) -> torch.Tensor:
        """Generate samples.

        Args:
            batch_size: Batch size.
            image_size: Image size.
            channels: Number of channels.
            guidance_scale: Guidance scale.

        Returns:
            torch.Tensor: Generated samples.
        """
        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale
        
        model = self.get_model()
        model.eval()
        
        # Start from noise
        x = torch.randn(batch_size, channels, image_size, image_size, device=self.device)
        
        # Sampling loop
        with torch.no_grad():
            for t in reversed(range(self.config.num_timesteps)):
                timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                
                # Model prediction
                model_output = model(x, timesteps)
                
                # Apply guidance if enabled
                if self.config.classifier_free_guidance and guidance_scale > 1.0:
                    uncond_output = model(x, timesteps, context=torch.zeros_like(context))
                    model_output = uncond_output + guidance_scale * (model_output - uncond_output)
                
                # Sampling step
                x = self.sampling_step(x, timesteps, model_output)
        
        return x


class DDPM(DiffusionModel):
    """Denoising Diffusion Probabilistic Models (DDPM)."""

    def __init__(self, config: DiffusionConfig):
        """Initialize DDPM.

        Args:
            config: Diffusion configuration.
        """
        super().__init__(config)
        self.model = UNet(
            in_channels=3,
            out_channels=3,
            model_channels=128,
            num_res_blocks=2,
            attention_resolutions=(16, 8),
            dropout=0.1,
            channel_mult=(1, 2, 4, 8),
            num_heads=8
        ).to(self.device)
        
        # EMA model
        if config.use_ema:
            self.ema_model = UNet(
                in_channels=3,
                out_channels=3,
                model_channels=128,
                num_res_blocks=2,
                attention_resolutions=(16, 8),
                dropout=0.1,
                channel_mult=(1, 2, 4, 8),
                num_heads=8
            ).to(self.device)
            self.ema_model.load_state_dict(self.model.state_dict())
            self.ema_model.eval()
        else:
            self.ema_model = None

    def get_model(self) -> nn.Module:
        """Get the model.

        Returns:
            nn.Module: The model.
        """
        return self.ema_model if self.ema_model is not None else self.model

    def training_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Training step.

        Args:
            batch: Input batch.
            optimizer: Optimizer.

        Returns:
            Dict[str, float]: Training metrics.
        """
        batch_size = batch.shape[0]
        timesteps = torch.randint(0, self.config.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise
        noisy_batch, noise = self.scheduler.add_noise(batch, timesteps)
        
        # Model prediction
        model_output = self.model(noisy_batch, timesteps)
        
        # Loss
        if self.config.loss_type == "mse":
            loss = F.mse_loss(model_output, noise)
        elif self.config.loss_type == "l1":
            loss = F.l1_loss(model_output, noise)
        elif self.config.loss_type == "huber":
            loss = F.smooth_l1_loss(model_output, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update EMA
        if self.ema_model is not None:
            with torch.no_grad():
                for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                    ema_param.data.mul_(self.config.ema_decay).add_(param.data, alpha=1 - self.config.ema_decay)
        
        return {"loss": loss.item()}

    def sampling_step(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """Sampling step.

        Args:
            x_t: Current tensor.
            timesteps: Timesteps.
            model_output: Model output.

        Returns:
            torch.Tensor: Next tensor.
        """
        # Compute mean and variance
        posterior_mean, posterior_variance, _ = self.scheduler.p_mean_variance(
            model_output, x_t, timesteps, self.config.clip_denoised
        )
        
        # Sample
        noise = torch.randn_like(x_t)
        x_prev = posterior_mean + torch.sqrt(posterior_variance) * noise
        
        return x_prev


class DDIM(DiffusionModel):
    """Denoising Diffusion Implicit Models (DDIM)."""

    def __init__(self, config: DiffusionConfig):
        """Initialize DDIM.

        Args:
            config: Diffusion configuration.
        """
        super().__init__(config)
        self.model = UNet(
            in_channels=3,
            out_channels=3,
            model_channels=128,
            num_res_blocks=2,
            attention_resolutions=(16, 8),
            dropout=0.1,
            channel_mult=(1, 2, 4, 8),
            num_heads=8
        ).to(self.device)

    def get_model(self) -> nn.Module:
        """Get the model.

        Returns:
            nn.Module: The model.
        """
        return self.model

    def training_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Training step.

        Args:
            batch: Input batch.
            optimizer: Optimizer.

        Returns:
            Dict[str, float]: Training metrics.
        """
        batch_size = batch.shape[0]
        timesteps = torch.randint(0, self.config.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise
        noisy_batch, noise = self.scheduler.add_noise(batch, timesteps)
        
        # Model prediction
        model_output = self.model(noisy_batch, timesteps)
        
        # Loss
        if self.config.loss_type == "mse":
            loss = F.mse_loss(model_output, noise)
        elif self.config.loss_type == "l1":
            loss = F.l1_loss(model_output, noise)
        elif self.config.loss_type == "huber":
            loss = F.smooth_l1_loss(model_output, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {"loss": loss.item()}

    def sampling_step(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """Sampling step.

        Args:
            x_t: Current tensor.
            timesteps: Timesteps.
            model_output: Model output.

        Returns:
            torch.Tensor: Next tensor.
        """
        return self.scheduler.ddim_step(model_output, x_t, timesteps, eta=0.0)


def create_diffusion_model(
    model_type: str,
    config: DiffusionConfig
) -> DiffusionModel:
    """Create diffusion model.

    Args:
        model_type: Type of diffusion model.
        config: Diffusion configuration.

    Returns:
        DiffusionModel: Created diffusion model.
    """
    if model_type == "ddpm":
        return DDPM(config)
    elif model_type == "ddim":
        return DDIM(config)
    else:
        raise ValueError(f"Unknown diffusion model type: {model_type}")


def create_noise_scheduler(config: DiffusionConfig) -> NoiseScheduler:
    """Create noise scheduler.

    Args:
        config: Diffusion configuration.

    Returns:
        NoiseScheduler: Created noise scheduler.
    """
    return NoiseScheduler(config)


def create_unet_model(
    in_channels: int = 3,
    out_channels: int = 3,
    model_channels: int = 128,
    num_res_blocks: int = 2,
    attention_resolutions: Tuple[int, ...] = (16, 8),
    dropout: float = 0.1,
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
    num_heads: int = 8
) -> UNet:
    """Create UNet model.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        model_channels: Model channels.
        num_res_blocks: Number of residual blocks.
        attention_resolutions: Attention resolutions.
        dropout: Dropout rate.
        channel_mult: Channel multipliers.
        num_heads: Number of attention heads.

    Returns:
        UNet: Created UNet model.
    """
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        num_heads=num_heads
    ) 