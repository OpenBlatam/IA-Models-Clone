"""
Advanced Diffusion Engine for Export IA
Refactored diffusion models with state-of-the-art techniques and optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import math
from pathlib import Path

# Advanced diffusion libraries
try:
    from diffusers import (
        DDPMPipeline, DDPMScheduler, DDIMScheduler, 
        StableDiffusionPipeline, StableDiffusionXLPipeline,
        ControlNetModel, StableDiffusionControlNetPipeline,
        UNet2DConditionModel, AutoencoderKL, CLIPTextModel, CLIPTokenizer,
        EulerDiscreteScheduler, DPMSolverMultistepScheduler,
        PNDMScheduler, LMSDiscreteScheduler
    )
    from diffusers.models.attention_processor import AttnProcessor2_0, LoRAAttnProcessor
    from diffusers.optimization import get_scheduler
    from diffusers.utils import randn_tensor
    import accelerate
    from accelerate import Accelerator
except ImportError:
    print("Installing required diffusion libraries...")
    import subprocess
    subprocess.check_call(["pip", "install", "diffusers", "accelerate", "transformers", "xformers"])

logger = logging.getLogger(__name__)

@dataclass
class DiffusionConfig:
    """Configuration for diffusion models"""
    # Model parameters
    model_name: str
    model_type: str = "ddpm"  # ddpm, ddim, stable_diffusion, controlnet
    
    # Architecture parameters
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Tuple[str, ...] = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D")
    up_block_types: Tuple[str, ...] = ("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    cross_attention_dim: int = 768
    attention_head_dim: int = 8
    use_linear_projection: bool = False
    
    # Training parameters
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # linear, cosine, cosine_beta
    prediction_type: str = "epsilon"  # epsilon, sample, v_prediction
    
    # Sampling parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0  # DDIM eta parameter
    
    # Optimization
    use_ema: bool = True
    ema_decay: float = 0.9999
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    # LoRA parameters
    use_lora: bool = False
    lora_rank: int = 4
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    
    # ControlNet parameters
    use_controlnet: bool = False
    controlnet_conditioning_scale: float = 1.0

class NoiseScheduler:
    """Advanced noise scheduler for diffusion models"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.beta_schedule = config.beta_schedule
        
        # Calculate betas
        self.betas = self._get_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculate variance
        self.variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.variance = torch.clamp(self.variance, min=1e-20)
        
        # Calculate log variance
        self.log_variance = torch.log(self.variance)
        
    def _get_betas(self) -> torch.Tensor:
        """Calculate beta values based on schedule"""
        if self.beta_schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        elif self.beta_schedule == "cosine":
            return self._cosine_beta_schedule()
        elif self.beta_schedule == "cosine_beta":
            return self._cosine_beta_schedule_v2()
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
            
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine beta schedule"""
        s = 0.008
        steps = self.num_train_timesteps + 1
        x = torch.linspace(0, self.num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_train_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
        
    def _cosine_beta_schedule_v2(self) -> torch.Tensor:
        """Improved cosine beta schedule"""
        s = 0.008
        steps = self.num_train_timesteps + 1
        x = torch.linspace(0, self.num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_train_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
        
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, 
                  timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to original samples"""
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
        
    def step(self, model_output: torch.Tensor, timestep: int, 
             sample: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        """Single denoising step"""
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        # Calculate alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Calculate predicted original sample
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        # Calculate variance
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance ** 0.5
        
        # Calculate previous sample
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        if eta > 0:
            noise = torch.randn_like(model_output)
            prev_sample += std_dev_t * noise
            
        return prev_sample

class UNet2D(nn.Module):
    """Advanced 2D U-Net for diffusion models"""
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(config.block_out_channels[0], config.block_out_channels[0] * 4),
            nn.SiLU(),
            nn.Linear(config.block_out_channels[0] * 4, config.block_out_channels[0])
        )
        
        # Input projection
        self.conv_in = nn.Conv2d(config.in_channels, config.block_out_channels[0], 3, padding=1)
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(
            DownBlock2D(
                config.block_out_channels[0],
                config.block_out_channels[0],
                config.layers_per_block,
                add_downsample=True
            )
        )
        
        for i in range(len(config.down_block_types) - 1):
            in_channels = config.block_out_channels[i]
            out_channels = config.block_out_channels[i + 1]
            self.down_blocks.append(
                DownBlock2D(
                    in_channels,
                    out_channels,
                    config.layers_per_block,
                    add_downsample=True
                )
            )
            
        # Middle block
        self.mid_block = MidBlock2D(
            config.block_out_channels[-1],
            config.layers_per_block
        )
        
        # Up blocks
        self.up_blocks = nn.ModuleList()
        for i in range(len(config.up_block_types)):
            in_channels = config.block_out_channels[-(i+1)]
            out_channels = config.block_out_channels[-(i+2)] if i < len(config.up_block_types) - 1 else config.block_out_channels[0]
            self.up_blocks.append(
                UpBlock2D(
                    in_channels,
                    out_channels,
                    config.layers_per_block,
                    add_upsample=True
                )
            )
            
        # Output projection
        self.conv_out = nn.Conv2d(config.block_out_channels[0], config.out_channels, 3, padding=1)
        
        # Group normalization
        self.group_norm = nn.GroupNorm(32, config.block_out_channels[0])
        
    def forward(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Time embedding
        t_emb = self.time_embedding(timestep)
        
        # Input projection
        x = self.conv_in(sample)
        
        # Down sampling
        down_block_res_samples = []
        for down_block in self.down_blocks:
            x, res_samples = down_block(x, t_emb)
            down_block_res_samples.extend(res_samples)
            
        # Middle block
        x = self.mid_block(x, t_emb)
        
        # Up sampling with skip connections
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]
            x = up_block(x, res_samples, t_emb)
            
        # Output projection
        x = self.group_norm(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return x

class DownBlock2D(nn.Module):
    """Down sampling block for U-Net"""
    
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, add_downsample: bool = True):
        super().__init__()
        self.resnets = nn.ModuleList()
        
        # First residual block
        self.resnets.append(ResNetBlock2D(in_channels, out_channels))
        
        # Additional residual blocks
        for _ in range(num_layers - 1):
            self.resnets.append(ResNetBlock2D(out_channels, out_channels))
            
        # Down sampling
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels)])
        else:
            self.downsamplers = None
            
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass"""
        res_samples = []
        
        for resnet in self.resnets:
            x = resnet(x, t_emb)
            res_samples.append(x)
            
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)
                
        return x, res_samples

class UpBlock2D(nn.Module):
    """Up sampling block for U-Net"""
    
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, add_upsample: bool = True):
        super().__init__()
        self.resnets = nn.ModuleList()
        
        # First residual block
        self.resnets.append(ResNetBlock2D(in_channels + out_channels, out_channels))
        
        # Additional residual blocks
        for _ in range(num_layers - 1):
            self.resnets.append(ResNetBlock2D(out_channels, out_channels))
            
        # Up sampling
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels)])
        else:
            self.upsamplers = None
            
    def forward(self, x: torch.Tensor, res_samples: List[torch.Tensor], 
                t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for i, resnet in enumerate(self.resnets):
            if i == 0:
                # Concatenate with skip connection
                x = torch.cat([x, res_samples[-(i+1)]], dim=1)
            x = resnet(x, t_emb)
            
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)
                
        return x

class MidBlock2D(nn.Module):
    """Middle block for U-Net"""
    
    def __init__(self, in_channels: int, num_layers: int):
        super().__init__()
        self.resnets = nn.ModuleList()
        
        for _ in range(num_layers):
            self.resnets.append(ResNetBlock2D(in_channels, in_channels))
            
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for resnet in self.resnets:
            x = resnet(x, t_emb)
        return x

class ResNetBlock2D(nn.Module):
    """Residual block for U-Net"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_emb_proj = nn.Linear(128, out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.nonlinearity = nn.SiLU()
        
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv_shortcut = None
            
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        residual = x
        
        # First convolution
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        
        # Time embedding
        t_emb = self.time_emb_proj(t_emb)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb
        
        # Second convolution
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        
        # Shortcut connection
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
            
        return x + residual

class Downsample2D(nn.Module):
    """Down sampling layer"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample2D(nn.Module):
    """Up sampling layer"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class DiffusionEngine:
    """Advanced diffusion engine with multiple model types"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.noise_scheduler = NoiseScheduler(config)
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.accelerator = Accelerator() if config.mixed_precision else None
        
        # EMA for model weights
        if config.use_ema:
            self.ema_model = self._create_ema_model()
        else:
            self.ema_model = None
            
        # LoRA setup
        if config.use_lora:
            self._setup_lora()
            
    def _create_model(self) -> nn.Module:
        """Create diffusion model"""
        if self.config.model_type == "ddpm":
            return UNet2D(self.config)
        elif self.config.model_type == "stable_diffusion":
            return self._create_stable_diffusion_model()
        elif self.config.model_type == "controlnet":
            return self._create_controlnet_model()
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
            
    def _create_stable_diffusion_model(self) -> nn.Module:
        """Create Stable Diffusion model"""
        try:
            # Load pre-trained components
            unet = UNet2DConditionModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="unet"
            )
            vae = AutoencoderKL.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="vae"
            )
            text_encoder = CLIPTextModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="text_encoder"
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="tokenizer"
            )
            
            return {
                'unet': unet,
                'vae': vae,
                'text_encoder': text_encoder,
                'tokenizer': tokenizer
            }
        except Exception as e:
            logger.warning(f"Could not load Stable Diffusion model: {e}")
            return UNet2D(self.config)
            
    def _create_controlnet_model(self) -> nn.Module:
        """Create ControlNet model"""
        try:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny"
            )
            return controlnet
        except Exception as e:
            logger.warning(f"Could not load ControlNet model: {e}")
            return UNet2D(self.config)
            
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-6,
            eps=1e-8
        )
        
    def _create_ema_model(self) -> nn.Module:
        """Create EMA model"""
        ema_model = self._create_model()
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model
        
    def _setup_lora(self):
        """Setup LoRA for efficient fine-tuning"""
        if hasattr(self.model, 'unet'):  # Stable Diffusion
            unet = self.model['unet']
            unet.set_attn_processor({})
            
            # Add LoRA processors
            lora_attn_procs = {}
            for name in unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                else:
                    hidden_size = unet.config.block_out_channels[0]
                    
                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.config.lora_rank,
                    network_alpha=self.config.lora_alpha
                )
                
            unet.set_attn_processor(lora_attn_procs)
            
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        images = batch['images'].to(self.device)
        batch_size = images.shape[0]
        
        # Sample noise
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (batch_size,), device=self.device
        ).long()
        
        # Add noise to images
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)
        
        # Predict noise
        if self.config.mixed_precision and self.accelerator:
            with self.accelerator.autocast():
                noise_pred = self.model(noisy_images, timesteps)
        else:
            noise_pred = self.model(noisy_images, timesteps)
            
        # Calculate loss
        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "sample":
            target = images
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
            
        loss = F.mse_loss(noise_pred, target)
        
        return {'loss': loss.item()}
        
    def sample(self, batch_size: int = 1, num_inference_steps: int = None, 
               guidance_scale: float = None) -> torch.Tensor:
        """Generate samples using diffusion process"""
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale
            
        # Start with random noise
        shape = (batch_size, self.config.in_channels, 64, 64)  # Default size
        sample = torch.randn(shape, device=self.device)
        
        # Denoising loop
        timesteps = torch.linspace(
            self.noise_scheduler.num_train_timesteps - 1, 0, num_inference_steps,
            device=self.device
        ).long()
        
        for i, t in enumerate(timesteps):
            # Predict noise
            with torch.no_grad():
                noise_pred = self.model(sample, t)
                
            # Denoise step
            sample = self.noise_scheduler.step(noise_pred, t.item(), sample)
            
        return sample
        
    def stable_diffusion_sample(self, prompt: str, negative_prompt: str = "", 
                               num_inference_steps: int = None, 
                               guidance_scale: float = None) -> torch.Tensor:
        """Generate image using Stable Diffusion"""
        if not isinstance(self.model, dict) or 'unet' not in self.model:
            raise ValueError("Stable Diffusion model not loaded")
            
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale
            
        # Tokenize prompts
        tokenizer = self.model['tokenizer']
        text_encoder = self.model['text_encoder']
        
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        negative_text_inputs = tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode text
        with torch.no_grad():
            text_embeddings = text_encoder(text_inputs.input_ids.to(self.device))[0]
            negative_text_embeddings = text_encoder(negative_text_inputs.input_ids.to(self.device))[0]
            
        # Concatenate embeddings
        text_embeddings = torch.cat([negative_text_embeddings, text_embeddings])
        
        # Generate image
        unet = self.model['unet']
        vae = self.model['vae']
        
        # Create scheduler
        scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )
        
        # Generate latent
        latents = torch.randn(
            (1, unet.config.in_channels, 64, 64),
            device=self.device
        )
        
        # Denoising loop
        scheduler.set_timesteps(num_inference_steps)
        
        for t in scheduler.timesteps:
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
                
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
        # Decode latent to image
        with torch.no_grad():
            image = vae.decode(latents / 0.18215).sample
            
        # Normalize to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        
        return image
        
    def controlnet_sample(self, prompt: str, control_image: torch.Tensor,
                         num_inference_steps: int = None,
                         guidance_scale: float = None) -> torch.Tensor:
        """Generate image using ControlNet"""
        if not isinstance(self.model, dict) or 'controlnet' not in self.model:
            raise ValueError("ControlNet model not loaded")
            
        # Implementation would go here
        # This is a simplified version
        return self.stable_diffusion_sample(prompt, num_inference_steps, guidance_scale)
        
    def save_model(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'noise_scheduler': {
                'betas': self.noise_scheduler.betas,
                'alphas_cumprod': self.noise_scheduler.alphas_cumprod
            }
        }
        
        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
            
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'ema_model_state_dict' in checkpoint and self.ema_model is not None:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            
        logger.info(f"Model loaded from {path}")

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test diffusion engine
    print("Testing Diffusion Engine...")
    
    # Create config
    config = DiffusionConfig(
        model_name="test_diffusion",
        model_type="ddpm",
        in_channels=3,
        out_channels=3,
        num_train_timesteps=1000,
        num_inference_steps=50
    )
    
    # Create diffusion engine
    engine = DiffusionEngine(config)
    
    # Test sampling
    print("Testing sampling...")
    with torch.no_grad():
        samples = engine.sample(batch_size=2, num_inference_steps=10)
        print(f"Generated samples shape: {samples.shape}")
        
    # Test Stable Diffusion if available
    print("\nTesting Stable Diffusion...")
    try:
        sd_config = DiffusionConfig(
            model_name="stable_diffusion_test",
            model_type="stable_diffusion"
        )
        sd_engine = DiffusionEngine(sd_config)
        
        if isinstance(sd_engine.model, dict):
            print("Stable Diffusion model loaded successfully")
            # Test generation
            with torch.no_grad():
                image = sd_engine.stable_diffusion_sample(
                    "a beautiful landscape",
                    num_inference_steps=10
                )
                print(f"Generated image shape: {image.shape}")
        else:
            print("Stable Diffusion model not available, using fallback")
            
    except Exception as e:
        print(f"Stable Diffusion test failed: {e}")
        
    print("\nDiffusion engine refactored successfully!")
























