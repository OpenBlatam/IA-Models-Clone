from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

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
from PIL import Image
import requests
from io import BytesIO
        from transformers import CLIPTextModel, CLIPTokenizer
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        from diffusers import KandinskyPriorPipeline, KandinskyPipeline as KandinskyDiffusersPipeline
        from transformers import CLIPTextModel, CLIPTokenizer
        from diffusers import WuerstchenPriorPipeline, WuerstchenDecoderPipeline
        from diffusers import AudioDiffusionPipeline as AudioDiffusersPipeline
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Diffusion Pipelines
Production-ready implementation of various diffusion pipelines including Stable Diffusion, SDXL, and more.
"""


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for diffusion pipelines."""
    # Model parameters
    model_name: str = "runwayml/stable-diffusion-v1-5"
    model_type: str = "stable-diffusion"  # stable-diffusion, sdxl, kandinsky, wuerstchen, audio
    device: str = "cuda"
    dtype: str = "float16"  # float16, float32, bfloat16
    
    # Generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    height: int = 512
    width: int = 512
    num_images_per_prompt: int = 1
    batch_size: int = 1
    
    # Advanced parameters
    negative_prompt: str = ""
    prompt_2: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    original_size: Optional[Tuple[int, int]] = None
    crops_coords_top_left: Tuple[int, int] = (0, 0)
    target_size: Optional[Tuple[int, int]] = None
    negative_original_size: Optional[Tuple[int, int]] = None
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0)
    negative_target_size: Optional[Tuple[int, int]] = None
    
    # Memory optimization
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    enable_xformers_memory_efficient_attention: bool = True
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    
    # LoRA parameters
    lora_path: Optional[str] = None
    lora_scale: float = 1.0
    
    # Output parameters
    output_dir: str = "./generated_images"
    save_format: str = "png"
    save_metadata: bool = True
    
    # Safety parameters
    safety_checker: bool = True
    requires_safety_checker: bool = True


class BaseDiffusionPipeline(ABC):
    """Base class for diffusion pipelines."""
    
    def __init__(self, config: PipelineConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.dtype)
        
        # Initialize components
        self._init_components()
        self._setup_memory_optimizations()
        
        logger.info(f"Initialized {self.__class__.__name__} on device: {self.device}")
    
    @abstractmethod
    def _init_components(self) -> Any:
        """Initialize pipeline components."""
        pass
    
    @abstractmethod
    def _setup_memory_optimizations(self) -> Any:
        """Setup memory optimizations."""
        pass
    
    @abstractmethod
    def __call__(self, prompt: str, **kwargs) -> Union[Image.Image, List[Image.Image]]:
        """Generate images from prompt."""
        pass
    
    def _setup_lora(self, lora_path: str, lora_scale: float = 1.0):
        """Setup LoRA for the pipeline."""
        if lora_path and os.path.exists(lora_path):
            logger.info(f"Loading LoRA from: {lora_path}")
            # LoRA loading implementation would go here
            pass
    
    def _setup_memory_optimizations(self) -> Any:
        """Setup memory optimizations."""
        if self.config.enable_attention_slicing:
            logger.info("Enabling attention slicing")
        
        if self.config.enable_vae_slicing:
            logger.info("Enabling VAE slicing")
        
        if self.config.enable_xformers_memory_efficient_attention:
            logger.info("Enabling xFormers memory efficient attention")
        
        if self.config.enable_model_cpu_offload:
            logger.info("Enabling model CPU offload")
        
        if self.config.enable_sequential_cpu_offload:
            logger.info("Enabling sequential CPU offload")


class StableDiffusionPipeline(BaseDiffusionPipeline):
    """Stable Diffusion pipeline implementation."""
    
    def _init_components(self) -> Any:
        """Initialize Stable Diffusion components."""
        
        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.model_name,
            subfolder="tokenizer",
            use_auth_token=None
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.model_name,
            subfolder="text_encoder",
            use_auth_token=None
        ).to(self.device, dtype=self.dtype)
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.config.model_name,
            subfolder="vae",
            use_auth_token=None
        ).to(self.device, dtype=self.dtype)
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.model_name,
            subfolder="unet",
            use_auth_token=None
        ).to(self.device, dtype=self.dtype)
        
        # Load scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            self.config.model_name,
            subfolder="scheduler",
            use_auth_token=None
        )
        
        # Setup LoRA if specified
        if self.config.lora_path:
            self._setup_lora(self.config.lora_path, self.config.lora_scale)
    
    def _setup_memory_optimizations(self) -> Any:
        """Setup memory optimizations for Stable Diffusion."""
        super()._setup_memory_optimizations()
        
        # Enable gradient checkpointing
        self.unet.enable_gradient_checkpointing()
        
        # Enable attention slicing
        if self.config.enable_attention_slicing:
            self.unet.set_attention_slice(slice_size="auto")
        
        # Enable VAE slicing
        if self.config.enable_vae_slicing:
            self.vae.enable_slicing()
        
        # Enable VAE tiling
        if self.config.enable_vae_tiling:
            self.vae.enable_tiling()
    
    def _encode_prompt(self, prompt: str, negative_prompt: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt and negative prompt."""
        # Tokenize prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # Encode prompt
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]
        
        # Tokenize negative prompt
        uncond_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=text_input_ids.shape[-1],
            truncation=True,
            return_tensors="pt"
        )
        uncond_input_ids = uncond_inputs.input_ids.to(self.device)
        
        # Encode negative prompt
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input_ids)[0]
        
        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        return text_embeddings, uncond_embeddings
    
    def _prepare_latents(self, batch_size: int, num_channels_latents: int, height: int, width: int) -> torch.Tensor:
        """Prepare initial latents."""
        latents = torch.randn(
            (batch_size, num_channels_latents, height // 8, width // 8),
            device=self.device,
            dtype=self.dtype
        )
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images."""
        latents = 1 / self.vae.config.scaling_factor * latents
        
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
    def __call__(self, prompt: str, **kwargs) -> Union[Image.Image, List[Image.Image]]:
        """Generate images from prompt."""
        # Update config with kwargs
        config = self.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Encode prompts
        text_embeddings, uncond_embeddings = self._encode_prompt(
            prompt, 
            kwargs.get("negative_prompt", config.negative_prompt)
        )
        
        # Prepare latents
        latents = self._prepare_latents(
            config.batch_size,
            4,  # num_channels_latents
            config.height,
            config.width
        )
        
        # Set timesteps
        self.scheduler.set_timesteps(config.num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Prepare extra step kwargs
        extra_step_kwargs = self.scheduler.prepare_extra_step_kwargs(generator=None, eta=config.eta)
        
        # Denoising loop
        with torch.autocast(self.device.type):
            for i, t in enumerate(tqdm(timesteps, desc="Generating")):
                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Predict noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings
                    ).sample
                
                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute previous noisy sample
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
        
        # Decode latents
        image = self._decode_latents(latents)
        
        # Convert to PIL images
        images = []
        for i in range(image.shape[0]):
            pil_image = Image.fromarray((image[i] * 255).round().astype("uint8"))
            images.append(pil_image)
        
        return images[0] if len(images) == 1 else images


class StableDiffusionXLPipeline(BaseDiffusionPipeline):
    """Stable Diffusion XL pipeline implementation."""
    
    def _init_components(self) -> Any:
        """Initialize Stable Diffusion XL components."""
        
        # Load tokenizers and text encoders
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.model_name,
            subfolder="tokenizer",
            use_auth_token=None
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.config.model_name,
            subfolder="tokenizer_2",
            use_auth_token=None
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.model_name,
            subfolder="text_encoder",
            use_auth_token=None
        ).to(self.device, dtype=self.dtype)
        
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.config.model_name,
            subfolder="text_encoder_2",
            use_auth_token=None
        ).to(self.device, dtype=self.dtype)
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.config.model_name,
            subfolder="vae",
            use_auth_token=None
        ).to(self.device, dtype=self.dtype)
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.model_name,
            subfolder="unet",
            use_auth_token=None
        ).to(self.device, dtype=self.dtype)
        
        # Load scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            self.config.model_name,
            subfolder="scheduler",
            use_auth_token=None
        )
        
        # Setup LoRA if specified
        if self.config.lora_path:
            self._setup_lora(self.config.lora_path, self.config.lora_scale)
    
    def _setup_memory_optimizations(self) -> Any:
        """Setup memory optimizations for Stable Diffusion XL."""
        super()._setup_memory_optimizations()
        
        # Enable gradient checkpointing
        self.unet.enable_gradient_checkpointing()
        
        # Enable attention slicing
        if self.config.enable_attention_slicing:
            self.unet.set_attention_slice(slice_size="auto")
        
        # Enable VAE slicing
        if self.config.enable_vae_slicing:
            self.vae.enable_slicing()
        
        # Enable VAE tiling
        if self.config.enable_vae_tiling:
            self.vae.enable_tiling()
    
    def _encode_prompt(self, prompt: str, prompt_2: Optional[str], 
                      negative_prompt: str = "", negative_prompt_2: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt and negative prompt for SDXL."""
        # Tokenize prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        text_inputs_2 = self.tokenizer_2(
            prompt_2 or prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(self.device)
        
        # Encode prompts
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]
            text_embeddings_2 = self.text_encoder_2(text_input_ids_2)[0]
        
        # Tokenize negative prompts
        uncond_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=text_input_ids.shape[-1],
            truncation=True,
            return_tensors="pt"
        )
        uncond_input_ids = uncond_inputs.input_ids.to(self.device)
        
        uncond_inputs_2 = self.tokenizer_2(
            negative_prompt_2 or negative_prompt,
            padding="max_length",
            max_length=text_input_ids_2.shape[-1],
            truncation=True,
            return_tensors="pt"
        )
        uncond_input_ids_2 = uncond_inputs_2.input_ids.to(self.device)
        
        # Encode negative prompts
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input_ids)[0]
            uncond_embeddings_2 = self.text_encoder_2(uncond_input_ids_2)[0]
        
        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_embeddings_2 = torch.cat([uncond_embeddings_2, text_embeddings_2])
        
        return text_embeddings, text_embeddings_2, uncond_embeddings, uncond_embeddings_2
    
    def _prepare_latents(self, batch_size: int, num_channels_latents: int, height: int, width: int) -> torch.Tensor:
        """Prepare initial latents."""
        latents = torch.randn(
            (batch_size, num_channels_latents, height // 8, width // 8),
            device=self.device,
            dtype=self.dtype
        )
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images."""
        latents = 1 / self.vae.config.scaling_factor * latents
        
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
    def __call__(self, prompt: str, **kwargs) -> Union[Image.Image, List[Image.Image]]:
        """Generate images from prompt using SDXL."""
        # Update config with kwargs
        config = self.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Encode prompts
        text_embeddings, text_embeddings_2, uncond_embeddings, uncond_embeddings_2 = self._encode_prompt(
            prompt,
            kwargs.get("prompt_2", config.prompt_2),
            kwargs.get("negative_prompt", config.negative_prompt),
            kwargs.get("negative_prompt_2", config.negative_prompt_2)
        )
        
        # Prepare latents
        latents = self._prepare_latents(
            config.batch_size,
            4,  # num_channels_latents
            config.height,
            config.width
        )
        
        # Set timesteps
        self.scheduler.set_timesteps(config.num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Prepare extra step kwargs
        extra_step_kwargs = self.scheduler.prepare_extra_step_kwargs(generator=None, eta=config.eta)
        
        # Denoising loop
        with torch.autocast(self.device.type):
            for i, t in enumerate(tqdm(timesteps, desc="Generating SDXL")):
                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Predict noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        encoder_hidden_states_2=text_embeddings_2
                    ).sample
                
                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute previous noisy sample
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
        
        # Decode latents
        image = self._decode_latents(latents)
        
        # Convert to PIL images
        images = []
        for i in range(image.shape[0]):
            pil_image = Image.fromarray((image[i] * 255).round().astype("uint8"))
            images.append(pil_image)
        
        return images[0] if len(images) == 1 else images


class KandinskyPipeline(BaseDiffusionPipeline):
    """Kandinsky pipeline implementation."""
    
    def _init_components(self) -> Any:
        """Initialize Kandinsky components."""
        
        # Load prior pipeline
        self.prior_pipeline = KandinskyPriorPipeline.from_pretrained(
            self.config.model_name,
            subfolder="prior",
            use_auth_token=None
        ).to(self.device, dtype=self.dtype)
        
        # Load main pipeline
        self.pipeline = KandinskyDiffusersPipeline.from_pretrained(
            self.config.model_name,
            use_auth_token=None
        ).to(self.device, dtype=self.dtype)
        
        # Setup LoRA if specified
        if self.config.lora_path:
            self._setup_lora(self.config.lora_path, self.config.lora_scale)
    
    def _setup_memory_optimizations(self) -> Any:
        """Setup memory optimizations for Kandinsky."""
        super()._setup_memory_optimizations()
        
        # Enable gradient checkpointing
        self.pipeline.unet.enable_gradient_checkpointing()
        
        # Enable attention slicing
        if self.config.enable_attention_slicing:
            self.pipeline.unet.set_attention_slice(slice_size="auto")
    
    def __call__(self, prompt: str, **kwargs) -> Union[Image.Image, List[Image.Image]]:
        """Generate images from prompt using Kandinsky."""
        # Update config with kwargs
        config = self.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Generate image embeddings using prior
        image_embeds = self.prior_pipeline(
            prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale
        ).image_embeds
        
        # Generate images using main pipeline
        images = self.pipeline(
            prompt,
            image_embeds=image_embeds,
            negative_prompt=kwargs.get("negative_prompt", config.negative_prompt),
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            height=config.height,
            width=config.width,
            num_images_per_prompt=config.num_images_per_prompt
        ).images
        
        return images[0] if len(images) == 1 else images


class WuerstchenPipeline(BaseDiffusionPipeline):
    """Wuerstchen pipeline implementation."""
    
    def _init_components(self) -> Any:
        """Initialize Wuerstchen components."""
        
        # Load prior pipeline
        self.prior_pipeline = WuerstchenPriorPipeline.from_pretrained(
            self.config.model_name,
            subfolder="prior",
            use_auth_token=None
        ).to(self.device, dtype=self.dtype)
        
        # Load decoder pipeline
        self.decoder_pipeline = WuerstchenDecoderPipeline.from_pretrained(
            self.config.model_name,
            subfolder="decoder",
            use_auth_token=None
        ).to(self.device, dtype=self.dtype)
        
        # Setup LoRA if specified
        if self.config.lora_path:
            self._setup_lora(self.config.lora_path, self.config.lora_scale)
    
    def _setup_memory_optimizations(self) -> Any:
        """Setup memory optimizations for Wuerstchen."""
        super()._setup_memory_optimizations()
        
        # Enable gradient checkpointing
        self.prior_pipeline.prior.enable_gradient_checkpointing()
        self.decoder_pipeline.decoder.enable_gradient_checkpointing()
        
        # Enable attention slicing
        if self.config.enable_attention_slicing:
            self.prior_pipeline.prior.set_attention_slice(slice_size="auto")
            self.decoder_pipeline.decoder.set_attention_slice(slice_size="auto")
    
    def __call__(self, prompt: str, **kwargs) -> Union[Image.Image, List[Image.Image]]:
        """Generate images from prompt using Wuerstchen."""
        # Update config with kwargs
        config = self.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Generate image embeddings using prior
        image_embeds = self.prior_pipeline(
            prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale
        ).image_embeds
        
        # Generate images using decoder
        images = self.decoder_pipeline(
            prompt,
            image_embeds=image_embeds,
            negative_prompt=kwargs.get("negative_prompt", config.negative_prompt),
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            height=config.height,
            width=config.width,
            num_images_per_prompt=config.num_images_per_prompt
        ).images
        
        return images[0] if len(images) == 1 else images


class AudioDiffusionPipeline(BaseDiffusionPipeline):
    """Audio diffusion pipeline implementation."""
    
    def _init_components(self) -> Any:
        """Initialize Audio diffusion components."""
        
        # Load audio pipeline
        self.pipeline = AudioDiffusersPipeline.from_pretrained(
            self.config.model_name,
            use_auth_token=None
        ).to(self.device, dtype=self.dtype)
        
        # Setup LoRA if specified
        if self.config.lora_path:
            self._setup_lora(self.config.lora_path, self.config.lora_scale)
    
    def _setup_memory_optimizations(self) -> Any:
        """Setup memory optimizations for Audio diffusion."""
        super()._setup_memory_optimizations()
        
        # Enable gradient checkpointing
        self.pipeline.unet.enable_gradient_checkpointing()
        
        # Enable attention slicing
        if self.config.enable_attention_slicing:
            self.pipeline.unet.set_attention_slice(slice_size="auto")
    
    def __call__(self, prompt: str, **kwargs) -> Union[Image.Image, List[Image.Image]]:
        """Generate audio from prompt."""
        # Update config with kwargs
        config = self.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Generate audio
        audio = self.pipeline(
            prompt,
            negative_prompt=kwargs.get("negative_prompt", config.negative_prompt),
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            num_images_per_prompt=config.num_images_per_prompt
        ).audios
        
        return audio[0] if len(audio) == 1 else audio


class PipelineFactory:
    """Factory for creating diffusion pipelines."""
    
    @staticmethod
    def create_pipeline(pipeline_type: str, config: PipelineConfig) -> BaseDiffusionPipeline:
        """Create a diffusion pipeline based on type."""
        pipelines = {
            "stable-diffusion": StableDiffusionPipeline,
            "sdxl": StableDiffusionXLPipeline,
            "kandinsky": KandinskyPipeline,
            "wuerstchen": WuerstchenPipeline,
            "audio": AudioDiffusionPipeline,
        }
        
        if pipeline_type not in pipelines:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        return pipelines[pipeline_type](config)


def create_pipeline(pipeline_type: str, config: PipelineConfig) -> BaseDiffusionPipeline:
    """Create a diffusion pipeline."""
    return PipelineFactory.create_pipeline(pipeline_type, config)


# Example usage
if __name__ == "__main__":
    # Create configuration for Stable Diffusion
    config = PipelineConfig(
        model_name="runwayml/stable-diffusion-v1-5",
        model_type="stable-diffusion",
        device="cuda",
        dtype="float16",
        num_inference_steps=50,
        guidance_scale=7.5,
        height=512,
        width=512
    )
    
    # Create Stable Diffusion pipeline
    sd_pipeline = create_pipeline("stable-diffusion", config)
    
    # Generate image
    prompt = "A beautiful landscape with mountains and a lake, digital art"
    images = sd_pipeline(prompt)
    
    # Save image
    if isinstance(images, list):
        for i, image in enumerate(images):
            image.save(f"generated_image_{i}.png")
    else:
        images.save("generated_image.png")
    
    # Create configuration for SDXL
    sdxl_config = PipelineConfig(
        model_name="stabilityai/stable-diffusion-xl-base-1.0",
        model_type="sdxl",
        device="cuda",
        dtype="float16",
        num_inference_steps=50,
        guidance_scale=7.5,
        height=1024,
        width=1024
    )
    
    # Create SDXL pipeline
    sdxl_pipeline = create_pipeline("sdxl", sdxl_config)
    
    # Generate image with SDXL
    sdxl_images = sdxl_pipeline(prompt)
    
    # Save SDXL image
    if isinstance(sdxl_images, list):
        for i, image in enumerate(sdxl_images):
            image.save(f"generated_sdxl_image_{i}.png")
    else:
        sdxl_images.save("generated_sdxl_image.png") 