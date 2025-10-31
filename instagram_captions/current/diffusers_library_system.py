"""
Diffusers Library System
Comprehensive implementation using the Diffusers library for diffusion models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    # Core diffusion components
    DiffusionPipeline, DDIMScheduler, DDPMScheduler, PNDMScheduler, 
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler,
    HeunDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, UniPCMultistepScheduler,
    
    # Model architectures
    UNet2DConditionModel, UNet2DModel, AutoencoderKL, VQModel,
    
    # Pipelines
    StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline, StableDiffusionInpaintPipelineLegacy,
    StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline,
    TextToImagePipeline, ImageToImagePipeline, InpaintPipeline,
    
    # ControlNet
    ControlNetModel, StableDiffusionControlNetPipeline, ControlNetPipeline,
    
    # LoRA and PEFT
    LoraLoaderMixin, AttnProcsLayers,
    
    # Utilities
    DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline,
    StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline,
    TextToImagePipeline, ImageToImagePipeline, InpaintPipeline,
    
    # Training
    DDPMScheduler, DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, HeunDiscreteScheduler,
    DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler, UniPCMultistepScheduler,
    
    # Model loading
    AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting,
    
    # Utilities
    load_image, save_image, make_image_grid, randn_tensor
)
from transformers import (
    CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel,
    T5EncoderModel, T5Tokenizer, CLIPVisionModel, CLIPImageProcessor
)
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import logging
import time
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import math
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


@dataclass
class DiffusersConfig:
    """Configuration for Diffusers library usage."""
    
    # Model settings
    model_name: str = "runwayml/stable-diffusion-v1-5"
    model_type: str = "text2img"  # text2img, img2img, inpaint, controlnet, xl
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Pipeline settings
    pipeline_type: str = "stable-diffusion"  # stable-diffusion, stable-diffusion-xl, controlnet
    use_safety_checker: bool = False
    use_attention_slicing: bool = True
    use_memory_efficient_attention: bool = True
    use_xformers: bool = True
    
    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    height: int = 512
    width: int = 512
    batch_size: int = 1
    num_images_per_prompt: int = 1
    
    # Scheduler settings
    scheduler_type: str = "ddim"  # ddim, ddpm, pndm, euler, euler_a, lms, heun, dpm_solver, kdpm2, unipc
    beta_start: float = 0.00085
    beta_end: float = 0.012
    num_train_timesteps: int = 1000
    
    # Training settings
    learning_rate: float = 1e-4
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    
    # LoRA settings
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # ControlNet settings
    use_controlnet: bool = False
    controlnet_model: str = "lllyasviel/sd-controlnet-canny"
    controlnet_conditioning_scale: float = 1.0
    
    # Memory optimization
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False


class DiffusersSchedulerManager:
    """Manager for different Diffusers schedulers."""
    
    def __init__(self, config: DiffusersConfig):
        self.config = config
        self.schedulers = {}
        self._initialize_schedulers()
    
    def _initialize_schedulers(self):
        """Initialize all available schedulers."""
        scheduler_configs = {
            "ddim": {
                "class": DDIMScheduler,
                "params": {
                    "num_train_timesteps": self.config.num_train_timesteps,
                    "beta_start": self.config.beta_start,
                    "beta_end": self.config.beta_end,
                    "beta_schedule": "linear",
                    "prediction_type": "epsilon"
                }
            },
            "ddpm": {
                "class": DDPMScheduler,
                "params": {
                    "num_train_timesteps": self.config.num_train_timesteps,
                    "beta_start": self.config.beta_start,
                    "beta_end": self.config.beta_end,
                    "beta_schedule": "linear"
                }
            },
            "pndm": {
                "class": PNDMScheduler,
                "params": {
                    "num_train_timesteps": self.config.num_train_timesteps,
                    "beta_start": self.config.beta_start,
                    "beta_end": self.config.beta_end,
                    "beta_schedule": "linear"
                }
            },
            "euler": {
                "class": EulerDiscreteScheduler,
                "params": {
                    "num_train_timesteps": self.config.num_train_timesteps,
                    "beta_start": self.config.beta_start,
                    "beta_end": self.config.beta_end,
                    "beta_schedule": "linear"
                }
            },
            "euler_a": {
                "class": EulerAncestralDiscreteScheduler,
                "params": {
                    "num_train_timesteps": self.config.num_train_timesteps,
                    "beta_start": self.config.beta_start,
                    "beta_end": self.config.beta_end,
                    "beta_schedule": "linear"
                }
            },
            "lms": {
                "class": LMSDiscreteScheduler,
                "params": {
                    "num_train_timesteps": self.config.num_train_timesteps,
                    "beta_start": self.config.beta_start,
                    "beta_end": self.config.beta_end,
                    "beta_schedule": "linear"
                }
            },
            "heun": {
                "class": HeunDiscreteScheduler,
                "params": {
                    "num_train_timesteps": self.config.num_train_timesteps,
                    "beta_start": self.config.beta_start,
                    "beta_end": self.config.beta_end,
                    "beta_schedule": "linear"
                }
            },
            "dpm_solver": {
                "class": DPMSolverMultistepScheduler,
                "params": {
                    "num_train_timesteps": self.config.num_train_timesteps,
                    "beta_start": self.config.beta_start,
                    "beta_end": self.config.beta_end,
                    "beta_schedule": "linear"
                }
            },
            "dpm_solver_single": {
                "class": DPMSolverSinglestepScheduler,
                "params": {
                    "num_train_timesteps": self.config.num_train_timesteps,
                    "beta_start": self.config.beta_start,
                    "beta_end": self.config.beta_end,
                    "beta_schedule": "linear"
                }
            },
            "kdpm2": {
                "class": KDPM2DiscreteScheduler,
                "params": {
                    "num_train_timesteps": self.config.num_train_timesteps,
                    "beta_start": self.config.beta_start,
                    "beta_end": self.config.beta_end,
                    "beta_schedule": "linear"
                }
            },
            "kdpm2_a": {
                "class": KDPM2AncestralDiscreteScheduler,
                "params": {
                    "num_train_timesteps": self.config.num_train_timesteps,
                    "beta_start": self.config.beta_start,
                    "beta_end": self.config.beta_end,
                    "beta_schedule": "linear"
                }
            },
            "unipc": {
                "class": UniPCMultistepScheduler,
                "params": {
                    "num_train_timesteps": self.config.num_train_timesteps,
                    "beta_start": self.config.beta_start,
                    "beta_end": self.config.beta_end,
                    "beta_schedule": "linear"
                }
            }
        }
        
        for name, config in scheduler_configs.items():
            try:
                scheduler_class = config["class"]
                scheduler_params = config["params"]
                self.schedulers[name] = scheduler_class(**scheduler_params)
                logging.info(f"Initialized {name} scheduler")
            except Exception as e:
                logging.warning(f"Failed to initialize {name} scheduler: {e}")
    
    def get_scheduler(self, scheduler_type: str = None) -> Any:
        """Get a specific scheduler."""
        scheduler_type = scheduler_type or self.config.scheduler_type
        if scheduler_type in self.schedulers:
            return self.schedulers[scheduler_type]
        else:
            logging.warning(f"Scheduler {scheduler_type} not found, using DDIM")
            return self.schedulers.get("ddim", self.schedulers["ddim"])
    
    def compare_schedulers(self, num_steps: int = 50) -> Dict[str, Any]:
        """Compare different schedulers."""
        results = {}
        
        for name, scheduler in self.schedulers.items():
            try:
                # Set timesteps
                scheduler.set_timesteps(num_steps)
                
                # Get timesteps
                timesteps = scheduler.timesteps
                
                results[name] = {
                    "timesteps": timesteps.cpu().numpy(),
                    "num_timesteps": len(timesteps),
                    "step_size": timesteps[0].item() - timesteps[1].item() if len(timesteps) > 1 else 0
                }
                
            except Exception as e:
                logging.warning(f"Error comparing scheduler {name}: {e}")
        
        return results


class DiffusersPipelineManager:
    """Manager for different Diffusers pipelines."""
    
    def __init__(self, config: DiffusersConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the appropriate pipeline based on configuration."""
        try:
            if self.config.pipeline_type == "stable-diffusion":
                self._initialize_stable_diffusion()
            elif self.config.pipeline_type == "stable-diffusion-xl":
                self._initialize_stable_diffusion_xl()
            elif self.config.pipeline_type == "controlnet":
                self._initialize_controlnet()
            else:
                self._initialize_stable_diffusion()
                
        except Exception as e:
            logging.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def _initialize_stable_diffusion(self):
        """Initialize Stable Diffusion pipeline."""
        # Load pipeline with optimizations
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
            safety_checker=None if not self.config.use_safety_checker else None,
            requires_safety_checker=False if not self.config.use_safety_checker else True
        )
        
        # Apply optimizations
        self._apply_optimizations()
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        logging.info(f"Initialized Stable Diffusion pipeline with {self.config.model_name}")
    
    def _initialize_stable_diffusion_xl(self):
        """Initialize Stable Diffusion XL pipeline."""
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
            safety_checker=None if not self.config.use_safety_checker else None,
            requires_safety_checker=False if not self.config.use_safety_checker else True
        )
        
        # Apply optimizations
        self._apply_optimizations()
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        logging.info("Initialized Stable Diffusion XL pipeline")
    
    def _initialize_controlnet(self):
        """Initialize ControlNet pipeline."""
        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            self.config.controlnet_model,
            torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32
        )
        
        # Load base pipeline
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.model_name,
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
            safety_checker=None if not self.config.use_safety_checker else None,
            requires_safety_checker=False if not self.config.use_safety_checker else True
        )
        
        # Apply optimizations
        self._apply_optimizations()
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        logging.info(f"Initialized ControlNet pipeline with {self.config.controlnet_model}")
    
    def _apply_optimizations(self):
        """Apply memory and performance optimizations."""
        if self.config.use_attention_slicing:
            self.pipeline.enable_attention_slicing()
            logging.info("Enabled attention slicing")
        
        if self.config.use_memory_efficient_attention and hasattr(self.pipeline, 'enable_memory_efficient_attention'):
            self.pipeline.enable_memory_efficient_attention()
            logging.info("Enabled memory efficient attention")
        
        if self.config.use_xformers and hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logging.info("Enabled xformers memory efficient attention")
            except:
                logging.warning("xformers not available, skipping")
        
        if self.config.enable_vae_slicing:
            self.pipeline.enable_vae_slicing()
            logging.info("Enabled VAE slicing")
        
        if self.config.enable_vae_tiling:
            self.pipeline.enable_vae_tiling()
            logging.info("Enabled VAE tiling")
        
        if self.config.enable_model_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
            logging.info("Enabled model CPU offload")
        
        if self.config.enable_sequential_cpu_offload:
            self.pipeline.enable_sequential_cpu_offload()
            logging.info("Enabled sequential CPU offload")
    
    def generate_text2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = None,
        guidance_scale: float = None,
        height: int = None,
        width: int = None,
        num_images_per_prompt: int = None
    ) -> List[Image.Image]:
        """Generate images from text prompt."""
        # Set parameters
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale or self.config.guidance_scale
        height = height or self.config.height
        width = width or self.config.width
        num_images_per_prompt = num_images_per_prompt or self.config.num_images_per_prompt
        
        # Generate images
        with torch.no_grad():
            images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_images_per_prompt=num_images_per_prompt
            ).images
        
        return images
    
    def generate_img2img(
        self,
        prompt: str,
        image: Union[Image.Image, torch.Tensor],
        negative_prompt: str = "",
        num_inference_steps: int = None,
        guidance_scale: float = None,
        strength: float = 0.8
    ) -> List[Image.Image]:
        """Generate images from input image."""
        # Convert to PIL if needed
        if isinstance(image, torch.Tensor):
            image = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
        
        # Set parameters
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale or self.config.guidance_scale
        
        # Generate images
        with torch.no_grad():
            images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength
            ).images
        
        return images
    
    def generate_inpaint(
        self,
        prompt: str,
        image: Union[Image.Image, torch.Tensor],
        mask: Union[Image.Image, torch.Tensor],
        negative_prompt: str = "",
        num_inference_steps: int = None,
        guidance_scale: float = None
    ) -> List[Image.Image]:
        """Generate images with inpainting."""
        # Convert to PIL if needed
        if isinstance(image, torch.Tensor):
            image = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
        if isinstance(mask, torch.Tensor):
            mask = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8))
        
        # Set parameters
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale or self.config.guidance_scale
        
        # Generate images
        with torch.no_grad():
            images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images
        
        return images
    
    def generate_with_controlnet(
        self,
        prompt: str,
        control_image: Union[Image.Image, torch.Tensor],
        negative_prompt: str = "",
        num_inference_steps: int = None,
        guidance_scale: float = None,
        controlnet_conditioning_scale: float = None
    ) -> List[Image.Image]:
        """Generate images with ControlNet."""
        # Convert to PIL if needed
        if isinstance(control_image, torch.Tensor):
            control_image = Image.fromarray((control_image.cpu().numpy() * 255).astype(np.uint8))
        
        # Set parameters
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale or self.config.guidance_scale
        controlnet_conditioning_scale = controlnet_conditioning_scale or self.config.controlnet_conditioning_scale
        
        # Generate images
        with torch.no_grad():
            images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale
            ).images
        
        return images


class LoRAManager:
    """Manager for LoRA (Low-Rank Adaptation) with Diffusers."""
    
    def __init__(self, config: DiffusersConfig):
        self.config = config
        self.device = torch.device(config.device)
    
    def add_lora_to_pipeline(self, pipeline: Any) -> Any:
        """Add LoRA to a pipeline."""
        if not self.config.use_lora:
            return pipeline
        
        try:
            # Add LoRA config
            lora_config = {
                "r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": self.config.lora_dropout,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }
            
            # Apply LoRA
            pipeline.unet = self._apply_lora_to_unet(pipeline.unet, lora_config)
            
            logging.info(f"Applied LoRA with r={self.config.lora_r}, alpha={self.config.lora_alpha}")
            
        except Exception as e:
            logging.warning(f"Failed to apply LoRA: {e}")
        
        return pipeline
    
    def _apply_lora_to_unet(self, unet: nn.Module, lora_config: Dict) -> nn.Module:
        """Apply LoRA to UNet model."""
        # This is a simplified implementation
        # In practice, you would use PEFT library for proper LoRA implementation
        
        for name, module in unet.named_modules():
            if "attn" in name and ("to_q" in name or "to_v" in name):
                # Apply LoRA to attention layers
                if hasattr(module, 'weight'):
                    original_weight = module.weight.data.clone()
                    
                    # Create low-rank adaptation
                    rank = lora_config["r"]
                    lora_A = nn.Parameter(torch.randn(rank, original_weight.shape[1]) * 0.01)
                    lora_B = nn.Parameter(torch.zeros(original_weight.shape[0], rank))
                    
                    # Store LoRA parameters
                    module.register_parameter(f"{name}_lora_A", lora_A)
                    module.register_parameter(f"{name}_lora_B", lora_B)
                    
                    # Override forward method
                    original_forward = module.forward
                    
                    def lora_forward(*args, **kwargs):
                        output = original_forward(*args, **kwargs)
                        if hasattr(module, f"{name}_lora_A") and hasattr(module, f"{name}_lora_B"):
                            lora_A = getattr(module, f"{name}_lora_A")
                            lora_B = getattr(module, f"{name}_lora_B")
                            lora_output = torch.matmul(torch.matmul(args[0], lora_A.T), lora_B.T)
                            output = output + lora_config["lora_alpha"] / rank * lora_output
                        return output
                    
                    module.forward = lora_forward
        
        return unet
    
    def save_lora_weights(self, pipeline: Any, save_path: str):
        """Save LoRA weights."""
        if not self.config.use_lora:
            return
        
        try:
            lora_state_dict = {}
            
            # Collect LoRA parameters
            for name, module in pipeline.unet.named_modules():
                if hasattr(module, f"{name}_lora_A"):
                    lora_state_dict[f"{name}_lora_A"] = getattr(module, f"{name}_lora_A").data
                if hasattr(module, f"{name}_lora_B"):
                    lora_state_dict[f"{name}_lora_B"] = getattr(module, f"{name}_lora_B").data
            
            # Save weights
            torch.save(lora_state_dict, save_path)
            logging.info(f"Saved LoRA weights to {save_path}")
            
        except Exception as e:
            logging.error(f"Failed to save LoRA weights: {e}")
    
    def load_lora_weights(self, pipeline: Any, load_path: str):
        """Load LoRA weights."""
        if not self.config.use_lora:
            return pipeline
        
        try:
            lora_state_dict = torch.load(load_path, map_location=self.device)
            
            # Load LoRA parameters
            for name, module in pipeline.unet.named_modules():
                if f"{name}_lora_A" in lora_state_dict:
                    module.register_parameter(f"{name}_lora_A", 
                                           nn.Parameter(lora_state_dict[f"{name}_lora_A"]))
                if f"{name}_lora_B" in lora_state_dict:
                    module.register_parameter(f"{name}_lora_B", 
                                           nn.Parameter(lora_state_dict[f"{name}_lora_B"]))
            
            logging.info(f"Loaded LoRA weights from {load_path}")
            
        except Exception as e:
            logging.error(f"Failed to load LoRA weights: {e}")
        
        return pipeline


class DiffusersTrainingManager:
    """Manager for training with Diffusers."""
    
    def __init__(self, config: DiffusersConfig):
        self.config = config
        self.device = torch.device(config.device)
    
    def prepare_for_training(self, pipeline: Any) -> Tuple[nn.Module, Any]:
        """Prepare pipeline for training."""
        # Set to training mode
        pipeline.unet.train()
        
        # Freeze VAE and text encoder
        pipeline.vae.eval()
        pipeline.text_encoder.eval()
        
        # Disable gradient computation for frozen components
        for param in pipeline.vae.parameters():
            param.requires_grad = False
        for param in pipeline.text_encoder.parameters():
            param.requires_grad = False
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            pipeline.unet.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08
        )
        
        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs
        )
        
        return pipeline.unet, optimizer, scheduler
    
    def training_step(
        self,
        unet: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        tokenizer: Any,
        scheduler: Any,
        optimizer: torch.optim.Optimizer,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Single training step."""
        # Move batch to device
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        
        # Encode images
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215
        
        # Encode text
        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids)[0]
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        
        # Add noise to latents
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return {"loss": loss.item()}


class DiffusersAnalyzer:
    """Analyzer for Diffusers pipeline performance."""
    
    def __init__(self, pipeline_manager: DiffusersPipelineManager):
        self.pipeline_manager = pipeline_manager
    
    def benchmark_generation(
        self,
        prompts: List[str],
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """Benchmark generation performance."""
        results = {
            "generation_times": [],
            "memory_usage": [],
            "image_qualities": []
        }
        
        for i in range(num_runs):
            for prompt in prompts:
                # Measure time
                start_time = time.time()
                
                # Generate image
                images = self.pipeline_manager.generate_text2img(
                    prompt=prompt,
                    num_inference_steps=20  # Use fewer steps for benchmarking
                )
                
                generation_time = time.time() - start_time
                results["generation_times"].append(generation_time)
                
                # Measure memory
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                    results["memory_usage"].append(memory_used)
                
                # Analyze image quality (simplified)
                if images:
                    image = images[0]
                    if isinstance(image, Image.Image):
                        # Convert to numpy for analysis
                        img_array = np.array(image)
                        quality_score = self._calculate_image_quality(img_array)
                        results["image_qualities"].append(quality_score)
        
        return results
    
    def _calculate_image_quality(self, image: np.ndarray) -> float:
        """Calculate a simple image quality metric."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Calculate sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Calculate contrast
        contrast = gray.std()
        
        # Combine metrics
        quality_score = (sharpness + contrast) / 2
        
        return float(quality_score)
    
    def compare_schedulers(
        self,
        prompt: str,
        schedulers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare different schedulers."""
        results = {}
        
        for name, scheduler in schedulers.items():
            try:
                # Set scheduler
                self.pipeline_manager.pipeline.scheduler = scheduler
                
                # Generate image
                start_time = time.time()
                images = self.pipeline_manager.generate_text2img(
                    prompt=prompt,
                    num_inference_steps=20
                )
                generation_time = time.time() - start_time
                
                results[name] = {
                    "generation_time": generation_time,
                    "num_images": len(images) if images else 0
                }
                
            except Exception as e:
                logging.warning(f"Error with scheduler {name}: {e}")
                results[name] = {"error": str(e)}
        
        return results
    
    def analyze_pipeline_components(self) -> Dict[str, Any]:
        """Analyze pipeline components."""
        pipeline = self.pipeline_manager.pipeline
        
        analysis = {
            "unet_parameters": sum(p.numel() for p in pipeline.unet.parameters()),
            "vae_parameters": sum(p.numel() for p in pipeline.vae.parameters()),
            "text_encoder_parameters": sum(p.numel() for p in pipeline.text_encoder.parameters()),
            "total_parameters": sum(p.numel() for p in pipeline.parameters()),
            "device": str(next(pipeline.parameters()).device),
            "dtype": str(next(pipeline.parameters()).dtype)
        }
        
        return analysis


class DiffusersUtilities:
    """Utility functions for Diffusers."""
    
    @staticmethod
    def create_image_grid(images: List[Image.Image], rows: int = None, cols: int = None) -> Image.Image:
        """Create a grid of images."""
        return make_image_grid(images, rows=rows, cols=cols)
    
    @staticmethod
    def save_images(images: List[Image.Image], save_dir: str, prefix: str = "generated"):
        """Save multiple images."""
        os.makedirs(save_dir, exist_ok=True)
        
        for i, image in enumerate(images):
            save_path = os.path.join(save_dir, f"{prefix}_{i:03d}.png")
            image.save(save_path)
        
        logging.info(f"Saved {len(images)} images to {save_dir}")
    
    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """Load image using Diffusers utilities."""
        return load_image(image_path)
    
    @staticmethod
    def save_image(image: Image.Image, save_path: str):
        """Save image using Diffusers utilities."""
        save_image(image, save_path)
    
    @staticmethod
    def create_random_tensor(shape: Tuple[int, ...], device: str = "cpu") -> torch.Tensor:
        """Create random tensor using Diffusers utilities."""
        return randn_tensor(shape, device=device)


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Diffusers Library Demonstration ===\n")
    
    # Configuration
    config = DiffusersConfig(
        model_name="runwayml/stable-diffusion-v1-5",
        pipeline_type="stable-diffusion",
        num_inference_steps=20,
        guidance_scale=7.5,
        use_lora=False,
        use_controlnet=False
    )
    
    # 1. Test scheduler manager
    print("1. Testing Scheduler Manager...")
    
    scheduler_manager = DiffusersSchedulerManager(config)
    
    # Get default scheduler
    default_scheduler = scheduler_manager.get_scheduler()
    print(f"Default scheduler: {type(default_scheduler).__name__}")
    
    # Compare schedulers
    scheduler_comparison = scheduler_manager.compare_schedulers(num_steps=20)
    print(f"Available schedulers: {list(scheduler_comparison.keys())}")
    
    # 2. Test pipeline manager
    print("\n2. Testing Pipeline Manager...")
    
    try:
        pipeline_manager = DiffusersPipelineManager(config)
        
        # Test text-to-image generation
        test_prompt = "A beautiful landscape with mountains and trees, high quality, detailed"
        test_negative_prompt = "blurry, low quality, distorted"
        
        print("Generating image...")
        images = pipeline_manager.generate_text2img(
            prompt=test_prompt,
            negative_prompt=test_negative_prompt,
            num_inference_steps=10  # Use fewer steps for testing
        )
        
        print(f"Generated {len(images)} images")
        
        # Save images
        DiffusersUtilities.save_images(images, "generated_images", "test")
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")
    
    # 3. Test LoRA manager
    print("\n3. Testing LoRA Manager...")
    
    lora_config = DiffusersConfig(
        model_name="runwayml/stable-diffusion-v1-5",
        use_lora=True,
        lora_r=16,
        lora_alpha=32
    )
    
    lora_manager = LoRAManager(lora_config)
    
    if 'pipeline_manager' in locals():
        # Apply LoRA to pipeline
        pipeline_with_lora = lora_manager.add_lora_to_pipeline(pipeline_manager.pipeline)
        print("Applied LoRA to pipeline")
        
        # Save LoRA weights
        lora_manager.save_lora_weights(pipeline_with_lora, "lora_weights.pt")
    
    # 4. Test training manager
    print("\n4. Testing Training Manager...")
    
    training_manager = DiffusersTrainingManager(config)
    
    if 'pipeline_manager' in locals():
        # Prepare for training
        unet, optimizer, scheduler = training_manager.prepare_for_training(pipeline_manager.pipeline)
        print(f"Prepared for training: UNet parameters = {sum(p.numel() for p in unet.parameters()):,}")
    
    # 5. Test analyzer
    print("\n5. Testing Diffusers Analyzer...")
    
    if 'pipeline_manager' in locals():
        analyzer = DiffusersAnalyzer(pipeline_manager)
        
        # Benchmark generation
        test_prompts = [
            "A cat sitting on a chair",
            "A beautiful sunset over the ocean",
            "A modern city skyline at night"
        ]
        
        benchmark_results = analyzer.benchmark_generation(test_prompts, num_runs=1)
        print(f"Average generation time: {np.mean(benchmark_results['generation_times']):.2f}s")
        
        # Analyze pipeline components
        component_analysis = analyzer.analyze_pipeline_components()
        print(f"Total parameters: {component_analysis['total_parameters']:,}")
        print(f"Device: {component_analysis['device']}")
    
    # 6. Test different pipeline types
    print("\n6. Testing Different Pipeline Types...")
    
    # Test ControlNet
    try:
        controlnet_config = DiffusersConfig(
            pipeline_type="controlnet",
            use_controlnet=True,
            controlnet_model="lllyasviel/sd-controlnet-canny"
        )
        
        controlnet_pipeline = DiffusersPipelineManager(controlnet_config)
        
        # Create dummy control image
        control_image = Image.new('RGB', (512, 512), color='white')
        
        # Generate with ControlNet
        controlnet_images = controlnet_pipeline.generate_with_controlnet(
            prompt="A beautiful painting of a landscape",
            control_image=control_image,
            num_inference_steps=10
        )
        
        print(f"Generated {len(controlnet_images)} ControlNet images")
        
    except Exception as e:
        print(f"ControlNet test failed: {e}")
    
    # 7. Test utilities
    print("\n7. Testing Diffusers Utilities...")
    
    # Create random tensor
    random_tensor = DiffusersUtilities.create_random_tensor((1, 3, 64, 64))
    print(f"Created random tensor: {random_tensor.shape}")
    
    # Create image grid (if we have images)
    if 'images' in locals() and images:
        image_grid = DiffusersUtilities.create_image_grid(images, rows=2, cols=2)
        print(f"Created image grid: {image_grid.size}")
    
    print("\n=== Demonstration Completed Successfully! ===")





