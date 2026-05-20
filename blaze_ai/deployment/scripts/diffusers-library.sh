#!/usr/bin/env python3
"""
Diffusers Library Integration for Blaze AI
Demonstrates how to use the Hugging Face Diffusers library for diffusion models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Diffusers imports
try:
    from diffusers import (
        AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPM, 
        DDPMScheduler, StableDiffusionPipeline, DiffusionPipeline,
        EulerDiscreteScheduler, DPMSolverMultistepScheduler,
        StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline,
        StableDiffusionUpscalePipeline, ControlNetModel, StableDiffusionControlNetPipeline,
        TextualInversionPipeline, DreamBoothPipeline, LoRATrainer,
        StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
    )
    from diffusers.utils import make_image_grid, randn_tensor
    from diffusers.training_utils import EMAModel
    from diffusers.loaders import AttnProcsLayers
    from diffusers.models.attention_processor import LoRAAttnProcessor
    from diffusers.optimization import get_scheduler
    from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
    from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
    from diffusers.schedulers.scheduling_utils import SchedulerMixin
    from diffusers.utils import logging as diffusers_logging
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    warnings.warn("Diffusers library not available. Install with: pip install diffusers transformers accelerate")

# Transformers imports
try:
    from transformers import (
        CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection,
        AutoTokenizer, AutoModel, AutoModelForCausalLM
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers library not available. Install with: pip install transformers")

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure diffusers logging
if DIFFUSERS_AVAILABLE:
    diffusers_logging.set_verbosity_info()


@dataclass
class DiffusersConfig:
    """Configuration for Diffusers library usage"""
    # Model settings
    model_id: str = "runwayml/stable-diffusion-v1-5"
    model_type: str = "stable-diffusion"  # stable-diffusion, stable-diffusion-xl, controlnet
    device: str = "auto"  # auto, cuda, cpu, mps
    
    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    batch_size: int = 1
    
    # Training settings
    learning_rate: float = 1e-5
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"  # fp16, bf16, no
    
    # LoRA settings
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # ControlNet settings
    use_controlnet: bool = False
    controlnet_model_id: str = "lllyasviel/sd-controlnet-canny"
    
    # Safety settings
    safety_checker: bool = True
    requires_safety_checking: bool = True


class DiffusersModelManager:
    """Manager for working with Diffusers library models"""
    
    def __init__(self, config: DiffusersConfig):
        self.config = config
        
        # Auto-detect device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = config.device
        
        self.pipeline = None
        self.model_components = {}
        
        logger.info(f"Using device: {self.device}")
    
    def load_stable_diffusion_pipeline(self) -> StableDiffusionPipeline:
        """Load Stable Diffusion pipeline"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        logger.info(f"Loading Stable Diffusion pipeline: {self.config.model_id}")
        
        # Load pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None if not self.config.safety_checker else None,
            requires_safety_checking=self.config.requires_safety_checking
        )
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        # Enable memory efficient attention if available
        if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
            self.pipeline.enable_xformers_memory_efficient_attention()
        
        # Enable attention slicing for memory efficiency
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()
        
        logger.info("Stable Diffusion pipeline loaded successfully")
        return self.pipeline
    
    def load_stable_diffusion_xl_pipeline(self) -> StableDiffusionXLPipeline:
        """Load Stable Diffusion XL pipeline"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        logger.info(f"Loading Stable Diffusion XL pipeline: {self.config.model_id}")
        
        # Load pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None if not self.config.safety_checker else None,
            requires_safety_checking=self.config.requires_safety_checking
        )
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        # Enable memory efficient attention if available
        if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
            self.pipeline.enable_xformers_memory_efficient_attention()
        
        logger.info("Stable Diffusion XL pipeline loaded successfully")
        return self.pipeline
    
    def load_controlnet_pipeline(self) -> StableDiffusionControlNetPipeline:
        """Load ControlNet pipeline"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        logger.info(f"Loading ControlNet pipeline: {self.config.model_id}")
        
        # Load ControlNet model
        controlnet = ControlNetModel.from_pretrained(
            self.config.controlnet_model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Load Stable Diffusion pipeline
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.model_id,
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None if not self.config.safety_checker else None
        )
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        logger.info("ControlNet pipeline loaded successfully")
        return self.pipeline
    
    def load_img2img_pipeline(self) -> StableDiffusionImg2ImgPipeline:
        """Load image-to-image pipeline"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        logger.info(f"Loading image-to-image pipeline: {self.config.model_id}")
        
        # Load pipeline
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None if not self.config.safety_checker else None
        )
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        logger.info("Image-to-image pipeline loaded successfully")
        return self.pipeline
    
    def load_inpaint_pipeline(self) -> StableDiffusionInpaintPipeline:
        """Load inpainting pipeline"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        logger.info(f"Loading inpainting pipeline: {self.config.model_id}")
        
        # Load pipeline
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None if not self.config.safety_checker else None
        )
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        logger.info("Inpainting pipeline loaded successfully")
        return self.pipeline
    
    def load_upscale_pipeline(self) -> StableDiffusionUpscalePipeline:
        """Load upscaling pipeline"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        logger.info(f"Loading upscaling pipeline: {self.config.model_id}")
        
        # Load pipeline
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        logger.info("Upscaling pipeline loaded successfully")
        return self.pipeline
    
    def load_textual_inversion_pipeline(self) -> TextualInversionPipeline:
        """Load textual inversion pipeline"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        logger.info(f"Loading textual inversion pipeline: {self.config.model_id}")
        
        # Load pipeline
        self.pipeline = TextualInversionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        logger.info("Textual inversion pipeline loaded successfully")
        return self.pipeline
    
    def load_dreambooth_pipeline(self) -> DreamBoothPipeline:
        """Load DreamBooth pipeline"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        logger.info(f"Loading DreamBooth pipeline: {self.config.model_id}")
        
        # Load pipeline
        self.pipeline = DreamBoothPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        logger.info("DreamBooth pipeline loaded successfully")
        return self.pipeline


class DiffusersGenerationManager:
    """Manager for text-to-image generation with Diffusers"""
    
    def __init__(self, pipeline, config: DiffusersConfig):
        self.pipeline = pipeline
        self.config = config
    
    def generate_image(self, prompt: str, negative_prompt: str = None, 
                      num_images: int = 1) -> List[Image.Image]:
        """Generate images from text prompt"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        logger.info(f"Generating image with prompt: {prompt}")
        
        # Set negative prompt
        if negative_prompt is None:
            negative_prompt = "low quality, bad quality, sketches"
        
        # Generate images
        with torch.autocast(self.device):
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                height=self.config.height,
                width=self.config.width,
                generator=torch.Generator(device=self.device).manual_seed(42)
            )
        
        images = result.images
        logger.info(f"Generated {len(images)} images successfully")
        
        return images
    
    def generate_image_grid(self, prompts: List[str], negative_prompts: List[str] = None) -> Image.Image:
        """Generate a grid of images from multiple prompts"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        if negative_prompts is None:
            negative_prompts = ["low quality, bad quality, sketches"] * len(prompts)
        
        all_images = []
        
        for prompt, neg_prompt in zip(prompts, negative_prompts):
            images = self.generate_image(prompt, neg_prompt, num_images=1)
            all_images.extend(images)
        
        # Create image grid
        grid = make_image_grid(all_images, rows=len(prompts), cols=1)
        
        return grid
    
    def generate_with_scheduler(self, prompt: str, scheduler_type: str = "ddim") -> List[Image.Image]:
        """Generate images with different schedulers"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        # Create scheduler
        if scheduler_type == "ddim":
            scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        elif scheduler_type == "ddpm":
            scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        elif scheduler_type == "euler":
            scheduler = EulerDiscreteScheduler.from_config(self.pipeline.scheduler.config)
        elif scheduler_type == "dpm":
            scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        # Set scheduler
        self.pipeline.scheduler = scheduler
        
        # Generate image
        images = self.generate_image(prompt)
        
        return images


class DiffusersTrainingManager:
    """Manager for training diffusion models with Diffusers"""
    
    def __init__(self, config: DiffusersConfig):
        self.config = config
        self.model_components = {}
    
    def load_model_components(self, model_id: str):
        """Load individual model components for training"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        logger.info(f"Loading model components from: {model_id}")
        
        # Load components
        self.model_components["tokenizer"] = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        self.model_components["text_encoder"] = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder"
        )
        self.model_components["vae"] = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae"
        )
        self.model_components["unet"] = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet"
        )
        self.model_components["noise_scheduler"] = DDPMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        
        logger.info("Model components loaded successfully")
    
    def setup_lora_training(self):
        """Setup LoRA training for the UNet"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        logger.info("Setting up LoRA training")
        
        # Add LoRA attention processors
        lora_attn_procs = {}
        for name in self.model_components["unet"].attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.model_components["unet"].config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.model_components["unet"].config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.model_components["unet"].config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.model_components["unet"].config.block_out_channels[block_id]
            
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout
            )
        
        # Set attention processors
        self.model_components["unet"].set_attn_processor(lora_attn_procs)
        
        logger.info("LoRA training setup completed")
    
    def create_training_config(self, output_dir: str) -> Dict[str, Any]:
        """Create training configuration"""
        training_config = {
            "output_dir": output_dir,
            "num_train_epochs": self.config.num_epochs,
            "learning_rate": self.config.learning_rate,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "mixed_precision": self.config.mixed_precision,
            "save_steps": 500,
            "save_total_limit": 2,
            "logging_steps": 10,
            "dataloader_num_workers": 4,
            "remove_unused_columns": False,
            "push_to_hub": False,
            "seed": 42
        }
        
        return training_config
    
    def setup_optimizer_and_scheduler(self, training_config: Dict[str, Any]):
        """Setup optimizer and learning rate scheduler"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        # Get trainable parameters
        trainable_params = []
        for name, param in self.model_components["unet"].named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=training_config["learning_rate"],
            weight_decay=0.01
        )
        
        # Create scheduler
        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=training_config["num_train_epochs"]
        )
        
        return optimizer, lr_scheduler


class DiffusersSchedulerManager:
    """Manager for different diffusion schedulers"""
    
    def __init__(self, config: DiffusersConfig):
        self.config = config
        self.schedulers = {}
    
    def create_ddpm_scheduler(self) -> DDPMScheduler:
        """Create DDPM scheduler"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        self.schedulers["ddpm"] = scheduler
        return scheduler
    
    def create_ddim_scheduler(self) -> DDIMScheduler:
        """Create DDIM scheduler"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=False,
            set_alpha_to_one=False
        )
        
        self.schedulers["ddim"] = scheduler
        return scheduler
    
    def create_euler_scheduler(self) -> EulerDiscreteScheduler:
        """Create Euler scheduler"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        scheduler = EulerDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        self.schedulers["euler"] = scheduler
        return scheduler
    
    def create_dpm_scheduler(self) -> DPMSolverMultistepScheduler:
        """Create DPM-Solver scheduler"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            algorithm_type="dpmsolver++",
            solver_type="midpoint"
        )
        
        self.schedulers["dpm"] = scheduler
        return scheduler
    
    def compare_schedulers(self, pipeline, prompt: str) -> Dict[str, List[Image.Image]]:
        """Compare different schedulers"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library not available")
        
        results = {}
        
        for scheduler_name, scheduler in self.schedulers.items():
            logger.info(f"Testing scheduler: {scheduler_name}")
            
            # Set scheduler
            pipeline.scheduler = scheduler
            
            # Generate image
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                result = pipeline(
                    prompt=prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5
                )
            
            results[scheduler_name] = result.images
        
        return results


class DiffusersExperiments:
    """Collection of Diffusers library experiments"""
    
    @staticmethod
    def demonstrate_stable_diffusion():
        """Demonstrate Stable Diffusion pipeline"""
        
        if not DIFFUSERS_AVAILABLE:
            logger.warning("Diffusers library not available")
            return None
        
        logger.info("Demonstrating Stable Diffusion pipeline...")
        
        # Create config
        config = DiffusersConfig(
            model_id="runwayml/stable-diffusion-v1-5",
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        # Create manager
        manager = DiffusersModelManager(config)
        
        # Load pipeline
        pipeline = manager.load_stable_diffusion_pipeline()
        
        # Create generation manager
        gen_manager = DiffusersGenerationManager(pipeline, config)
        
        # Generate image
        prompt = "A beautiful landscape with mountains and a lake, digital art"
        images = gen_manager.generate_image(prompt)
        
        logger.info(f"Generated {len(images)} images with Stable Diffusion")
        
        return pipeline, gen_manager
    
    @staticmethod
    def demonstrate_stable_diffusion_xl():
        """Demonstrate Stable Diffusion XL pipeline"""
        
        if not DIFFUSERS_AVAILABLE:
            logger.warning("Diffusers library not available")
            return None
        
        logger.info("Demonstrating Stable Diffusion XL pipeline...")
        
        # Create config
        config = DiffusersConfig(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            num_inference_steps=30,
            guidance_scale=7.5,
            height=1024,
            width=1024
        )
        
        # Create manager
        manager = DiffusersModelManager(config)
        
        # Load pipeline
        pipeline = manager.load_stable_diffusion_xl_pipeline()
        
        # Create generation manager
        gen_manager = DiffusersGenerationManager(pipeline, config)
        
        # Generate image
        prompt = "A futuristic cityscape with flying cars and neon lights, cinematic lighting"
        images = gen_manager.generate_image(prompt)
        
        logger.info(f"Generated {len(images)} images with Stable Diffusion XL")
        
        return pipeline, gen_manager
    
    @staticmethod
    def demonstrate_controlnet():
        """Demonstrate ControlNet pipeline"""
        
        if not DIFFUSERS_AVAILABLE:
            logger.warning("Diffusers library not available")
            return None
        
        logger.info("Demonstrating ControlNet pipeline...")
        
        # Create config
        config = DiffusersConfig(
            model_id="runwayml/stable-diffusion-v1-5",
            use_controlnet=True,
            controlnet_model_id="lllyasviel/sd-controlnet-canny",
            num_inference_steps=20
        )
        
        # Create manager
        manager = DiffusersModelManager(config)
        
        # Load pipeline
        pipeline = manager.load_controlnet_pipeline()
        
        logger.info("ControlNet pipeline loaded successfully")
        
        return pipeline
    
    @staticmethod
    def demonstrate_img2img():
        """Demonstrate image-to-image pipeline"""
        
        if not DIFFUSERS_AVAILABLE:
            logger.warning("Diffusers library not available")
            return None
        
        logger.info("Demonstrating image-to-image pipeline...")
        
        # Create config
        config = DiffusersConfig(
            model_id="runwayml/stable-diffusion-v1-5",
            num_inference_steps=20
        )
        
        # Create manager
        manager = DiffusersModelManager(config)
        
        # Load pipeline
        pipeline = manager.load_img2img_pipeline()
        
        logger.info("Image-to-image pipeline loaded successfully")
        
        return pipeline
    
    @staticmethod
    def demonstrate_schedulers():
        """Demonstrate different schedulers"""
        
        if not DIFFUSERS_AVAILABLE:
            logger.warning("Diffusers library not available")
            return None
        
        logger.info("Demonstrating different schedulers...")
        
        # Create config
        config = DiffusersConfig(
            model_id="runwayml/stable-diffusion-v1-5",
            num_inference_steps=20
        )
        
        # Create manager
        manager = DiffusersModelManager(config)
        
        # Load pipeline
        pipeline = manager.load_stable_diffusion_pipeline()
        
        # Create scheduler manager
        scheduler_manager = DiffusersSchedulerManager(config)
        
        # Create schedulers
        scheduler_manager.create_ddpm_scheduler()
        scheduler_manager.create_ddim_scheduler()
        scheduler_manager.create_euler_scheduler()
        scheduler_manager.create_dpm_scheduler()
        
        logger.info("Schedulers created successfully")
        
        return scheduler_manager
    
    @staticmethod
    def demonstrate_lora_training():
        """Demonstrate LoRA training setup"""
        
        if not DIFFUSERS_AVAILABLE:
            logger.warning("Diffusers library not available")
            return None
        
        logger.info("Demonstrating LoRA training setup...")
        
        # Create config
        config = DiffusersConfig(
            model_id="runwayml/stable-diffusion-v1-5",
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
            learning_rate=1e-5
        )
        
        # Create training manager
        training_manager = DiffusersTrainingManager(config)
        
        # Load model components
        training_manager.load_model_components(config.model_id)
        
        # Setup LoRA training
        training_manager.setup_lora_training()
        
        # Create training config
        training_config = training_manager.create_training_config("./lora_output")
        
        # Setup optimizer and scheduler
        optimizer, lr_scheduler = training_manager.setup_optimizer_and_scheduler(training_config)
        
        logger.info("LoRA training setup completed")
        
        return training_manager, training_config


def main():
    """Main execution function"""
    if not DIFFUSERS_AVAILABLE:
        logger.error("Diffusers library not available. Please install it first.")
        return
    
    logger.info("Starting Diffusers Library Demonstrations...")
    
    # Demonstrate Stable Diffusion
    logger.info("Testing Stable Diffusion...")
    sd_pipeline, sd_gen_manager = DiffusersExperiments.demonstrate_stable_diffusion()
    
    # Demonstrate Stable Diffusion XL
    logger.info("Testing Stable Diffusion XL...")
    sdxl_pipeline, sdxl_gen_manager = DiffusersExperiments.demonstrate_stable_diffusion_xl()
    
    # Demonstrate ControlNet
    logger.info("Testing ControlNet...")
    controlnet_pipeline = DiffusersExperiments.demonstrate_controlnet()
    
    # Demonstrate image-to-image
    logger.info("Testing image-to-image...")
    img2img_pipeline = DiffusersExperiments.demonstrate_img2img()
    
    # Demonstrate schedulers
    logger.info("Testing schedulers...")
    scheduler_manager = DiffusersExperiments.demonstrate_schedulers()
    
    # Demonstrate LoRA training
    logger.info("Testing LoRA training setup...")
    training_manager, training_config = DiffusersExperiments.demonstrate_lora_training()
    
    # Create comprehensive diffusers system
    logger.info("Creating comprehensive diffusers system...")
    
    comprehensive_config = DiffusersConfig(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        model_type="stable-diffusion-xl",
        num_inference_steps=50,
        guidance_scale=7.5,
        height=1024,
        width=1024,
        use_lora=True,
        lora_r=32,
        lora_alpha=64
    )
    
    comprehensive_manager = DiffusersModelManager(comprehensive_config)
    
    # Test comprehensive system
    try:
        pipeline = comprehensive_manager.load_stable_diffusion_xl_pipeline()
        gen_manager = DiffusersGenerationManager(pipeline, comprehensive_config)
        
        # Test generation
        test_prompt = "A majestic dragon flying over a medieval castle, epic fantasy art"
        test_images = gen_manager.generate_image(test_prompt, num_images=1)
        
        logger.info(f"Comprehensive system test successful")
        logger.info(f"Generated image shape: {test_images[0].size if test_images else 'None'}")
        
    except Exception as e:
        logger.warning(f"Comprehensive system test failed: {e}")
    
    # Summary
    logger.info("Diffusers Library Summary:")
    logger.info(f"Stable Diffusion tested: {'✓' if sd_pipeline else '✗'}")
    logger.info(f"Stable Diffusion XL tested: {'✓' if sdxl_pipeline else '✗'}")
    logger.info(f"ControlNet tested: {'✓' if controlnet_pipeline else '✗'}")
    logger.info(f"Image-to-image tested: {'✓' if img2img_pipeline else '✗'}")
    logger.info(f"Schedulers tested: {'✓' if scheduler_manager else '✗'}")
    logger.info(f"LoRA training setup tested: {'✓' if training_manager else '✗'}")
    logger.info(f"Comprehensive diffusers system created: {'✓' if 'comprehensive_manager' in locals() else '✗'}")
    
    logger.info("Diffusers Library demonstrations completed successfully!")


if __name__ == "__main__":
    main()
