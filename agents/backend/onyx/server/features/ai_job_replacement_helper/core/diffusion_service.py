"""
Diffusion Service - Servicio avanzado de modelos de difusión
=============================================================

Sistema profesional para trabajar con modelos de difusión usando Diffusers.
"""

import logging
import time
import io
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available")

# Try to import diffusers
try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        PNDMScheduler,
    )
    import torch
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers library not available. Using simulation mode.")


@dataclass
class DiffusionConfig:
    """Configuración de difusión"""
    model_id: str
    scheduler: str = "DPMSolverMultistepScheduler"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    use_gpu: bool = True
    torch_dtype: str = "float16"  # float16, float32


@dataclass
class GeneratedImage:
    """Imagen generada"""
    image_data: bytes
    prompt: str
    negative_prompt: Optional[str]
    config: DiffusionConfig
    generation_time: float
    seed: Optional[int] = None
    width: int = 512
    height: int = 512


class DiffusionService:
    """Servicio avanzado de modelos de difusión"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.pipelines: Dict[str, Any] = {}
        self.img2img_pipelines: Dict[str, Any] = {}
        self.inpaint_pipelines: Dict[str, Any] = {}
        self.configs: Dict[str, DiffusionConfig] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if DIFFUSERS_AVAILABLE else None
        logger.info(f"DiffusionService initialized on device: {self.device}")
    
    def _get_scheduler(self, scheduler_name: str):
        """Obtener scheduler"""
        if not DIFFUSERS_AVAILABLE:
            return None
        
        schedulers = {
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "PNDMScheduler": PNDMScheduler,
        }
        
        return schedulers.get(scheduler_name, DPMSolverMultistepScheduler)
    
    def load_pipeline(self, config: DiffusionConfig) -> bool:
        """
        Cargar pipeline de difusión con optimizaciones.
        
        Args:
            config: Configuración del pipeline
        
        Returns:
            True si se cargó exitosamente
        """
        try:
            if DIFFUSERS_AVAILABLE:
                device = self.device if config.use_gpu else torch.device("cpu")
                
                # Determine dtype
                dtype_map = {
                    "float16": torch.float16,
                    "float32": torch.float32,
                }
                torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)
                
                logger.info(f"Loading diffusion pipeline {config.model_id}...")
                
                # Load pipeline with error handling
                try:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        config.model_id,
                        torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
                        safety_checker=None,  # Disable for faster inference
                        requires_safety_checker=False,
                    )
                except Exception as e:
                    logger.warning(f"Failed to load with safety_checker=None: {e}")
                    # Retry without disabling safety checker
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        config.model_id,
                        torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
                    )
                
                # Set scheduler
                scheduler_class = self._get_scheduler(config.scheduler)
                if scheduler_class:
                    pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)
                    logger.info(f"Scheduler set to {config.scheduler}")
                
                # Move to device
                pipeline = pipeline.to(device)
                
                # Memory optimizations
                if device.type == "cuda":
                    # Enable attention slicing to reduce memory
                    try:
                        pipeline.enable_attention_slicing()
                        logger.info("Attention slicing enabled")
                    except Exception as e:
                        logger.warning(f"Could not enable attention slicing: {e}")
                    
                    # Enable CPU offload for memory efficiency
                    try:
                        pipeline.enable_model_cpu_offload()
                        logger.info("Model CPU offload enabled")
                    except Exception as e:
                        logger.warning(f"Could not enable CPU offload: {e}")
                
                self.pipelines[config.model_id] = pipeline
            else:
                # Simulation mode
                logger.warning("Diffusers not available, using simulation mode")
                self.pipelines[config.model_id] = {"loaded": True}
            
            self.configs[config.model_id] = config
            
            logger.info(f"Diffusion pipeline {config.model_id} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading pipeline {config.model_id}: {e}", exc_info=True)
            return False
    
    def generate_image(
        self,
        prompt: str,
        model_id: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> GeneratedImage:
        """Generar imagen con modelo de difusión"""
        start_time = time.time()
        
        config = self.configs.get(model_id)
        if not config:
            raise ValueError(f"Pipeline {model_id} not loaded")
        
        if DIFFUSERS_AVAILABLE and model_id in self.pipelines:
            try:
                pipeline = self.pipelines[model_id]
                
                # Setup generator
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                
                # Generate image
                image = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps or config.num_inference_steps,
                    guidance_scale=guidance_scale or config.guidance_scale,
                    width=width or config.width,
                    height=height or config.height,
                    generator=generator,
                ).images[0]
                
                # Convert to bytes
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_data = buffer.getvalue()
                
            except Exception as e:
                logger.error(f"Error generating image: {e}")
                image_data = b"error"
        else:
            # Simulation mode
            image_data = b"fake_image_data"
        
        generation_time = time.time() - start_time
        
        return GeneratedImage(
            image_data=image_data,
            prompt=prompt,
            negative_prompt=negative_prompt,
            config=config,
            generation_time=generation_time,
            seed=seed,
            width=width or config.width,
            height=height or config.height,
        )
    
    def load_img2img_pipeline(self, model_id: str) -> bool:
        """Cargar pipeline img2img"""
        try:
            if DIFFUSERS_AVAILABLE:
                config = self.configs.get(model_id)
                if not config:
                    return False
                
                device = self.device if config.use_gpu else torch.device("cpu")
                torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
                
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
                )
                pipeline = pipeline.to(device)
                pipeline.enable_attention_slicing()
                
                self.img2img_pipelines[model_id] = pipeline
                return True
            else:
                self.img2img_pipelines[model_id] = {"loaded": True}
                return True
                
        except Exception as e:
            logger.error(f"Error loading img2img pipeline: {e}")
            return False
    
    def img2img(
        self,
        prompt: str,
        init_image: bytes,
        model_id: str,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None
    ) -> GeneratedImage:
        """Generar imagen desde imagen inicial"""
        start_time = time.time()
        
        config = self.configs.get(model_id)
        if not config:
            raise ValueError(f"Pipeline {model_id} not loaded")
        
        if DIFFUSERS_AVAILABLE and model_id in self.img2img_pipelines:
            try:
                pipeline = self.img2img_pipelines[model_id]
                
                # Load init image
                init_image_pil = Image.open(io.BytesIO(init_image)).convert("RGB")
                
                # Generate
                image = pipeline(
                    prompt=prompt,
                    image=init_image_pil,
                    strength=strength,
                    num_inference_steps=num_inference_steps or config.num_inference_steps,
                    guidance_scale=guidance_scale or config.guidance_scale,
                ).images[0]
                
                # Convert to bytes
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_data = buffer.getvalue()
                
            except Exception as e:
                logger.error(f"Error in img2img: {e}")
                image_data = b"error"
        else:
            image_data = b"fake_img2img_data"
        
        generation_time = time.time() - start_time
        
        return GeneratedImage(
            image_data=image_data,
            prompt=prompt,
            negative_prompt=None,
            config=config,
            generation_time=generation_time,
        )
    
    def load_inpaint_pipeline(self, model_id: str) -> bool:
        """Cargar pipeline de inpainting"""
        try:
            if DIFFUSERS_AVAILABLE:
                config = self.configs.get(model_id)
                if not config:
                    return False
                
                device = self.device if config.use_gpu else torch.device("cpu")
                torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
                
                pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
                )
                pipeline = pipeline.to(device)
                pipeline.enable_attention_slicing()
                
                self.inpaint_pipelines[model_id] = pipeline
                return True
            else:
                self.inpaint_pipelines[model_id] = {"loaded": True}
                return True
                
        except Exception as e:
            logger.error(f"Error loading inpaint pipeline: {e}")
            return False
    
    def inpainting(
        self,
        prompt: str,
        init_image: bytes,
        mask_image: bytes,
        model_id: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None
    ) -> GeneratedImage:
        """Inpainting de imagen"""
        start_time = time.time()
        
        config = self.configs.get(model_id)
        if not config:
            raise ValueError(f"Pipeline {model_id} not loaded")
        
        if DIFFUSERS_AVAILABLE and model_id in self.inpaint_pipelines:
            try:
                pipeline = self.inpaint_pipelines[model_id]
                
                # Load images
                init_image_pil = Image.open(io.BytesIO(init_image)).convert("RGB")
                mask_image_pil = Image.open(io.BytesIO(mask_image)).convert("RGB")
                
                # Generate
                image = pipeline(
                    prompt=prompt,
                    image=init_image_pil,
                    mask_image=mask_image_pil,
                    num_inference_steps=num_inference_steps or config.num_inference_steps,
                    guidance_scale=guidance_scale or config.guidance_scale,
                ).images[0]
                
                # Convert to bytes
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_data = buffer.getvalue()
                
            except Exception as e:
                logger.error(f"Error in inpainting: {e}")
                image_data = b"error"
        else:
            image_data = b"fake_inpaint_data"
        
        generation_time = time.time() - start_time
        
        return GeneratedImage(
            image_data=image_data,
            prompt=prompt,
            negative_prompt=None,
            config=config,
            generation_time=generation_time,
        )
