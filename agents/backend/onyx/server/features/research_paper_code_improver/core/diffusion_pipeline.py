"""
Diffusion Model Pipeline - Pipeline para modelos de difusión
=============================================================
"""

import logging
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler
)
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from PIL import Image
import io

logger = logging.getLogger(__name__)


@dataclass
class DiffusionConfig:
    """Configuración de modelo de difusión"""
    model_id: str = "runwayml/stable-diffusion-v1-5"
    use_xl: bool = False
    scheduler_type: str = "DPMSolverMultistep"  # DDIM, DPMSolverMultistep, EulerAncestral
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: str = "float16" if torch.cuda.is_available() else "float32"


class DiffusionPipeline:
    """Pipeline para modelos de difusión"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.pipeline = None
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.torch_dtype)
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Carga el pipeline de difusión"""
        try:
            if self.config.use_xl:
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.config.model_id,
                    torch_dtype=self.dtype
                )
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.config.model_id,
                    torch_dtype=self.dtype
                )
            
            # Configurar scheduler
            self._setup_scheduler()
            
            # Mover a device
            self.pipeline = self.pipeline.to(self.device)
            
            # Optimizar si es posible
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
            if hasattr(self.pipeline, "enable_model_cpu_offload"):
                self.pipeline.enable_model_cpu_offload()
            
            logger.info(f"Pipeline de difusión cargado: {self.config.model_id}")
        except Exception as e:
            logger.error(f"Error cargando pipeline: {e}")
            raise
    
    def _setup_scheduler(self):
        """Configura el scheduler"""
        scheduler_map = {
            "DDIM": DDIMScheduler,
            "DPMSolverMultistep": DPMSolverMultistepScheduler,
            "EulerAncestral": EulerAncestralDiscreteScheduler
        }
        
        scheduler_class = scheduler_map.get(self.config.scheduler_type)
        if scheduler_class:
            self.pipeline.scheduler = scheduler_class.from_config(
                self.pipeline.scheduler.config
            )
            logger.info(f"Scheduler configurado: {self.config.scheduler_type}")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """Genera imágenes desde un prompt"""
        if self.pipeline is None:
            raise ValueError("Pipeline no cargado")
        
        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Generar imágenes
        images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            num_images_per_prompt=num_images,
            height=height,
            width=width,
            generator=generator
        ).images
        
        return images
    
    def img2img(
        self,
        prompt: str,
        init_image: Image.Image,
        strength: float = 0.8,
        negative_prompt: Optional[str] = None
    ) -> Image.Image:
        """Genera imagen desde otra imagen (img2img)"""
        if self.pipeline is None:
            raise ValueError("Pipeline no cargado")
        
        if not hasattr(self.pipeline, "img2img"):
            raise ValueError("Pipeline no soporta img2img")
        
        image = self.pipeline(
            prompt=prompt,
            image=init_image,
            strength=strength,
            negative_prompt=negative_prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale
        ).images[0]
        
        return image
    
    def inpainting(
        self,
        prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        negative_prompt: Optional[str] = None
    ) -> Image.Image:
        """Inpainting (rellenar áreas de imagen)"""
        if self.pipeline is None:
            raise ValueError("Pipeline no cargado")
        
        if not hasattr(self.pipeline, "inpaint"):
            raise ValueError("Pipeline no soporta inpainting")
        
        result = self.pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale
        ).images[0]
        
        return result




