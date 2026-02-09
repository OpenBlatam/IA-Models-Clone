"""
Diffusion Models System - Sistema de modelos de difusión para generación de imágenes 3D
==========================================================================================
Integración con Diffusers para generar imágenes y modelos 3D
"""

import logging
import torch
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from PIL import Image
import numpy as np

try:
    from diffusers import (
        StableDiffusionPipeline, StableDiffusionXLPipeline,
        DDIMScheduler, DDPMScheduler, PNDMScheduler,
        EulerDiscreteScheduler, DPMSolverMultistepScheduler
    )
    from diffusers.utils import export_to_video
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("Diffusers library not available")

logger = logging.getLogger(__name__)


class SchedulerType(str, Enum):
    """Tipos de schedulers soportados"""
    DDIM = "ddim"
    DDPM = "ddpm"
    PNDM = "pndm"
    EULER = "euler"
    DPMSOLVER = "dpm_solver"


@dataclass
class DiffusionConfig:
    """Configuración de modelo de difusión"""
    model_name: str = "runwayml/stable-diffusion-v1-5"
    use_xl: bool = False
    scheduler_type: SchedulerType = SchedulerType.DPMSOLVER
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    use_gpu: bool = True
    use_attention_slicing: bool = True
    use_vae_slicing: bool = True


class DiffusionModelsSystem:
    """Sistema de modelos de difusión"""
    
    def __init__(self):
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library is required")
        
        self.pipelines: Dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def load_pipeline(self, pipeline_id: str, config: DiffusionConfig) -> Dict[str, Any]:
        """Carga un pipeline de difusión"""
        try:
            # Seleccionar pipeline
            if config.use_xl:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    config.model_name,
                    torch_dtype=torch.float16 if config.use_gpu else torch.float32
                )
            else:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    config.model_name,
                    torch_dtype=torch.float16 if config.use_gpu else torch.float32
                )
            
            # Configurar scheduler
            scheduler = self._get_scheduler(config.scheduler_type, pipeline.scheduler.config)
            pipeline.scheduler = scheduler
            
            # Optimizaciones
            if config.use_attention_slicing:
                pipeline.enable_attention_slicing()
            if config.use_vae_slicing:
                pipeline.enable_vae_slicing()
            
            # Mover a GPU si está disponible
            if config.use_gpu:
                pipeline = pipeline.to(self.device)
            
            self.pipelines[pipeline_id] = {
                "pipeline": pipeline,
                "config": config
            }
            
            logger.info(f"Pipeline {pipeline_id} loaded successfully")
            
            return {
                "pipeline_id": pipeline_id,
                "status": "loaded",
                "device": str(self.device),
                "model_name": config.model_name
            }
        
        except Exception as e:
            logger.error(f"Error loading pipeline {pipeline_id}: {e}")
            raise
    
    def _get_scheduler(self, scheduler_type: SchedulerType, config: Dict[str, Any]):
        """Obtiene scheduler según tipo"""
        if scheduler_type == SchedulerType.DDIM:
            return DDIMScheduler.from_config(config)
        elif scheduler_type == SchedulerType.DDPM:
            return DDPMScheduler.from_config(config)
        elif scheduler_type == SchedulerType.PNDM:
            return PNDMScheduler.from_config(config)
        elif scheduler_type == SchedulerType.EULER:
            return EulerDiscreteScheduler.from_config(config)
        elif scheduler_type == SchedulerType.DPMSOLVER:
            return DPMSolverMultistepScheduler.from_config(config)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def generate_image(
        self,
        pipeline_id: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Genera imagen desde prompt"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not loaded")
        
        pipeline_data = self.pipelines[pipeline_id]
        pipeline = pipeline_data["pipeline"]
        config = pipeline_data["config"]
        
        # Parámetros de generación
        generation_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": kwargs.get("num_inference_steps", config.num_inference_steps),
            "guidance_scale": kwargs.get("guidance_scale", config.guidance_scale),
            "height": kwargs.get("height", config.height),
            "width": kwargs.get("width", config.width),
            "num_images_per_prompt": num_images
        }
        
        # Generar
        logger.info(f"Generating image with prompt: {prompt[:50]}...")
        with torch.autocast(device_type="cuda" if config.use_gpu else "cpu"):
            images = pipeline(**generation_kwargs).images
        
        # Convertir a formato base64 o guardar
        image_paths = []
        for i, image in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"storage/generated_images/{pipeline_id}_{timestamp}_{i}.png"
            image.save(path)
            image_paths.append(path)
        
        return {
            "pipeline_id": pipeline_id,
            "prompt": prompt,
            "images": image_paths,
            "num_images": len(images),
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_3d_model_description(
        self,
        pipeline_id: str,
        product_description: str,
        style: Optional[str] = None
    ) -> Dict[str, Any]:
        """Genera descripción visual de modelo 3D"""
        enhanced_prompt = f"3D model, {product_description}"
        if style:
            enhanced_prompt += f", {style} style"
        enhanced_prompt += ", high quality, detailed, professional"
        
        return self.generate_image(pipeline_id, enhanced_prompt)
    
    def get_pipeline_info(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información del pipeline"""
        if pipeline_id not in self.pipelines:
            return None
        
        pipeline_data = self.pipelines[pipeline_id]
        
        return {
            "pipeline_id": pipeline_id,
            "config": pipeline_data["config"].__dict__,
            "device": str(self.device)
        }




