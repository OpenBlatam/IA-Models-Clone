"""
Diffusion Models Integration Module
====================================

Integración profesional con Hugging Face Diffusers.
Soporta diferentes pipelines de difusión y schedulers.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        DDPMScheduler,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        DiffusionPipeline
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("Diffusers library not available.")

logger = logging.getLogger(__name__)


class DiffusionPipelineWrapper:
    """
    Wrapper profesional para pipelines de difusión.
    
    Soporta:
    - Stable Diffusion
    - Stable Diffusion XL
    - Custom schedulers
    - Optimizaciones de memoria
    """
    
    def __init__(
        self,
        model_id: str,
        pipeline_type: str = "stable-diffusion",
        use_fp16: bool = True,
        device: str = "cuda",
        scheduler_type: Optional[str] = None
    ):
        """
        Inicializar pipeline de difusión.
        
        Args:
            model_id: ID del modelo (HuggingFace)
            pipeline_type: Tipo de pipeline ("stable-diffusion", "stable-diffusion-xl")
            use_fp16: Usar FP16 para ahorrar memoria
            device: Dispositivo
            scheduler_type: Tipo de scheduler (opcional)
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library is required")
        
        self.model_id = model_id
        self.pipeline_type = pipeline_type
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        
        # Cargar pipeline
        torch_dtype = torch.float16 if self.use_fp16 else torch.float32
        
        if pipeline_type == "stable-diffusion":
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None
            )
        elif pipeline_type == "stable-diffusion-xl":
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None
            )
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        # Cambiar scheduler si se especifica
        if scheduler_type:
            self.set_scheduler(scheduler_type)
        
        # Optimizaciones
        if device == "cuda":
            self.pipeline = self.pipeline.to(device)
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
        
        logger.info(f"DiffusionPipelineWrapper initialized: {model_id} on {device}")
    
    def set_scheduler(self, scheduler_type: str):
        """
        Cambiar scheduler del pipeline.
        
        Args:
            scheduler_type: Tipo de scheduler ("ddpm", "ddim", "dpm", "euler")
        """
        if scheduler_type == "ddpm":
            self.pipeline.scheduler = DDPMScheduler.from_config(
                self.pipeline.scheduler.config
            )
        elif scheduler_type == "ddim":
            self.pipeline.scheduler = DDIMScheduler.from_config(
                self.pipeline.scheduler.config
            )
        elif scheduler_type == "dpm":
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
        elif scheduler_type == "euler":
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        logger.info(f"Scheduler changed to {scheduler_type}")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generar imagen con el pipeline.
        
        Args:
            prompt: Prompt de texto
            negative_prompt: Negative prompt (opcional)
            num_inference_steps: Número de pasos de inferencia
            guidance_scale: Guidance scale
            height: Altura de imagen
            width: Ancho de imagen
            seed: Semilla para reproducibilidad
            
        Returns:
            Imagen generada como numpy array
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        with torch.autocast(device_type=self.device, dtype=torch.float16 if self.use_fp16 else torch.float32):
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator
            )
        
        image = output.images[0]
        
        # Convertir a numpy
        if hasattr(image, 'numpy'):
            return np.array(image)
        else:
            return np.array(image)
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[np.ndarray]:
        """
        Generar múltiples imágenes.
        
        Args:
            prompts: Lista de prompts
            **kwargs: Argumentos adicionales para generate()
            
        Returns:
            Lista de imágenes generadas
        """
        images = []
        for prompt in prompts:
            image = self.generate(prompt, **kwargs)
            images.append(image)
        
        return images
    
    def save_model(self, output_dir: str):
        """
        Guardar modelo.
        
        Args:
            output_dir: Directorio de salida
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.pipeline.save_pretrained(str(output_path))
        logger.info(f"Model saved to {output_path}")

