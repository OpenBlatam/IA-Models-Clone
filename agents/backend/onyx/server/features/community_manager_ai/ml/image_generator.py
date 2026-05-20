"""
Image Generator - Generador de Imágenes con Diffusion Models
============================================================

Generación de imágenes para memes y contenido usando Stable Diffusion.
"""

import logging
import torch
from typing import Dict, Any, Optional, List
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler
)
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generador de imágenes usando diffusion models"""
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        use_xl: bool = False,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Inicializar generador de imágenes
        
        Args:
            model_name: Nombre del modelo
            use_xl: Usar Stable Diffusion XL
            device: Dispositivo
            dtype: Tipo de datos (float16 para ahorrar memoria)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.use_xl = use_xl
        
        if dtype is None and self.device == "cuda":
            dtype = torch.float16  # Usar half precision en GPU
        
        self.dtype = dtype or torch.float32
        
        try:
            if use_xl:
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype
                )
            else:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype
                )
            
            # Optimizar scheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Mover a dispositivo
            if self.device == "cuda":
                self.pipe = self.pipe.to(self.device)
                # Habilitar optimizaciones de memoria
                self.pipe.enable_attention_slicing()
                if hasattr(self.pipe, "enable_vae_slicing"):
                    self.pipe.enable_vae_slicing()
            
            logger.info(f"Image Generator inicializado con {model_name} en {self.device}")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            self.pipe = None
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None
    ) -> Optional[Image.Image]:
        """
        Generar imagen
        
        Args:
            prompt: Prompt descriptivo
            negative_prompt: Prompt negativo
            num_inference_steps: Pasos de inferencia
            guidance_scale: Escala de guía
            width: Ancho de imagen
            height: Alto de imagen
            seed: Semilla para reproducibilidad
            
        Returns:
            Imagen generada o None
        """
        if not self.pipe:
            logger.error("Pipeline no inicializado")
            return None
        
        try:
            # Generador para reproducibilidad
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generar imagen
            with torch.autocast(device_type=self.device, dtype=self.dtype):
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                ).images[0]
            
            logger.info(f"Imagen generada: {prompt[:50]}...")
            return image
            
        except Exception as e:
            logger.error(f"Error generando imagen: {e}")
            return None
    
    def generate_meme_image(
        self,
        meme_text: str,
        style: str = "funny cartoon"
    ) -> Optional[Image.Image]:
        """
        Generar imagen para meme
        
        Args:
            meme_text: Texto del meme
            style: Estilo de imagen
            
        Returns:
            Imagen generada
        """
        prompt = f"{style}, {meme_text}, meme style, colorful, high quality"
        negative_prompt = "blurry, low quality, distorted, text"
        
        return self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,  # Menos pasos para más velocidad
            guidance_scale=7.0
        )
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Optional[Image.Image]]:
        """
        Generar múltiples imágenes
        
        Args:
            prompts: Lista de prompts
            **kwargs: Argumentos adicionales para generate
            
        Returns:
            Lista de imágenes
        """
        images = []
        for prompt in prompts:
            image = self.generate(prompt, **kwargs)
            images.append(image)
        return images




