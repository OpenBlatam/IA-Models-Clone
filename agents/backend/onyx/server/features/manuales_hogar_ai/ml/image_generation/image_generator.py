"""
Generador de Imágenes con Diffusion Models
===========================================

Generador de imágenes ilustrativas para manuales usando Stable Diffusion.
"""

import logging
import torch
from typing import Optional, List, Dict, Any
from PIL import Image
import io
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generador de imágenes usando diffusion models."""
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        use_xl: bool = False,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """
        Inicializar generador de imágenes.
        
        Args:
            model_name: Nombre del modelo
            use_xl: Usar Stable Diffusion XL
            device: Dispositivo (cuda/cpu)
            torch_dtype: Tipo de datos de torch
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_xl = use_xl
        
        if torch_dtype is None:
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype
        
        logger.info(f"Inicializando generador de imágenes: {model_name} en {self.device}")
        
        try:
            if use_xl:
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=self.torch_dtype,
                    use_safetensors=True
                )
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=self.torch_dtype,
                    use_safetensors=True
                )
            
            # Optimizar scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            self.pipeline = self.pipeline.to(self.device)
            
            # Optimización de memoria
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_vae_slicing()
            
            logger.info("Pipeline de generación de imágenes inicializado")
        
        except Exception as e:
            logger.error(f"Error inicializando pipeline: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generar imagen desde prompt.
        
        Args:
            prompt: Prompt de texto
            negative_prompt: Prompt negativo (opcional)
            num_inference_steps: Pasos de inferencia
            guidance_scale: Escala de guía
            width: Ancho de imagen
            height: Alto de imagen
            seed: Semilla para reproducibilidad
        
        Returns:
            Imagen generada
        """
        try:
            # Generador para reproducibilidad
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generar imagen
            with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                )
            
            image = result.images[0]
            logger.info(f"Imagen generada: {width}x{height}")
            return image
        
        except Exception as e:
            logger.error(f"Error generando imagen: {str(e)}")
            raise
    
    def generate_manual_illustration(
        self,
        step_description: str,
        category: str = "general",
        style: str = "lego_instruction"
    ) -> Image.Image:
        """
        Generar ilustración para paso de manual.
        
        Args:
            step_description: Descripción del paso
            category: Categoría del oficio
            style: Estilo de ilustración
        
        Returns:
            Imagen generada
        """
        # Construir prompt optimizado
        style_prompts = {
            "lego_instruction": "LEGO instruction manual style, clear step-by-step illustration, isometric view, bright colors, simple shapes",
            "technical_diagram": "technical diagram, engineering drawing style, clear labels, professional",
            "realistic": "realistic photo, high quality, detailed, professional photography"
        }
        
        style_prompt = style_prompts.get(style, style_prompts["lego_instruction"])
        
        prompt = f"{step_description}, {style_prompt}, {category} repair manual illustration, clear and easy to understand"
        
        negative_prompt = "blurry, low quality, distorted, unclear, confusing, complex"
        
        return self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=512,
            height=512
        )
    
    def generate_multiple(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Image.Image]:
        """
        Generar múltiples imágenes.
        
        Args:
            prompts: Lista de prompts
            **kwargs: Parámetros de generación
        
        Returns:
            Lista de imágenes
        """
        images = []
        for prompt in prompts:
            try:
                image = self.generate(prompt, **kwargs)
                images.append(image)
            except Exception as e:
                logger.error(f"Error generando imagen para prompt '{prompt[:50]}...': {str(e)}")
                continue
        
        return images




