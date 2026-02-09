"""
Servicio de modelos de difusión para generación de contenido visual
"""

import logging
import torch
from typing import Dict, Any, List, Optional
from PIL import Image
import io

from ..config import get_settings

logger = logging.getLogger(__name__)


class DiffusionService:
    """Servicio para modelos de difusión"""
    
    def __init__(self):
        self.settings = get_settings()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Carga pipeline de difusión"""
        try:
            from diffusers import StableDiffusionPipeline
            
            logger.info(f"Cargando pipeline de difusión en {self.device}...")
            
            # Usar modelo más ligero para desarrollo
            model_id = "runwayml/stable-diffusion-v1-5"
            
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # Desactivar para desarrollo
                requires_safety_checker=False
            )
            self.pipeline = self.pipeline.to(self.device)
            
            # Optimización para inferencia
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_xformers_memory_efficient_attention()
            
            logger.info("Pipeline de difusión cargado exitosamente")
            
        except ImportError:
            logger.warning("diffusers no instalado, funcionalidad limitada")
            self.pipeline = None
        except Exception as e:
            logger.warning(f"No se pudo cargar pipeline de difusión: {e}")
            self.pipeline = None
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None
    ) -> Optional[Image.Image]:
        """
        Genera imagen usando modelo de difusión
        
        Args:
            prompt: Prompt de texto
            negative_prompt: Prompt negativo
            num_inference_steps: Pasos de inferencia
            guidance_scale: Escala de guía
            height: Altura de imagen
            width: Ancho de imagen
            seed: Semilla para reproducibilidad
            
        Returns:
            Imagen generada o None
        """
        if not self.pipeline:
            logger.error("Pipeline de difusión no disponible")
            return None
        
        try:
            # Establecer semilla si se proporciona
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None
            
            # Generar imagen
            logger.info(f"Generando imagen con prompt: {prompt[:50]}...")
            
            with torch.autocast(self.device):
                image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator
                ).images[0]
            
            logger.info("Imagen generada exitosamente")
            return image
            
        except Exception as e:
            logger.error(f"Error generando imagen: {e}", exc_info=True)
            return None
    
    def generate_image_from_identity(
        self,
        identity_description: str,
        content_style: str = "authentic",
        additional_prompt: str = ""
    ) -> Optional[Image.Image]:
        """
        Genera imagen basada en descripción de identidad
        
        Args:
            identity_description: Descripción de la identidad
            content_style: Estilo de contenido
            additional_prompt: Prompt adicional
            
        Returns:
            Imagen generada
        """
        # Construir prompt combinado
        prompt = f"{identity_description}, {content_style} style"
        if additional_prompt:
            prompt += f", {additional_prompt}"
        
        return self.generate_image(prompt)
    
    def image_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Convierte imagen a bytes"""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()


# Singleton global
_diffusion_service: Optional[DiffusionService] = None


def get_diffusion_service() -> DiffusionService:
    """Obtiene instancia singleton del servicio de difusión"""
    global _diffusion_service
    if _diffusion_service is None:
        _diffusion_service = DiffusionService()
    return _diffusion_service




