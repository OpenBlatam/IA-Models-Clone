"""
Modelos de Difusión para Visualizaciones
========================================
Generación de visualizaciones usando diffusion models
"""

from typing import Dict, Any, List, Optional, Tuple
import structlog
import torch
import torch.nn as nn
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderKL
)
from PIL import Image
import numpy as np
import io

logger = structlog.get_logger()


class PsychologicalVisualizationGenerator:
    """
    Generador de visualizaciones psicológicas usando diffusion models
    """
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        use_xl: bool = False,
        device: Optional[str] = None
    ):
        """
        Inicializar generador
        
        Args:
            model_name: Nombre del modelo de difusión
            use_xl: Usar Stable Diffusion XL
            device: Dispositivo (cuda/cpu)
        """
        self.model_name = model_name
        self.use_xl = use_xl
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        
        try:
            if use_xl:
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            self.pipeline = self.pipeline.to(self.device)
            
            # Optimizar scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            logger.info("Diffusion model loaded", model=model_name, device=self.device)
        except Exception as e:
            logger.warning("Could not load diffusion model", error=str(e))
            self.pipeline = None
    
    def generate_profile_visualization(
        self,
        personality_traits: Dict[str, float],
        prompt_enhancement: Optional[str] = None
    ) -> Optional[Image.Image]:
        """
        Generar visualización del perfil psicológico
        
        Args:
            personality_traits: Rasgos de personalidad
            prompt_enhancement: Mejora del prompt (opcional)
            
        Returns:
            Imagen generada o None
        """
        if self.pipeline is None:
            logger.warning("Diffusion pipeline not available")
            return None
        
        try:
            # Crear prompt basado en rasgos
            prompt = self._create_personality_prompt(personality_traits, prompt_enhancement)
            
            # Generar imagen
            with torch.autocast(self.device):
                image = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                ).images[0]
            
            logger.info("Profile visualization generated", prompt=prompt[:50])
            return image
            
        except Exception as e:
            logger.error("Error generating visualization", error=str(e))
            return None
    
    def _create_personality_prompt(
        self,
        traits: Dict[str, float],
        enhancement: Optional[str] = None
    ) -> str:
        """Crear prompt basado en rasgos de personalidad"""
        # Identificar rasgo dominante
        dominant_trait = max(traits.items(), key=lambda x: x[1])
        trait_name, trait_value = dominant_trait
        
        # Mapear rasgos a descripciones visuales
        trait_descriptions = {
            "openness": "creative, imaginative, abstract art",
            "conscientiousness": "organized, structured, geometric patterns",
            "extraversion": "vibrant, energetic, colorful",
            "agreeableness": "harmonious, soft, peaceful",
            "neuroticism": "intense, dynamic, contrasting"
        }
        
        base_description = trait_descriptions.get(trait_name, "psychological profile")
        
        prompt = f"Abstract psychological visualization, {base_description}, "
        prompt += f"{trait_name} trait: {trait_value:.2f}, "
        prompt += "professional, high quality, detailed"
        
        if enhancement:
            prompt += f", {enhancement}"
        
        return prompt
    
    def generate_sentiment_visualization(
        self,
        sentiment_data: Dict[str, Any]
    ) -> Optional[Image.Image]:
        """
        Generar visualización de sentimientos
        
        Args:
            sentiment_data: Datos de sentimiento
            
        Returns:
            Imagen generada o None
        """
        if self.pipeline is None:
            return None
        
        try:
            sentiment = sentiment_data.get("sentiment", "neutral")
            confidence = sentiment_data.get("confidence", 0.5)
            
            color_map = {
                "positive": "warm, bright, optimistic colors",
                "negative": "cool, dark, melancholic tones",
                "neutral": "balanced, gray, neutral palette"
            }
            
            prompt = f"Emotional visualization, {color_map.get(sentiment, 'neutral')}, "
            prompt += f"sentiment: {sentiment}, confidence: {confidence:.2f}, "
            prompt += "abstract art, psychological representation"
            
            with torch.autocast(self.device):
                image = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=25,
                    guidance_scale=6.0,
                    height=512,
                    width=512
                ).images[0]
            
            return image
            
        except Exception as e:
            logger.error("Error generating sentiment visualization", error=str(e))
            return None


class DiffusionImageProcessor:
    """Procesador de imágenes generadas por diffusion"""
    
    @staticmethod
    def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
        """
        Convertir imagen a bytes
        
        Args:
            image: Imagen PIL
            format: Formato (PNG, JPEG)
            
        Returns:
            Bytes de la imagen
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """
        Convertir imagen a base64
        
        Args:
            image: Imagen PIL
            
        Returns:
            String base64
        """
        import base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


# Instancia global del generador
visualization_generator = PsychologicalVisualizationGenerator()




