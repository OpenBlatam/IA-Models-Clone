"""
Diffusion Models
================

Sistema de modelos de difusión usando Diffusers.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

try:
    import torch
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        DDIMScheduler,
        DDPMScheduler,
        UNet2DConditionModel
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    torch = None
    StableDiffusionPipeline = None
    StableDiffusionXLPipeline = None
    DDIMScheduler = None
    DDPMScheduler = None
    UNet2DConditionModel = None

logger = logging.getLogger(__name__)


class DiffusionModelType(Enum):
    """Tipo de modelo de difusión."""
    STABLE_DIFFUSION = "stable_diffusion"
    STABLE_DIFFUSION_XL = "stable_diffusion_xl"
    CUSTOM = "custom"


class DiffusionModelManager:
    """
    Gestor de modelos de difusión.
    
    Maneja creación y uso de modelos de difusión.
    """
    
    def __init__(self):
        """Inicializar gestor."""
        if not DIFFUSERS_AVAILABLE:
            logger.warning("Diffusers library not available. Install with: pip install diffusers")
        
        self.pipelines: Dict[str, Any] = {}
        self.device = "cuda" if DIFFUSERS_AVAILABLE else "cpu"
    
    def load_pipeline(
        self,
        model_type: DiffusionModelType,
        model_id: str,
        use_fp16: bool = True
    ) -> str:
        """
        Cargar pipeline de difusión.
        
        Args:
            model_type: Tipo de modelo
            model_id: ID del modelo (HuggingFace)
            use_fp16: Usar FP16
            
        Returns:
            ID del pipeline
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library is required")
        
        import uuid
        pipeline_id = str(uuid.uuid4())
        
        try:
            if model_type == DiffusionModelType.STABLE_DIFFUSION:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if use_fp16 else torch.float32
                )
            elif model_type == DiffusionModelType.STABLE_DIFFUSION_XL:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if use_fp16 else torch.float32
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            pipeline = pipeline.to(self.device)
            self.pipelines[pipeline_id] = pipeline
            
            logger.info(f"Loaded {model_type.value} pipeline: {model_id}")
            
            return pipeline_id
            
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            raise
    
    def generate(
        self,
        pipeline_id: str,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Generar imagen.
        
        Args:
            pipeline_id: ID del pipeline
            prompt: Prompt de texto
            num_inference_steps: Número de pasos de inferencia
            guidance_scale: Escala de guía
            negative_prompt: Prompt negativo (opcional)
            **kwargs: Argumentos adicionales
            
        Returns:
            Imagen generada
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        
        pipeline = self.pipelines[pipeline_id]
        
        try:
            result = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                **kwargs
            )
            
            return result.images[0] if hasattr(result, 'images') else result
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "total_pipelines": len(self.pipelines),
            "device": self.device
        }


# Instancia global
_diffusion_manager: Optional[DiffusionModelManager] = None


def get_diffusion_manager() -> DiffusionModelManager:
    """Obtener instancia global del gestor de difusión."""
    global _diffusion_manager
    if _diffusion_manager is None:
        _diffusion_manager = DiffusionModelManager()
    return _diffusion_manager

