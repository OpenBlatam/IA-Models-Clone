"""
Diffusion Utils - Utilidades de Diffusion Models
================================================

Utilidades para trabajar con diffusion models usando Diffusers.
"""

import logging
import torch
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Intentar importar diffusers
try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        DDIMScheduler,
        DDPMScheduler,
        PNDMScheduler,
        EulerDiscreteScheduler,
        DPMSolverMultistepScheduler
    )
    _has_diffusers = True
except ImportError:
    _has_diffusers = False
    logger.warning("diffusers library not available")


class DiffusionPipelineManager:
    """
    Gestor de pipelines de diffusion models.
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        pipeline_type: str = "stable-diffusion",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_safetensors: bool = True
    ):
        """
        Inicializar gestor de pipeline.
        
        Args:
            model_id: ID del modelo
            pipeline_type: Tipo de pipeline (stable-diffusion, stable-diffusion-xl)
            device: Dispositivo
            use_safetensors: Usar safetensors
        """
        if not _has_diffusers:
            raise ImportError("diffusers library is required")
        
        self.model_id = model_id
        self.pipeline_type = pipeline_type
        self.device = torch.device(device)
        self.use_safetensors = use_safetensors
        self.pipeline = None
    
    def load_pipeline(
        self,
        scheduler_type: str = "ddim",
        torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ) -> None:
        """
        Cargar pipeline de diffusion.
        
        Args:
            scheduler_type: Tipo de scheduler
            torch_dtype: Tipo de datos de PyTorch
        """
        if self.pipeline_type == "stable-diffusion-xl":
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                use_safetensors=self.use_safetensors
            )
        else:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                use_safetensors=self.use_safetensors
            )
        
        # Configurar scheduler
        self.set_scheduler(scheduler_type)
        
        # Mover a dispositivo
        self.pipeline = self.pipeline.to(self.device)
        
        # Optimización
        if self.device.type == "cuda":
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_xformers_memory_efficient_attention()
        
        logger.info(f"Pipeline loaded: {self.model_id} on {self.device}")
    
    def set_scheduler(self, scheduler_type: str) -> None:
        """
        Configurar scheduler.
        
        Args:
            scheduler_type: Tipo de scheduler (ddim, ddpm, pndm, euler, dpm)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not loaded")
        
        scheduler_map = {
            "ddim": DDIMScheduler,
            "ddpm": DDPMScheduler,
            "pndm": PNDMScheduler,
            "euler": EulerDiscreteScheduler,
            "dpm": DPMSolverMultistepScheduler
        }
        
        scheduler_class = scheduler_map.get(scheduler_type.lower())
        if scheduler_class is None:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        self.pipeline.scheduler = scheduler_class.from_config(
            self.pipeline.scheduler.config
        )
        logger.info(f"Scheduler set to: {scheduler_type}")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None
    ) -> List[Any]:
        """
        Generar imágenes.
        
        Args:
            prompt: Prompt de texto
            negative_prompt: Prompt negativo
            num_inference_steps: Pasos de inferencia
            guidance_scale: Escala de guía
            height: Altura de imagen
            width: Ancho de imagen
            num_images_per_prompt: Número de imágenes
            seed: Semilla para reproducibilidad
            
        Returns:
            Lista de imágenes generadas
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not loaded")
        
        # Generador para reproducibilidad
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generar
        images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator
        ).images
        
        return images
    
    def save_images(self, images: List[Any], output_dir: str, prefix: str = "generated") -> List[str]:
        """
        Guardar imágenes generadas.
        
        Args:
            images: Lista de imágenes
            output_dir: Directorio de salida
            prefix: Prefijo de archivo
            
        Returns:
            Lista de rutas guardadas
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for i, image in enumerate(images):
            path = output_path / f"{prefix}_{i+1}.png"
            image.save(path)
            saved_paths.append(str(path))
        
        logger.info(f"Saved {len(images)} images to {output_dir}")
        return saved_paths


class NoiseScheduler:
    """
    Gestor de noise schedulers para diffusion.
    """
    
    @staticmethod
    def create_scheduler(
        scheduler_type: str,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear"
    ):
        """
        Crear scheduler de noise.
        
        Args:
            scheduler_type: Tipo de scheduler
            num_train_timesteps: Número de timesteps de entrenamiento
            beta_start: Beta inicial
            beta_end: Beta final
            beta_schedule: Schedule de beta
            
        Returns:
            Scheduler configurado
        """
        if not _has_diffusers:
            raise ImportError("diffusers library is required")
        
        config = {
            "num_train_timesteps": num_train_timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "beta_schedule": beta_schedule
        }
        
        scheduler_map = {
            "ddim": DDIMScheduler,
            "ddpm": DDPMScheduler,
            "pndm": PNDMScheduler,
            "euler": EulerDiscreteScheduler,
            "dpm": DPMSolverMultistepScheduler
        }
        
        scheduler_class = scheduler_map.get(scheduler_type.lower())
        if scheduler_class is None:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        return scheduler_class(**config)




