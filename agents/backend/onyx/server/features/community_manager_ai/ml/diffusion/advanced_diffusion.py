"""
Advanced Diffusion - Diffusion Avanzado
========================================

Funcionalidades avanzadas para diffusion models.
"""

import logging
import torch
from typing import Optional, List, Dict, Any
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline
)
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class AdvancedDiffusionPipeline:
    """Pipeline avanzado de diffusion con múltiples características"""
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        use_xl: bool = False,
        use_controlnet: bool = False,
        controlnet_model: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Inicializar pipeline avanzado
        
        Args:
            model_name: Nombre del modelo
            use_xl: Usar SDXL
            use_controlnet: Usar ControlNet
            controlnet_model: Modelo de ControlNet
            device: Dispositivo
            dtype: Tipo de datos
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        
        try:
            if use_controlnet and controlnet_model:
                # Pipeline con ControlNet
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_model,
                    torch_dtype=self.dtype
                )
                self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    model_name,
                    controlnet=controlnet,
                    torch_dtype=self.dtype
                )
            elif use_xl:
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
                self.pipe.scheduler.config,
                use_karras_sigmas=True
            )
            
            # Optimizaciones
            if self.device == "cuda":
                self.pipe = self.pipe.to(self.device)
                self.pipe.enable_attention_slicing()
                if hasattr(self.pipe, "enable_vae_slicing"):
                    self.pipe.enable_vae_slicing()
                if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                    except Exception:
                        pass
            
            logger.info(f"Advanced Diffusion Pipeline inicializado en {self.device}")
        except Exception as e:
            logger.error(f"Error inicializando pipeline: {e}")
            self.pipe = None
    
    def generate_with_control(
        self,
        prompt: str,
        control_image: Image.Image,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0
    ) -> Optional[Image.Image]:
        """
        Generar con ControlNet
        
        Args:
            prompt: Prompt de texto
            control_image: Imagen de control
            negative_prompt: Prompt negativo
            num_inference_steps: Pasos de inferencia
            guidance_scale: Escala de guía
            controlnet_conditioning_scale: Escala de ControlNet
            
        Returns:
            Imagen generada
        """
        if not self.pipe or not hasattr(self.pipe, "controlnet"):
            logger.error("ControlNet no disponible")
            return None
        
        try:
            image = self.pipe(
                prompt=prompt,
                image=control_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale
            ).images[0]
            
            return image
        except Exception as e:
            logger.error(f"Error generando con ControlNet: {e}")
            return None
    
    def img2img(
        self,
        prompt: str,
        init_image: Image.Image,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> Optional[Image.Image]:
        """
        Image-to-image generation
        
        Args:
            prompt: Prompt de texto
            init_image: Imagen inicial
            strength: Fuerza de transformación (0-1)
            num_inference_steps: Pasos de inferencia
            guidance_scale: Escala de guía
            
        Returns:
            Imagen generada
        """
        if not self.pipe:
            return None
        
        try:
            # Redimensionar si es necesario
            init_image = init_image.resize((512, 512))
            
            image = self.pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            return image
        except Exception as e:
            logger.error(f"Error en img2img: {e}")
            return None
    
    def inpainting(
        self,
        prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> Optional[Image.Image]:
        """
        Inpainting (rellenar áreas de imagen)
        
        Args:
            prompt: Prompt de texto
            image: Imagen base
            mask_image: Máscara (áreas a rellenar)
            num_inference_steps: Pasos de inferencia
            guidance_scale: Escala de guía
            
        Returns:
            Imagen con inpainting
        """
        if not self.pipe:
            return None
        
        try:
            # Usar pipeline de inpainting si está disponible
            from diffusers import StableDiffusionInpaintPipeline
            
            inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=self.dtype
            ).to(self.device)
            
            image = inpaint_pipe(
                prompt=prompt,
                image=image,
                mask_image=mask_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            return image
        except Exception as e:
            logger.error(f"Error en inpainting: {e}")
            return None
    
    def change_scheduler(self, scheduler_type: str = "dpm"):
        """
        Cambiar scheduler
        
        Args:
            scheduler_type: Tipo de scheduler (dpm, euler, ddim)
        """
        if not self.pipe:
            return
        
        scheduler_map = {
            "dpm": DPMSolverMultistepScheduler,
            "euler": EulerAncestralDiscreteScheduler,
            "ddim": DDIMScheduler
        }
        
        scheduler_class = scheduler_map.get(scheduler_type.lower())
        if scheduler_class:
            self.pipe.scheduler = scheduler_class.from_config(
                self.pipe.scheduler.config
            )
            logger.info(f"Scheduler cambiado a {scheduler_type}")




