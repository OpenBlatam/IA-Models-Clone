"""
Modular Integration with HuggingFace Diffusers
Provides seamless integration with diffusion models
"""

from typing import Optional, Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)

try:
    from diffusers import (
        DiffusionPipeline,
        DDIMScheduler,
        DDPMScheduler,
        PNDMScheduler,
        EulerDiscreteScheduler,
        DPMSolverMultistepScheduler,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        AudioLDMPipeline,
        UNet2DConditionModel
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers library not available")

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class DiffusionSchedulerFactory:
    """Factory for creating diffusion schedulers"""
    
    @staticmethod
    def create(
        scheduler_type: str,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear"
    ):
        """
        Create scheduler based on type
        
        Args:
            scheduler_type: Type of scheduler ("ddim", "ddpm", "pndm", "euler", "dpm")
            num_train_timesteps: Number of training timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            beta_schedule: Beta schedule type
        
        Returns:
            Scheduler instance
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library required")
        
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == "ddim":
            return DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                clip_sample=False
            )
        
        elif scheduler_type == "ddpm":
            return DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule
            )
        
        elif scheduler_type == "pndm":
            return PNDMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule
            )
        
        elif scheduler_type == "euler":
            return EulerDiscreteScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule
            )
        
        elif scheduler_type == "dpm":
            return DPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class DiffusionPipelineWrapper:
    """
    Wrapper for diffusion pipelines with modular configuration
    """
    
    def __init__(
        self,
        pipeline_type: str = "stable_diffusion",
        model_id: Optional[str] = None,
        scheduler_type: str = "ddim",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ):
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers library required")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.pipeline_type = pipeline_type
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load pipeline
        self._load_pipeline(model_id, scheduler_type)
    
    def _load_pipeline(self, model_id: Optional[str], scheduler_type: str):
        """Load diffusion pipeline"""
        try:
            if self.pipeline_type == "stable_diffusion":
                if model_id is None:
                    model_id = "runwayml/stable-diffusion-v1-5"
                
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.torch_dtype
                )
            
            elif self.pipeline_type == "stable_diffusion_xl":
                if model_id is None:
                    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
                
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.torch_dtype
                )
            
            elif self.pipeline_type == "audio_ldm":
                if model_id is None:
                    model_id = "cvssp/audioldm-s-full"
                
                self.pipeline = AudioLDMPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.torch_dtype
                )
            
            else:
                raise ValueError(f"Unknown pipeline type: {self.pipeline_type}")
            
            # Set scheduler
            if scheduler_type != "default":
                scheduler = DiffusionSchedulerFactory.create(scheduler_type)
                self.pipeline.scheduler = scheduler
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            logger.info(f"Loaded {self.pipeline_type} pipeline on {self.device}")
        
        except Exception as e:
            logger.error(f"Error loading pipeline: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate using diffusion pipeline
        
        Args:
            prompt: Text prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            negative_prompt: Negative prompt
            seed: Random seed
            **kwargs: Additional pipeline arguments
        
        Returns:
            Generation results
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        try:
            # Generate
            result = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                **kwargs
            )
            
            return {
                "success": True,
                "images": result.images if hasattr(result, 'images') else None,
                "audio": result.audios if hasattr(result, 'audios') else None,
                "prompt": prompt,
                "num_inference_steps": num_inference_steps
            }
        
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt
            }



