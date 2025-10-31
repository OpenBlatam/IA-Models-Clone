"""
Diffusion Pipeline System using Diffusers Library
Implements StableDiffusionPipeline and StableDiffusionXLPipeline
"""

import torch
import torch.nn.functional as F
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler
)
from diffusers.utils import randn_tensor
from typing import Optional, Union, List, Dict, Any
import logging
import gc
from pathlib import Path
import numpy as np
from PIL import Image
import json

class DiffusionPipelineSystem:
    """Production-ready diffusion pipeline system"""
    
    def __init__(self, 
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 xl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 torch_dtype: torch.dtype = torch.float16,
                 enable_attention_slicing: bool = True,
                 enable_vae_slicing: bool = True,
                 enable_model_cpu_offload: bool = False,
                 enable_sequential_cpu_offload: bool = False):
        
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_id = model_id
        self.xl_model_id = xl_model_id
        
        # Initialize pipelines
        self.sd_pipeline = None
        self.sdxl_pipeline = None
        
        # Performance optimizations
        self.enable_attention_slicing = enable_attention_slicing
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_sequential_cpu_offload = enable_sequential_cpu_offload
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize pipelines
        self._initialize_pipelines()
    
    def _initialize_pipelines(self):
        """Initialize Stable Diffusion pipelines with optimizations"""
        try:
            # Initialize SD 1.5 pipeline
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Apply optimizations
            if self.enable_attention_slicing:
                self.sd_pipeline.enable_attention_slicing()
            
            if self.enable_vae_slicing:
                self.sd_pipeline.enable_vae_slicing()
            
            if self.enable_model_cpu_offload:
                self.sd_pipeline.enable_model_cpu_offload()
            
            if self.enable_sequential_cpu_offload:
                self.sd_pipeline.enable_sequential_cpu_offload()
            
            self.sd_pipeline = self.sd_pipeline.to(self.device)
            
            # Initialize SDXL pipeline
            self.sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.xl_model_id,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Apply same optimizations
            if self.enable_attention_slicing:
                self.sdxl_pipeline.enable_attention_slicing()
            
            if self.enable_vae_slicing:
                self.sdxl_pipeline.enable_vae_slicing()
            
            if self.enable_model_cpu_offload:
                self.sdxl_pipeline.enable_model_cpu_offload()
            
            if self.enable_sequential_cpu_offload:
                self.sdxl_pipeline.enable_sequential_cpu_offload()
            
            self.sdxl_pipeline = self.sdxl_pipeline.to(self.device)
            
            self.logger.info("Diffusion pipelines initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipelines: {e}")
            raise
    
    def generate_image_sd(self,
                         prompt: str,
                         negative_prompt: str = "",
                         num_inference_steps: int = 50,
                         guidance_scale: float = 7.5,
                         width: int = 512,
                         height: int = 512,
                         num_images: int = 1,
                         seed: Optional[int] = None,
                         scheduler: str = "default") -> List[Image.Image]:
        """Generate images using Stable Diffusion 1.5"""
        
        if self.sd_pipeline is None:
            raise RuntimeError("SD pipeline not initialized")
        
        # Set scheduler
        if scheduler != "default":
            self._set_scheduler(self.sd_pipeline, scheduler)
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        try:
            with torch.autocast(self.device):
                images = self.sd_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    num_images_per_prompt=num_images
                ).images
            
            self.logger.info(f"Generated {len(images)} SD images")
            return images
            
        except Exception as e:
            self.logger.error(f"Error generating SD images: {e}")
            raise
    
    def generate_image_sdxl(self,
                           prompt: str,
                           negative_prompt: str = "",
                           num_inference_steps: int = 50,
                           guidance_scale: float = 7.5,
                           width: int = 1024,
                           height: int = 1024,
                           num_images: int = 1,
                           seed: Optional[int] = None,
                           scheduler: str = "default") -> List[Image.Image]:
        """Generate images using Stable Diffusion XL"""
        
        if self.sdxl_pipeline is None:
            raise RuntimeError("SDXL pipeline not initialized")
        
        # Set scheduler
        if scheduler != "default":
            self._set_scheduler(self.sdxl_pipeline, scheduler)
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        try:
            with torch.autocast(self.device):
                images = self.sdxl_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    num_images_per_prompt=num_images
                ).images
            
            self.logger.info(f"Generated {len(images)} SDXL images")
            return images
            
        except Exception as e:
            self.logger.error(f"Error generating SDXL images: {e}")
            raise
    
    def _set_scheduler(self, pipeline, scheduler_name: str):
        """Set custom scheduler for pipeline"""
        schedulers = {
            "dpm": DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config),
            "euler": EulerDiscreteScheduler.from_config(pipeline.scheduler.config),
            "ddim": DDIMScheduler.from_config(pipeline.scheduler.config),
            "pndm": PNDMScheduler.from_config(pipeline.scheduler.config)
        }
        
        if scheduler_name in schedulers:
            pipeline.scheduler = schedulers[scheduler_name]
            self.logger.info(f"Set scheduler to {scheduler_name}")
    
    def generate_with_controlnet(self, 
                                prompt: str,
                                control_image: Image.Image,
                                controlnet_type: str = "canny",
                                **kwargs) -> List[Image.Image]:
        """Generate images with ControlNet (placeholder for future implementation)"""
        self.logger.warning("ControlNet not yet implemented")
        return self.generate_image_sd(prompt, **kwargs)
    
    def generate_with_lora(self,
                           prompt: str,
                           lora_path: str,
                           **kwargs) -> List[Image.Image]:
        """Generate images with LoRA fine-tuned model"""
        try:
            # Load LoRA weights
            self.sd_pipeline.load_lora_weights(lora_path)
            self.logger.info(f"Loaded LoRA weights from {lora_path}")
            
            # Generate images
            images = self.generate_image_sd(prompt, **kwargs)
            
            # Unload LoRA weights to free memory
            self.sd_pipeline.unload_lora_weights()
            
            return images
            
        except Exception as e:
            self.logger.error(f"Error with LoRA generation: {e}")
            raise
    
    def batch_generate(self,
                       prompts: List[str],
                       pipeline_type: str = "sd",
                       **kwargs) -> Dict[str, List[Image.Image]]:
        """Generate images for multiple prompts in batch"""
        
        results = {}
        
        for i, prompt in enumerate(prompts):
            try:
                if pipeline_type == "sdxl":
                    images = self.generate_image_sdxl(prompt, **kwargs)
                else:
                    images = self.generate_image_sd(prompt, **kwargs)
                
                results[f"prompt_{i}"] = images
                
                # Clear cache between generations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.error(f"Error in batch generation for prompt {i}: {e}")
                results[f"prompt_{i}"] = []
        
        return results
    
    def save_pipeline_config(self, save_path: str):
        """Save pipeline configuration"""
        config = {
            "model_id": self.model_id,
            "xl_model_id": self.xl_model_id,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "enable_attention_slicing": self.enable_attention_slicing,
            "enable_vae_slicing": self.enable_vae_slicing,
            "enable_model_cpu_offload": self.enable_model_cpu_offload,
            "enable_sequential_cpu_offload": self.enable_sequential_cpu_offload
        }
        
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Pipeline config saved to {save_path}")
    
    def cleanup(self):
        """Clean up resources and free memory"""
        try:
            # Clear pipelines
            if self.sd_pipeline:
                del self.sd_pipeline
                self.sd_pipeline = None
            
            if self.sdxl_pipeline:
                del self.sdxl_pipeline
                self.sdxl_pipeline = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("Pipeline system cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and status"""
        return {
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "sd_pipeline_loaded": self.sd_pipeline is not None,
            "sdxl_pipeline_loaded": self.sdxl_pipeline is not None,
            "cuda_available": torch.cuda.is_available(),
            "cuda_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "cuda_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }

def main():
    """Example usage of the diffusion pipeline system"""
    
    # Initialize system
    pipeline_system = DiffusionPipelineSystem(
        enable_attention_slicing=True,
        enable_vae_slicing=True
    )
    
    try:
        # Generate SD image
        sd_images = pipeline_system.generate_image_sd(
            prompt="A beautiful sunset over mountains, digital art",
            num_inference_steps=30,
            guidance_scale=7.5,
            seed=42
        )
        
        # Save SD image
        if sd_images:
            sd_images[0].save("sd_generated.png")
            print("SD image saved as sd_generated.png")
        
        # Generate SDXL image
        sdxl_images = pipeline_system.generate_image_sdxl(
            prompt="A futuristic cityscape at night, cinematic lighting",
            num_inference_steps=40,
            guidance_scale=8.0,
            seed=123
        )
        
        # Save SDXL image
        if sdxl_images:
            sdxl_images[0].save("sdxl_generated.png")
            print("SDXL image saved as sdxl_generated.png")
        
        # Batch generation
        prompts = [
            "A cat sitting on a windowsill",
            "A robot in a garden",
            "A spaceship in orbit"
        ]
        
        batch_results = pipeline_system.batch_generate(
            prompts=prompts,
            pipeline_type="sd",
            num_inference_steps=25
        )
        
        print(f"Batch generation completed: {len(batch_results)} results")
        
        # Save batch results
        for prompt_name, images in batch_results.items():
            if images:
                images[0].save(f"{prompt_name}.png")
        
        # Get system info
        system_info = pipeline_system.get_system_info()
        print("System info:", system_info)
        
    except Exception as e:
        print(f"Error in main: {e}")
    
    finally:
        # Cleanup
        pipeline_system.cleanup()

if __name__ == "__main__":
    main()


