"""
Diffusion Models for Progress Visualization and Image Generation
Using HuggingFace Diffusers library for Stable Diffusion
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)

try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        UNet2DConditionModel
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class RecoveryProgressVisualizer(nn.Module):
    """
    Generate progress visualization images using Stable Diffusion
    Creates motivational and progress-tracking images
    """
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True,
        scheduler_type: str = "dpm"
    ):
        """
        Initialize diffusion model for progress visualization
        
        Args:
            model_name: HuggingFace model name
            device: PyTorch device
            use_mixed_precision: Use FP16 for faster generation
            scheduler_type: Scheduler type (dpm, euler)
        """
        super().__init__()
        
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library is required")
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        
        try:
            # Load pipeline
            torch_dtype = torch.float16 if self.use_mixed_precision else torch.float32
            
            if "xl" in model_name.lower():
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    use_safetensors=True
                )
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    use_safetensors=True
                )
            
            self.pipeline = self.pipeline.to(self.device)
            
            # Set scheduler with proper configuration
            if scheduler_type == "dpm":
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config,
                    use_karras_sigmas=True,  # Better quality
                    algorithm_type="dpmsolver++"
                )
            elif scheduler_type == "euler":
                self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    self.pipeline.scheduler.config,
                    use_karras_sigmas=True
                )
            else:
                # Default: keep original scheduler
                logger.info(f"Using default scheduler: {type(self.pipeline.scheduler).__name__}")
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("XFormers memory efficient attention enabled")
                except Exception as e:
                    logger.warning(f"XFormers not available: {e}")
            
            # Enable attention slicing for lower memory usage
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
            
            logger.info(
                f"RecoveryProgressVisualizer initialized on {self.device} "
                f"(mixed_precision={self.use_mixed_precision})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize diffusion model: {e}")
            raise
    
    def generate_progress_image(
        self,
        prompt: str,
        days_sober: int,
        progress_score: float,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate progress visualization image
        
        Args:
            prompt: Text prompt for image generation
            days_sober: Days of sobriety
            progress_score: Progress score (0-1)
            negative_prompt: Negative prompt to avoid certain elements
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for prompt adherence
            height: Image height
            width: Image width
            seed: Random seed for reproducibility
            
        Returns:
            Generated PIL Image
        """
        try:
            # Enhance prompt with progress information
            enhanced_prompt = (
                f"{prompt}, celebrating {days_sober} days of recovery, "
                f"progress: {progress_score:.0%}, motivational, inspiring, "
                f"positive, hopeful, professional illustration"
            )
            
            if negative_prompt is None:
                negative_prompt = (
                    "negative, sad, depressing, failure, relapse, "
                    "unprofessional, low quality, blurry"
                )
            
            # Set seed for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate image with optimized settings
            with torch.autocast(device_type=self.device.type, enabled=self.use_mixed_precision):
                image = self.pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                    output_type="pil",  # Explicit output type
                    return_dict=False  # Faster for single image
                )[0]
            
            logger.info(f"Generated progress image for {days_sober} days")
            return image
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}", exc_info=True)
            # Return a placeholder image
            return Image.new('RGB', (width, height), color=(200, 200, 200))
    
    def generate_milestone_image(
        self,
        milestone: str,
        achievement: str,
        num_inference_steps: int = 50
    ) -> Image.Image:
        """Generate milestone celebration image"""
        prompt = (
            f"Celebrating milestone: {milestone}, achievement: {achievement}, "
            f"trophy, certificate, success, celebration, professional design"
        )
        return self.generate_progress_image(
            prompt=prompt,
            days_sober=0,
            progress_score=1.0,
            num_inference_steps=num_inference_steps
        )
    
    def generate_motivational_image(
        self,
        message: str,
        num_inference_steps: int = 50
    ) -> Image.Image:
        """Generate motivational image with text message"""
        prompt = (
            f"Inspirational image: {message}, motivational quote, "
            f"beautiful landscape, sunrise, hope, strength, professional"
        )
        return self.generate_progress_image(
            prompt=prompt,
            days_sober=0,
            progress_score=0.5,
            num_inference_steps=num_inference_steps
        )


class RecoveryChartGenerator(nn.Module):
    """
    Generate progress charts and visualizations using diffusion models
    Creates data-driven visualizations
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True
    ):
        """
        Initialize chart generator
        
        Args:
            device: PyTorch device
            use_mixed_precision: Use mixed precision
        """
        super().__init__()
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision
        
        logger.info("RecoveryChartGenerator initialized")
    
    def generate_progress_chart(
        self,
        progress_data: List[Dict[str, float]],
        chart_type: str = "line"
    ) -> Image.Image:
        """
        Generate progress chart visualization
        
        Args:
            progress_data: List of progress data points
            chart_type: Type of chart (line, bar, radar)
            
        Returns:
            Generated chart image
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Extract data
            days = [d.get("day", i) for i, d in enumerate(progress_data)]
            scores = [d.get("score", 0.0) for d in progress_data]
            
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == "line":
                ax.plot(days, scores, marker='o', linewidth=2, markersize=8)
                ax.fill_between(days, scores, alpha=0.3)
            elif chart_type == "bar":
                ax.bar(days, scores, alpha=0.7)
            elif chart_type == "radar":
                # Convert to radar chart
                categories = list(progress_data[0].keys())
                categories.remove("day")
                values = [[d.get(cat, 0.0) for cat in categories] for d in progress_data]
                
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                values = values[0] + values[0][:1]  # Close the circle
                angles += angles[:1]
                
                ax = plt.subplot(111, projection='polar')
                ax.plot(angles, values, 'o-', linewidth=2)
                ax.fill(angles, values, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
            
            ax.set_xlabel("Days")
            ax.set_ylabel("Progress Score")
            ax.set_title("Recovery Progress")
            ax.grid(True, alpha=0.3)
            
            # Convert to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            plt.close()
            
            return image
            
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return Image.new('RGB', (800, 600), color=(255, 255, 255))


def create_progress_visualizer(
    model_name: str = "runwayml/stable-diffusion-v1-5",
    device: Optional[torch.device] = None,
    use_mixed_precision: bool = True
) -> RecoveryProgressVisualizer:
    """Factory function for progress visualizer"""
    return RecoveryProgressVisualizer(
        model_name=model_name,
        device=device,
        use_mixed_precision=use_mixed_precision
    )


def create_chart_generator(
    device: Optional[torch.device] = None
) -> RecoveryChartGenerator:
    """Factory function for chart generator"""
    return RecoveryChartGenerator(device=device)






