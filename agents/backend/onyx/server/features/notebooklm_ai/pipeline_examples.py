from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from dataclasses import dataclass
from diffusion_pipelines import (
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Diffusion Pipeline Examples
==========================

Comprehensive examples demonstrating different diffusion pipeline types:
- StableDiffusionPipeline
- StableDiffusionXLPipeline
- StableDiffusionImg2ImgPipeline
- StableDiffusionInpaintPipeline
- StableDiffusionControlNetPipeline
- Custom pipelines and advanced features

Features: Practical usage scenarios, batch processing, performance optimization,
error handling, and production-ready implementations.
"""


# Import our pipeline implementation
    DiffusionPipelineManager, PipelineConfig, GenerationRequest,
    CustomPipelineFactory, PipelinePerformanceMonitor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineExample:
    """Example configuration for pipeline demonstration."""
    name: str
    description: str
    pipeline_type: str
    model_name: str
    prompts: List[str]
    config_overrides: Dict[str, Any] = None


class PipelineExamples:
    """Comprehensive examples for different pipeline types."""
    
    def __init__(self) -> Any:
        self.config = PipelineConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            enable_attention_slicing=True,
            enable_xformers_memory_efficient_attention=True,
            enable_vae_slicing=True,
            max_workers=4
        )
        self.manager = DiffusionPipelineManager(self.config)
        self.monitor = PipelinePerformanceMonitor()
        
    async def run_stable_diffusion_examples(self) -> Any:
        """Run Stable Diffusion pipeline examples."""
        logger.info("Running Stable Diffusion examples...")
        
        # Load pipeline
        pipeline_key = await self.manager.load_stable_diffusion_pipeline()
        
        # Example 1: Basic text-to-image generation
        basic_request = GenerationRequest(
            prompt="A beautiful sunset over mountains, digital art, high quality",
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=30,
            guidance_scale=7.5,
            height=512,
            width=512
        )
        
        self.monitor.start_monitoring(pipeline_key)
        images = await self.manager.generate_image(pipeline_key, basic_request)
        metrics = self.monitor.end_monitoring(pipeline_key)
        
        logger.info(f"Basic generation completed in {metrics['duration']:.2f}s")
        logger.info(f"Memory usage: {metrics['peak_memory'] / 1024 / 1024:.1f} MB")
        
        # Example 2: High-quality generation with more steps
        high_quality_request = GenerationRequest(
            prompt="A detailed portrait of a wise old wizard, fantasy art, intricate details",
            negative_prompt="cartoon, anime, blurry, low quality",
            num_inference_steps=50,
            guidance_scale=8.5,
            height=768,
            width=512
        )
        
        self.monitor.start_monitoring(pipeline_key)
        high_quality_images = await self.manager.generate_image(pipeline_key, high_quality_request)
        metrics = self.monitor.end_monitoring(pipeline_key)
        
        logger.info(f"High-quality generation completed in {metrics['duration']:.2f}s")
        
        # Example 3: Batch processing
        batch_requests = [
            GenerationRequest(
                prompt="A serene lake with mountains in the background",
                num_inference_steps=25,
                guidance_scale=7.0
            ),
            GenerationRequest(
                prompt="A futuristic cityscape at night with neon lights",
                num_inference_steps=25,
                guidance_scale=7.0
            ),
            GenerationRequest(
                prompt="A cozy cottage in a magical forest",
                num_inference_steps=25,
                guidance_scale=7.0
            )
        ]
        
        self.monitor.start_monitoring(pipeline_key)
        batch_results = await self.manager.batch_generate(pipeline_key, batch_requests)
        metrics = self.monitor.end_monitoring(pipeline_key)
        
        logger.info(f"Batch generation completed in {metrics['duration']:.2f}s")
        logger.info(f"Generated {sum(len(images) for images in batch_results)} images")
        
        return {
            "basic": images,
            "high_quality": high_quality_images,
            "batch": batch_results
        }
    
    async def run_stable_diffusion_xl_examples(self) -> Any:
        """Run Stable Diffusion XL pipeline examples."""
        logger.info("Running Stable Diffusion XL examples...")
        
        # Load XL pipeline
        xl_pipeline_key = await self.manager.load_stable_diffusion_xl_pipeline()
        
        # Example 1: High-resolution generation
        xl_request = GenerationRequest(
            prompt="A majestic dragon flying over a medieval castle, epic fantasy scene",
            negative_prompt="cartoon, anime, blurry, low quality, distorted",
            num_inference_steps=30,
            guidance_scale=7.5,
            height=1024,
            width=1024,
            original_size=(1024, 1024),
            target_size=(1024, 1024)
        )
        
        self.monitor.start_monitoring(xl_pipeline_key)
        xl_images = await self.manager.generate_image(xl_pipeline_key, xl_request)
        metrics = self.monitor.end_monitoring(xl_pipeline_key)
        
        logger.info(f"XL generation completed in {metrics['duration']:.2f}s")
        
        # Example 2: Different aspect ratios
        landscape_request = GenerationRequest(
            prompt="A panoramic view of the Grand Canyon at sunset",
            negative_prompt="blurry, low quality",
            num_inference_steps=25,
            guidance_scale=7.0,
            height=768,
            width=1024,
            original_size=(1024, 768),
            target_size=(1024, 768)
        )
        
        self.monitor.start_monitoring(xl_pipeline_key)
        landscape_images = await self.manager.generate_image(xl_pipeline_key, landscape_request)
        metrics = self.monitor.end_monitoring(xl_pipeline_key)
        
        logger.info(f"Landscape XL generation completed in {metrics['duration']:.2f}s")
        
        return {
            "standard": xl_images,
            "landscape": landscape_images
        }
    
    async def run_img2img_examples(self) -> Any:
        """Run Img2Img pipeline examples."""
        logger.info("Running Img2Img examples...")
        
        # Load img2img pipeline
        img2img_pipeline_key = await self.manager.load_img2img_pipeline()
        
        # Create a sample input image
        input_image = self._create_sample_image()
        
        # Example 1: Style transfer
        style_request = GenerationRequest(
            prompt="Turn this into a Van Gogh painting style",
            image=input_image,
            strength=0.8,
            num_inference_steps=40,
            guidance_scale=7.5
        )
        
        self.monitor.start_monitoring(img2img_pipeline_key)
        style_images = await self.manager.generate_image(img2img_pipeline_key, style_request)
        metrics = self.monitor.end_monitoring(img2img_pipeline_key)
        
        logger.info(f"Style transfer completed in {metrics['duration']:.2f}s")
        
        # Example 2: Light strength transformation
        light_request = GenerationRequest(
            prompt="Make this image brighter and more vibrant",
            image=input_image,
            strength=0.6,
            num_inference_steps=30,
            guidance_scale=6.0
        )
        
        self.monitor.start_monitoring(img2img_pipeline_key)
        light_images = await self.manager.generate_image(img2img_pipeline_key, light_request)
        metrics = self.monitor.end_monitoring(img2img_pipeline_key)
        
        logger.info(f"Light transformation completed in {metrics['duration']:.2f}s")
        
        # Example 3: Strong transformation
        strong_request = GenerationRequest(
            prompt="Transform this into a cyberpunk cityscape",
            image=input_image,
            strength=0.9,
            num_inference_steps=50,
            guidance_scale=8.0
        )
        
        self.monitor.start_monitoring(img2img_pipeline_key)
        strong_images = await self.manager.generate_image(img2img_pipeline_key, strong_request)
        metrics = self.monitor.end_monitoring(img2img_pipeline_key)
        
        logger.info(f"Strong transformation completed in {metrics['duration']:.2f}s")
        
        return {
            "style_transfer": style_images,
            "light_transformation": light_images,
            "strong_transformation": strong_images
        }
    
    async def run_inpaint_examples(self) -> Any:
        """Run Inpaint pipeline examples."""
        logger.info("Running Inpaint examples...")
        
        # Load inpaint pipeline
        inpaint_pipeline_key = await self.manager.load_inpaint_pipeline()
        
        # Create sample image and mask
        original_image = self._create_sample_image()
        mask_image = self._create_sample_mask(original_image)
        
        # Example 1: Object removal and replacement
        inpaint_request = GenerationRequest(
            prompt="A beautiful flower garden in the center",
            image=original_image,
            mask_image=mask_image,
            num_inference_steps=40,
            guidance_scale=7.5
        )
        
        self.monitor.start_monitoring(inpaint_pipeline_key)
        inpainted_images = await self.manager.generate_image(inpaint_pipeline_key, inpaint_request)
        metrics = self.monitor.end_monitoring(inpaint_pipeline_key)
        
        logger.info(f"Inpainting completed in {metrics['duration']:.2f}s")
        
        # Example 2: Background replacement
        background_mask = self._create_background_mask(original_image)
        background_request = GenerationRequest(
            prompt="A futuristic city skyline as background",
            image=original_image,
            mask_image=background_mask,
            num_inference_steps=35,
            guidance_scale=7.0
        )
        
        self.monitor.start_monitoring(inpaint_pipeline_key)
        background_images = await self.manager.generate_image(inpaint_pipeline_key, background_request)
        metrics = self.monitor.end_monitoring(inpaint_pipeline_key)
        
        logger.info(f"Background replacement completed in {metrics['duration']:.2f}s")
        
        return {
            "object_replacement": inpainted_images,
            "background_replacement": background_images
        }
    
    async def run_controlnet_examples(self) -> Any:
        """Run ControlNet pipeline examples."""
        logger.info("Running ControlNet examples...")
        
        # Load ControlNet pipeline with Canny edge detection
        controlnet_pipeline_key = await self.manager.load_controlnet_pipeline(
            model_name="runwayml/stable-diffusion-v1-5",
            controlnet_model_name="lllyasviel/control_v11p_sd15_canny"
        )
        
        # Create control image (Canny edges)
        control_image = self._create_canny_control_image()
        
        # Example 1: Edge-guided generation
        edge_request = GenerationRequest(
            prompt="A beautiful landscape with mountains and lake",
            control_image=control_image,
            controlnet_conditioning_scale=1.0,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            num_inference_steps=30,
            guidance_scale=7.5
        )
        
        self.monitor.start_monitoring(controlnet_pipeline_key)
        edge_images = await self.manager.generate_image(controlnet_pipeline_key, edge_request)
        metrics = self.monitor.end_monitoring(controlnet_pipeline_key)
        
        logger.info(f"Edge-guided generation completed in {metrics['duration']:.2f}s")
        
        # Example 2: Different conditioning scales
        scale_request = GenerationRequest(
            prompt="A futuristic cityscape",
            control_image=control_image,
            controlnet_conditioning_scale=0.5,  # Less strict control
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            num_inference_steps=25,
            guidance_scale=7.0
        )
        
        self.monitor.start_monitoring(controlnet_pipeline_key)
        scale_images = await self.manager.generate_image(controlnet_pipeline_key, scale_request)
        metrics = self.monitor.end_monitoring(controlnet_pipeline_key)
        
        logger.info(f"Scale-controlled generation completed in {metrics['duration']:.2f}s")
        
        return {
            "edge_guided": edge_images,
            "scale_controlled": scale_images
        }
    
    async def run_custom_pipeline_examples(self) -> Any:
        """Run custom pipeline examples."""
        logger.info("Running custom pipeline examples...")
        
        # Example 1: Text-to-video pipeline
        try:
            video_pipeline = await CustomPipelineFactory.create_text_to_video_pipeline()
            logger.info("Text-to-video pipeline loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load text-to-video pipeline: {e}")
            video_pipeline = None
        
        # Example 2: Upscale pipeline
        try:
            upscale_pipeline = await CustomPipelineFactory.create_upscale_pipeline()
            logger.info("Upscale pipeline loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load upscale pipeline: {e}")
            upscale_pipeline = None
        
        return {
            "video_pipeline": video_pipeline,
            "upscale_pipeline": upscale_pipeline
        }
    
    async def run_performance_comparison(self) -> Any:
        """Run performance comparison between different pipelines."""
        logger.info("Running performance comparison...")
        
        # Load all pipeline types
        pipelines = {}
        
        # Stable Diffusion
        pipelines["stable_diffusion"] = await self.manager.load_stable_diffusion_pipeline()
        
        # Stable Diffusion XL
        pipelines["stable_diffusion_xl"] = await self.manager.load_stable_diffusion_xl_pipeline()
        
        # Img2Img
        pipelines["img2img"] = await self.manager.load_img2img_pipeline()
        
        # Test request
        test_request = GenerationRequest(
            prompt="A beautiful landscape",
            negative_prompt="blurry, low quality",
            num_inference_steps=30,
            guidance_scale=7.5,
            height=512,
            width=512
        )
        
        # Performance comparison
        results = {}
        for name, pipeline_key in pipelines.items():
            logger.info(f"Testing {name}...")
            
            self.monitor.start_monitoring(pipeline_key)
            try:
                images = await self.manager.generate_image(pipeline_key, test_request)
                metrics = self.monitor.end_monitoring(pipeline_key)
                
                results[name] = {
                    "success": True,
                    "duration": metrics["duration"],
                    "memory_usage": metrics["peak_memory"],
                    "image_count": len(images)
                }
                
                logger.info(f"{name}: {metrics['duration']:.2f}s, {metrics['peak_memory'] / 1024 / 1024:.1f} MB")
                
            except Exception as e:
                metrics = self.monitor.end_monitoring(pipeline_key)
                results[name] = {
                    "success": False,
                    "error": str(e),
                    "duration": metrics.get("duration", 0),
                    "memory_usage": metrics.get("peak_memory", 0)
                }
                logger.error(f"{name} failed: {e}")
        
        return results
    
    def _create_sample_image(self) -> Image.Image:
        """Create a sample image for testing."""
        # Create a simple gradient image
        width, height = 512, 512
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        
        for y in range(height):
            for x in range(width):
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = int(255 * (x + y) / (width + height))
                draw.point((x, y), fill=(r, g, b))
        
        return image
    
    def _create_sample_mask(self, image: Image.Image) -> Image.Image:
        """Create a sample mask for inpainting."""
        width, height = image.size
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Create a circular mask in the center
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4
        
        draw.ellipse(
            (center_x - radius, center_y - radius, 
             center_x + radius, center_y + radius),
            fill=255
        )
        
        return mask
    
    def _create_background_mask(self, image: Image.Image) -> Image.Image:
        """Create a background mask for inpainting."""
        width, height = image.size
        mask = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(mask)
        
        # Create a rectangular mask for the center object
        center_x, center_y = width // 2, height // 2
        rect_width, rect_height = width // 3, height // 3
        
        draw.rectangle(
            (center_x - rect_width // 2, center_y - rect_height // 2,
             center_x + rect_width // 2, center_y + rect_height // 2),
            fill=0
        )
        
        return mask
    
    def _create_canny_control_image(self) -> Image.Image:
        """Create a Canny edge control image."""
        # Create a simple geometric pattern
        width, height = 512, 512
        image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Draw some geometric shapes
        draw.rectangle((100, 100, 200, 200), outline=(0, 0, 0), width=3)
        draw.ellipse((300, 100, 400, 200), outline=(0, 0, 0), width=3)
        draw.polygon([(250, 300), (350, 250), (450, 350)], outline=(0, 0, 0), width=3)
        
        # Apply edge detection effect
        image = image.filter(ImageFilter.FIND_EDGES)
        
        return image
    
    async def cleanup(self) -> Any:
        """Clean up resources."""
        self.manager.cleanup()


async def main():
    """Run all pipeline examples."""
    examples = PipelineExamples()
    
    try:
        logger.info("Starting pipeline examples...")
        
        # Run all examples
        results = {}
        
        # Stable Diffusion examples
        results["stable_diffusion"] = await examples.run_stable_diffusion_examples()
        
        # Stable Diffusion XL examples
        results["stable_diffusion_xl"] = await examples.run_stable_diffusion_xl_examples()
        
        # Img2Img examples
        results["img2img"] = await examples.run_img2img_examples()
        
        # Inpaint examples
        results["inpaint"] = await examples.run_inpaint_examples()
        
        # ControlNet examples
        results["controlnet"] = await examples.run_controlnet_examples()
        
        # Custom pipeline examples
        results["custom"] = await examples.run_custom_pipeline_examples()
        
        # Performance comparison
        results["performance"] = await examples.run_performance_comparison()
        
        logger.info("All examples completed successfully!")
        
        # Print performance summary
        if "performance" in results:
            logger.info("\nPerformance Summary:")
            for pipeline, metrics in results["performance"].items():
                if metrics["success"]:
                    logger.info(f"{pipeline}: {metrics['duration']:.2f}s, {metrics['memory_usage'] / 1024 / 1024:.1f} MB")
                else:
                    logger.info(f"{pipeline}: FAILED - {metrics['error']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise
    finally:
        await examples.cleanup()


match __name__:
    case "__main__":
    asyncio.run(main()) 