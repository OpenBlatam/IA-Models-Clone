from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import (
from diffusers.utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from PIL import Image
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
from diffusion_models import DiffusionModelManager, DiffusionConfig, GenerationConfig, CustomDiffusionPipeline
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Diffusion Models Examples - Production Workflows
===============================================

Comprehensive examples demonstrating diffusion model usage with Diffusers library.
Includes: Stable Diffusion, DDPM, DDIM, custom training, schedulers, and optimization.
"""

    StableDiffusionPipeline, DDIMPipeline, DDPMPipeline,
    UNet2DConditionModel, AutoencoderKL, DDIMScheduler,
    DDPMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline, ControlNetModel
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiffusionExamples:
    """Comprehensive diffusion model examples."""
    
    def __init__(self) -> Any:
        self.config = DiffusionConfig()
        self.manager = DiffusionModelManager(self.config)
    
    async def example_stable_diffusion_basic(self) -> Any:
        """Basic Stable Diffusion image generation."""
        logger.info("Running basic Stable Diffusion example")
        
        # Load model
        model_key = await self.manager.load_stable_diffusion()
        
        # Generate image
        config = GenerationConfig(
            prompt="A beautiful sunset over mountains, digital art style",
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=30,
            guidance_scale=7.5,
            height=512,
            width=512,
            seed=42
        )
        
        images = await self.manager.generate_image(model_key, config)
        
        if images:
            images[0].save("stable_diffusion_basic.png")
            logger.info("Generated: stable_diffusion_basic.png")
        
        return images
    
    async def example_stable_diffusion_batch(self) -> Any:
        """Batch image generation with multiple prompts."""
        logger.info("Running batch Stable Diffusion example")
        
        model_key = await self.manager.load_stable_diffusion()
        
        configs = [
            GenerationConfig(
                prompt="A majestic dragon flying over a medieval castle",
                num_inference_steps=25,
                guidance_scale=8.0,
                seed=123
            ),
            GenerationConfig(
                prompt="A futuristic cityscape with flying cars and neon lights",
                num_inference_steps=25,
                guidance_scale=8.0,
                seed=456
            ),
            GenerationConfig(
                prompt="A serene forest with ancient trees and magical creatures",
                num_inference_steps=25,
                guidance_scale=8.0,
                seed=789
            )
        ]
        
        all_images = await self.manager.batch_generate(model_key, configs)
        
        for i, images in enumerate(all_images):
            if images:
                images[0].save(f"batch_generation_{i}.png")
                logger.info(f"Generated: batch_generation_{i}.png")
        
        return all_images
    
    async def example_ddim_generation(self) -> Any:
        """DDIM model generation example."""
        logger.info("Running DDIM generation example")
        
        model_key = await self.manager.load_ddim()
        
        config = GenerationConfig(
            prompt="A simple geometric pattern",
            num_inference_steps=50,
            guidance_scale=1.0,
            height=256,
            width=256,
            seed=999
        )
        
        images = await self.manager.generate_image(model_key, config)
        
        if images:
            images[0].save("ddim_generation.png")
            logger.info("Generated: ddim_generation.png")
        
        return images
    
    async def example_ddpm_generation(self) -> Any:
        """DDPM model generation example."""
        logger.info("Running DDPM generation example")
        
        model_key = await self.manager.load_ddpm()
        
        config = GenerationConfig(
            prompt="A simple pattern",
            num_inference_steps=1000,
            guidance_scale=1.0,
            height=32,
            width=32,
            seed=111
        )
        
        images = await self.manager.generate_image(model_key, config)
        
        if images:
            images[0].save("ddpm_generation.png")
            logger.info("Generated: ddpm_generation.png")
        
        return images
    
    async def example_custom_pipeline(self) -> Any:
        """Custom diffusion pipeline example."""
        logger.info("Running custom pipeline example")
        
        pipeline = CustomDiffusionPipeline(self.config)
        await pipeline.load_components("runwayml/stable-diffusion-v1-5")
        pipeline.set_scheduler("ddim")
        
        # Generate latents
        latents = await pipeline.generate_latents(
            prompt="A beautiful landscape painting",
            height=512,
            width=512,
            num_inference_steps=30,
            guidance_scale=7.5,
            seed=555
        )
        
        # Decode to image
        image = await pipeline.decode_latents(latents)
        image.save("custom_pipeline.png")
        logger.info("Generated: custom_pipeline.png")
        
        return image
    
    async def example_img2img_generation(self) -> Any:
        """Image-to-image generation example."""
        logger.info("Running img2img generation example")
        
        # Create a simple input image
        input_image = Image.new('RGB', (512, 512), color='red')
        input_image.save("input_image.png")
        
        # Load img2img pipeline
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
        
        # Generate
        result = pipeline(
            prompt="A beautiful sunset landscape",
            image=input_image,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=30
        )
        
        if result.images:
            result.images[0].save("img2img_result.png")
            logger.info("Generated: img2img_result.png")
        
        return result.images
    
    async def example_inpainting(self) -> Any:
        """Inpainting example."""
        logger.info("Running inpainting example")
        
        # Create a simple image and mask
        image = Image.new('RGB', (512, 512), color='blue')
        mask = Image.new('L', (512, 512), color=0)
        
        # Create a mask in the center
        mask_array = np.array(mask)
        mask_array[200:300, 200:300] = 255
        mask = Image.fromarray(mask_array)
        
        image.save("inpaint_original.png")
        mask.save("inpaint_mask.png")
        
        # Load inpainting pipeline
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
        
        # Inpaint
        result = pipeline(
            prompt="A beautiful flower",
            image=image,
            mask_image=mask,
            guidance_scale=7.5,
            num_inference_steps=30
        )
        
        if result.images:
            result.images[0].save("inpaint_result.png")
            logger.info("Generated: inpaint_result.png")
        
        return result.images
    
    async def example_scheduler_comparison(self) -> Any:
        """Compare different schedulers."""
        logger.info("Running scheduler comparison example")
        
        model_key = await self.manager.load_stable_diffusion()
        pipeline = self.manager.models[model_key]
        
        schedulers = {
            "Euler": EulerDiscreteScheduler.from_config(pipeline.scheduler.config),
            "DDIM": DDIMScheduler.from_config(pipeline.scheduler.config),
            "DPM-Solver": DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        }
        
        results = {}
        
        for name, scheduler in schedulers.items():
            pipeline.scheduler = scheduler
            
            config = GenerationConfig(
                prompt="A serene mountain landscape",
                num_inference_steps=30,
                guidance_scale=7.5,
                seed=42
            )
            
            start_time = time.time()
            images = await self.manager.generate_image(model_key, config)
            generation_time = time.time() - start_time
            
            if images:
                images[0].save(f"scheduler_{name.lower()}.png")
                results[name] = {
                    "image": images[0],
                    "time": generation_time
                }
                logger.info(f"Generated: scheduler_{name.lower()}.png (Time: {generation_time:.2f}s)")
        
        return results
    
    async def example_memory_optimization(self) -> Any:
        """Memory optimization example."""
        logger.info("Running memory optimization example")
        
        # Load multiple models
        sd_key = await self.manager.load_stable_diffusion()
        ddim_key = await self.manager.load_ddim()
        
        # Generate with first model
        config = GenerationConfig(
            prompt="A beautiful artwork",
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        images1 = await self.manager.generate_image(sd_key, config)
        
        # Optimize memory
        self.manager.optimize_memory(sd_key)
        
        # Generate with second model
        images2 = await self.manager.generate_image(ddim_key, config)
        
        if images1:
            images1[0].save("memory_opt_sd.png")
        if images2:
            images2[0].save("memory_opt_ddim.png")
        
        logger.info("Generated: memory_opt_sd.png, memory_opt_ddim.png")
        
        return images1, images2
    
    async def example_control_net(self) -> Any:
        """ControlNet example (if available)."""
        logger.info("Running ControlNet example")
        
        try:
            # Load ControlNet
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Load pipeline
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if torch.cuda.is_available():
                pipeline = pipeline.to("cuda")
            
            # Create a simple control image (edge detection would be used in practice)
            control_image = Image.new('L', (512, 512), color=128)
            control_image.save("control_image.png")
            
            # Generate
            result = pipeline(
                prompt="A beautiful landscape",
                image=control_image,
                guidance_scale=7.5,
                num_inference_steps=30
            )
            
            if result.images:
                result.images[0].save("controlnet_result.png")
                logger.info("Generated: controlnet_result.png")
            
            return result.images
            
        except Exception as e:
            logger.warning(f"ControlNet not available: {e}")
            return None
    
    async def example_performance_benchmark(self) -> Any:
        """Performance benchmarking example."""
        logger.info("Running performance benchmark")
        
        model_key = await self.manager.load_stable_diffusion()
        
        configs = [
            GenerationConfig(
                prompt="A beautiful artwork",
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=i
            ) for i in range(5)
        ]
        
        # Single generation timing
        start_time = time.time()
        single_images = await self.manager.generate_image(model_key, configs[0])
        single_time = time.time() - start_time
        
        # Batch generation timing
        start_time = time.time()
        batch_images = await self.manager.batch_generate(model_key, configs)
        batch_time = time.time() - start_time
        
        logger.info(f"Single generation: {single_time:.2f}s")
        logger.info(f"Batch generation: {batch_time:.2f}s")
        logger.info(f"Speedup: {single_time * len(configs) / batch_time:.2f}x")
        
        return {
            "single_time": single_time,
            "batch_time": batch_time,
            "speedup": single_time * len(configs) / batch_time
        }
    
    async def run_all_examples(self) -> Any:
        """Run all examples."""
        logger.info("Starting all diffusion examples")
        
        results = {}
        
        try:
            results["basic"] = await self.example_stable_diffusion_basic()
        except Exception as e:
            logger.error(f"Basic example failed: {e}")
        
        try:
            results["batch"] = await self.example_stable_diffusion_batch()
        except Exception as e:
            logger.error(f"Batch example failed: {e}")
        
        try:
            results["ddim"] = await self.example_ddim_generation()
        except Exception as e:
            logger.error(f"DDIM example failed: {e}")
        
        try:
            results["ddpm"] = await self.example_ddpm_generation()
        except Exception as e:
            logger.error(f"DDPM example failed: {e}")
        
        try:
            results["custom"] = await self.example_custom_pipeline()
        except Exception as e:
            logger.error(f"Custom pipeline example failed: {e}")
        
        try:
            results["img2img"] = await self.example_img2img_generation()
        except Exception as e:
            logger.error(f"Img2img example failed: {e}")
        
        try:
            results["inpainting"] = await self.example_inpainting()
        except Exception as e:
            logger.error(f"Inpainting example failed: {e}")
        
        try:
            results["schedulers"] = await self.example_scheduler_comparison()
        except Exception as e:
            logger.error(f"Scheduler comparison failed: {e}")
        
        try:
            results["memory"] = await self.example_memory_optimization()
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
        
        try:
            results["controlnet"] = await self.example_control_net()
        except Exception as e:
            logger.error(f"ControlNet example failed: {e}")
        
        try:
            results["benchmark"] = await self.example_performance_benchmark()
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
        
        logger.info("All examples completed")
        return results
    
    def cleanup(self) -> Any:
        """Cleanup resources."""
        self.manager.cleanup()


async def main():
    """Main function to run examples."""
    examples = DiffusionExamples()
    
    try:
        results = await examples.run_all_examples()
        logger.info("Examples completed successfully")
        return results
    except Exception as e:
        logger.error(f"Examples failed: {e}")
        raise
    finally:
        examples.cleanup()


match __name__:
    case "__main__":
    asyncio.run(main()) 