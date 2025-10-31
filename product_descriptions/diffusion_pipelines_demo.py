from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import cv2
import psutil
import gc
from diffusion_pipelines import (
from typing import Any, List, Dict, Optional
"""
Advanced Diffusion Pipelines Demo
================================

This demo showcases the comprehensive diffusion pipelines implementation,
including:

1. StableDiffusionPipeline - Standard text-to-image generation
2. StableDiffusionXLPipeline - High-quality XL generation
3. StableDiffusionImg2ImgPipeline - Image-to-image transformation
4. StableDiffusionInpaintPipeline - Inpainting and editing
5. StableDiffusionControlNetPipeline - ControlNet for precise control
6. Performance comparisons and optimizations

Features:
- Practical examples for each pipeline type
- Performance benchmarking
- Memory usage monitoring
- Quality comparisons
- Error handling demonstrations
- Security considerations

Author: AI Assistant
License: MIT
"""



# Import our diffusion pipelines
    PipelineType, SchedulerType, PipelineConfig, GenerationConfig,
    DiffusionPipelineFactory, AdvancedPipelineManager,
    create_pipeline, create_pipeline_manager,
    get_available_pipeline_types, get_available_scheduler_types
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiffusionPipelinesDemo:
    """
    Comprehensive demo for diffusion pipelines.
    
    This class provides demonstrations of:
    - Different pipeline types
    - Performance comparisons
    - Memory management
    - Quality assessments
    - Practical examples
    """
    
    def __init__(self, output_dir: str = "diffusion_pipelines_demo_output"):
        
    """__init__ function."""
self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create pipeline manager
        self.pipeline_manager = create_pipeline_manager()
        
        # Default configurations
        self.default_pipeline_config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION,
            model_id="runwayml/stable-diffusion-v1-5",
            scheduler_type=SchedulerType.DDIM,
            num_inference_steps=20,  # Reduced for faster demo
            guidance_scale=7.5,
            height=512,
            width=512
        )
        
        self.default_generation_config = GenerationConfig(
            prompt="A beautiful landscape with mountains and a lake, high quality, detailed",
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512
        )
        
        logger.info("DiffusionPipelinesDemo initialized")
    
    def create_test_image(self, size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """Create a test image for img2img and inpainting demos."""
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple scene
        # Sky
        draw.rectangle([0, 0, size[0], size[1]//2], fill='lightblue')
        
        # Sun
        sun_center = (size[0]//4, size[1]//4)
        sun_radius = 30
        draw.ellipse([
            sun_center[0] - sun_radius, sun_center[1] - sun_radius,
            sun_center[0] + sun_radius, sun_center[1] + sun_radius
        ], fill='yellow')
        
        # Mountains
        mountain_points = [
            (0, size[1]//2),
            (size[0]//3, size[1]//3),
            (size[0]//2, size[1]//2),
            (2*size[0]//3, size[1]//4),
            (size[0], size[1]//2),
            (size[0], size[1]),
            (0, size[1])
        ]
        draw.polygon(mountain_points, fill='gray')
        
        # Lake
        lake_points = [
            (0, size[1]//2),
            (size[0], size[1]//2),
            (size[0], size[1]),
            (0, size[1])
        ]
        draw.polygon(lake_points, fill='blue')
        
        return img
    
    def create_test_mask(self, size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """Create a test mask for inpainting demo."""
        mask = Image.new('L', size, color=0)
        draw = ImageDraw.Draw(mask)
        
        # Create a circular mask in the center
        center = (size[0]//2, size[1]//2)
        radius = 100
        draw.ellipse([
            center[0] - radius, center[1] - radius,
            center[0] + radius, center[1] + radius
        ], fill=255)
        
        return mask
    
    def demo_stable_diffusion_pipeline(self) -> Dict[str, Any]:
        """Demonstrate StableDiffusionPipeline."""
        logger.info("Demonstrating StableDiffusionPipeline...")
        
        # Create pipeline
        config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION,
            model_id="runwayml/stable-diffusion-v1-5",
            scheduler_type=SchedulerType.DDIM,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        pipeline = DiffusionPipelineFactory.create(config)
        pipeline.load_pipeline()
        
        # Generate images
        generation_config = GenerationConfig(
            prompt="A majestic dragon flying over a medieval castle, epic fantasy art",
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512
        )
        
        start_time = time.time()
        result = pipeline.generate(generation_config)
        end_time = time.time()
        
        # Save results
        for i, image in enumerate(result.images):
            image.save(self.output_dir / f"stable_diffusion_result_{i}.png")
        
        # Cleanup
        pipeline.cleanup()
        
        return {
            "pipeline_type": "stable_diffusion",
            "processing_time": result.processing_time,
            "memory_usage": result.memory_usage,
            "num_images": len(result.images),
            "metadata": result.metadata
        }
    
    def demo_stable_diffusion_xl_pipeline(self) -> Dict[str, Any]:
        """Demonstrate StableDiffusionXLPipeline."""
        logger.info("Demonstrating StableDiffusionXLPipeline...")
        
        # Create pipeline
        config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION_XL,
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            scheduler_type=SchedulerType.DPM_SOLVER,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        pipeline = DiffusionPipelineFactory.create(config)
        pipeline.load_pipeline()
        
        # Generate images
        generation_config = GenerationConfig(
            prompt="A futuristic cityscape with flying cars and neon lights, cinematic lighting",
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512
        )
        
        start_time = time.time()
        result = pipeline.generate(generation_config)
        end_time = time.time()
        
        # Save results
        for i, image in enumerate(result.images):
            image.save(self.output_dir / f"stable_diffusion_xl_result_{i}.png")
        
        # Cleanup
        pipeline.cleanup()
        
        return {
            "pipeline_type": "stable_diffusion_xl",
            "processing_time": result.processing_time,
            "memory_usage": result.memory_usage,
            "num_images": len(result.images),
            "metadata": result.metadata
        }
    
    def demo_img2img_pipeline(self) -> Dict[str, Any]:
        """Demonstrate StableDiffusionImg2ImgPipeline."""
        logger.info("Demonstrating StableDiffusionImg2ImgPipeline...")
        
        # Create test image
        test_image = self.create_test_image()
        test_image.save(self.output_dir / "img2img_input.png")
        
        # Create pipeline
        config = PipelineConfig(
            pipeline_type=PipelineType.IMG2IMG,
            model_id="runwayml/stable-diffusion-v1-5",
            scheduler_type=SchedulerType.DDIM,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        pipeline = DiffusionPipelineFactory.create(config)
        pipeline.load_pipeline()
        
        # Generate images
        generation_config = GenerationConfig(
            prompt="Transform this into a magical fantasy landscape with unicorns and rainbows",
            negative_prompt="blurry, low quality, distorted",
            image=test_image,
            strength=0.8,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        start_time = time.time()
        result = pipeline.generate(generation_config)
        end_time = time.time()
        
        # Save results
        for i, image in enumerate(result.images):
            image.save(self.output_dir / f"img2img_result_{i}.png")
        
        # Cleanup
        pipeline.cleanup()
        
        return {
            "pipeline_type": "img2img",
            "processing_time": result.processing_time,
            "memory_usage": result.memory_usage,
            "num_images": len(result.images),
            "metadata": result.metadata
        }
    
    def demo_inpaint_pipeline(self) -> Dict[str, Any]:
        """Demonstrate StableDiffusionInpaintPipeline."""
        logger.info("Demonstrating StableDiffusionInpaintPipeline...")
        
        # Create test image and mask
        test_image = self.create_test_image()
        test_mask = self.create_test_mask()
        
        test_image.save(self.output_dir / "inpaint_input.png")
        test_mask.save(self.output_dir / "inpaint_mask.png")
        
        # Create pipeline
        config = PipelineConfig(
            pipeline_type=PipelineType.INPAINT,
            model_id="runwayml/stable-diffusion-inpainting",
            scheduler_type=SchedulerType.DDIM,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        pipeline = DiffusionPipelineFactory.create(config)
        pipeline.load_pipeline()
        
        # Generate images
        generation_config = GenerationConfig(
            prompt="A beautiful castle in the center of the image",
            negative_prompt="blurry, low quality, distorted",
            image=test_image,
            mask_image=test_mask,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        start_time = time.time()
        result = pipeline.generate(generation_config)
        end_time = time.time()
        
        # Save results
        for i, image in enumerate(result.images):
            image.save(self.output_dir / f"inpaint_result_{i}.png")
        
        # Cleanup
        pipeline.cleanup()
        
        return {
            "pipeline_type": "inpaint",
            "processing_time": result.processing_time,
            "memory_usage": result.memory_usage,
            "num_images": len(result.images),
            "metadata": result.metadata
        }
    
    def demo_controlnet_pipeline(self) -> Dict[str, Any]:
        """Demonstrate StableDiffusionControlNetPipeline."""
        logger.info("Demonstrating StableDiffusionControlNetPipeline...")
        
        # Create a simple edge image for ControlNet
        edge_image = Image.new('RGB', (512, 512), color='black')
        draw = ImageDraw.Draw(edge_image)
        
        # Draw some edges
        draw.line([(100, 100), (400, 100)], fill='white', width=3)
        draw.line([(100, 100), (100, 400)], fill='white', width=3)
        draw.line([(400, 100), (400, 400)], fill='white', width=3)
        draw.line([(100, 400), (400, 400)], fill='white', width=3)
        
        edge_image.save(self.output_dir / "controlnet_input.png")
        
        # Create pipeline
        config = PipelineConfig(
            pipeline_type=PipelineType.CONTROLNET,
            model_id="runwayml/stable-diffusion-v1-5",
            scheduler_type=SchedulerType.DDIM,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        pipeline = DiffusionPipelineFactory.create(
            config, 
            controlnet_model_id="lllyasviel/sd-controlnet-canny"
        )
        pipeline.load_pipeline()
        
        # Generate images
        generation_config = GenerationConfig(
            prompt="A beautiful modern building with clean architecture",
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512,
            controlnet_conditioning_scale=1.0
        )
        
        start_time = time.time()
        result = pipeline.generate(generation_config)
        end_time = time.time()
        
        # Save results
        for i, image in enumerate(result.images):
            image.save(self.output_dir / f"controlnet_result_{i}.png")
        
        # Cleanup
        pipeline.cleanup()
        
        return {
            "pipeline_type": "controlnet",
            "processing_time": result.processing_time,
            "memory_usage": result.memory_usage,
            "num_images": len(result.images),
            "metadata": result.metadata
        }
    
    def compare_pipeline_performance(self) -> Dict[str, Dict]:
        """Compare performance of different pipeline types."""
        logger.info("Comparing pipeline performance...")
        
        pipeline_configs = [
            {
                "name": "stable_diffusion",
                "config": PipelineConfig(
                    pipeline_type=PipelineType.STABLE_DIFFUSION,
                    model_id="runwayml/stable-diffusion-v1-5",
                    scheduler_type=SchedulerType.DDIM,
                    num_inference_steps=20
                )
            },
            {
                "name": "stable_diffusion_xl",
                "config": PipelineConfig(
                    pipeline_type=PipelineType.STABLE_DIFFUSION_XL,
                    model_id="stabilityai/stable-diffusion-xl-base-1.0",
                    scheduler_type=SchedulerType.DPM_SOLVER,
                    num_inference_steps=20
                )
            },
            {
                "name": "img2img",
                "config": PipelineConfig(
                    pipeline_type=PipelineType.IMG2IMG,
                    model_id="runwayml/stable-diffusion-v1-5",
                    scheduler_type=SchedulerType.DDIM,
                    num_inference_steps=20
                )
            }
        ]
        
        results = {}
        
        for pipeline_info in pipeline_configs:
            name = pipeline_info["name"]
            config = pipeline_info["config"]
            
            logger.info(f"Testing {name} pipeline...")
            
            try:
                pipeline = DiffusionPipelineFactory.create(config)
                pipeline.load_pipeline()
                
                # Test generation
                generation_config = GenerationConfig(
                    prompt="A beautiful landscape",
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                )
                
                if name == "img2img":
                    test_image = self.create_test_image()
                    generation_config.image = test_image
                    generation_config.strength = 0.8
                
                result = pipeline.generate(generation_config)
                
                results[name] = {
                    "processing_time": result.processing_time,
                    "memory_usage": result.memory_usage,
                    "num_images": len(result.images),
                    "success": True
                }
                
                # Save sample image
                if result.images:
                    result.images[0].save(self.output_dir / f"performance_test_{name}.png")
                
                pipeline.cleanup()
                
            except Exception as e:
                logger.error(f"Failed to test {name} pipeline: {e}")
                results[name] = {
                    "processing_time": 0.0,
                    "memory_usage": 0.0,
                    "num_images": 0,
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def demo_pipeline_manager(self) -> Any:
        """Demonstrate the advanced pipeline manager."""
        logger.info("Demonstrating AdvancedPipelineManager...")
        
        # Add different pipelines
        pipeline_configs = [
            ("sd_v1_5", PipelineConfig(
                pipeline_type=PipelineType.STABLE_DIFFUSION,
                model_id="runwayml/stable-diffusion-v1-5",
                scheduler_type=SchedulerType.DDIM
            )),
            ("sd_xl", PipelineConfig(
                pipeline_type=PipelineType.STABLE_DIFFUSION_XL,
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                scheduler_type=SchedulerType.DPM_SOLVER
            )),
            ("img2img", PipelineConfig(
                pipeline_type=PipelineType.IMG2IMG,
                model_id="runwayml/stable-diffusion-v1-5",
                scheduler_type=SchedulerType.DDIM
            ))
        ]
        
        for name, config in pipeline_configs:
            try:
                self.pipeline_manager.add_pipeline(name, config)
                logger.info(f"Added pipeline: {name}")
            except Exception as e:
                logger.error(f"Failed to add pipeline {name}: {e}")
        
        # Get pipeline info
        info = self.pipeline_manager.get_pipeline_info()
        logger.info(f"Pipeline info: {info}")
        
        # Test generation with different pipelines
        generation_config = GenerationConfig(
            prompt="A serene mountain landscape at sunset",
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512
        )
        
        results = {}
        
        for name in info.keys():
            try:
                if name == "img2img":
                    # Add image for img2img
                    test_image = self.create_test_image()
                    generation_config.image = test_image
                    generation_config.strength = 0.8
                
                result = self.pipeline_manager.generate(generation_config, pipeline_name=name)
                
                # Save result
                if result.images:
                    result.images[0].save(self.output_dir / f"manager_test_{name}.png")
                
                results[name] = {
                    "processing_time": result.processing_time,
                    "memory_usage": result.memory_usage,
                    "num_images": len(result.images)
                }
                
            except Exception as e:
                logger.error(f"Failed to generate with {name}: {e}")
                results[name] = {"error": str(e)}
        
        # Cleanup
        self.pipeline_manager.cleanup_all()
        
        return results
    
    def visualize_performance_comparison(self, performance_data: Dict[str, Dict]):
        """Visualize performance comparison results."""
        logger.info("Creating performance comparison visualization...")
        
        # Prepare data
        pipeline_names = []
        processing_times = []
        memory_usage = []
        
        for name, data in performance_data.items():
            if data.get("success", True):
                pipeline_names.append(name)
                processing_times.append(data["processing_time"])
                memory_usage.append(data["memory_usage"])
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Processing time comparison
        bars1 = ax1.bar(pipeline_names, processing_times, color='skyblue', alpha=0.7)
        ax1.set_title("Processing Time Comparison")
        ax1.set_ylabel("Time (seconds)")
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time_val in zip(bars1, processing_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{time_val:.2f}s', ha='center', va='bottom')
        
        # Memory usage comparison
        bars2 = ax2.bar(pipeline_names, memory_usage, color='lightcoral', alpha=0.7)
        ax2.set_title("Memory Usage Comparison")
        ax2.set_ylabel("Memory (MB)")
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mem_val in zip(bars2, memory_usage):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{mem_val:.1f}MB', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance comparison saved to {self.output_dir / 'performance_comparison.png'}")
    
    def run_comprehensive_demo(self) -> Any:
        """Run the comprehensive demo."""
        logger.info("Starting comprehensive diffusion pipelines demo...")
        
        results = {}
        
        # 1. Demo individual pipelines
        logger.info("\n" + "="*50)
        logger.info("1. DEMONSTRATING INDIVIDUAL PIPELINES")
        logger.info("="*50)
        
        try:
            results["stable_diffusion"] = self.demo_stable_diffusion_pipeline()
        except Exception as e:
            logger.error(f"Stable Diffusion demo failed: {e}")
            results["stable_diffusion"] = {"error": str(e)}
        
        try:
            results["stable_diffusion_xl"] = self.demo_stable_diffusion_xl_pipeline()
        except Exception as e:
            logger.error(f"Stable Diffusion XL demo failed: {e}")
            results["stable_diffusion_xl"] = {"error": str(e)}
        
        try:
            results["img2img"] = self.demo_img2img_pipeline()
        except Exception as e:
            logger.error(f"Img2Img demo failed: {e}")
            results["img2img"] = {"error": str(e)}
        
        try:
            results["inpaint"] = self.demo_inpaint_pipeline()
        except Exception as e:
            logger.error(f"Inpaint demo failed: {e}")
            results["inpaint"] = {"error": str(e)}
        
        try:
            results["controlnet"] = self.demo_controlnet_pipeline()
        except Exception as e:
            logger.error(f"ControlNet demo failed: {e}")
            results["controlnet"] = {"error": str(e)}
        
        # 2. Performance comparison
        logger.info("\n" + "="*50)
        logger.info("2. PERFORMANCE COMPARISON")
        logger.info("="*50)
        
        performance_results = self.compare_pipeline_performance()
        results["performance_comparison"] = performance_results
        
        # 3. Pipeline manager demo
        logger.info("\n" + "="*50)
        logger.info("3. PIPELINE MANAGER DEMO")
        logger.info("="*50)
        
        manager_results = self.demo_pipeline_manager()
        results["pipeline_manager"] = manager_results
        
        # 4. Visualize results
        logger.info("\n" + "="*50)
        logger.info("4. CREATING VISUALIZATIONS")
        logger.info("="*50)
        
        self.visualize_performance_comparison(performance_results)
        
        # 5. Generate summary report
        self._generate_summary_report(results)
        
        logger.info("\n" + "="*50)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info(f"Output saved to: {self.output_dir}")
        logger.info("="*50)
        
        return results
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate a comprehensive summary report."""
        report_path = self.output_dir / "demo_summary_report.txt"
        
        with open(report_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("DIFFUSION PIPELINES DEMO SUMMARY\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("=" * 40 + "\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Individual pipeline results
            f.write("1. INDIVIDUAL PIPELINE RESULTS\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("-" * 30 + "\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            for pipeline_name, result in results.items():
                if pipeline_name in ["performance_comparison", "pipeline_manager"]:
                    continue
                    
                f.write(f"{pipeline_name.upper()}:\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if "error" in result:
                    f.write(f"  - Status: FAILED\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    f.write(f"  - Error: {result['error']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                else:
                    f.write(f"  - Status: SUCCESS\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    f.write(f"  - Processing time: {result['processing_time']:.3f}s\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    f.write(f"  - Memory usage: {result['memory_usage']:.1f}MB\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    f.write(f"  - Images generated: {result['num_images']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write("\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Performance comparison
            f.write("2. PERFORMANCE COMPARISON\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("-" * 25 + "\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            perf_results = results.get("performance_comparison", {})
            for pipeline_name, perf_data in perf_results.items():
                f.write(f"{pipeline_name.upper()}:\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if perf_data.get("success", False):
                    f.write(f"  - Processing time: {perf_data['processing_time']:.3f}s\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    f.write(f"  - Memory usage: {perf_data['memory_usage']:.1f}MB\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    f.write(f"  - Images generated: {perf_data['num_images']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                else:
                    f.write(f"  - Status: FAILED\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    f.write(f"  - Error: {perf_data.get('error', 'Unknown error')}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write("\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Pipeline manager results
            f.write("3. PIPELINE MANAGER RESULTS\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("-" * 28 + "\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            manager_results = results.get("pipeline_manager", {})
            for pipeline_name, manager_data in manager_results.items():
                f.write(f"{pipeline_name.upper()}:\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if "error" in manager_data:
                    f.write(f"  - Status: FAILED\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    f.write(f"  - Error: {manager_data['error']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                else:
                    f.write(f"  - Processing time: {manager_data['processing_time']:.3f}s\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    f.write(f"  - Memory usage: {manager_data['memory_usage']:.1f}MB\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    f.write(f"  - Images generated: {manager_data['num_images']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write("\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        logger.info(f"Summary report saved to: {report_path}")


async def main():
    """Main demo function."""
    logger.info("Starting Diffusion Pipelines Demo")
    
    # Create demo instance
    demo = DiffusionPipelinesDemo()
    
    # Run comprehensive demo
    results = demo.run_comprehensive_demo()
    
    logger.info("Demo completed successfully!")
    return results


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 