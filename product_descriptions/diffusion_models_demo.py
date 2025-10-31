from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import json
from diffusion_models import (
from transformers_manager import TransformersManager, ModelConfig, ModelType
        from PIL import Image, ImageDraw
        from PIL import Image, ImageDraw
        from PIL import Image, ImageDraw, ImageFilter
from typing import Any, List, Dict, Optional
"""
Diffusion Models Demo for Cybersecurity Applications
===================================================

This demo showcases the comprehensive diffusion models implementation
for cybersecurity applications, including text-to-image generation,
security visualizations, and integration with existing infrastructure.

Features Demonstrated:
- Text-to-image generation for security reports
- Security-focused prompt engineering
- Image-to-image transformation
- Inpainting for data reconstruction
- ControlNet integration
- Performance optimization
- Memory management
- Integration with transformers

Author: AI Assistant
License: MIT
"""


# Import our diffusion models manager
    DiffusionModelsManager, DiffusionConfig, GenerationConfig,
    ImageToImageConfig, InpaintingConfig, ControlNetConfig,
    DiffusionTask, SchedulerType, SecurityPromptEngine
)

# Import transformers manager for integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiffusionModelsDemo:
    """Comprehensive demo for diffusion models in cybersecurity."""
    
    def __init__(self) -> Any:
        """Initialize the demo."""
        self.diffusion_manager = DiffusionModelsManager()
        self.transformers_manager = TransformersManager()
        self.output_dir = Path("diffusion_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Demo configurations
        self.demo_configs = {
            "security_visualization": {
                "threat_types": ["malware_analysis", "network_security", "threat_hunting", "incident_response"],
                "severities": ["low", "medium", "high", "critical"],
                "styles": ["technical", "detailed", "simple"]
            },
            "performance_testing": {
                "batch_sizes": [1, 2, 4],
                "inference_steps": [10, 20, 30],
                "guidance_scales": [5.0, 7.5, 10.0]
            }
        }
    
    async def run_comprehensive_demo(self) -> Any:
        """Run the complete diffusion models demo."""
        logger.info("🚀 Starting Comprehensive Diffusion Models Demo")
        
        try:
            # 1. Basic Text-to-Image Generation
            await self.demo_text_to_image_generation()
            
            # 2. Security Visualization Generation
            await self.demo_security_visualizations()
            
            # 3. Image-to-Image Transformation
            await self.demo_image_to_image()
            
            # 4. Inpainting Demo
            await self.demo_inpainting()
            
            # 5. ControlNet Integration
            await self.demo_controlnet()
            
            # 6. Performance Testing
            await self.demo_performance_testing()
            
            # 7. Integration with Transformers
            await self.demo_transformers_integration()
            
            # 8. Memory Management
            await self.demo_memory_management()
            
            logger.info("✅ Comprehensive Diffusion Models Demo Completed Successfully")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {str(e)}")
            raise
    
    async def demo_text_to_image_generation(self) -> Any:
        """Demonstrate basic text-to-image generation."""
        logger.info("📝 Demo: Basic Text-to-Image Generation")
        
        # Load text-to-image pipeline
        config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.TEXT_TO_IMAGE,
            scheduler=SchedulerType.DPM_SOLVER
        )
        
        pipeline_key = f"{config.model_name}_{config.task.value}"
        await self.diffusion_manager.load_pipeline(config)
        
        # Test prompts
        test_prompts = [
            "cybersecurity dashboard with network monitoring, professional technical diagram, clean visualization",
            "malware analysis workflow, security investigation, digital forensics, technical diagram",
            "firewall configuration diagram, network security infrastructure, professional technical illustration"
        ]
        
        for i, prompt in enumerate(test_prompts):
            generation_config = GenerationConfig(
                prompt=prompt,
                negative_prompt="cartoon, anime, artistic, decorative, colorful, playful",
                num_inference_steps=20,
                guidance_scale=7.5,
                width=512,
                height=512,
                seed=42 + i
            )
            
            result = await self.diffusion_manager.generate_image(pipeline_key, generation_config)
            
            # Save images
            for j, image in enumerate(result.images):
                filename = f"text_to_image_{i}_{j}.png"
                image.save(self.output_dir / filename)
                logger.info(f"  💾 Saved: {filename}")
            
            logger.info(f"  ⏱️  Generation time: {result.processing_time:.2f}s")
    
    async def demo_security_visualizations(self) -> Any:
        """Demonstrate security-focused visualizations."""
        logger.info("🔒 Demo: Security Visualizations")
        
        threat_types = self.demo_configs["security_visualization"]["threat_types"]
        severities = self.demo_configs["security_visualization"]["severities"]
        styles = self.demo_configs["security_visualization"]["styles"]
        
        for threat_type in threat_types:
            for severity in severities:
                for style in styles:
                    logger.info(f"  Generating: {threat_type} - {severity} - {style}")
                    
                    result = await self.diffusion_manager.generate_security_visualization(
                        threat_type=threat_type,
                        severity=severity,
                        style=style
                    )
                    
                    # Save image
                    filename = f"security_{threat_type}_{severity}_{style}.png"
                    result.images[0].save(self.output_dir / filename)
                    
                    logger.info(f"    💾 Saved: {filename}")
                    logger.info(f"    ⏱️  Generation time: {result.processing_time:.2f}s")
    
    async def demo_image_to_image(self) -> Any:
        """Demonstrate image-to-image transformation."""
        logger.info("🔄 Demo: Image-to-Image Transformation")
        
        # Load image-to-image pipeline
        config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.IMAGE_TO_IMAGE
        )
        
        pipeline_key = f"{config.model_name}_{config.task.value}"
        await self.diffusion_manager.load_pipeline(config)
        
        # Create a simple test image (you can replace with actual image loading)
        
        # Create a simple network diagram
        test_image = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(test_image)
        draw.rectangle([100, 100, 412, 412], outline='black', width=2)
        draw.text((256, 256), "Network Diagram", fill='black', anchor='mm')
        
        # Save original
        test_image.save(self.output_dir / "original_network.png")
        
        # Transform the image
        img2img_config = ImageToImageConfig(
            prompt="enhanced cybersecurity network diagram, professional technical visualization, clean design",
            negative_prompt="cartoon, anime, artistic, decorative",
            image=test_image,
            strength=0.7,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        result = await self.diffusion_manager.generate_image_to_image(pipeline_key, img2img_config)
        
        # Save transformed image
        result.images[0].save(self.output_dir / "transformed_network.png")
        logger.info("  💾 Saved: transformed_network.png")
        logger.info(f"  ⏱️  Transformation time: {result.processing_time:.2f}s")
    
    async def demo_inpainting(self) -> Any:
        """Demonstrate inpainting capabilities."""
        logger.info("🎨 Demo: Inpainting")
        
        # Load inpainting pipeline
        config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.INPAINTING
        )
        
        pipeline_key = f"{config.model_name}_{config.task.value}"
        await self.diffusion_manager.load_pipeline(config)
        
        # Create test image and mask
        
        # Create a security dashboard image
        image = Image.new('RGB', (512, 512), color='lightblue')
        draw = ImageDraw.Draw(image)
        draw.rectangle([50, 50, 462, 462], fill='white', outline='black', width=2)
        draw.text((256, 100), "Security Dashboard", fill='black', anchor='mm')
        draw.text((256, 200), "Status: Normal", fill='green', anchor='mm')
        draw.text((256, 300), "Threats: 0", fill='green', anchor='mm')
        
        # Create mask (area to inpaint)
        mask = Image.new('L', (512, 512), color=0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([200, 250, 312, 350], fill=255)  # Mask the middle area
        
        # Save original and mask
        image.save(self.output_dir / "original_dashboard.png")
        mask.save(self.output_dir / "inpaint_mask.png")
        
        # Perform inpainting
        inpaint_config = InpaintingConfig(
            prompt="security alert notification, warning message, red alert",
            negative_prompt="cartoon, anime, artistic",
            image=image,
            mask_image=mask,
            mask_strength=0.8,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        result = await self.diffusion_manager.generate_image(pipeline_key, inpaint_config)
        
        # Save inpainted image
        result.images[0].save(self.output_dir / "inpainted_dashboard.png")
        logger.info("  💾 Saved: inpainted_dashboard.png")
        logger.info(f"  ⏱️  Inpainting time: {result.processing_time:.2f}s")
    
    async def demo_controlnet(self) -> Any:
        """Demonstrate ControlNet integration."""
        logger.info("🎛️ Demo: ControlNet Integration")
        
        # Load ControlNet pipeline
        config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.CONTROLNET
        )
        
        pipeline_key = f"{config.model_name}_{config.task.value}"
        await self.diffusion_manager.load_pipeline(config)
        
        # Create control image (edge detection)
        
        # Create a simple diagram
        control_image = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(control_image)
        
        # Draw network nodes and connections
        nodes = [(100, 100), (400, 100), (250, 300), (100, 400), (400, 400)]
        for node in nodes:
            draw.ellipse([node[0]-20, node[1]-20, node[0]+20, node[1]+20], fill='black')
        
        # Draw connections
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        for start, end in connections:
            draw.line([nodes[start], nodes[end]], fill='black', width=3)
        
        # Apply edge detection
        control_image = control_image.filter(ImageFilter.FIND_EDGES)
        control_image = control_image.convert('L')  # Convert to grayscale
        
        # Save control image
        control_image.save(self.output_dir / "control_edges.png")
        
        # Generate with ControlNet
        control_config = ControlNetConfig(
            prompt="cybersecurity network topology, professional technical diagram, clean visualization",
            negative_prompt="cartoon, anime, artistic, decorative",
            control_image=control_image,
            controlnet_conditioning_scale=1.0,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        result = await self.diffusion_manager.generate_image(pipeline_key, control_config)
        
        # Save generated image
        result.images[0].save(self.output_dir / "controlnet_generated.png")
        logger.info("  💾 Saved: controlnet_generated.png")
        logger.info(f"  ⏱️  ControlNet generation time: {result.processing_time:.2f}s")
    
    async def demo_performance_testing(self) -> Any:
        """Demonstrate performance testing and optimization."""
        logger.info("⚡ Demo: Performance Testing")
        
        # Load pipeline
        config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.TEXT_TO_IMAGE
        )
        
        pipeline_key = f"{config.model_name}_{config.task.value}"
        await self.diffusion_manager.load_pipeline(config)
        
        performance_results = []
        
        # Test different configurations
        batch_sizes = self.demo_configs["performance_testing"]["batch_sizes"]
        inference_steps = self.demo_configs["performance_testing"]["inference_steps"]
        guidance_scales = self.demo_configs["performance_testing"]["guidance_scales"]
        
        for batch_size in batch_sizes:
            for steps in inference_steps:
                for guidance in guidance_scales:
                    logger.info(f"  Testing: batch={batch_size}, steps={steps}, guidance={guidance}")
                    
                    generation_config = GenerationConfig(
                        prompt="cybersecurity visualization, technical diagram",
                        negative_prompt="cartoon, anime, artistic",
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        num_images_per_prompt=batch_size
                    )
                    
                    start_time = time.time()
                    result = await self.diffusion_manager.generate_image(pipeline_key, generation_config)
                    end_time = time.time()
                    
                    performance_results.append({
                        "batch_size": batch_size,
                        "inference_steps": steps,
                        "guidance_scale": guidance,
                        "processing_time": result.processing_time,
                        "total_time": end_time - start_time,
                        "throughput": batch_size / result.processing_time,
                        "memory_usage": result.memory_usage
                    })
                    
                    logger.info(f"    ⏱️  Time: {result.processing_time:.2f}s")
                    logger.info(f"    📊 Throughput: {batch_size / result.processing_time:.2f} images/s")
        
        # Save performance results
        with open(self.output_dir / "performance_results.json", "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(performance_results, f, indent=2)
        
        logger.info("  💾 Saved: performance_results.json")
    
    async def demo_transformers_integration(self) -> Any:
        """Demonstrate integration with transformers manager."""
        logger.info("🔗 Demo: Transformers Integration")
        
        # Load a text classification model for security analysis
        model_config = ModelConfig(
            model_name="microsoft/DialoGPT-medium",
            model_type=ModelType.CAUSAL_LANGUAGE_MODEL
        )
        
        model_key = f"{model_config.model_name}_{model_config.model_type.value}"
        await self.transformers_manager.load_model(model_config)
        
        # Generate security-related text
        security_prompts = [
            "Analyze this cybersecurity threat:",
            "Generate a security report for:",
            "Describe the network vulnerability:"
        ]
        
        for prompt in security_prompts:
            # Tokenize with transformers
            tokenized = await self.transformers_manager.tokenize_text(
                prompt,
                model_config.model_name,
                max_length=100
            )
            
            logger.info(f"  📝 Prompt: {prompt}")
            logger.info(f"    Token count: {tokenized.token_count}")
            logger.info(f"    Processing time: {tokenized.processing_time:.4f}s")
            
            # Run inference
            result = await self.transformers_manager.run_inference(
                model_key,
                prompt
            )
            
            logger.info(f"    Inference time: {result.processing_time:.4f}s")
            logger.info(f"    Memory usage: {result.memory_usage['rss_mb']:.1f} MB")
    
    async def demo_memory_management(self) -> Any:
        """Demonstrate memory management capabilities."""
        logger.info("🧠 Demo: Memory Management")
        
        # Test memory usage with multiple pipelines
        pipeline_configs = [
            DiffusionConfig(
                model_name="runwayml/stable-diffusion-v1-5",
                task=DiffusionTask.TEXT_TO_IMAGE
            ),
            DiffusionConfig(
                model_name="runwayml/stable-diffusion-v1-5",
                task=DiffusionTask.IMAGE_TO_IMAGE
            ),
            DiffusionConfig(
                model_name="runwayml/stable-diffusion-v1-5",
                task=DiffusionTask.INPAINTING
            )
        ]
        
        loaded_pipelines = []
        
        for i, config in enumerate(pipeline_configs):
            logger.info(f"  Loading pipeline {i+1}/{len(pipeline_configs)}")
            
            pipeline = await self.diffusion_manager.load_pipeline(config)
            loaded_pipelines.append(pipeline)
            
            # Check memory usage
            metrics = self.diffusion_manager.get_metrics()
            for key, value in metrics.items():
                if "memory_usage" in value:
                    logger.info(f"    Memory usage: {value['memory_usage']['rss_mb']:.1f} MB")
        
        logger.info(f"  📊 Total loaded pipelines: {len(self.diffusion_manager.list_loaded_pipelines())}")
        
        # Clear cache
        logger.info("  🧹 Clearing cache...")
        self.diffusion_manager.clear_cache()
        
        logger.info(f"  📊 Pipelines after cache clear: {len(self.diffusion_manager.list_loaded_pipelines())}")
    
    async def demo_security_prompt_engineering(self) -> Any:
        """Demonstrate security-focused prompt engineering."""
        logger.info("🎯 Demo: Security Prompt Engineering")
        
        # Test different security scenarios
        security_scenarios = [
            {
                "threat_type": "malware_analysis",
                "severity": "critical",
                "style": "technical",
                "description": "Critical malware analysis visualization"
            },
            {
                "threat_type": "network_security",
                "severity": "high",
                "style": "detailed",
                "description": "High-priority network security breach"
            },
            {
                "threat_type": "threat_hunting",
                "severity": "medium",
                "style": "simple",
                "description": "Medium-level threat hunting process"
            }
        ]
        
        for scenario in security_scenarios:
            logger.info(f"  🎯 Scenario: {scenario['description']}")
            
            # Generate prompts
            positive_prompt, negative_prompt = SecurityPromptEngine.generate_security_prompt(
                scenario["threat_type"],
                scenario["severity"],
                scenario["style"]
            )
            
            logger.info(f"    Positive prompt: {positive_prompt}")
            logger.info(f"    Negative prompt: {negative_prompt}")
            
            # Generate visualization
            result = await self.diffusion_manager.generate_security_visualization(
                threat_type=scenario["threat_type"],
                severity=scenario["severity"],
                style=scenario["style"]
            )
            
            # Save image
            filename = f"security_prompt_{scenario['threat_type']}_{scenario['severity']}_{scenario['style']}.png"
            result.images[0].save(self.output_dir / filename)
            logger.info(f"    💾 Saved: {filename}")


async def main():
    """Main demo function."""
    demo = DiffusionModelsDemo()
    
    print("🎨 Diffusion Models Demo for Cybersecurity Applications")
    print("=" * 60)
    print()
    
    await demo.run_comprehensive_demo()
    
    print()
    print("📁 Generated files saved in: diffusion_outputs/")
    print("📊 Check performance_results.json for detailed metrics")
    print("✅ Demo completed successfully!")


match __name__:
    case "__main__":
    asyncio.run(main()) 