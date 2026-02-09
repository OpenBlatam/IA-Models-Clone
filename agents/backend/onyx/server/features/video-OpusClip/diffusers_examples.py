#!/usr/bin/env python3
"""
Diffusers Examples for Video-OpusClip

Comprehensive examples demonstrating Diffusers library usage
in the Video-OpusClip system for AI image and video generation.
"""

import torch
import time
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from PIL import Image
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# BASIC IMAGE GENERATION EXAMPLES
# =============================================================================

def example_basic_image_generation():
    """Example 1: Basic image generation from text prompt."""
    
    print("🎨 Example 1: Basic Image Generation")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline
        
        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        
        # Generate image
        prompt = "A beautiful sunset over mountains, high quality, detailed"
        
        with torch.autocast(device) if device == "cuda" else torch.no_grad():
            image = pipeline(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
        
        # Save result
        image.save("example_1_basic_image.png")
        print(f"✅ Generated: example_1_basic_image.png")
        
        return image
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def example_image_generation_with_parameters():
    """Example 2: Image generation with different parameters."""
    
    print("\n🎨 Example 2: Image Generation with Parameters")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline
        
        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        
        # Test different parameters
        base_prompt = "A cat playing with a laser pointer"
        
        # High quality settings
        high_quality = pipeline(
            prompt=base_prompt,
            height=768,
            width=768,
            num_inference_steps=50,
            guidance_scale=8.5
        ).images[0]
        high_quality.save("example_2_high_quality.png")
        
        # Fast settings
        fast_generation = pipeline(
            prompt=base_prompt,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=7.0
        ).images[0]
        fast_generation.save("example_2_fast_generation.png")
        
        # Creative settings
        creative = pipeline(
            prompt=base_prompt,
            height=512,
            width=512,
            num_inference_steps=30,
            guidance_scale=6.0,
            num_return_sequences=1
        ).images[0]
        creative.save("example_2_creative.png")
        
        print("✅ Generated: high_quality.png, fast_generation.png, creative.png")
        
        return [high_quality, fast_generation, creative]
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# =============================================================================
# VIDEO GENERATION EXAMPLES
# =============================================================================

def example_basic_video_generation():
    """Example 3: Basic video generation from text prompt."""
    
    print("\n🎬 Example 3: Basic Video Generation")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline
        from moviepy.editor import ImageSequenceClip
        
        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        
        # Generate video frames
        prompt = "A cat playing with a ball"
        num_frames = 15
        fps = 5
        
        frames = []
        
        for i in range(num_frames):
            # Add temporal context
            temporal_prompt = f"{prompt}, frame {i+1} of {num_frames}"
            
            with torch.autocast(device) if device == "cuda" else torch.no_grad():
                image = pipeline(
                    temporal_prompt,
                    height=512,
                    width=512,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
            
            frames.append(np.array(image))
            print(f"  Generated frame {i+1}/{num_frames}")
        
        # Create video
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile("example_3_basic_video.mp4", verbose=False, logger=None)
        
        print("✅ Generated: example_3_basic_video.mp4")
        
        return clip
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def example_advanced_video_generation():
    """Example 4: Advanced video generation with motion control."""
    
    print("\n🎬 Example 4: Advanced Video Generation")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline
        from moviepy.editor import ImageSequenceClip
        
        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        
        # Generate video with motion control
        base_prompt = "A bird flying through clouds"
        num_frames = 20
        fps = 8
        
        frames = []
        
        for i in range(num_frames):
            # Calculate motion progress
            progress = i / (num_frames - 1)
            
            # Add motion context
            motion_descriptions = [
                "beginning of flight",
                "early flight",
                "mid flight",
                "late flight",
                "end of flight"
            ]
            
            motion_index = int(progress * (len(motion_descriptions) - 1))
            motion_desc = motion_descriptions[motion_index]
            
            motion_prompt = f"{base_prompt}, {motion_desc}, motion intensity: 0.7"
            
            with torch.autocast(device) if device == "cuda" else torch.no_grad():
                image = pipeline(
                    motion_prompt,
                    height=512,
                    width=512,
                    num_inference_steps=25,
                    guidance_scale=7.5
                ).images[0]
            
            frames.append(np.array(image))
            print(f"  Generated frame {i+1}/{num_frames} ({motion_desc})")
        
        # Create video
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile("example_4_advanced_video.mp4", verbose=False, logger=None)
        
        print("✅ Generated: example_4_advanced_video.mp4")
        
        return clip
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# =============================================================================
# OPTIMIZATION EXAMPLES
# =============================================================================

def example_memory_optimization():
    """Example 5: Memory optimization techniques."""
    
    print("\n⚡ Example 5: Memory Optimization")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline
        
        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Apply memory optimizations
        if device == "cuda":
            pipeline.enable_attention_slicing()
            pipeline.enable_vae_slicing()
            pipeline.enable_model_cpu_offload()
            print("✅ Memory optimizations enabled")
        
        pipeline = pipeline.to(device)
        
        # Generate image with memory optimization
        prompt = "A detailed landscape painting"
        
        with torch.autocast(device) if device == "cuda" else torch.no_grad():
            image = pipeline(
                prompt=prompt,
                height=768,
                width=768,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
        
        image.save("example_5_memory_optimized.png")
        print("✅ Generated: example_5_memory_optimized.png")
        
        return image
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def example_speed_optimization():
    """Example 6: Speed optimization techniques."""
    
    print("\n⚡ Example 6: Speed Optimization")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        
        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Apply speed optimizations
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        if device == "cuda":
            pipeline.enable_xformers_memory_efficient_attention()
        
        pipeline = pipeline.to(device)
        
        print("✅ Speed optimizations enabled")
        
        # Generate image with speed optimization
        prompt = "A fast car on a highway"
        
        start_time = time.time()
        
        with torch.autocast(device) if device == "cuda" else torch.no_grad():
            image = pipeline(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
        
        generation_time = time.time() - start_time
        
        image.save("example_6_speed_optimized.png")
        print(f"✅ Generated: example_6_speed_optimized.png in {generation_time:.2f}s")
        
        return image, generation_time
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

# =============================================================================
# BATCH PROCESSING EXAMPLES
# =============================================================================

def example_batch_image_generation():
    """Example 7: Batch image generation."""
    
    print("\n📦 Example 7: Batch Image Generation")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline
        
        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        
        # Batch prompts
        prompts = [
            "A beautiful sunset over mountains",
            "A cat playing with a laser pointer",
            "A dog running in a park",
            "A bird flying through clouds"
        ]
        
        batch_size = 2
        all_images = []
        
        print(f"Processing {len(prompts)} prompts in batches of {batch_size}")
        
        start_time = time.time()
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            print(f"  Processing batch {i//batch_size + 1}: {batch_prompts}")
            
            with torch.autocast(device) if device == "cuda" else torch.no_grad():
                batch_images = pipeline(
                    prompt=batch_prompts,
                    height=512,
                    width=512,
                    num_inference_steps=25,
                    guidance_scale=7.5
                ).images
            
            all_images.extend(batch_images)
        
        total_time = time.time() - start_time
        
        # Save batch images
        for i, (image, prompt) in enumerate(zip(all_images, prompts)):
            filename = f"example_7_batch_{i+1}.png"
            image.save(filename)
            print(f"  Saved: {filename} ({prompt})")
        
        print(f"✅ Batch processing completed in {total_time:.2f}s")
        print(f"📊 Average time per image: {total_time/len(prompts):.2f}s")
        
        return all_images, total_time
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

# =============================================================================
# INTEGRATION EXAMPLES
# =============================================================================

def example_video_opusclip_integration():
    """Example 8: Integration with Video-OpusClip system."""
    
    print("\n🔗 Example 8: Video-OpusClip Integration")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline
        
        class VideoOpusClipDiffusersIntegration:
            """Integration class for Video-OpusClip system."""
            
            def __init__(self):
                self.pipeline = None
                self.setup_pipeline()
            
            def setup_pipeline(self):
                """Setup optimized pipeline."""
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Apply optimizations
                if device == "cuda":
                    self.pipeline.enable_attention_slicing()
                    self.pipeline.enable_vae_slicing()
                
                self.pipeline = self.pipeline.to(device)
                print("✅ Pipeline setup for Video-OpusClip integration")
            
            def generate_viral_thumbnail(self, video_description: str):
                """Generate viral thumbnail for video."""
                
                prompt = f"Viral thumbnail for video: {video_description}, high quality, eye-catching, trending"
                
                with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
                    image = self.pipeline(
                        prompt=prompt,
                        height=1280,
                        width=720,
                        num_inference_steps=30,
                        guidance_scale=7.5
                    ).images[0]
                
                return image
            
            def generate_video_intro(self, video_description: str, duration: int = 3):
                """Generate video intro frames."""
                
                intro_prompt = f"Dynamic intro for video: {video_description}, cinematic, engaging"
                num_frames = duration * 10  # 10 fps
                
                frames = []
                
                for i in range(num_frames):
                    temporal_prompt = f"{intro_prompt}, frame {i+1} of {num_frames}"
                    
                    with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
                        image = self.pipeline(
                            temporal_prompt,
                            height=720,
                            width=1280,
                            num_inference_steps=20,
                            guidance_scale=7.5
                        ).images[0]
                    
                    frames.append(image)
                
                return frames
        
        # Test integration
        integration = VideoOpusClipDiffusersIntegration()
        
        # Generate viral thumbnail
        thumbnail = integration.generate_viral_thumbnail("Funny cat compilation video")
        thumbnail.save("example_8_viral_thumbnail.png")
        print("✅ Generated: example_8_viral_thumbnail.png")
        
        # Generate video intro
        intro_frames = integration.generate_video_intro("Amazing dog tricks", duration=2)
        
        # Save intro frames
        for i, frame in enumerate(intro_frames):
            frame.save(f"example_8_intro_frame_{i+1}.png")
        
        print(f"✅ Generated: {len(intro_frames)} intro frames")
        
        return thumbnail, intro_frames
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

# =============================================================================
# ADVANCED FEATURES EXAMPLES
# =============================================================================

def example_controlnet_integration():
    """Example 9: ControlNet integration for precise control."""
    
    print("\n🎯 Example 9: ControlNet Integration")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        import cv2
        
        # Create a simple control image (canny edges)
        control_image = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.rectangle(control_image, (100, 100), (400, 400), (255, 255, 255), 2)
        cv2.circle(control_image, (256, 256), 50, (255, 255, 255), 2)
        
        control_image = Image.fromarray(control_image)
        control_image.save("example_9_control_image.png")
        
        # Load ControlNet model
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
        
        # Create pipeline
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        
        # Process control image
        control_image = cv2.Canny(np.array(control_image), 100, 200)
        control_image = Image.fromarray(control_image)
        
        # Generate image with control
        prompt = "A beautiful painting with geometric shapes"
        
        with torch.autocast(device) if device == "cuda" else torch.no_grad():
            image = pipeline(
                prompt=prompt,
                image=control_image,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
        
        image.save("example_9_controlnet_result.png")
        print("✅ Generated: example_9_controlnet_result.png")
        
        return image
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def example_custom_scheduler():
    """Example 10: Custom scheduler configuration."""
    
    print("\n⚙️ Example 10: Custom Scheduler")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        
        # Create custom scheduler
        scheduler_config = {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "num_train_timesteps": 1000,
            "clip_sample": False,
            "set_alpha_to_one": False,
        }
        
        custom_scheduler = DDIMScheduler(**scheduler_config)
        
        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Apply custom scheduler
        pipeline.scheduler = custom_scheduler
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        
        # Generate image with custom scheduler
        prompt = "A surreal landscape with custom scheduling"
        
        with torch.autocast(device) if device == "cuda" else torch.no_grad():
            image = pipeline(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
        
        image.save("example_10_custom_scheduler.png")
        print("✅ Generated: example_10_custom_scheduler.png")
        
        return image
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# =============================================================================
# PERFORMANCE MONITORING EXAMPLES
# =============================================================================

def example_performance_monitoring():
    """Example 11: Performance monitoring and metrics."""
    
    print("\n📊 Example 11: Performance Monitoring")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline
        import psutil
        import GPUtil
        
        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        
        # Monitor system resources
        def get_system_metrics():
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available / (1024**3)  # GB
            }
            
            if torch.cuda.is_available():
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics["gpu_memory_percent"] = gpu.memoryUtil * 100
                    metrics["gpu_memory_used"] = gpu.memoryUsed
                    metrics["gpu_memory_total"] = gpu.memoryTotal
            
            return metrics
        
        # Generate with monitoring
        prompt = "A detailed portrait with performance monitoring"
        
        print("📊 System metrics before generation:")
        before_metrics = get_system_metrics()
        for key, value in before_metrics.items():
            print(f"  {key}: {value}")
        
        start_time = time.time()
        
        with torch.autocast(device) if device == "cuda" else torch.no_grad():
            image = pipeline(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
        
        generation_time = time.time() - start_time
        
        print(f"\n📊 System metrics after generation:")
        after_metrics = get_system_metrics()
        for key, value in after_metrics.items():
            print(f"  {key}: {value}")
        
        print(f"\n⏱️ Generation time: {generation_time:.2f} seconds")
        
        # Save performance report
        performance_report = {
            "generation_time": generation_time,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "prompt": prompt
        }
        
        with open("example_11_performance_report.json", "w") as f:
            json.dump(performance_report, f, indent=2)
        
        image.save("example_11_performance_monitored.png")
        print("✅ Generated: example_11_performance_monitored.png")
        print("✅ Saved: example_11_performance_report.json")
        
        return image, performance_report
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_examples():
    """Run all Diffusers examples."""
    
    print("🚀 Diffusers Examples for Video-OpusClip")
    print("=" * 60)
    
    results = {}
    
    # Basic examples
    results['basic_image'] = example_basic_image_generation()
    results['image_parameters'] = example_image_generation_with_parameters()
    
    # Video examples
    results['basic_video'] = example_basic_video_generation()
    results['advanced_video'] = example_advanced_video_generation()
    
    # Optimization examples
    results['memory_optimization'] = example_memory_optimization()
    results['speed_optimization'] = example_speed_optimization()
    
    # Batch processing
    results['batch_processing'] = example_batch_image_generation()
    
    # Integration examples
    results['integration'] = example_video_opusclip_integration()
    
    # Advanced features
    results['controlnet'] = example_controlnet_integration()
    results['custom_scheduler'] = example_custom_scheduler()
    
    # Performance monitoring
    results['performance'] = example_performance_monitoring()
    
    # Summary
    print("\n📊 Examples Summary")
    print("=" * 60)
    
    successful_examples = sum(1 for result in results.values() if result is not None)
    total_examples = len(results)
    
    print(f"✅ Successful examples: {successful_examples}/{total_examples}")
    
    for name, result in results.items():
        status = "✅" if result is not None else "❌"
        print(f"{status} {name.replace('_', ' ').title()}")
    
    print(f"\n📁 Generated files:")
    for file in Path(".").glob("example_*.png"):
        print(f"  📄 {file}")
    for file in Path(".").glob("example_*.mp4"):
        print(f"  🎬 {file}")
    for file in Path(".").glob("example_*.json"):
        print(f"  📊 {file}")
    
    return results

if __name__ == "__main__":
    # Run all examples
    results = run_all_examples()
    
    print("\n🔧 Next Steps:")
    print("1. Explore the generated images and videos")
    print("2. Read the DIFFUSERS_GUIDE.md for detailed usage")
    print("3. Run quick_start_diffusers.py for basic setup")
    print("4. Integrate with your Video-OpusClip workflow") 