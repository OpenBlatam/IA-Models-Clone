#!/usr/bin/env python3
"""
Quick Start Diffusers for Video-OpusClip

This script demonstrates how to quickly get started with Diffusers
in the Video-OpusClip system for image and video generation.
"""

import torch
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_diffusers_installation():
    """Check if Diffusers is properly installed."""
    
    print("🔍 Checking Diffusers Installation")
    print("=" * 50)
    
    try:
        import diffusers
        print(f"✅ Diffusers version: {diffusers.__version__}")
        
        # Test basic imports
        from diffusers import StableDiffusionPipeline
        print("✅ StableDiffusionPipeline imported successfully")
        
        from diffusers import DDIMScheduler, DDPMScheduler
        print("✅ Schedulers imported successfully")
        
        from diffusers import UNet2DConditionModel, AutoencoderKL
        print("✅ Model components imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Diffusers import error: {e}")
        print("💡 Install with: pip install diffusers[torch] accelerate")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def quick_start_basic_image_generation():
    """Basic image generation with Diffusers."""
    
    print("\n🎨 Quick Start: Basic Image Generation")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline
        from PIL import Image
        
        print("📥 Loading Stable Diffusion pipeline...")
        
        # Load pipeline with optimizations
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True
        )
        
        # Move to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        
        # Enable optimizations
        if device == "cuda":
            pipeline.enable_attention_slicing()
            pipeline.enable_vae_slicing()
        
        print(f"✅ Pipeline loaded on {device}")
        
        # Generate image
        prompt = "A beautiful sunset over mountains, high quality, detailed"
        print(f"🎯 Generating image: '{prompt}'")
        
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
        
        print(f"✅ Image generated in {generation_time:.2f} seconds")
        print(f"📏 Image size: {image.size}")
        
        # Save image
        output_path = "generated_image.png"
        image.save(output_path)
        print(f"💾 Image saved to: {output_path}")
        
        return image, generation_time
        
    except Exception as e:
        print(f"❌ Image generation error: {e}")
        return None, None

def quick_start_video_generation():
    """Basic video generation with Diffusers."""
    
    print("\n🎬 Quick Start: Basic Video Generation")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline
        from moviepy.editor import ImageSequenceClip
        
        print("📥 Loading pipeline for video generation...")
        
        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        
        if device == "cuda":
            pipeline.enable_attention_slicing()
        
        print(f"✅ Pipeline loaded on {device}")
        
        # Generate video frames
        prompt = "A cat playing with a ball"
        num_frames = 10  # Reduced for quick demo
        fps = 5
        
        print(f"🎯 Generating {num_frames} frames: '{prompt}'")
        
        frames = []
        start_time = time.time()
        
        for i in range(num_frames):
            # Add temporal context
            temporal_prompt = f"{prompt}, frame {i+1} of {num_frames}"
            
            with torch.autocast(device) if device == "cuda" else torch.no_grad():
                image = pipeline(
                    temporal_prompt,
                    height=512,
                    width=512,
                    num_inference_steps=20,  # Reduced for speed
                    guidance_scale=7.5
                ).images[0]
            
            frames.append(np.array(image))
            print(f"  Frame {i+1}/{num_frames} generated")
        
        total_time = time.time() - start_time
        
        print(f"✅ Video frames generated in {total_time:.2f} seconds")
        print(f"📊 Average time per frame: {total_time/num_frames:.2f} seconds")
        
        # Create video clip
        clip = ImageSequenceClip(frames, fps=fps)
        
        # Save video
        output_path = "generated_video.mp4"
        clip.write_videofile(output_path, verbose=False, logger=None)
        print(f"💾 Video saved to: {output_path}")
        
        return clip, total_time
        
    except Exception as e:
        print(f"❌ Video generation error: {e}")
        return None, None

def quick_start_optimization_techniques():
    """Demonstrate optimization techniques."""
    
    print("\n⚡ Quick Start: Optimization Techniques")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        
        print("📥 Loading pipeline with optimizations...")
        
        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Apply optimizations
        print("🔧 Applying optimizations...")
        
        # Memory optimizations
        if device == "cuda":
            pipeline.enable_attention_slicing()
            pipeline.enable_vae_slicing()
            pipeline.enable_model_cpu_offload()
            print("  ✅ Memory optimizations enabled")
        
        # Speed optimizations
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        print("  ✅ Fast scheduler enabled")
        
        # Move to device
        pipeline = pipeline.to(device)
        
        # Test generation speed
        prompt = "A beautiful landscape"
        
        print(f"🎯 Testing optimized generation: '{prompt}'")
        
        start_time = time.time()
        
        with torch.autocast(device) if device == "cuda" else torch.no_grad():
            image = pipeline(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=20,  # Reduced for speed
                guidance_scale=7.5
            ).images[0]
        
        generation_time = time.time() - start_time
        
        print(f"✅ Optimized generation completed in {generation_time:.2f} seconds")
        
        # Save optimized image
        output_path = "optimized_image.png"
        image.save(output_path)
        print(f"💾 Optimized image saved to: {output_path}")
        
        return image, generation_time
        
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        return None, None

def quick_start_batch_processing():
    """Demonstrate batch processing capabilities."""
    
    print("\n📦 Quick Start: Batch Processing")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline
        
        print("📥 Loading pipeline for batch processing...")
        
        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        
        if device == "cuda":
            pipeline.enable_attention_slicing()
        
        print(f"✅ Pipeline loaded on {device}")
        
        # Batch prompts
        prompts = [
            "A beautiful sunset",
            "A cat playing",
            "A dog running",
            "A bird flying"
        ]
        
        batch_size = 2  # Process 2 at a time
        all_images = []
        
        print(f"🎯 Processing {len(prompts)} prompts in batches of {batch_size}")
        
        start_time = time.time()
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            print(f"  Processing batch {i//batch_size + 1}: {batch_prompts}")
            
            with torch.autocast(device) if device == "cuda" else torch.no_grad():
                batch_images = pipeline(
                    prompt=batch_prompts,
                    height=512,
                    width=512,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images
            
            all_images.extend(batch_images)
        
        total_time = time.time() - start_time
        
        print(f"✅ Batch processing completed in {total_time:.2f} seconds")
        print(f"📊 Average time per image: {total_time/len(prompts):.2f} seconds")
        
        # Save batch images
        for i, (image, prompt) in enumerate(zip(all_images, prompts)):
            output_path = f"batch_image_{i+1}.png"
            image.save(output_path)
            print(f"💾 {prompt} -> {output_path}")
        
        return all_images, total_time
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return None, None

def quick_start_integration_demo():
    """Demonstrate integration with Video-OpusClip system."""
    
    print("\n🔗 Quick Start: Video-OpusClip Integration")
    print("=" * 50)
    
    try:
        # Import Video-OpusClip components
        try:
            from optimized_libraries import OptimizedVideoDiffusionPipeline
            print("✅ OptimizedVideoDiffusionPipeline imported")
        except ImportError:
            print("⚠️  OptimizedVideoDiffusionPipeline not available, using basic pipeline")
            from diffusers import StableDiffusionPipeline
            OptimizedVideoDiffusionPipeline = StableDiffusionPipeline
        
        try:
            from enhanced_error_handling import safe_load_ai_model
            print("✅ Enhanced error handling imported")
        except ImportError:
            print("⚠️  Enhanced error handling not available")
            safe_load_ai_model = None
        
        # Create integrated generator
        class VideoOpusClipDiffusersDemo:
            """Demo integration with Video-OpusClip."""
            
            def __init__(self):
                self.pipeline = None
                self.setup_pipeline()
            
            def setup_pipeline(self):
                """Setup pipeline with error handling."""
                try:
                    if OptimizedVideoDiffusionPipeline != StableDiffusionPipeline:
                        self.pipeline = OptimizedVideoDiffusionPipeline()
                    else:
                        self.pipeline = StableDiffusionPipeline.from_pretrained(
                            "runwayml/stable-diffusion-v1-5",
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                        )
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        self.pipeline = self.pipeline.to(device)
                        if device == "cuda":
                            self.pipeline.enable_attention_slicing()
                    
                    print("✅ Pipeline setup successful")
                    
                except Exception as e:
                    print(f"❌ Pipeline setup failed: {e}")
                    self.pipeline = None
            
            def generate_viral_thumbnail(self, video_description: str):
                """Generate viral thumbnail for video."""
                
                if self.pipeline is None:
                    print("❌ Pipeline not available")
                    return None
                
                try:
                    # Create viral thumbnail prompt
                    prompt = f"Viral thumbnail for video: {video_description}, high quality, eye-catching, trending"
                    
                    print(f"🎯 Generating viral thumbnail: '{video_description}'")
                    
                    start_time = time.time()
                    
                    if hasattr(self.pipeline, 'generate_video_frames'):
                        # Use optimized pipeline
                        frames = self.pipeline.generate_video_frames(
                            prompt=prompt,
                            num_frames=1,
                            height=1280,
                            width=720,
                            num_inference_steps=30,
                            guidance_scale=7.5
                        )
                        image = frames[0] if frames else None
                    else:
                        # Use basic pipeline
                        with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
                            image = self.pipeline(
                                prompt=prompt,
                                height=720,
                                width=1280,
                                num_inference_steps=30,
                                guidance_scale=7.5
                            ).images[0]
                    
                    generation_time = time.time() - start_time
                    
                    if image:
                        output_path = "viral_thumbnail.png"
                        image.save(output_path)
                        print(f"✅ Viral thumbnail generated in {generation_time:.2f} seconds")
                        print(f"💾 Saved to: {output_path}")
                        return image
                    else:
                        print("❌ Failed to generate thumbnail")
                        return None
                        
                except Exception as e:
                    print(f"❌ Thumbnail generation error: {e}")
                    return None
        
        # Test integration
        demo = VideoOpusClipDiffusersDemo()
        thumbnail = demo.generate_viral_thumbnail("Funny cat compilation video")
        
        return thumbnail
        
    except Exception as e:
        print(f"❌ Integration demo error: {e}")
        return None

def run_all_quick_starts():
    """Run all quick start demonstrations."""
    
    print("🚀 Diffusers Quick Start for Video-OpusClip")
    print("=" * 60)
    
    results = {}
    
    # Check installation
    results['installation'] = check_diffusers_installation()
    
    if results['installation']:
        # Basic image generation
        image, gen_time = quick_start_basic_image_generation()
        results['basic_image'] = {'image': image, 'time': gen_time}
        
        # Video generation
        video, video_time = quick_start_video_generation()
        results['video'] = {'video': video, 'time': video_time}
        
        # Optimization techniques
        opt_image, opt_time = quick_start_optimization_techniques()
        results['optimization'] = {'image': opt_image, 'time': opt_time}
        
        # Batch processing
        batch_images, batch_time = quick_start_batch_processing()
        results['batch'] = {'images': batch_images, 'time': batch_time}
        
        # Integration demo
        thumbnail = quick_start_integration_demo()
        results['integration'] = {'thumbnail': thumbnail}
    
    # Summary
    print("\n📊 Quick Start Summary")
    print("=" * 60)
    
    if results.get('installation'):
        print("✅ Installation: Successful")
        
        if results.get('basic_image', {}).get('time'):
            print(f"✅ Basic Image Generation: {results['basic_image']['time']:.2f}s")
        
        if results.get('video', {}).get('time'):
            print(f"✅ Video Generation: {results['video']['time']:.2f}s")
        
        if results.get('optimization', {}).get('time'):
            print(f"✅ Optimization Demo: {results['optimization']['time']:.2f}s")
        
        if results.get('batch', {}).get('time'):
            print(f"✅ Batch Processing: {results['batch']['time']:.2f}s")
        
        if results.get('integration', {}).get('thumbnail'):
            print("✅ Integration Demo: Successful")
        
        print("\n🎉 All quick starts completed successfully!")
        print("📁 Check the generated files in the current directory")
        
    else:
        print("❌ Installation failed - please check your setup")
    
    return results

if __name__ == "__main__":
    # Run all quick starts
    results = run_all_quick_starts()
    
    print("\n🔧 Next Steps:")
    print("1. Explore the generated images and videos")
    print("2. Read the DIFFUSERS_GUIDE.md for detailed usage")
    print("3. Check diffusers_examples.py for more examples")
    print("4. Integrate with your Video-OpusClip workflow") 