# Diffusers Guide for Video-OpusClip

Complete guide to using the Hugging Face Diffusers library in your Video-OpusClip system for advanced AI image and video generation.

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Core Components](#core-components)
4. [Image Generation](#image-generation)
5. [Video Generation](#video-generation)
6. [Optimization Techniques](#optimization-techniques)
7. [Integration with Video-OpusClip](#integration-with-video-opusclip)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

## Overview

The Diffusers library provides state-of-the-art diffusion models for image and video generation that are essential for your Video-OpusClip system. It enables:

- **Image Generation**: Create stunning images from text prompts
- **Video Generation**: Generate videos from text or image inputs
- **Image-to-Image**: Transform existing images with new styles
- **Inpainting**: Fill in missing parts of images
- **ControlNet**: Precise control over image generation
- **Model Fine-tuning**: Customize models for specific use cases

## Installation & Setup

### Current Dependencies

Your Video-OpusClip system already includes Diffusers in the requirements:

```txt
# From requirements_complete.txt
diffusers>=0.18.0
accelerate>=0.20.0
transformers>=4.30.0
```

### Installation Commands

```bash
# Install basic Diffusers
pip install diffusers

# Install with all optimizations
pip install diffusers[torch] accelerate

# Install for production
pip install diffusers[torch,accelerate] transformers

# Install from your requirements
pip install -r requirements_complete.txt
```

### Verify Installation

```python
import diffusers
print(f"Diffusers version: {diffusers.__version__}")

# Test basic functionality
from diffusers import StableDiffusionPipeline
print("✅ Diffusers installation successful!")
```

## Core Components

### 1. Pipelines

```python
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    DDIMPipeline,
    DDPMPipeline
)

# Basic pipeline
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# XL pipeline for higher quality
xl_pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
```

### 2. Schedulers

```python
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)

# Different schedulers for different use cases
ddim_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
dpm_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
```

### 3. Models

```python
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    CLIPTextModel,
    CLIPTokenizer
)

# Individual model components
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
```

## Image Generation

### Basic Image Generation

```python
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_image(prompt: str, height: int = 512, width: int = 512):
    """Generate image from text prompt."""
    
    # Load pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    # Generate image
    with torch.autocast(device) if device == "cuda" else torch.no_grad():
        image = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
    
    return image

# Usage
image = generate_image("A beautiful sunset over mountains")
image.save("generated_image.png")
```

### Optimized Image Generation

```python
class OptimizedImageGenerator:
    """Optimized image generator with caching and performance features."""
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_pipeline()
    
    def load_pipeline(self):
        """Load and optimize pipeline."""
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if self.device == "cuda" else None
        )
        
        # Optimize for performance
        if self.device == "cuda":
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_model_cpu_offload()
        
        self.pipeline = self.pipeline.to(self.device)
    
    def generate_image(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = None
    ):
        """Generate optimized image."""
        
        if seed is not None:
            torch.manual_seed(seed)
        
        with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
            image = self.pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        
        return image

# Usage
generator = OptimizedImageGenerator()
image = generator.generate_image("A cat playing with a laser pointer")
```

### Batch Image Generation

```python
def generate_batch_images(
    prompts: List[str],
    pipeline,
    batch_size: int = 4,
    **kwargs
):
    """Generate multiple images in batches."""
    
    all_images = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
            batch_images = pipeline(
                prompt=batch_prompts,
                **kwargs
            ).images
        
        all_images.extend(batch_images)
    
    return all_images

# Usage
prompts = [
    "A beautiful sunset",
    "A cat playing",
    "A dog running",
    "A bird flying"
]
images = generate_batch_images(prompts, pipeline)
```

## Video Generation

### Basic Video Generation

```python
def generate_video_frames(
    prompt: str,
    num_frames: int = 30,
    fps: int = 10,
    pipeline=None
):
    """Generate video frames from text prompt."""
    
    if pipeline is None:
        pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    
    frames = []
    
    # Generate frames with temporal consistency
    for i in range(num_frames):
        # Add temporal context to prompt
        temporal_prompt = f"{prompt}, frame {i+1} of {num_frames}"
        
        with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
            image = pipeline(
                temporal_prompt,
                height=512,
                width=512,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
        
        frames.append(image)
    
    return frames

# Usage
frames = generate_video_frames("A cat playing with a ball")
```

### Advanced Video Generation

```python
import numpy as np
from moviepy.editor import ImageSequenceClip

class VideoGenerator:
    """Advanced video generator with temporal consistency."""
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = self.pipeline.to(self.device)
    
    def generate_video(
        self,
        prompt: str,
        duration: int = 5,
        fps: int = 10,
        height: int = 512,
        width: int = 512,
        motion_strength: float = 0.5
    ):
        """Generate video with motion control."""
        
        num_frames = duration * fps
        frames = []
        
        for i in range(num_frames):
            # Create motion-aware prompt
            progress = i / (num_frames - 1)
            motion_prompt = self._add_motion_context(prompt, progress, motion_strength)
            
            with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
                image = self.pipeline(
                    motion_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
            
            frames.append(np.array(image))
        
        # Create video clip
        clip = ImageSequenceClip(frames, fps=fps)
        return clip
    
    def _add_motion_context(self, prompt: str, progress: float, motion_strength: float):
        """Add motion context to prompt."""
        
        motion_descriptions = [
            "beginning of motion",
            "early motion",
            "mid motion",
            "late motion",
            "end of motion"
        ]
        
        motion_index = int(progress * (len(motion_descriptions) - 1))
        motion_desc = motion_descriptions[motion_index]
        
        return f"{prompt}, {motion_desc}, motion intensity: {motion_strength}"

# Usage
generator = VideoGenerator()
video = generator.generate_video("A cat playing with a laser pointer", duration=3)
video.write_videofile("generated_video.mp4")
```

### Image-to-Video Generation

```python
def generate_video_from_image(
    image: Image.Image,
    prompt: str,
    num_frames: int = 30,
    motion_strength: float = 0.5
):
    """Generate video from existing image."""
    
    from diffusers import StableDiffusionImg2ImgPipeline
    
    # Load img2img pipeline
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    
    frames = []
    
    for i in range(num_frames):
        # Add motion context
        motion_prompt = f"{prompt}, frame {i+1}, motion: {motion_strength}"
        
        with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
            result = pipeline(
                prompt=motion_prompt,
                image=image,
                strength=0.3,  # How much to change the image
                guidance_scale=7.5
            ).images[0]
        
        frames.append(result)
    
    return frames

# Usage
from PIL import Image
input_image = Image.open("cat.jpg")
video_frames = generate_video_from_image(input_image, "Cat moving around")
```

## Optimization Techniques

### Memory Optimization

```python
def optimize_pipeline_memory(pipeline):
    """Optimize pipeline for memory efficiency."""
    
    # Enable attention slicing
    pipeline.enable_attention_slicing()
    
    # Enable VAE slicing
    pipeline.enable_vae_slicing()
    
    # Enable model CPU offload
    pipeline.enable_model_cpu_offload()
    
    # Use memory efficient attention
    pipeline.enable_xformers_memory_efficient_attention()
    
    return pipeline

# Usage
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline = optimize_pipeline_memory(pipeline)
```

### Speed Optimization

```python
def optimize_pipeline_speed(pipeline):
    """Optimize pipeline for speed."""
    
    # Use faster scheduler
    from diffusers import DPMSolverMultistepScheduler
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    
    # Enable memory efficient attention
    pipeline.enable_xformers_memory_efficient_attention()
    
    # Use half precision
    pipeline = pipeline.half()
    
    return pipeline

# Usage
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline = optimize_pipeline_speed(pipeline)
```

### Caching and Batching

```python
class CachedDiffusionGenerator:
    """Diffusion generator with caching for performance."""
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id)
        self.cache = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = self.pipeline.to(self.device)
    
    def generate_with_cache(self, prompt: str, **kwargs):
        """Generate image with caching."""
        
        # Create cache key
        cache_key = hash(f"{prompt}_{kwargs}")
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate new image
        with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
            image = self.pipeline(prompt=prompt, **kwargs).images[0]
        
        # Cache result
        self.cache[cache_key] = image
        
        return image

# Usage
generator = CachedDiffusionGenerator()
image1 = generator.generate_with_cache("A cat", num_inference_steps=30)
image2 = generator.generate_with_cache("A cat", num_inference_steps=30)  # Uses cache
```

## Integration with Video-OpusClip

### Integration with Existing Components

```python
from optimized_libraries import OptimizedVideoDiffusionPipeline
from enhanced_error_handling import safe_load_ai_model, safe_model_inference

class VideoOpusClipDiffusers:
    """Diffusers integration for Video-OpusClip system."""
    
    def __init__(self):
        self.image_generator = None
        self.video_generator = None
        self.setup_generators()
    
    def setup_generators(self):
        """Setup diffusion generators."""
        try:
            self.image_generator = OptimizedVideoDiffusionPipeline()
            self.video_generator = OptimizedVideoDiffusionPipeline()
        except Exception as e:
            logger.error(f"Failed to setup generators: {e}")
    
    def generate_viral_thumbnail(self, video_description: str):
        """Generate viral thumbnail for video."""
        
        # Create thumbnail prompt
        thumbnail_prompt = f"Viral thumbnail for video: {video_description}, high quality, eye-catching, trending"
        
        try:
            frames = self.image_generator.generate_video_frames(
                prompt=thumbnail_prompt,
                num_frames=1,
                height=1280,
                width=720,
                num_inference_steps=30,
                guidance_scale=7.5
            )
            return frames[0] if frames else None
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
            return None
    
    def generate_video_intro(self, video_description: str, duration: int = 3):
        """Generate video intro."""
        
        intro_prompt = f"Dynamic intro for video: {video_description}, cinematic, engaging"
        
        try:
            frames = self.video_generator.generate_video_frames(
                prompt=intro_prompt,
                num_frames=duration * 10,  # 10 fps
                height=720,
                width=1280,
                num_inference_steps=20,
                guidance_scale=7.5
            )
            return frames
        except Exception as e:
            logger.error(f"Intro generation failed: {e}")
            return None

# Usage
video_opus = VideoOpusClipDiffusers()
thumbnail = video_opus.generate_viral_thumbnail("Funny cat compilation")
intro_frames = video_opus.generate_video_intro("Amazing dog tricks")
```

### API Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    generation_type: str = "image"  # "image" or "video"
    height: int = 512
    width: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5

@app.post("/generate")
async def generate_content(request: GenerationRequest):
    """Generate image or video via API."""
    
    try:
        if request.generation_type == "image":
            generator = OptimizedImageGenerator()
            image = generator.generate_image(
                prompt=request.prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale
            )
            return {"type": "image", "data": image}
        
        elif request.generation_type == "video":
            generator = VideoGenerator()
            video = generator.generate_video(
                prompt=request.prompt,
                height=request.height,
                width=request.width
            )
            return {"type": "video", "data": video}
        
        else:
            raise HTTPException(status_code=400, detail="Invalid generation type")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Usage
# POST /generate
# {
#   "prompt": "A beautiful sunset",
#   "generation_type": "image",
#   "height": 512,
#   "width": 512
# }
```

## Advanced Features

### ControlNet Integration

```python
def generate_with_controlnet(
    prompt: str,
    control_image: Image.Image,
    control_type: str = "canny"
):
    """Generate image with ControlNet for precise control."""
    
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers.utils import load_image
    import cv2
    import numpy as np
    
    # Load ControlNet model
    if control_type == "canny":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
    elif control_type == "pose":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")
    else:
        raise ValueError(f"Unsupported control type: {control_type}")
    
    # Create pipeline
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet
    )
    
    # Process control image
    if control_type == "canny":
        control_image = cv2.Canny(np.array(control_image), 100, 200)
        control_image = Image.fromarray(control_image)
    
    # Generate image
    with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
        image = pipeline(
            prompt=prompt,
            image=control_image,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
    
    return image

# Usage
control_image = Image.open("pose.jpg")
result = generate_with_controlnet("A person dancing", control_image, "pose")
```

### Model Fine-tuning

```python
def fine_tune_diffusion_model(
    base_model: str = "runwayml/stable-diffusion-v1-5",
    training_data: List[str] = None,
    output_dir: str = "./fine_tuned_model"
):
    """Fine-tune diffusion model on custom data."""
    
    from diffusers import StableDiffusionPipeline
    from diffusers.training_utils import train_dreambooth
    
    # Load base model
    pipeline = StableDiffusionPipeline.from_pretrained(base_model)
    
    # Setup training
    training_args = {
        "pretrained_model_name_or_path": base_model,
        "instance_data_dir": "./training_data",
        "output_dir": output_dir,
        "instance_prompt": "a photo of sks person",
        "resolution": 512,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "learning_rate": 5e-6,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "num_class_images": 50,
        "max_train_steps": 400,
    }
    
    # Train model
    train_dreambooth(**training_args)
    
    return output_dir

# Usage
# fine_tune_diffusion_model(training_data=["path/to/images"])
```

### Custom Schedulers

```python
def create_custom_scheduler():
    """Create custom scheduler for specific use cases."""
    
    from diffusers import DDIMScheduler
    
    # Custom scheduler configuration
    scheduler_config = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "num_train_timesteps": 1000,
        "clip_sample": False,
        "set_alpha_to_one": False,
    }
    
    scheduler = DDIMScheduler(**scheduler_config)
    return scheduler

# Usage
custom_scheduler = create_custom_scheduler()
pipeline.scheduler = custom_scheduler
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```python
   # Solution: Enable memory optimizations
   pipeline.enable_attention_slicing()
   pipeline.enable_vae_slicing()
   pipeline.enable_model_cpu_offload()
   ```

2. **Slow Generation**
   ```python
   # Solution: Use faster scheduler
   from diffusers import DPMSolverMultistepScheduler
   pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
   ```

3. **Poor Quality Output**
   ```python
   # Solution: Adjust parameters
   image = pipeline(
       prompt=prompt,
       num_inference_steps=50,  # More steps
       guidance_scale=8.5,      # Higher guidance
       height=768,              # Higher resolution
       width=768
   )
   ```

4. **Model Loading Errors**
   ```python
   # Solution: Clear cache and retry
   from diffusers import clear_cache
   clear_cache()
   
   # Or use specific model revision
   pipeline = StableDiffusionPipeline.from_pretrained("model_name", revision="main")
   ```

### Performance Optimization

```python
# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    image = pipeline(prompt)

# Batch processing
def process_batch(prompts, batch_size=4):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        # Process batch
        results.extend(process_single_batch(batch))
    return results
```

## Examples

### Complete Image Generation System

```python
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import logging
from typing import List, Dict, Optional
import time

class CompleteDiffusionSystem:
    """Complete diffusion system for Video-OpusClip."""
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_pipeline()
    
    def setup_pipeline(self):
        """Setup optimized pipeline."""
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if self.device == "cuda" else None
        )
        
        # Optimize for performance
        if self.device == "cuda":
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_model_cpu_offload()
        
        self.pipeline = self.pipeline.to(self.device)
    
    def generate_viral_thumbnail(self, video_description: str):
        """Generate viral thumbnail for video."""
        
        prompt = f"Viral thumbnail for video: {video_description}, high quality, trending, eye-catching"
        
        start_time = time.time()
        
        with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
            image = self.pipeline(
                prompt=prompt,
                height=1280,
                width=720,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
        
        generation_time = time.time() - start_time
        
        return {
            "image": image,
            "generation_time": generation_time,
            "prompt": prompt
        }
    
    def generate_video_frames(self, prompt: str, num_frames: int = 30):
        """Generate video frames with temporal consistency."""
        
        frames = []
        start_time = time.time()
        
        for i in range(num_frames):
            # Add temporal context
            temporal_prompt = f"{prompt}, frame {i+1} of {num_frames}, temporal consistency"
            
            with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
                image = self.pipeline(
                    temporal_prompt,
                    height=512,
                    width=512,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
            
            frames.append(image)
        
        total_time = time.time() - start_time
        
        return {
            "frames": frames,
            "total_time": total_time,
            "avg_time_per_frame": total_time / num_frames
        }
    
    def batch_generate_images(self, prompts: List[str], batch_size: int = 4):
        """Generate multiple images in batches."""
        
        all_images = []
        start_time = time.time()
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
                batch_images = self.pipeline(
                    prompt=batch_prompts,
                    height=512,
                    width=512,
                    num_inference_steps=30,
                    guidance_scale=7.5
                ).images
            
            all_images.extend(batch_images)
        
        total_time = time.time() - start_time
        
        return {
            "images": all_images,
            "total_time": total_time,
            "avg_time_per_image": total_time / len(prompts)
        }

# Usage
system = CompleteDiffusionSystem()

# Generate thumbnail
thumbnail_result = system.generate_viral_thumbnail("Funny cat compilation")
thumbnail_result["image"].save("thumbnail.png")

# Generate video frames
video_result = system.generate_video_frames("A cat playing with a ball", num_frames=30)

# Batch generate images
prompts = ["Beautiful sunset", "Cat playing", "Dog running", "Bird flying"]
batch_result = system.batch_generate_images(prompts)
```

This comprehensive guide covers all aspects of using Diffusers in your Video-OpusClip system. The library provides powerful capabilities for image and video generation that are essential for creating engaging, viral content. 