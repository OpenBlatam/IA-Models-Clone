# Advanced Diffusers Library Integration Guide

This guide covers the advanced features of the Diffusers library that have been integrated into the ads feature system, including fast generation models, custom schedulers, and advanced optimization techniques.

## Table of Contents

1. [Overview](#overview)
2. [Fast Generation Models](#fast-generation-models)
3. [Advanced Schedulers](#advanced-schedulers)
4. [LoRA and Textual Inversion](#lora-and-textual-inversion)
5. [Memory Optimizations](#memory-optimizations)
6. [API Endpoints](#api-endpoints)
7. [Performance Tips](#performance-tips)
8. [Troubleshooting](#troubleshooting)

## Overview

The system now supports advanced Diffusers library features including:

- **LCM (Latent Consistency Models)**: Ultra-fast generation (4-8 steps)
- **TCD (Trajectory Consistency Distillation)**: Very fast generation (1-4 steps)
- **Advanced Schedulers**: DDIM, PNDM, Euler, DPM++, Heun, KDPM2, UniPC
- **LoRA Support**: Low-Rank Adaptation for model fine-tuning
- **Textual Inversion**: Custom text embeddings
- **Memory Optimizations**: xformers, attention slicing, VAE slicing
- **Model Variants**: FP16, sequential CPU offload

## Fast Generation Models

### LCM (Latent Consistency Model)

LCM models can generate high-quality images in just 4-8 steps, making them ideal for real-time applications.

```python
# Example API call
POST /api/diffusion/lcm
{
    "prompt": "A beautiful sunset over mountains",
    "negative_prompt": "blurry, low quality",
    "width": 512,
    "height": 512,
    "num_images": 1,
    "guidance_scale": 7.5,
    "num_inference_steps": 4,
    "model_name": "SimianLuo/LCM_Dreamshaper_v7"
}
```

**Available LCM Models:**
- `SimianLuo/LCM_Dreamshaper_v7` - General purpose
- `SimianLuo/LCM_SDXL` - SDXL version
- `latent-consistency/lcm-sdxl` - Official SDXL LCM

### TCD (Trajectory Consistency Distillation)

TCD models are even faster, generating images in 1-4 steps with good quality.

```python
# Example API call
POST /api/diffusion/tcd
{
    "prompt": "A futuristic cityscape at night",
    "negative_prompt": "cartoon, anime",
    "width": 512,
    "height": 512,
    "num_images": 1,
    "guidance_scale": 7.5,
    "num_inference_steps": 1,
    "model_name": "h1t/TCD-SD15"
}
```

**Available TCD Models:**
- `h1t/TCD-SD15` - Stable Diffusion 1.5 version
- `h1t/TCD-SDXL` - SDXL version

## Advanced Schedulers

The system supports multiple advanced schedulers optimized for different use cases:

### Scheduler Types

1. **DDIM**: Deterministic, good for interpolation
2. **PNDM**: Fast, good quality
3. **Euler**: Balanced speed and quality
4. **DPM++**: High quality, slower
5. **Heun**: High quality, good for img2img
6. **KDPM2**: Fast, good quality
7. **UniPC**: Fast, good for real-time
8. **LCM**: Ultra-fast (4-8 steps)
9. **TCD**: Very fast (1-4 steps)

### Using Custom Schedulers

```python
# Example API call
POST /api/diffusion/custom-scheduler
{
    "prompt": "A detailed portrait of a woman",
    "negative_prompt": "blurry, distorted",
    "width": 512,
    "height": 512,
    "num_images": 1,
    "guidance_scale": 7.5,
    "num_inference_steps": 50,
    "scheduler_type": "DPM++",
    "model_name": "runwayml/stable-diffusion-v1-5"
}
```

### Optimal Scheduler Selection

The system automatically selects optimal schedulers based on task and quality requirements:

```python
# Fast generation
scheduler = DiffusionSchedulerFactory.get_optimal_scheduler_for_task("text2img", "fast")
# Returns: "LCM"

# High quality
scheduler = DiffusionSchedulerFactory.get_optimal_scheduler_for_task("img2img", "high")
# Returns: "Heun"

# Balanced
scheduler = DiffusionSchedulerFactory.get_optimal_scheduler_for_task("text2img", "balanced")
# Returns: "Euler"
```

## LoRA and Textual Inversion

### LoRA (Low-Rank Adaptation)

LoRA allows fine-tuning models with minimal memory usage:

```python
# Example API call
POST /api/diffusion/advanced
{
    "prompt": "A photo of <lora:style:1.0> woman",
    "negative_prompt": "blurry, low quality",
    "width": 512,
    "height": 512,
    "num_images": 1,
    "guidance_scale": 7.5,
    "num_inference_steps": 50,
    "model_name": "runwayml/stable-diffusion-v1-5",
    "use_lora": true,
    "lora_path": "/path/to/lora/weights.safetensors"
}
```

### Textual Inversion

Textual Inversion allows creating custom text embeddings:

```python
# Example API call
POST /api/diffusion/advanced
{
    "prompt": "A photo of <my-concept> in a garden",
    "negative_prompt": "blurry, low quality",
    "width": 512,
    "height": 512,
    "num_images": 1,
    "guidance_scale": 7.5,
    "num_inference_steps": 50,
    "model_name": "runwayml/stable-diffusion-v1-5",
    "use_textual_inversion": true,
    "textual_inversion_path": "/path/to/textual_inversion.bin"
}
```

## Memory Optimizations

The system implements several memory optimization techniques:

### Automatic Optimizations

1. **FP16 Precision**: Uses half-precision when GPU is available
2. **Attention Slicing**: Reduces memory usage for large images
3. **VAE Slicing**: Processes VAE in chunks
4. **Model CPU Offload**: Offloads unused components to CPU
5. **Sequential CPU Offload**: For very large models (SDXL)
6. **xformers**: Memory-efficient attention (if available)

### Memory Usage by Model

| Model | Memory (FP16) | Memory (FP32) | Optimizations |
|-------|---------------|---------------|---------------|
| SD 1.5 | ~4GB | ~8GB | Standard |
| SD 2.1 | ~5GB | ~10GB | Standard |
| SDXL | ~8GB | ~16GB | Sequential CPU offload |
| LCM | ~4GB | ~8GB | Standard |
| TCD | ~4GB | ~8GB | Standard |

## API Endpoints

### Fast Generation Endpoints

```python
# LCM Generation
POST /api/diffusion/lcm
Content-Type: application/json

# TCD Generation
POST /api/diffusion/tcd
Content-Type: application/json

# Custom Scheduler
POST /api/diffusion/custom-scheduler
Content-Type: application/json

# Advanced Options
POST /api/diffusion/advanced
Content-Type: application/json
```

### Model Information

```python
# Get available models
GET /api/diffusion/models

# Response includes:
{
    "models": {
        "text_to_image": [...],
        "image_to_image": [...],
        "inpainting": [...],
        "controlnet": [...],
        "fast_generation": {
            "lcm": [...],
            "tcd": [...]
        },
        "schedulers": [...]
    }
}
```

## Performance Tips

### 1. Choose the Right Model

- **Real-time applications**: Use LCM or TCD
- **High quality**: Use SDXL with DPM++ scheduler
- **Balanced**: Use SD 1.5 with Euler scheduler

### 2. Optimize Parameters

```python
# Fast generation
{
    "num_inference_steps": 4,  # LCM
    "guidance_scale": 7.5
}

# High quality
{
    "num_inference_steps": 50,  # Standard
    "guidance_scale": 7.5
}

# Very fast
{
    "num_inference_steps": 1,  # TCD
    "guidance_scale": 7.5
}
```

### 3. Use Caching

The system automatically caches results for 1 hour. Use consistent prompts and parameters to benefit from caching.

### 4. Batch Processing

Use the batch endpoint for multiple generations:

```python
POST /api/diffusion/batch-generate
{
    "requests": [
        {
            "prompt": "Image 1",
            "model_name": "SimianLuo/LCM_Dreamshaper_v7"
        },
        {
            "prompt": "Image 2",
            "model_name": "h1t/TCD-SD15"
        }
    ]
}
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce image size (512x512 instead of 1024x1024)
   - Use FP16 models
   - Enable memory optimizations
   - Use smaller models (SD 1.5 instead of SDXL)

2. **Slow Generation**
   - Use LCM or TCD models
   - Reduce inference steps
   - Use faster schedulers (Euler, PNDM)

3. **Poor Quality**
   - Increase inference steps
   - Use higher quality models (SDXL)
   - Use DPM++ or Heun schedulers
   - Improve prompts

4. **Model Loading Issues**
   - Check internet connection for model downloads
   - Ensure sufficient disk space
   - Verify model paths

### Performance Monitoring

```python
# Get generation statistics
GET /api/diffusion/stats

# Response includes:
{
    "total_cache_entries": 150,
    "cache_hits": 45,
    "cache_misses": 105,
    "cache_hit_rate": 0.3,
    "loaded_models": 3,
    "device": "cuda:0"
}
```

### Health Check

```python
# Check service health
GET /api/diffusion/health

# Response includes:
{
    "status": "healthy",
    "service": "diffusion",
    "loaded_models": 3,
    "device": "cuda:0",
    "cache_entries": 150
}
```

## Advanced Configuration

### Environment Variables

```bash
# Model cache directory
export HF_HOME=/path/to/model/cache

# Enable xformers
export XFORMERS_FORCE_DISABLE_TRITON=0

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Custom Model Loading

```python
# Load custom model
pipeline = StableDiffusionPipeline.from_pretrained(
    "/path/to/custom/model",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
)
```

## Integration Examples

### Python Client

```python
import requests
import base64
from PIL import Image
import io

def generate_image_with_lcm(prompt: str):
    response = requests.post(
        "http://localhost:8000/api/diffusion/lcm",
        json={
            "prompt": prompt,
            "num_inference_steps": 4,
            "model_name": "SimianLuo/LCM_Dreamshaper_v7"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        # Decode base64 images
        images = []
        for img_data in result['images']:
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)
        return images
    else:
        raise Exception(f"Generation failed: {response.text}")

# Usage
images = generate_image_with_lcm("A beautiful landscape")
```

### JavaScript Client

```javascript
async function generateImageWithLCM(prompt) {
    const response = await fetch('/api/diffusion/lcm', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            prompt: prompt,
            num_inference_steps: 4,
            model_name: 'SimianLuo/LCM_Dreamshaper_v7'
        })
    });
    
    if (response.ok) {
        const result = await response.json();
        return result.images.map(imgData => {
            // Convert base64 to image
            return `data:image/png;base64,${imgData}`;
        });
    } else {
        throw new Error('Generation failed');
    }
}

// Usage
const images = await generateImageWithLCM('A beautiful landscape');
```

## Conclusion

The advanced Diffusers integration provides:

1. **Ultra-fast generation** with LCM and TCD models
2. **Flexible scheduling** with multiple advanced schedulers
3. **Memory efficiency** with automatic optimizations
4. **Customization** with LoRA and Textual Inversion
5. **Production readiness** with caching, monitoring, and error handling

This system is optimized for both development and production use, providing the best balance of speed, quality, and resource efficiency for image generation tasks. 