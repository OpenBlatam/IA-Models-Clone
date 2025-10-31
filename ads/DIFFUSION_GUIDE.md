# Advanced Diffusion Models Guide

This guide covers the comprehensive diffusion models implementation for ads generation and image manipulation, including text-to-image, image-to-image, inpainting, and ControlNet capabilities.

## Overview

The diffusion system provides:
- **Text-to-Image Generation**: Create images from text descriptions
- **Image-to-Image Transformation**: Transform existing images with prompts
- **Image Inpainting**: Fill masked areas with generated content
- **ControlNet Integration**: Precise control over image generation
- **Batch Processing**: Efficient batch generation for multiple requests
- **Caching & Optimization**: Performance optimization and result caching

## Architecture

### Core Components

1. **DiffusionModelManager**: Manages multiple diffusion pipelines
2. **ImageProcessor**: Handles image loading, processing, and manipulation
3. **DiffusionService**: Main service orchestrating all generation tasks
4. **DiffusionSchedulerFactory**: Creates and configures schedulers
5. **GenerationParams**: Structured parameters for image generation

### Supported Models

- **Stable Diffusion v1.5**: Primary text-to-image model
- **Stable Diffusion v2.1**: Enhanced quality model
- **Stable Diffusion Inpainting**: Specialized inpainting model
- **ControlNet Models**: Canny edges, depth, pose, segmentation

## Text-to-Image Generation

### Basic Generation

```python
from onyx.server.features.ads.diffusion_service import DiffusionService, GenerationParams

diffusion_service = DiffusionService()

# Create generation parameters
params = GenerationParams(
    prompt="A beautiful advertisement for a premium coffee brand, professional photography",
    negative_prompt="blurry, low quality, distorted",
    width=512,
    height=512,
    num_images=1,
    guidance_scale=7.5,
    num_inference_steps=50,
    seed=42
)

# Generate images
images = await diffusion_service.generate_text_to_image(
    params=params,
    model_name="runwayml/stable-diffusion-v1-5"
)
```

### Advanced Prompts

```python
# Product advertisement prompt
product_prompt = """
Professional advertisement for {product_name}:
- High-quality product photography
- Clean, modern composition
- Professional lighting
- Brand colors: {brand_colors}
- Target audience: {target_audience}
- Call-to-action: {cta_text}
- Style: {style_description}
"""

params = GenerationParams(
    prompt=product_prompt.format(
        product_name="Premium Coffee",
        brand_colors="gold and brown",
        target_audience="young professionals",
        cta_text="Experience the difference",
        style_description="minimalist, elegant"
    ),
    guidance_scale=8.0,
    num_inference_steps=75
)
```

### Style Presets

```python
# Different style presets for ads
style_presets = {
    "professional": "professional photography, clean composition, business aesthetic",
    "lifestyle": "lifestyle photography, natural lighting, authentic moments",
    "luxury": "luxury photography, premium materials, sophisticated composition",
    "minimalist": "minimalist design, clean lines, simple composition",
    "vintage": "vintage aesthetic, retro styling, classic composition"
}

params = GenerationParams(
    prompt=f"Product advertisement: {base_prompt}",
    style_preset=style_presets["professional"]
)
```

## Image-to-Image Transformation

### Basic Transformation

```python
# Load initial image
init_image = Image.open("product_photo.jpg")

# Create transformation parameters
params = GenerationParams(
    prompt="Transform this into a professional advertisement with modern styling",
    negative_prompt="blurry, amateur, low quality",
    width=512,
    height=512,
    guidance_scale=7.5,
    num_inference_steps=50
)

# Generate transformed images
transformed_images = await diffusion_service.generate_image_to_image(
    init_image=init_image,
    params=params,
    model_name="runwayml/stable-diffusion-v1-5"
)
```

### Style Transfer

```python
# Apply different styles to the same image
styles = {
    "modern": "modern, contemporary, sleek design",
    "vintage": "vintage, retro, classic styling",
    "luxury": "luxury, premium, sophisticated",
    "minimalist": "minimalist, clean, simple"
}

for style_name, style_prompt in styles.items():
    params = GenerationParams(
        prompt=f"Transform into {style_prompt} advertisement",
        guidance_scale=8.0,
        num_inference_steps=60
    )
    
    styled_images = await diffusion_service.generate_image_to_image(
        init_image=init_image,
        params=params
    )
```

## Image Inpainting

### Basic Inpainting

```python
# Load image and create mask
image = Image.open("advertisement.jpg")
mask = diffusion_service.image_processor.create_mask(image, mask_type="center")

# Create inpainting parameters
params = GenerationParams(
    prompt="Add a call-to-action button in the center",
    negative_prompt="blurry, distorted, low quality",
    width=512,
    height=512,
    guidance_scale=7.5,
    num_inference_steps=50
)

# Inpaint the image
inpainted_images = await diffusion_service.inpaint_image(
    image=image,
    mask=mask,
    params=params,
    model_name="runwayml/stable-diffusion-inpainting"
)
```

### Advanced Masking

```python
# Create custom mask
def create_custom_mask(image, region):
    """Create custom mask for specific region."""
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    
    if region == "top_left":
        draw.rectangle([0, 0, image.width//2, image.height//2], fill=255)
    elif region == "bottom_right":
        draw.rectangle([image.width//2, image.height//2, image.width, image.height], fill=255)
    
    return mask

# Use custom mask for inpainting
custom_mask = create_custom_mask(image, "top_left")
params = GenerationParams(
    prompt="Add brand logo in the top left corner",
    guidance_scale=8.0
)

inpainted = await diffusion_service.inpaint_image(
    image=image,
    mask=custom_mask,
    params=params
)
```

## ControlNet Integration

### Canny Edge Control

```python
# Load control image
control_image = Image.open("product_sketch.jpg")

# Apply Canny edge detection
canny_image = diffusion_service.image_processor.apply_canny_edge_detection(
    control_image,
    low_threshold=100,
    high_threshold=200
)

# Generate with ControlNet
params = GenerationParams(
    prompt="Professional product advertisement following the edge structure",
    negative_prompt="blurry, distorted",
    guidance_scale=8.0,
    num_inference_steps=60
)

controlled_images = await diffusion_service.generate_with_controlnet(
    control_image=canny_image,
    params=params,
    controlnet_type="canny"
)
```

### Depth Control

```python
# Apply depth estimation
depth_image = diffusion_service.image_processor.apply_depth_estimation(control_image)

# Generate with depth control
params = GenerationParams(
    prompt="Create advertisement with proper depth and perspective",
    guidance_scale=7.5
)

depth_controlled_images = await diffusion_service.generate_with_controlnet(
    control_image=depth_image,
    params=params,
    controlnet_type="depth"
)
```

## Batch Processing

### Multiple Generations

```python
# Create batch of generation requests
batch_requests = [
    GenerationParams(
        prompt="Professional coffee advertisement",
        style_preset="modern"
    ),
    GenerationParams(
        prompt="Luxury coffee advertisement",
        style_preset="luxury"
    ),
    GenerationParams(
        prompt="Minimalist coffee advertisement",
        style_preset="minimalist"
    )
]

# Process batch
batch_results = []
for params in batch_requests:
    images = await diffusion_service.generate_text_to_image(params)
    batch_results.extend(images)
```

### Parallel Processing

```python
import asyncio

async def generate_multiple_variations(base_prompt, variations):
    """Generate multiple variations of the same prompt."""
    tasks = []
    
    for i, variation in enumerate(variations):
        params = GenerationParams(
            prompt=f"{base_prompt} {variation}",
            seed=i * 100  # Different seed for each variation
        )
        task = diffusion_service.generate_text_to_image(params)
        tasks.append(task)
    
    # Run all generations in parallel
    results = await asyncio.gather(*tasks)
    return results

# Generate variations
variations = [
    "with warm lighting",
    "with cool lighting",
    "with natural lighting",
    "with dramatic lighting"
]

all_variations = await generate_multiple_variations(
    "Professional coffee advertisement",
    variations
)
```

## API Endpoints

### Text-to-Image Generation

```bash
POST /ads/v2/diffusion/text-to-image
{
    "prompt": "Professional advertisement for premium coffee",
    "negative_prompt": "blurry, low quality",
    "width": 512,
    "height": 512,
    "num_images": 1,
    "guidance_scale": 7.5,
    "num_inference_steps": 50,
    "seed": 42,
    "model_name": "runwayml/stable-diffusion-v1-5"
}
```

### Image-to-Image Transformation

```bash
POST /ads/v2/diffusion/image-to-image
Content-Type: multipart/form-data

{
    "prompt": "Transform into professional advertisement",
    "negative_prompt": "blurry, amateur",
    "width": 512,
    "height": 512,
    "guidance_scale": 7.5,
    "strength": 0.8
}

init_image: [file upload]
```

### Image Inpainting

```bash
POST /ads/v2/diffusion/inpaint
Content-Type: multipart/form-data

{
    "prompt": "Add call-to-action button",
    "negative_prompt": "blurry, distorted",
    "width": 512,
    "height": 512,
    "mask_type": "center"
}

image: [file upload]
mask: [optional file upload]
```

### ControlNet Generation

```bash
POST /ads/v2/diffusion/controlnet
Content-Type: multipart/form-data

{
    "prompt": "Professional advertisement following structure",
    "negative_prompt": "blurry, distorted",
    "width": 512,
    "height": 512,
    "controlnet_type": "canny"
}

control_image: [file upload]
```

### Batch Generation

```bash
POST /ads/v2/diffusion/batch-generate
{
    "requests": [
        {
            "prompt": "Modern coffee advertisement",
            "guidance_scale": 7.5
        },
        {
            "prompt": "Luxury coffee advertisement",
            "guidance_scale": 8.0
        }
    ],
    "priority": "normal"
}
```

## Performance Optimization

### Caching Strategy

- **Result Caching**: Cache generated images based on parameters
- **Model Caching**: Keep models loaded in memory
- **Pipeline Caching**: Cache pipeline configurations
- **Redis Integration**: Distributed caching for production

### Memory Management

```python
# Optimize memory usage
def optimize_memory_usage():
    """Optimize memory usage for diffusion models."""
    if torch.cuda.is_available():
        # Enable memory efficient attention
        torch.backends.cuda.enable_attention_slicing()
        
        # Enable VAE slicing
        torch.backends.cuda.enable_vae_slicing()
        
        # Clear cache periodically
        torch.cuda.empty_cache()

# Use in generation
async def generate_with_optimization(params):
    optimize_memory_usage()
    return await diffusion_service.generate_text_to_image(params)
```

### Batch Size Optimization

```python
# Dynamic batch size based on memory
def get_optimal_batch_size():
    """Get optimal batch size based on available memory."""
    if torch.cuda.is_available():
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if memory_gb >= 24:
            return 4
        elif memory_gb >= 16:
            return 2
        else:
            return 1
    return 1
```

## Monitoring and Analytics

### Generation Statistics

```python
# Get generation statistics
stats = await diffusion_service.get_generation_stats()

print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Loaded models: {stats['loaded_models']}")
print(f"Total cache entries: {stats['total_cache_entries']}")
```

### Performance Metrics

- **Generation Time**: Time per image generation
- **Cache Hit Rate**: Percentage of cached results
- **Memory Usage**: GPU/CPU memory utilization
- **Model Loading Time**: Time to load models
- **Error Rate**: Failed generation attempts

### Health Monitoring

```bash
GET /ads/v2/diffusion/health
```

Response:
```json
{
    "status": "healthy",
    "service": "diffusion",
    "loaded_models": 3,
    "device": "cuda",
    "cache_entries": 150,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## Best Practices

### Prompt Engineering

1. **Be Specific**: Use detailed, specific prompts
2. **Include Style**: Specify artistic style and composition
3. **Use Negative Prompts**: Avoid unwanted elements
4. **Iterate**: Test different prompt variations
5. **Combine Elements**: Mix product, style, and context

### Quality Optimization

1. **Higher Steps**: Use 50-75 steps for better quality
2. **Optimal Guidance**: Use 7.5-8.5 for balanced results
3. **Consistent Seeds**: Use fixed seeds for reproducible results
4. **Proper Sizing**: Use 512x512 or 768x768 for best results
5. **Post-processing**: Apply additional image processing if needed

### Production Considerations

1. **Model Selection**: Choose appropriate models for use case
2. **Caching**: Implement proper caching strategies
3. **Error Handling**: Handle generation failures gracefully
4. **Rate Limiting**: Implement rate limiting for API endpoints
5. **Monitoring**: Monitor performance and resource usage

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or image resolution
2. **Slow Generation**: Enable attention slicing and VAE slicing
3. **Poor Quality**: Increase inference steps and guidance scale
4. **Model Loading Errors**: Check model availability and dependencies
5. **Cache Issues**: Clear cache or increase cache size

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('onyx.server.features.ads.diffusion_service').setLevel(logging.DEBUG)

# Monitor memory usage
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### Performance Profiling

```python
import time

async def profile_generation():
    start_time = time.time()
    images = await diffusion_service.generate_text_to_image(params)
    generation_time = time.time() - start_time
    
    print(f"Generation took {generation_time:.2f} seconds")
    print(f"Images generated: {len(images)}")
    print(f"Time per image: {generation_time / len(images):.2f} seconds")
```

## Future Enhancements

1. **Multi-Model Support**: Support for additional diffusion models
2. **Advanced ControlNet**: More ControlNet types and configurations
3. **Style Transfer**: Advanced style transfer capabilities
4. **Real-time Generation**: Stream generation progress
5. **Custom Training**: Fine-tune models for specific domains
6. **Advanced Caching**: Intelligent caching with similarity search
7. **Distributed Processing**: Multi-GPU and distributed generation

This comprehensive diffusion system provides the foundation for advanced image generation and manipulation in ads creation, with support for multiple generation modes, optimization strategies, and production-ready features. 