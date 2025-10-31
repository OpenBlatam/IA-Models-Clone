import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
import logging
from typing import Tuple, Optional
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import enhanced logging
try:
    from .logging_config import EnhancedLogger, ErrorMessages, log_error_with_context
    logger = EnhancedLogger("gradio_demo")
except ImportError:
    logger = logging.getLogger(__name__)

# Import error factories
try:
    from .error_factories import (
        error_factory, context_manager,
        create_validation_error, create_processing_error, create_inference_error,
        create_resource_error, create_error_context
    )
except ImportError:
    error_factory = None
    context_manager = None

class ImageGenerationError(Exception):
    """Custom exception for image generation errors."""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_prompt(prompt: str) -> None:
    """Validate the input prompt with comprehensive guard clauses."""
    # GUARD CLAUSE: Empty or None prompt
    if not prompt:
        raise ValidationError("Prompt cannot be empty")
    
    # GUARD CLAUSE: Wrong data type
    if not isinstance(prompt, str):
        raise ValidationError("Prompt must be a string")
    
    # GUARD CLAUSE: Empty after stripping
    if len(prompt.strip()) == 0:
        raise ValidationError("Prompt cannot be empty")
    
    # GUARD CLAUSE: Too long
    if len(prompt) > 1000:
        raise ValidationError("Prompt too long (max 1000 characters)")
    
    # HAPPY PATH: Prompt is valid (implicit return)

def validate_parameters(guidance_scale: float, num_inference_steps: int) -> None:
    """Validate generation parameters with comprehensive guard clauses."""
    # GUARD CLAUSE: Wrong guidance scale type
    if not isinstance(guidance_scale, (int, float)):
        raise ValidationError("Guidance scale must be a number")
    
    # GUARD CLAUSE: Guidance scale out of range
    if guidance_scale < 1.0 or guidance_scale > 20.0:
        raise ValidationError("Guidance scale must be between 1.0 and 20.0")
    
    # GUARD CLAUSE: Wrong inference steps type
    if not isinstance(num_inference_steps, int):
        raise ValidationError("Inference steps must be an integer")
    
    # GUARD CLAUSE: Inference steps out of range
    if num_inference_steps < 10 or num_inference_steps > 100:
        raise ValidationError("Inference steps must be between 10 and 100")
    
    # HAPPY PATH: Parameters are valid (implicit return)

# Initialize pipeline
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    
    logger.info("Stable Diffusion pipeline loaded successfully")
    
except Exception as e:
    logger.error(f"Failed to load Stable Diffusion pipeline: {e}")
    pipe = None

def generate_image(prompt: str, guidance_scale: float = 7.5, num_inference_steps: int = 30) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Generate image with comprehensive guard clauses and validation.
    Enhanced with proper error logging, user-friendly error messages, and error factories.
    
    Args:
        prompt: Text prompt for image generation
        guidance_scale: Guidance scale for generation (1.0-20.0)
        num_inference_steps: Number of inference steps (10-100)
        
    Returns:
        Tuple of (image, error_message)
    """
    # Set operation context for error tracking
    if context_manager:
        context_manager.set_operation_context("image_generation", "gradio_demo", "generate_image")
        context_manager.start_timing()
    
    # GUARD CLAUSE: Check for None/empty prompt
    if prompt is None:
        context = create_error_context(operation="image_generation", component="gradio_demo", step="prompt_validation")
        error = create_validation_error("prompt", None, "Prompt cannot be None", context)
        logger.warning(error.message, details={"field": "prompt", "value": None})
        return None, error.message
    
    if not prompt or not prompt.strip():
        context = create_error_context(operation="image_generation", component="gradio_demo", step="prompt_validation")
        error = create_validation_error("prompt", prompt, "Prompt cannot be empty", context)
        logger.warning(error.message, details={"field": "prompt", "value": prompt})
        return None, error.message
    
    # GUARD CLAUSE: Validate data types
    if not isinstance(prompt, str):
        context = create_error_context(operation="image_generation", component="gradio_demo", step="type_validation")
        error = create_validation_error("prompt", type(prompt).__name__, "Prompt must be a string", context)
        logger.warning(error.message, details={"field": "prompt", "value": type(prompt).__name__})
        return None, error.message
    
    if not isinstance(guidance_scale, (int, float)):
        context = create_error_context(operation="image_generation", component="gradio_demo", step="type_validation")
        error = create_validation_error("guidance_scale", type(guidance_scale).__name__, "Guidance scale must be a number", context)
        logger.warning(error.message, details={"field": "guidance_scale", "value": type(guidance_scale).__name__})
        return None, error.message
    
    if not isinstance(num_inference_steps, int):
        context = create_error_context(operation="image_generation", component="gradio_demo", step="type_validation")
        error = create_validation_error("num_inference_steps", type(num_inference_steps).__name__, "Inference steps must be an integer", context)
        logger.warning(error.message, details={"field": "num_inference_steps", "value": type(num_inference_steps).__name__})
        return None, error.message
    
    # GUARD CLAUSE: Validate parameter ranges
    if guidance_scale < 1.0 or guidance_scale > 20.0:
        context = create_error_context(
            operation="image_generation", 
            component="gradio_demo", 
            step="parameter_validation",
            guidance_scale=guidance_scale,
            range="1.0-20.0"
        )
        error = create_validation_error("guidance_scale", guidance_scale, "Guidance scale must be between 1.0 and 20.0", context)
        logger.warning(error.message, details={"field": "guidance_scale", "value": guidance_scale, "range": "1.0-20.0"})
        return None, error.message
    
    if num_inference_steps < 10 or num_inference_steps > 100:
        context = create_error_context(
            operation="image_generation", 
            component="gradio_demo", 
            step="parameter_validation",
            num_inference_steps=num_inference_steps,
            range="10-100"
        )
        error = create_validation_error("num_inference_steps", num_inference_steps, "Inference steps must be between 10 and 100", context)
        logger.warning(error.message, details={"field": "num_inference_steps", "value": num_inference_steps, "range": "10-100"})
        return None, error.message
    
    # GUARD CLAUSE: Check prompt length
    if len(prompt) > 1000:
        context = create_error_context(
            operation="image_generation", 
            component="gradio_demo", 
            step="prompt_validation",
            prompt_length=len(prompt),
            max_length=1000
        )
        error = create_validation_error("prompt", len(prompt), "Prompt too long (max 1000 characters)", context)
        logger.warning(error.message, details={"field": "prompt", "length": len(prompt), "max_length": 1000})
        return None, error.message
    
    # GUARD CLAUSE: Check if pipeline is available
    if pipe is None:
        context = create_error_context(operation="image_generation", component="gradio_demo", step="pipeline_check")
        error = create_processing_error("Image generation pipeline not available", "pipeline_initialization", context)
        logger.error(error.message, details={"component": "stable_diffusion_pipeline"})
        return None, error.message
    
    # GUARD CLAUSE: Check GPU memory if using CUDA
    if device == "cuda":
        try:
            if torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > 0.95:
                context = create_error_context(
                    operation="image_generation", 
                    component="gradio_demo", 
                    step="resource_check",
                    resource="gpu_memory",
                    usage="critical"
                )
                error = create_resource_error("GPU memory usage critical", "gpu_memory", "critical", "available", context)
                logger.warning(error.message, details={"resource": "gpu_memory", "usage": "critical"})
                return None, error.message
        except Exception as e:
            context = create_error_context(operation="image_generation", component="gradio_demo", step="resource_check", resource="gpu_memory")
            logger.warning("Could not check GPU memory", error=e, details={"resource": "gpu_memory"})
    
    # HAPPY PATH: Generate image and return result
    try:
        # Generate image
        with torch.autocast(device) if device == "cuda" else torch.no_grad():
            result = pipe(
                prompt=prompt.strip(),
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
            # GUARD CLAUSE: Check result immediately
            if not result or not hasattr(result, 'images'):
                context = create_error_context(operation="image_generation", component="gradio_demo", step="result_validation")
                error = create_processing_error("Invalid result from pipeline", "image_generation", context)
                logger.error(error.message, details={"operation": "image_generation"})
                return None, error.message
            
            if not result.images:
                context = create_error_context(operation="image_generation", component="gradio_demo", step="result_validation")
                error = create_processing_error("No images generated", "image_generation", context)
                logger.error(error.message, details={"operation": "image_generation"})
                return None, error.message
            
            image = result.images[0]
            
            # GUARD CLAUSE: Validate generated image immediately
            if image is None:
                context = create_error_context(operation="image_generation", component="gradio_demo", step="image_validation")
                error = create_processing_error("Generated image is None", "image_generation", context)
                logger.error(error.message, details={"operation": "image_generation"})
                return None, error.message
            
            if not hasattr(image, 'size') or image.size[0] == 0 or image.size[1] == 0:
                context = create_error_context(
                    operation="image_generation", 
                    component="gradio_demo", 
                    step="image_validation",
                    image_size=getattr(image, 'size', None)
                )
                error = create_processing_error("Generated image has invalid dimensions", "image_generation", context)
                logger.error(error.message, details={"operation": "image_generation", "image_size": getattr(image, 'size', None)})
                return None, error.message
            
            # Validate image is actually a PIL Image
            if not isinstance(image, Image.Image):
                context = create_error_context(
                    operation="image_generation", 
                    component="gradio_demo", 
                    step="image_validation",
                    image_type=type(image).__name__
                )
                error = create_processing_error("Generated image is not a valid image type", "image_generation", context)
                logger.error(error.message, details={"operation": "image_generation", "image_type": type(image).__name__})
                return None, error.message
        
        # End error tracking
        if context_manager:
            context_manager.end_timing()
        
        logger.info("Image generated successfully", details={"prompt_length": len(prompt), "prompt_preview": prompt[:50]})
        return image, None
        
    except torch.cuda.OutOfMemoryError as e:
        context = create_error_context(
            operation="image_generation", 
            component="gradio_demo", 
            step="gpu_processing",
            resource="gpu_memory",
            suggestion="Try reducing image size or batch size"
        )
        error = create_resource_error("GPU memory insufficient", "gpu_memory", "insufficient", "required", context)
        logger.error(error.message, error=e, details={"resource": "gpu_memory", "suggestion": "Try reducing image size or batch size"})
        return None, error.message
    except ValidationError as e:
        # Enrich error with context
        if context_manager:
            enrich_error_with_context(e, context_manager.get_context())
        logger.warning("Validation error during image generation", error=e, details={"operation": "image_generation"})
        return None, str(e)
    except ImageGenerationError as e:
        context = create_error_context(operation="image_generation", component="gradio_demo", step="generation")
        error = create_processing_error("Image generation failed", "image_generation", context)
        logger.error(error.message, error=e, details={"operation": "image_generation"})
        return None, error.message
    except Exception as e:
        context = create_error_context(operation="image_generation", component="gradio_demo", step="unexpected_error")
        error = create_processing_error("Unexpected error during image generation", "image_generation", context)
        logger.error(error.message, error=e, details={"operation": "image_generation"})
        return None, error.message

# Interfaz Gradio mejorada
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Describe la imagen..."),
        gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="Guidance Scale"),
        gr.Slider(10, 50, value=30, step=1, label="Inference Steps")
    ],
    outputs=[
        gr.Image(type="pil", label="Imagen generada"),
        gr.Textbox(label="Mensaje de error", visible=True)
    ],
    title="Stable Diffusion Demo",
    description="Genera imágenes a partir de texto usando Diffusers y Stable Diffusion."
)

if __name__ == "__main__":
    demo.launch() 