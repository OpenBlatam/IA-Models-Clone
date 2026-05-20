"""
Gradio Utilities
Enhanced Gradio integration with error handling and validation.
"""

import gradio as gr
from typing import Callable, Optional, Dict, Any, List
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def gradio_error_handler(func: Callable) -> Callable:
    """
    Decorator for Gradio functions with error handling.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Gradio function error: {e}", exc_info=True)
            error_message = f"Error: {str(e)}"
            return error_message
    
    return wrapper


def create_text_generation_interface(
    generate_fn: Callable,
    title: str = "Text Generation",
    description: str = "Generate text using language models",
    examples: Optional[List[str]] = None,
    **kwargs
) -> gr.Interface:
    """
    Create a text generation Gradio interface with validation.
    
    Args:
        generate_fn: Function that takes prompt and returns generated text
        title: Interface title
        description: Interface description
        examples: Example prompts
        **kwargs: Additional Gradio parameters
        
    Returns:
        Gradio Interface
    """
    @gradio_error_handler
    def generate_wrapper(prompt: str, max_length: int, temperature: float, **params):
        """Wrapper with validation."""
        if not prompt or not prompt.strip():
            return "Error: Prompt cannot be empty"
        
        if max_length < 1 or max_length > 2048:
            return "Error: max_length must be between 1 and 2048"
        
        if temperature < 0 or temperature > 2:
            return "Error: temperature must be between 0 and 2"
        
        return generate_fn(prompt, max_length=max_length, temperature=temperature, **params)
    
    interface = gr.Interface(
        fn=generate_wrapper,
        inputs=[
            gr.Textbox(
                label="Prompt",
                placeholder="Enter your text prompt here...",
                lines=5,
            ),
            gr.Slider(
                minimum=10,
                maximum=512,
                value=100,
                step=10,
                label="Max Length",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Temperature",
            ),
        ],
        outputs=gr.Textbox(label="Generated Text", lines=10),
        title=title,
        description=description,
        examples=examples or [],
        **kwargs
    )
    
    return interface


def create_image_generation_interface(
    generate_fn: Callable,
    title: str = "Image Generation",
    description: str = "Generate images from text prompts",
    examples: Optional[List[str]] = None,
    **kwargs
) -> gr.Interface:
    """
    Create an image generation Gradio interface.
    
    Args:
        generate_fn: Function that takes prompt and returns PIL Image
        title: Interface title
        description: Interface description
        examples: Example prompts
        **kwargs: Additional Gradio parameters
        
    Returns:
        Gradio Interface
    """
    @gradio_error_handler
    def generate_wrapper(prompt: str, negative_prompt: str, num_steps: int, **params):
        """Wrapper with validation."""
        if not prompt or not prompt.strip():
            return None
        
        if num_steps < 1 or num_steps > 100:
            return None
        
        return generate_fn(
            prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_steps,
            **params
        )
    
    interface = gr.Interface(
        fn=generate_wrapper,
        inputs=[
            gr.Textbox(
                label="Prompt",
                placeholder="A beautiful landscape...",
                lines=3,
            ),
            gr.Textbox(
                label="Negative Prompt (optional)",
                placeholder="blurry, low quality",
                lines=2,
            ),
            gr.Slider(
                minimum=10,
                maximum=100,
                value=50,
                step=5,
                label="Inference Steps",
            ),
        ],
        outputs=gr.Image(label="Generated Image"),
        title=title,
        description=description,
        examples=examples or [],
        **kwargs
    )
    
    return interface


def create_chat_interface(
    chat_fn: Callable,
    title: str = "Chat Interface",
    description: str = "Chat with a language model",
    **kwargs
) -> gr.ChatInterface:
    """
    Create a chat interface with Gradio.
    
    Args:
        chat_fn: Function that takes message and history, returns response
        title: Interface title
        description: Interface description
        **kwargs: Additional Gradio parameters
        
    Returns:
        Gradio ChatInterface
    """
    @gradio_error_handler
    def chat_wrapper(message: str, history: List[List[str]]):
        """Wrapper with validation."""
        if not message or not message.strip():
            return ""
        
        response = chat_fn(message, history)
        return response
    
    interface = gr.ChatInterface(
        fn=chat_wrapper,
        title=title,
        description=description,
        **kwargs
    )
    
    return interface


def create_batch_interface(
    batch_fn: Callable,
    title: str = "Batch Processing",
    description: str = "Process multiple inputs",
    **kwargs
) -> gr.Interface:
    """
    Create a batch processing interface.
    
    Args:
        batch_fn: Function that takes list of inputs and returns list of outputs
        title: Interface title
        description: Interface description
        **kwargs: Additional Gradio parameters
        
    Returns:
        Gradio Interface
    """
    @gradio_error_handler
    def batch_wrapper(inputs: str):
        """Wrapper with validation."""
        if not inputs:
            return "Error: No inputs provided"
        
        # Parse inputs (one per line)
        input_list = [line.strip() for line in inputs.split("\n") if line.strip()]
        
        if not input_list:
            return "Error: No valid inputs"
        
        if len(input_list) > 100:
            return "Error: Maximum 100 inputs allowed"
        
        results = batch_fn(input_list)
        
        # Format results
        if isinstance(results, list):
            return "\n".join(str(r) for r in results)
        return str(results)
    
    interface = gr.Interface(
        fn=batch_wrapper,
        inputs=gr.Textbox(
            label="Inputs (one per line)",
            lines=10,
            placeholder="Input 1\nInput 2\nInput 3",
        ),
        outputs=gr.Textbox(
            label="Results",
            lines=10,
        ),
        title=title,
        description=description,
        **kwargs
    )
    
    return interface



