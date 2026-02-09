"""
Gradio Demo for Diffusion Models
=================================

Interactive interface for image generation with diffusion models.
"""

import gradio as gr
from typing import Optional
import logging

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

try:
    from ..models.diffusion_models import DiffusionModel
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False
    DiffusionModel = None


def create_diffusion_demo(
    model: DiffusionModel,
    title: str = "Diffusion Model Demo",
    description: str = "Generate images from text prompts using diffusion models"
) -> gr.Blocks:
    """
    Create Gradio demo for diffusion model.
    
    Args:
        model: Diffusion model instance
        title: Demo title
        description: Demo description
    
    Returns:
        Gradio Blocks interface
    """
    if not DIFFUSION_AVAILABLE:
        raise ImportError("Diffusion models not available")
    
    def generate_image(
        prompt: str,
        negative_prompt: str,
        num_steps: int,
        guidance_scale: float,
        width: int,
        height: int
    ):
        """Generate image from prompt."""
        try:
            if not prompt.strip():
                return None, "Please enter a prompt"
            
            result = model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            )
            
            images = result["images"]
            if images:
                return images[0], f"Generated in {result['num_inference_steps']} steps"
            return None, "Generation failed"
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return None, f"Error: {str(e)}"
    
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\n{description}")
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="A beautiful landscape with mountains...",
                    lines=3
                )
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt (optional)",
                    placeholder="blurry, low quality...",
                    lines=2
                )
                
                with gr.Row():
                    num_steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Inference Steps"
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale"
                    )
                
                with gr.Row():
                    width = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Width"
                    )
                    height = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Height"
                    )
                
                generate_btn = gr.Button("Generate Image", variant="primary")
            
            with gr.Column():
                image_output = gr.Image(label="Generated Image")
                status_text = gr.Textbox(label="Status", interactive=False)
        
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                negative_prompt_input,
                num_steps,
                guidance_scale,
                width,
                height
            ],
            outputs=[image_output, status_text]
        )
        
        gr.Examples(
            examples=[
                ["A futuristic city at sunset, cyberpunk style, neon lights"],
                ["A serene forest with sunlight filtering through trees"],
                ["An astronaut floating in space, detailed, 4k"]
            ],
            inputs=[prompt_input]
        )
    
    return demo



