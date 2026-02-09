"""
Gradio Interfaces for AI Services

Creates interactive web interfaces for:
- Text generation
- Image generation (diffusion)
- Sentiment analysis
- Content moderation
- Semantic search
"""

import logging
from typing import Optional, List, Tuple
import gradio as gr
import torch
from PIL import Image

from .embedding_service import EmbeddingService
from .sentiment_service import SentimentService
from .moderation_service import ModerationService
from .text_generation_service import TextGenerationService
from .diffusion_service import DiffusionService

logger = logging.getLogger(__name__)


class GradioInterface:
    """
    Gradio interface for AI services
    
    Creates interactive demos for all AI capabilities.
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        sentiment_service: Optional[SentimentService] = None,
        moderation_service: Optional[ModerationService] = None,
        text_generation_service: Optional[TextGenerationService] = None,
        diffusion_service: Optional[DiffusionService] = None
    ):
        """
        Initialize Gradio interface
        
        Args:
            embedding_service: Optional embedding service
            sentiment_service: Optional sentiment service
            moderation_service: Optional moderation service
            text_generation_service: Optional text generation service
            diffusion_service: Optional diffusion service
        """
        self.embedding_service = embedding_service
        self.sentiment_service = sentiment_service
        self.moderation_service = moderation_service
        self.text_generation_service = text_generation_service
        self.diffusion_service = diffusion_service
    
    def create_text_generation_interface(self) -> gr.Blocks:
        """Create interface for text generation"""
        
        def generate_text(
            prompt: str,
            max_length: int,
            temperature: float,
            top_p: float,
            num_sequences: int
        ) -> str:
            if not self.text_generation_service:
                return "Text generation service not available"
            
            try:
                result = self.text_generation_service.generate_text(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_sequences
                )
                
                generated = result.get("generated_text", "")
                if isinstance(generated, list):
                    return "\n\n---\n\n".join(generated)
                return generated
            except Exception as e:
                return f"Error: {str(e)}"
        
        with gr.Blocks(title="Text Generation") as interface:
            gr.Markdown("# Text Generation with LLMs")
            gr.Markdown("Generate text using language models")
            
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=5
                    )
                    
                    with gr.Row():
                        max_length = gr.Slider(
                            label="Max Length",
                            minimum=10,
                            maximum=500,
                            value=200,
                            step=10
                        )
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1
                        )
                    
                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-p",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.05
                        )
                        num_sequences = gr.Slider(
                            label="Number of Sequences",
                            minimum=1,
                            maximum=5,
                            value=1,
                            step=1
                        )
                    
                    generate_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    output = gr.Textbox(
                        label="Generated Text",
                        lines=10
                    )
            
            generate_btn.click(
                fn=generate_text,
                inputs=[prompt_input, max_length, temperature, top_p, num_sequences],
                outputs=output
            )
        
        return interface
    
    def create_sentiment_interface(self) -> gr.Blocks:
        """Create interface for sentiment analysis"""
        
        def analyze_sentiment(text: str) -> Tuple[str, float]:
            if not self.sentiment_service:
                return "Service not available", 0.0
            
            try:
                result = self.sentiment_service.analyze_sentiment(text)
                label = result.get("label", "unknown")
                score = result.get("score", 0.0)
                return label, score
            except Exception as e:
                return f"Error: {str(e)}", 0.0
        
        with gr.Blocks(title="Sentiment Analysis") as interface:
            gr.Markdown("# Sentiment Analysis")
            gr.Markdown("Analyze the sentiment of text")
            
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Text",
                        placeholder="Enter text to analyze...",
                        lines=5
                    )
                    analyze_btn = gr.Button("Analyze", variant="primary")
                
                with gr.Column():
                    label_output = gr.Textbox(label="Sentiment")
                    score_output = gr.Slider(
                        label="Confidence Score",
                        minimum=0.0,
                        maximum=1.0,
                        interactive=False
                    )
            
            analyze_btn.click(
                fn=analyze_sentiment,
                inputs=text_input,
                outputs=[label_output, score_output]
            )
        
        return interface
    
    def create_moderation_interface(self) -> gr.Blocks:
        """Create interface for content moderation"""
        
        def moderate_content(text: str) -> Tuple[bool, float, str]:
            if not self.moderation_service:
                return False, 0.0, "Service not available"
            
            try:
                result = self.moderation_service.moderate_content(text)
                is_toxic = result.get("is_toxic", False)
                score = result.get("toxicity_score", 0.0)
                flags = result.get("flags", [])
                
                flags_str = ", ".join([f"{f['label']}: {f['score']:.2f}" for f in flags]) if flags else "None"
                
                return is_toxic, score, flags_str
            except Exception as e:
                return False, 0.0, f"Error: {str(e)}"
        
        with gr.Blocks(title="Content Moderation") as interface:
            gr.Markdown("# Content Moderation")
            gr.Markdown("Check text for toxic or inappropriate content")
            
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Text to Check",
                        placeholder="Enter text to moderate...",
                        lines=5
                    )
                    moderate_btn = gr.Button("Moderate", variant="primary")
                
                with gr.Column():
                    is_toxic_output = gr.Checkbox(label="Is Toxic", interactive=False)
                    score_output = gr.Slider(
                        label="Toxicity Score",
                        minimum=0.0,
                        maximum=1.0,
                        interactive=False
                    )
                    flags_output = gr.Textbox(label="Flags", lines=3)
            
            moderate_btn.click(
                fn=moderate_content,
                inputs=text_input,
                outputs=[is_toxic_output, score_output, flags_output]
            )
        
        return interface
    
    def create_diffusion_interface(self) -> gr.Blocks:
        """Create interface for image generation"""
        
        def generate_image(
            prompt: str,
            negative_prompt: str,
            num_steps: int,
            guidance_scale: float,
            seed: int
        ) -> Image.Image:
            if not self.diffusion_service:
                return None
            
            try:
                images = self.diffusion_service.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    seed=seed if seed >= 0 else None
                )
                return images[0] if images else None
            except Exception as e:
                logger.error(f"Error generating image: {e}")
                return None
        
        with gr.Blocks(title="Image Generation") as interface:
            gr.Markdown("# Image Generation with Stable Diffusion")
            gr.Markdown("Generate images from text prompts")
            
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="A beautiful landscape...",
                        lines=3
                    )
                    negative_prompt_input = gr.Textbox(
                        label="Negative Prompt (optional)",
                        placeholder="blurry, low quality...",
                        lines=2
                    )
                    
                    with gr.Row():
                        num_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=5
                        )
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5
                        )
                    
                    seed_input = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1
                    )
                    
                    generate_btn = gr.Button("Generate Image", variant="primary")
                
                with gr.Column():
                    image_output = gr.Image(label="Generated Image")
            
            generate_btn.click(
                fn=generate_image,
                inputs=[prompt_input, negative_prompt_input, num_steps, guidance_scale, seed_input],
                outputs=image_output
            )
        
        return interface
    
    def create_combined_interface(self, port: int = 7860, share: bool = False) -> None:
        """
        Create combined interface with all services
        
        Args:
            port: Port to run the interface on
            share: Whether to create a public link
        """
        interfaces = []
        
        if self.text_generation_service:
            interfaces.append(("Text Generation", self.create_text_generation_interface()))
        
        if self.sentiment_service:
            interfaces.append(("Sentiment Analysis", self.create_sentiment_interface()))
        
        if self.moderation_service:
            interfaces.append(("Content Moderation", self.create_moderation_interface()))
        
        if self.diffusion_service:
            interfaces.append(("Image Generation", self.create_diffusion_interface()))
        
        if not interfaces:
            logger.warning("No services available for Gradio interface")
            return
        
        # Create tabbed interface
        with gr.Blocks(title="Lovable Community AI Services") as demo:
            gr.Markdown("# Lovable Community AI Services")
            gr.Markdown("Interactive demos for AI capabilities")
            
            with gr.Tabs():
                for name, interface in interfaces:
                    with gr.Tab(name):
                        # Extract the content from the interface
                        interface.render()
        
        demo.launch(server_port=port, share=share)
        logger.info(f"Gradio interface launched on port {port}")















