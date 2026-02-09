"""
Gradio Integration for Interactive Music Generation Demos

Implements:
- User-friendly interfaces for model inference
- Real-time visualization
- Proper error handling and input validation
- Multiple model support
"""

import logging
from typing import Optional, Dict, Any, Tuple
import gradio as gr
import numpy as np
import torch
from pathlib import Path

from .music_generator import get_music_generator
from .diffusion_generator import DiffusionMusicGenerator

logger = logging.getLogger(__name__)


class MusicGenerationInterface:
    """
    Gradio interface for music generation.
    """
    
    def __init__(
        self,
        generator_type: str = "standard",
        model_name: Optional[str] = None
    ):
        """
        Initialize Gradio interface.
        
        Args:
            generator_type: Type of generator (standard, diffusion)
            model_name: Model name to use
        """
        self.generator_type = generator_type
        self.model_name = model_name
        self.generator = None
        self._load_generator()
    
    def _load_generator(self) -> None:
        """Load music generator."""
        try:
            if self.generator_type == "standard":
                self.generator = get_music_generator()
            elif self.generator_type == "diffusion":
                self.generator = DiffusionMusicGenerator(
                    model_name=self.model_name
                )
            else:
                raise ValueError(f"Unknown generator type: {self.generator_type}")
            
            logger.info("Generator loaded for Gradio interface")
        except Exception as e:
            logger.error(f"Error loading generator: {e}", exc_info=True)
            raise
    
    def generate_music(
        self,
        prompt: str,
        duration: int,
        temperature: float,
        guidance_scale: float,
        top_k: int,
        top_p: float
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Generate music from prompt.
        
        Args:
            prompt: Text prompt
            duration: Duration in seconds
            temperature: Sampling temperature
            guidance_scale: Guidance scale
            top_k: Top-k sampling
            top_p: Top-p sampling
            
        Returns:
            Tuple of (audio_array, error_message)
        """
        if not prompt or not prompt.strip():
            return None, "Please provide a text prompt"
        
        if duration <= 0 or duration > 300:
            return None, "Duration must be between 1 and 300 seconds"
        
        try:
            # Generate audio
            if self.generator_type == "standard":
                audio = self.generator.generate_from_text(
                    text=prompt,
                    duration=duration,
                    temperature=temperature,
                    guidance_scale=guidance_scale,
                    top_k=top_k,
                    top_p=top_p
                )
            else:
                audio = self.generator.generate(
                    text=prompt,
                    duration=duration,
                    temperature=temperature,
                    guidance_scale=guidance_scale
                )
            
            # Ensure audio is in correct format for Gradio
            if len(audio.shape) == 1:
                audio = audio.reshape(1, -1)
            
            # Normalize audio
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
            
            return audio, None
            
        except torch.cuda.OutOfMemoryError:
            return None, "GPU out of memory. Try reducing duration or batch size."
        except Exception as e:
            logger.error(f"Error generating music: {e}", exc_info=True)
            return None, f"Error: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Music Generation with AI") as interface:
            gr.Markdown(
                """
                # 🎵 Music Generation with AI
                
                Generate music from text descriptions using state-of-the-art AI models.
                
                ### How to use:
                1. Enter a text description of the music you want
                2. Adjust generation parameters (optional)
                3. Click "Generate Music"
                4. Listen to the generated audio
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(
                        label="Music Description",
                        placeholder="e.g., Upbeat electronic music with synthesizers and drums",
                        lines=3
                    )
                    
                    duration_slider = gr.Slider(
                        minimum=5,
                        maximum=60,
                        value=30,
                        step=5,
                        label="Duration (seconds)"
                    )
                    
                    with gr.Accordion("Advanced Parameters", open=False):
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Temperature"
                        )
                        
                        guidance_scale_slider = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=3.0,
                            step=0.5,
                            label="Guidance Scale"
                        )
                        
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=500,
                            value=250,
                            step=10,
                            label="Top-K"
                        )
                        
                        top_p_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                            label="Top-P (Nucleus Sampling)"
                        )
                    
                    generate_btn = gr.Button("Generate Music", variant="primary")
                
                with gr.Column(scale=1):
                    audio_output = gr.Audio(
                        label="Generated Music",
                        type="numpy"
                    )
                    
                    error_output = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                    
                    with gr.Accordion("Model Information", open=False):
                        model_info = gr.Markdown(
                            f"""
                            **Generator Type:** {self.generator_type}
                            **Model:** {self.model_name or 'Default'}
                            **Device:** {self.generator.device if hasattr(self.generator, 'device') else 'Unknown'}
                            """
                        )
            
            # Connect inputs to generation function
            generate_btn.click(
                fn=self.generate_music,
                inputs=[
                    prompt_input,
                    duration_slider,
                    temperature_slider,
                    guidance_scale_slider,
                    top_k_slider,
                    top_p_slider
                ],
                outputs=[audio_output, error_output]
            )
            
            # Example prompts
            gr.Markdown("### Example Prompts:")
            examples = gr.Examples(
                examples=[
                    ["Upbeat electronic music with synthesizers and drums"],
                    ["Calm acoustic guitar melody, folk style"],
                    ["Energetic rock song with electric guitar and bass"],
                    ["Jazz piano piece with smooth saxophone"],
                    ["Cinematic orchestral music, epic and dramatic"]
                ],
                inputs=prompt_input
            )
        
        return interface
    
    def launch(
        self,
        share: bool = False,
        server_name: str = "0.0.0.0",
        server_port: int = 7860
    ) -> None:
        """
        Launch Gradio interface.
        
        Args:
            share: Create public link
            server_name: Server hostname
            server_port: Server port
        """
        interface = self.create_interface()
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port
        )


def create_batch_generation_interface() -> gr.Blocks:
    """
    Create interface for batch music generation.
    
    Returns:
        Gradio Blocks interface
    """
    generator = get_music_generator()
    
    def generate_batch(
        prompts: str,
        duration: int
    ) -> Tuple[list, Optional[str]]:
        """
        Generate multiple music tracks.
        
        Args:
            prompts: Newline-separated prompts
            duration: Duration for each track
            
        Returns:
            Tuple of (audio_list, error_message)
        """
        prompt_list = [p.strip() for p in prompts.split('\n') if p.strip()]
        
        if not prompt_list:
            return [], "Please provide at least one prompt"
        
        if len(prompt_list) > 10:
            return [], "Maximum 10 prompts allowed"
        
        try:
            audio_list = []
            for prompt in prompt_list:
                audio = generator.generate_from_text(
                    text=prompt,
                    duration=duration
                )
                audio_list.append((32000, audio))  # (sample_rate, audio)
            
            return audio_list, None
            
        except Exception as e:
            logger.error(f"Error in batch generation: {e}", exc_info=True)
            return [], f"Error: {str(e)}"
    
    with gr.Blocks(title="Batch Music Generation") as interface:
        gr.Markdown("# Batch Music Generation")
        
        with gr.Row():
            with gr.Column():
                prompts_input = gr.Textbox(
                    label="Prompts (one per line)",
                    lines=10,
                    placeholder="Enter multiple prompts, one per line"
                )
                
                duration_slider = gr.Slider(
                    minimum=5,
                    maximum=60,
                    value=30,
                    step=5,
                    label="Duration (seconds)"
                )
                
                generate_btn = gr.Button("Generate All", variant="primary")
            
            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated Tracks",
                    type="numpy"
                )
                
                error_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
        
        generate_btn.click(
            fn=generate_batch,
            inputs=[prompts_input, duration_slider],
            outputs=[audio_output, error_output]
        )
    
    return interface


if __name__ == "__main__":
    # Example usage
    interface = MusicGenerationInterface()
    interface.launch(share=False, server_port=7860)













