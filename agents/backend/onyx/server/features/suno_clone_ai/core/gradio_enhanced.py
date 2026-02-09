"""
Enhanced Gradio Interface for Music Generation

Implements:
- User-friendly interfaces for model inference
- Real-time visualization
- Proper error handling and input validation
- Multiple model support
- Experiment tracking integration
"""

import logging
from typing import Optional, Dict, Any, Tuple, List
import gradio as gr
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime

from .music_generator import get_music_generator
from .diffusion_generator import DiffusionMusicGenerator

logger = logging.getLogger(__name__)


class EnhancedMusicGenerationInterface:
    """
    Enhanced Gradio interface for music generation with advanced features.
    """
    
    def __init__(
        self,
        generator_type: str = "standard",
        model_name: Optional[str] = None,
        enable_visualization: bool = True,
        enable_analysis: bool = True
    ):
        """
        Initialize enhanced Gradio interface.
        
        Args:
            generator_type: Type of generator (standard, diffusion)
            model_name: Model name to use
            enable_visualization: Enable audio visualization
            enable_analysis: Enable audio analysis
        """
        self.generator_type = generator_type
        self.model_name = model_name
        self.enable_visualization = enable_visualization
        self.enable_analysis = enable_analysis
        self.generator = None
        self.generation_history = []
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
            
            logger.info("Generator loaded for enhanced Gradio interface")
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
        top_p: float,
        num_inference_steps: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[str], Optional[Dict]]:
        """
        Generate music from prompt with enhanced error handling.
        
        Args:
            prompt: Text prompt
            duration: Duration in seconds
            temperature: Sampling temperature
            guidance_scale: Guidance scale
            top_k: Top-k sampling
            top_p: Top-p sampling
            num_inference_steps: Number of inference steps (for diffusion)
            
        Returns:
            Tuple of (audio_array, error_message, analysis_dict)
        """
        # Input validation
        if not prompt or not prompt.strip():
            return None, "Please provide a text prompt", None
        
        if duration <= 0 or duration > 300:
            return None, "Duration must be between 1 and 300 seconds", None
        
        if temperature < 0.1 or temperature > 2.0:
            return None, "Temperature must be between 0.1 and 2.0", None
        
        try:
            # Generate audio
            start_time = datetime.now()
            
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
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Ensure audio is in correct format for Gradio
            if len(audio.shape) == 1:
                audio = audio.reshape(1, -1)
            
            # Normalize audio
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
            
            # Audio analysis
            analysis = None
            if self.enable_analysis:
                try:
                    analysis = self._analyze_audio(audio)
                    analysis['generation_time'] = generation_time
                except Exception as e:
                    logger.warning(f"Error in audio analysis: {e}")
            
            # Store in history
            self.generation_history.append({
                'prompt': prompt,
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'generation_time': generation_time
            })
            
            return audio, None, analysis
            
        except torch.cuda.OutOfMemoryError:
            return None, "GPU out of memory. Try reducing duration or batch size.", None
        except ValueError as e:
            return None, f"Validation error: {str(e)}", None
        except Exception as e:
            logger.error(f"Error generating music: {e}", exc_info=True)
            return None, f"Error: {str(e)}", None
    
    def _analyze_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze generated audio.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        try:
            import librosa
            
            # Ensure mono
            if len(audio.shape) > 1:
                audio_mono = np.mean(audio, axis=0)
            else:
                audio_mono = audio.flatten()
            
            # Basic statistics
            analysis['duration'] = len(audio_mono) / 32000  # Assuming 32kHz
            analysis['rms'] = float(np.sqrt(np.mean(audio_mono ** 2)))
            analysis['peak'] = float(np.max(np.abs(audio_mono)))
            
            # Spectral features
            stft = librosa.stft(audio_mono, n_fft=2048)
            magnitude = np.abs(stft)
            
            # Spectral centroid
            analysis['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(
                S=magnitude
            )))
            
            # Zero crossing rate
            analysis['zcr'] = float(np.mean(librosa.feature.zero_crossing_rate(audio_mono)))
            
        except ImportError:
            logger.warning("librosa not available for detailed analysis")
            analysis = {
                'duration': len(audio.flatten()) / 32000,
                'rms': float(np.sqrt(np.mean(audio.flatten() ** 2)))
            }
        except Exception as e:
            logger.warning(f"Error in audio analysis: {e}")
        
        return analysis
    
    def create_interface(self) -> gr.Blocks:
        """
        Create enhanced Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="Enhanced Music Generation with AI",
            theme=gr.themes.Soft()
        ) as interface:
            gr.Markdown(
                """
                # 🎵 Enhanced Music Generation with AI
                
                Generate high-quality music from text descriptions using state-of-the-art AI models.
                
                ### Features:
                - Multiple generation models
                - Real-time audio visualization
                - Audio analysis and metrics
                - Advanced parameter control
                - Generation history tracking
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(
                        label="Music Description",
                        placeholder="e.g., Upbeat electronic music with synthesizers and drums",
                        lines=4,
                        max_lines=10
                    )
                    
                    with gr.Row():
                        duration_slider = gr.Slider(
                            minimum=5,
                            maximum=120,
                            value=30,
                            step=5,
                            label="Duration (seconds)"
                        )
                    
                    with gr.Accordion("Generation Parameters", open=False):
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Temperature",
                            info="Controls randomness. Higher = more creative"
                        )
                        
                        guidance_scale_slider = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=3.0,
                            step=0.5,
                            label="Guidance Scale",
                            info="How closely to follow the prompt"
                        )
                        
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=500,
                            value=250,
                            step=10,
                            label="Top-K",
                            info="Number of top tokens to consider"
                        )
                        
                        top_p_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                            label="Top-P (Nucleus Sampling)",
                            info="Cumulative probability threshold"
                        )
                    
                    if self.generator_type == "diffusion":
                        num_steps_slider = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Inference Steps",
                            info="Number of diffusion steps"
                        )
                    else:
                        num_steps_slider = gr.Slider(
                            visible=False,
                            value=50
                        )
                    
                    generate_btn = gr.Button(
                        "Generate Music",
                        variant="primary",
                        size="lg"
                    )
                    
                    clear_btn = gr.Button("Clear", variant="secondary")
                
                with gr.Column(scale=1):
                    audio_output = gr.Audio(
                        label="Generated Music",
                        type="numpy",
                        format="wav"
                    )
                    
                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=2
                    )
                    
                    if self.enable_analysis:
                        analysis_output = gr.JSON(
                            label="Audio Analysis",
                            visible=True
                        )
                    else:
                        analysis_output = gr.JSON(visible=False)
                    
                    with gr.Accordion("Model Information", open=False):
                        model_info = gr.Markdown(
                            f"""
                            **Generator Type:** {self.generator_type}
                            **Model:** {self.model_name or 'Default'}
                            **Device:** {self.generator.device if hasattr(self.generator, 'device') else 'Unknown'}
                            **Mixed Precision:** {getattr(self.generator, 'use_mixed_precision', False)}
                            """
                        )
                    
                    with gr.Accordion("Generation History", open=False):
                        history_output = gr.JSON(
                            value=self.generation_history[-10:] if self.generation_history else [],
                            label="Recent Generations"
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
                    top_p_slider,
                    num_steps_slider
                ],
                outputs=[audio_output, status_output, analysis_output]
            ).then(
                fn=lambda: self.generation_history[-10:] if self.generation_history else [],
                outputs=history_output
            )
            
            clear_btn.click(
                fn=lambda: (None, "", None, []),
                outputs=[audio_output, status_output, analysis_output, history_output]
            )
            
            # Example prompts
            gr.Markdown("### Example Prompts:")
            examples = gr.Examples(
                examples=[
                    ["Upbeat electronic music with synthesizers and drums, energetic and modern"],
                    ["Calm acoustic guitar melody, folk style, peaceful and relaxing"],
                    ["Energetic rock song with electric guitar and bass, powerful and driving"],
                    ["Jazz piano piece with smooth saxophone, sophisticated and smooth"],
                    ["Cinematic orchestral music, epic and dramatic, with strings and brass"],
                    ["Ambient electronic soundscape, atmospheric and ethereal"],
                    ["Latin dance music with percussion and brass, festive and rhythmic"]
                ],
                inputs=prompt_input
            )
        
        return interface
    
    def launch(
        self,
        share: bool = False,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        show_error: bool = True
    ) -> None:
        """
        Launch enhanced Gradio interface.
        
        Args:
            share: Create public link
            server_name: Server hostname
            server_port: Server port
            show_error: Show error details
        """
        interface = self.create_interface()
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=show_error
        )


def create_batch_generation_interface() -> gr.Blocks:
    """
    Create enhanced interface for batch music generation.
    
    Returns:
        Gradio Blocks interface
    """
    generator = get_music_generator()
    
    def generate_batch(
        prompts: str,
        duration: int,
        temperature: float = 1.0
    ) -> Tuple[List, Optional[str]]:
        """
        Generate multiple music tracks.
        
        Args:
            prompts: Newline-separated prompts
            duration: Duration for each track
            temperature: Sampling temperature
            
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
            for i, prompt in enumerate(prompt_list):
                try:
                    audio = generator.generate_from_text(
                        text=prompt,
                        duration=duration,
                        temperature=temperature
                    )
                    audio_list.append((32000, audio))  # (sample_rate, audio)
                except Exception as e:
                    logger.error(f"Error generating for prompt {i+1}: {e}")
                    continue
            
            if not audio_list:
                return [], "Failed to generate any audio"
            
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
                
                with gr.Row():
                    duration_slider = gr.Slider(
                        minimum=5,
                        maximum=60,
                        value=30,
                        step=5,
                        label="Duration (seconds)"
                    )
                    
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature"
                    )
                
                generate_btn = gr.Button("Generate All", variant="primary")
            
            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated Tracks",
                    type="numpy"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
        
        generate_btn.click(
            fn=generate_batch,
            inputs=[prompts_input, duration_slider, temperature_slider],
            outputs=[audio_output, status_output]
        )
    
    return interface


if __name__ == "__main__":
    # Example usage
    interface = EnhancedMusicGenerationInterface()
    interface.launch(share=False, server_port=7860)



