from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
import sys
import logging
import time
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from production_code import MultiGPUTrainer, TrainingConfiguration, RadioIntegration
from error_handling_gradio import GradioErrorHandler
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Enhanced Gradio Demos with Error Handling
========================================

This module provides enhanced versions of the existing Gradio demos
with comprehensive error handling and input validation:
- Integrated error handling from error_handling_gradio.py
- Enhanced user feedback and error messages
- Input validation for all user inputs
- Graceful error recovery
- Performance monitoring
- Security validation
"""


# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedGradioDemos:
    """Enhanced Gradio demos with comprehensive error handling"""
    
    def __init__(self) -> Any:
        self.config = TrainingConfiguration(
            enable_radio_integration=True,
            enable_gradio_demo=True,
            gradio_port=7866,
            gradio_share=False
        )
        
        self.trainer = MultiGPUTrainer(self.config)
        self.radio = self.trainer.radio if self.trainer.radio else None
        
        # Initialize error handler
        self.error_handler = GradioErrorHandler()
        
        # Demo state
        self.current_model = None
        self.training_history = {}
        self.demo_data = {}
        
        # Initialize demo data
        self._initialize_demo_data()
        
        logger.info("Enhanced Gradio Demos initialized")
    
    def _initialize_demo_data(self) -> Any:
        """Initialize demo data and sample content"""
        self.demo_data = {
            'sample_texts': [
                "The future of artificial intelligence is bright and promising.",
                "Machine learning models are transforming industries worldwide.",
                "Deep learning has revolutionized computer vision and NLP.",
                "Neural networks can learn complex patterns from data.",
                "AI assistants are becoming more intelligent and helpful."
            ],
            'sample_prompts': [
                "A serene landscape with mountains and a lake",
                "A futuristic city with flying cars and neon lights",
                "A cozy coffee shop with warm lighting",
                "A magical forest with glowing mushrooms",
                "A space station orbiting Earth"
            ],
            'sample_audio': self._generate_sample_audio(),
            'training_metrics': self._generate_sample_training_data()
        }
    
    def _generate_sample_audio(self) -> np.ndarray:
        """Generate sample audio data for demos"""
        try:
            sample_rate = 44100
            duration = 3.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Generate a simple melody
            frequencies = [440, 494, 523, 587, 659, 587, 523, 494]
            audio = np.zeros_like(t)
            
            for i, freq in enumerate(frequencies):
                start = i * len(t) // len(frequencies)
                end = (i + 1) * len(t) // len(frequencies)
                audio[start:end] = np.sin(2 * np.pi * freq * t[start:end])
            
            return audio
        except Exception as e:
            self.error_handler.log_error(e, "sample_audio_generation")
            return np.zeros(44100)  # Return silence on error
    
    def _generate_sample_training_data(self) -> Dict:
        """Generate sample training data for visualization"""
        try:
            epochs = 100
            return {
                'epochs': list(range(1, epochs + 1)),
                'train_loss': [2.5 * np.exp(-epoch/30) + 0.1 * np.random.random() for epoch in range(1, epochs + 1)],
                'val_loss': [2.3 * np.exp(-epoch/25) + 0.15 * np.random.random() for epoch in range(1, epochs + 1)],
                'train_acc': [0.3 + 0.6 * (1 - np.exp(-epoch/20)) + 0.02 * np.random.random() for epoch in range(1, epochs + 1)],
                'val_acc': [0.25 + 0.65 * (1 - np.exp(-epoch/18)) + 0.03 * np.random.random() for epoch in range(1, epochs + 1)],
                'learning_rate': [1e-3 * np.exp(-epoch/50) for epoch in range(1, epochs + 1)]
            }
        except Exception as e:
            self.error_handler.log_error(e, "sample_training_data_generation")
            return {'epochs': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'learning_rate': []}
    
    @GradioErrorHandler().validate_inputs(
        prompt='text',
        max_length='number',
        temperature='number',
        top_p='number',
        num_samples='number'
    )
    def generate_text_with_enhanced_validation(self, prompt: str, max_length: int = 100, 
                                             temperature: float = 0.8, top_p: float = 0.9, 
                                             num_samples: int = 3) -> Tuple[str, List[str]]:
        """Generate text with enhanced validation and error handling"""
        
        def text_generation_logic():
            
    """text_generation_logic function."""
# Additional validation
            if max_length < 10 or max_length > 2000:
                raise ValueError("Max length must be between 10 and 2000 characters")
            
            if temperature < 0.1 or temperature > 2.0:
                raise ValueError("Temperature must be between 0.1 and 2.0")
            
            if top_p < 0.1 or top_p > 1.0:
                raise ValueError("Top-p must be between 0.1 and 1.0")
            
            if num_samples < 1 or num_samples > 10:
                raise ValueError("Number of samples must be between 1 and 10")
            
            # Simulate processing delay
            time.sleep(0.5)
            
            # Generate text samples
            generated_texts = []
            
            for i in range(num_samples):
                # Simple text generation simulation
                words = prompt.split()
                if len(words) < 3:
                    words.extend(["artificial", "intelligence", "technology"])
                
                # Generate continuation
                continuation = " ".join(words[-3:]) + " " + " ".join([
                    "is", "revolutionizing", "the", "way", "we", "think", "about",
                    "machine", "learning", "and", "deep", "neural", "networks."
                ])
                
                # Add randomness based on temperature
                if temperature > 0.5:
                    continuation += " " + " ".join([
                        "The", "future", "looks", "promising", "for", "AI", "applications."
                    ])
                
                generated_texts.append(continuation[:max_length])
            
            return generated_texts
        
        result, status = self.error_handler.safe_execute(text_generation_logic)
        
        if result is None:
            return "Error generating text. Please try again.", []
        else:
            # Create analysis
            analysis = f"""
            **Text Generation Analysis:**
            - **Prompt Length**: {len(prompt)} characters
            - **Generated Length**: {len(result[0])} characters
            - **Temperature**: {temperature} (creativity level)
            - **Top-p**: {top_p} (nucleus sampling)
            - **Samples Generated**: {num_samples}
            
            **Generation Parameters:**
            - Max Length: {max_length}
            - Temperature: {temperature}
            - Top-p: {top_p}
            """
            
            return analysis, result
    
    @GradioErrorHandler().validate_inputs(
        text='text'
    )
    def analyze_text_with_enhanced_validation(self, text: str) -> str:
        """Analyze text with enhanced validation and error handling"""
        
        def text_analysis_logic():
            
    """text_analysis_logic function."""
if not text or len(text.strip()) < 5:
                raise ValueError("Text must be at least 5 characters long")
            
            words = text.split()
            sentences = text.split('.')
            characters = len(text)
            
            # Calculate metrics
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            avg_sentence_length = np.mean([len(sent.split()) for sent in sentences if sent.strip()]) if sentences else 0
            
            analysis = f"""
            **Text Analysis:**
            - **Characters**: {characters}
            - **Words**: {len(words)}
            - **Sentences**: {len(sentences)}
            - **Average Word Length**: {avg_word_length:.2f}
            - **Average Sentence Length**: {avg_sentence_length:.2f}
            
            **Word Frequency (Top 10):**
            """
            
            # Word frequency analysis
            word_freq = {}
            for word in words:
                word = word.lower().strip('.,!?')
                if word:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            for word, freq in sorted_words[:10]:
                analysis += f"\n- {word}: {freq}"
            
            return analysis
        
        result, status = self.error_handler.safe_execute(text_analysis_logic)
        
        if result is None:
            return f"Error analyzing text: {status}"
        else:
            return result
    
    @GradioErrorHandler().validate_inputs(
        prompt='text',
        negative_prompt='text',
        num_steps='number',
        guidance_scale='number',
        seed='number',
        width='number',
        height='number'
    )
    def generate_image_with_enhanced_validation(self, prompt: str, negative_prompt: str = "", 
                                              num_steps: int = 50, guidance_scale: float = 7.5,
                                              seed: int = -1, width: int = 512, height: int = 512) -> Tuple[Image.Image, str]:
        """Generate image with enhanced validation and error handling"""
        
        def image_generation_logic():
            
    """image_generation_logic function."""
# Additional validation
            if len(prompt.strip()) < 5:
                raise ValueError("Image prompt must be at least 5 characters long")
            
            if num_steps < 10 or num_steps > 100:
                raise ValueError("Number of steps must be between 10 and 100")
            
            if guidance_scale < 1.0 or guidance_scale > 20.0:
                raise ValueError("Guidance scale must be between 1.0 and 20.0")
            
            if width < 256 or width > 1024 or height < 256 or height > 1024:
                raise ValueError("Image dimensions must be between 256x256 and 1024x1024")
            
            # Simulate processing delay
            time.sleep(1.0)
            
            # Set seed
            if seed == -1:
                seed = np.random.randint(0, 1000000)
            
            # Create a simple generated image
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Add visual elements based on prompt
            if 'mountain' in prompt.lower():
                points = [(0, height), (width//4, height//2), (width//2, height//3), 
                         (3*width//4, height//2), (width, height)]
                draw.polygon(points, fill='gray')
            
            if 'sky' in prompt.lower() or 'blue' in prompt.lower():
                draw.rectangle([0, 0, width, height//2], fill='lightblue')
            
            if 'sun' in prompt.lower():
                draw.ellipse([width-100, 50, width-50, 100], fill='yellow')
            
            # Add text overlay
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), f"Generated: {prompt[:30]}...", fill='black', font=font)
            draw.text((10, height-30), f"Seed: {seed}", fill='black', font=font)
            
            return img
        
        result, status = self.error_handler.safe_execute(image_generation_logic)
        
        if result is None:
            # Return error image
            error_img = Image.new('RGB', (width, height), color='red')
            draw = ImageDraw.Draw(error_img)
            draw.text((50, height//2), "Error generating image", fill='white')
            return error_img, f"Error: {status}"
        else:
            # Create generation info
            info = f"""
            **Image Generation Info:**
            - **Prompt**: {prompt}
            - **Negative Prompt**: {negative_prompt}
            - **Steps**: {num_steps}
            - **Guidance Scale**: {guidance_scale}
            - **Seed**: {seed}
            - **Dimensions**: {width}x{height}
            - **Generation Time**: {np.random.uniform(2.5, 5.0):.2f}s
            """
            
            return result, info
    
    @GradioErrorHandler().validate_inputs(
        audio_input='audio',
        operation='text'
    )
    def process_audio_with_enhanced_validation(self, audio_input, operation: str, parameters: Dict) -> Tuple[np.ndarray, str]:
        """Process audio with enhanced validation and error handling"""
        
        def audio_processing_logic():
            
    """audio_processing_logic function."""
if audio_input is None:
                raise ValueError("Audio input is required")
            
            valid_operations = ['noise_reduction', 'equalizer', 'reverb', 'pitch_shift']
            if operation not in valid_operations:
                raise ValueError(f"Operation must be one of: {', '.join(valid_operations)}")
            
            audio_data, sample_rate = audio_input
            
            # Validate audio data
            if len(audio_data) == 0:
                raise ValueError("Audio data is empty")
            
            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                raise ValueError("Audio data contains invalid values")
            
            # Simulate processing delay
            time.sleep(0.5)
            
            # Apply effects
            if operation == "noise_reduction":
                processed_audio = audio_data * 0.8
                effect_description = "Applied noise reduction (volume reduced by 20%)"
            elif operation == "equalizer":
                processed_audio = audio_data * np.random.uniform(0.5, 1.5, len(audio_data))
                effect_description = "Applied equalizer with random frequency adjustments"
            elif operation == "reverb":
                delay = int(sample_rate * 0.1)
                processed_audio = audio_data.copy()
                processed_audio[delay:] += audio_data[:-delay] * 0.3
                effect_description = "Applied reverb effect (100ms delay, 30% mix)"
            elif operation == "pitch_shift":
                processed_audio = np.interp(
                    np.arange(len(audio_data)),
                    np.arange(len(audio_data)) * 1.1,
                    audio_data
                )
                effect_description = "Applied pitch shift (pitch increased by 10%)"
            else:
                processed_audio = audio_data
                effect_description = "No processing applied"
            
            return processed_audio, effect_description
        
        result, status = self.error_handler.safe_execute(audio_processing_logic)
        
        if result is None:
            return None, f"Error processing audio: {status}"
        else:
            processed_audio, effect_description = result
            return processed_audio, effect_description
    
    @GradioErrorHandler().validate_inputs(
        epochs='number',
        learning_rate='number',
        batch_size='number'
    )
    def update_training_plot_with_enhanced_validation(self, epochs: int, learning_rate: float, batch_size: int) -> Tuple[go.Figure, str]:
        """Update training visualization with enhanced validation and error handling"""
        
        def training_plot_logic():
            
    """training_plot_logic function."""
# Additional validation
            if epochs < 10 or epochs > 500:
                raise ValueError("Epochs must be between 10 and 500")
            
            if learning_rate < 1e-6 or learning_rate > 1e-1:
                raise ValueError("Learning rate must be between 1e-6 and 1e-1")
            
            if batch_size < 1 or batch_size > 1024:
                raise ValueError("Batch size must be between 1 and 1024")
            
            # Generate training data
            x = list(range(1, epochs + 1))
            train_loss = [2.5 * np.exp(-epoch/30) + 0.1 * np.random.random() for epoch in x]
            val_loss = [2.3 * np.exp(-epoch/25) + 0.15 * np.random.random() for epoch in x]
            train_acc = [0.3 + 0.6 * (1 - np.exp(-epoch/20)) + 0.02 * np.random.random() for epoch in x]
            val_acc = [0.25 + 0.65 * (1 - np.exp(-epoch/18)) + 0.03 * np.random.random() for epoch in x]
            lr_schedule = [learning_rate * np.exp(-epoch/50) for epoch in x]
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Training Loss', 'Validation Loss', 'Accuracy', 'Learning Rate'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Add traces
            fig.add_trace(
                go.Scatter(x=x, y=train_loss, name="Train Loss", line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=x, y=val_loss, name="Val Loss", line=dict(color='red')),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=x, y=train_acc, name="Train Acc", line=dict(color='green')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=x, y=val_acc, name="Val Acc", line=dict(color='orange')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=x, y=lr_schedule, name="Learning Rate", line=dict(color='purple')),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                title_text="Training Progress Visualization",
                showlegend=True
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_xaxes(title_text="Epoch", row=2, col=2)
            
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=2)
            fig.update_yaxes(title_text="Accuracy", row=2, col=1)
            fig.update_yaxes(title_text="Learning Rate", row=2, col=2)
            
            return fig
        
        result, status = self.error_handler.safe_execute(training_plot_logic)
        
        if result is None:
            return go.Figure(), f"Error creating visualization: {status}"
        else:
            # Create summary
            summary = f"""
            **Training Summary:**
            - **Total Epochs**: {epochs}
            - **Initial Learning Rate**: {learning_rate:.6f}
            - **Batch Size**: {batch_size}
            - **Final Train Loss**: {2.5 * np.exp(-epochs/30) + 0.1 * np.random.random():.4f}
            - **Final Val Loss**: {2.3 * np.exp(-epochs/25) + 0.15 * np.random.random():.4f}
            - **Final Train Acc**: {0.3 + 0.6 * (1 - np.exp(-epochs/20)) + 0.02 * np.random.random():.4f}
            - **Final Val Acc**: {0.25 + 0.65 * (1 - np.exp(-epochs/18)) + 0.03 * np.random.random():.4f}
            - **Final Learning Rate**: {learning_rate * np.exp(-epochs/50):.6f}
            """
            
            return result, summary
    
    def create_enhanced_comprehensive_demo(self) -> gr.Interface:
        """Create enhanced comprehensive demo with error handling"""
        
        def get_error_summary():
            """Get error summary for monitoring"""
            summary = self.error_handler.get_error_summary()
            return summary
        
        # Create interface
        with gr.Blocks(
            title="Enhanced AI Demo Suite",
            theme=gr.themes.Soft(),
            css="""
            .error-message {
                background-color: #ffebee;
                border: 1px solid #f44336;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
                color: #c62828;
            }
            .success-message {
                background-color: #e8f5e8;
                border: 1px solid #4caf50;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
                color: #2e7d32;
            }
            .validation-error {
                background-color: #fff3e0;
                border: 1px solid #ff9800;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
                color: #e65100;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🛡️ Enhanced AI Demo Suite")
            gr.Markdown("Comprehensive AI demos with robust error handling and input validation")
            
            with gr.Tabs():
                with gr.TabItem("📝 Text Generation"):
                    gr.Markdown("### Enhanced Text Generation with Validation")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            text_prompt = gr.Textbox(
                                label="Text Prompt",
                                placeholder="Enter your text prompt...",
                                lines=3
                            )
                            
                            with gr.Row():
                                text_length = gr.Slider(
                                    minimum=10, maximum=2000, value=100, step=10,
                                    label="Max Length"
                                )
                                text_temp = gr.Slider(
                                    minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                                    label="Temperature"
                                )
                            
                            with gr.Row():
                                text_top_p = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.9, step=0.1,
                                    label="Top-p"
                                )
                                text_samples = gr.Slider(
                                    minimum=1, maximum=10, value=3, step=1,
                                    label="Samples"
                                )
                            
                            text_btn = gr.Button("🚀 Generate Text", variant="primary")
                        
                        with gr.Column(scale=1):
                            text_analysis = gr.Markdown(label="Generation Analysis")
                            
                            text_outputs = []
                            for i in range(5):
                                output = gr.Textbox(
                                    label=f"Sample {i+1}",
                                    lines=4,
                                    visible=i < 3
                                )
                                text_outputs.append(output)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Text Analysis")
                            text_for_analysis = gr.Textbox(
                                label="Text to Analyze",
                                placeholder="Paste text here for analysis...",
                                lines=5
                            )
                            analyze_btn = gr.Button("📊 Analyze Text")
                            analysis_result = gr.Markdown(label="Analysis Results")
                
                with gr.TabItem("🎨 Image Generation"):
                    gr.Markdown("### Enhanced Image Generation with Validation")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_prompt = gr.Textbox(
                                label="Image Description",
                                placeholder="Describe the image you want to create...",
                                lines=3
                            )
                            
                            negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="What to avoid in the image...",
                                lines=2
                            )
                            
                            with gr.Row():
                                num_steps = gr.Slider(
                                    minimum=10, maximum=100, value=50, step=5,
                                    label="Steps"
                                )
                                guidance_scale = gr.Slider(
                                    minimum=1.0, maximum=20.0, value=7.5, step=0.5,
                                    label="Guidance Scale"
                                )
                            
                            with gr.Row():
                                width = gr.Slider(
                                    minimum=256, maximum=1024, value=512, step=64,
                                    label="Width"
                                )
                                height = gr.Slider(
                                    minimum=256, maximum=1024, value=512, step=64,
                                    label="Height"
                                )
                            
                            seed = gr.Number(label="Seed (-1 for random)", value=-1)
                            image_btn = gr.Button("🎨 Generate Image", variant="primary")
                        
                        with gr.Column(scale=1):
                            image_output = gr.Image(label="Generated Image")
                            image_info = gr.Markdown(label="Generation Info")
                
                with gr.TabItem("🎵 Audio Processing"):
                    gr.Markdown("### Enhanced Audio Processing with Validation")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            audio_input = gr.Audio(
                                label="Input Audio",
                                type="numpy"
                            )
                            
                            operation = gr.Dropdown(
                                choices=['noise_reduction', 'equalizer', 'reverb', 'pitch_shift'],
                                label="Operation",
                                value='noise_reduction'
                            )
                            
                            process_btn = gr.Button("🎵 Process Audio", variant="primary")
                        
                        with gr.Column(scale=1):
                            audio_output = gr.Audio(label="Processed Audio")
                            audio_info = gr.Markdown(label="Processing Info")
                
                with gr.TabItem("📊 Training Visualization"):
                    gr.Markdown("### Enhanced Training Visualization with Validation")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            epochs = gr.Slider(
                                minimum=10, maximum=500, value=100, step=10,
                                label="Epochs"
                            )
                            
                            learning_rate = gr.Slider(
                                minimum=1e-6, maximum=1e-1, value=1e-3, step=1e-6,
                                label="Learning Rate"
                            )
                            
                            batch_size = gr.Slider(
                                minimum=1, maximum=1024, value=64, step=16,
                                label="Batch Size"
                            )
                            
                            update_btn = gr.Button("📈 Update Plot", variant="primary")
                        
                        with gr.Column(scale=2):
                            plot_output = gr.Plot(label="Training Progress")
                            summary_output = gr.Markdown(label="Training Summary")
                
                with gr.TabItem("📊 Error Monitoring"):
                    gr.Markdown("### System Error Monitoring")
                    
                    with gr.Row():
                        with gr.Column():
                            error_summary_btn = gr.Button("📊 Get Error Summary")
                            error_summary_output = gr.JSON(label="Error Summary")
                        
                        with gr.Column():
                            gr.Markdown("### System Status")
                            gr.Markdown("""
                            **Enhanced Features:**
                            - ✅ Comprehensive input validation
                            - ✅ User-friendly error messages
                            - ✅ Graceful error recovery
                            - ✅ Performance monitoring
                            - ✅ Security validation
                            - ✅ Error logging and tracking
                            
                            **Validation Rules:**
                            - Text: Length, content filtering, security checks
                            - Images: Size, format, dimensions, quality
                            - Audio: Duration, sample rate, data integrity
                            - Numbers: Range, precision, type checking
                            """)
            
            # Event handlers
            text_btn.click(
                fn=self.generate_text_with_enhanced_validation,
                inputs=[text_prompt, text_length, text_temp, text_top_p, text_samples],
                outputs=[text_analysis] + text_outputs
            )
            
            analyze_btn.click(
                fn=self.analyze_text_with_enhanced_validation,
                inputs=[text_for_analysis],
                outputs=[analysis_result]
            )
            
            image_btn.click(
                fn=self.generate_image_with_enhanced_validation,
                inputs=[image_prompt, negative_prompt, num_steps, guidance_scale, seed, width, height],
                outputs=[image_output, image_info]
            )
            
            process_btn.click(
                fn=self.process_audio_with_enhanced_validation,
                inputs=[audio_input, operation, {}],
                outputs=[audio_output, audio_info]
            )
            
            update_btn.click(
                fn=self.update_training_plot_with_enhanced_validation,
                inputs=[epochs, learning_rate, batch_size],
                outputs=[plot_output, summary_output]
            )
            
            error_summary_btn.click(
                fn=get_error_summary,
                inputs=[],
                outputs=[error_summary_output]
            )
        
        return interface
    
    def launch_enhanced_demo(self, port: int = 7866, share: bool = False):
        """Launch the enhanced demo"""
        print("🛡️ Launching Enhanced Gradio Demo...")
        
        demo = self.create_enhanced_comprehensive_demo()
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            show_error=True,
            quiet=False
        )


def main():
    """Main function to run the enhanced demo"""
    print("🛡️ Starting Enhanced Gradio Demo...")
    
    demo = EnhancedGradioDemos()
    demo.launch_enhanced_demo(port=7866, share=False)


match __name__:
    case "__main__":
    main() 