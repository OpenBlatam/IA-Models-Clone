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
import asyncio
import threading
import time
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from production_code import (
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Interactive Gradio Demos for Model Inference and Visualization
=============================================================

This module provides comprehensive interactive demos for:
- Text generation and analysis
- Image generation with diffusion models
- Audio processing and radio integration
- Model training visualization
- Performance monitoring
- Real-time inference
"""


# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    MultiGPUTrainer, TrainingConfiguration, RadioIntegration,
    PerformanceOptimizer, EarlyStopping, LearningRateMonitor
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveGradioDemos:
    """Comprehensive interactive Gradio demos for model inference and visualization"""
    
    def __init__(self) -> Any:
        self.config = TrainingConfiguration(
            enable_radio_integration=True,
            enable_gradio_demo=True,
            gradio_port=7860,
            gradio_share=False
        )
        
        self.trainer = MultiGPUTrainer(self.config)
        self.radio = self.trainer.radio if self.trainer.radio else None
        
        # Demo state
        self.current_model = None
        self.training_history = {}
        self.demo_data = {}
        
        # Initialize demo data
        self._initialize_demo_data()
        
        logger.info("Interactive Gradio Demos initialized")
    
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
    
    def _generate_sample_training_data(self) -> Dict:
        """Generate sample training data for visualization"""
        epochs = 100
        return {
            'epochs': list(range(1, epochs + 1)),
            'train_loss': [2.5 * np.exp(-epoch/30) + 0.1 * np.random.random() for epoch in range(1, epochs + 1)],
            'val_loss': [2.3 * np.exp(-epoch/25) + 0.15 * np.random.random() for epoch in range(1, epochs + 1)],
            'train_acc': [0.3 + 0.6 * (1 - np.exp(-epoch/20)) + 0.02 * np.random.random() for epoch in range(1, epochs + 1)],
            'val_acc': [0.25 + 0.65 * (1 - np.exp(-epoch/18)) + 0.03 * np.random.random() for epoch in range(1, epochs + 1)],
            'learning_rate': [1e-3 * np.exp(-epoch/50) for epoch in range(1, epochs + 1)]
        }
    
    def create_text_generation_demo(self) -> gr.Interface:
        """Create interactive text generation demo"""
        
        def generate_text(prompt: str, max_length: int = 100, temperature: float = 0.8, 
                         top_p: float = 0.9, num_samples: int = 3) -> Tuple[str, List[str]]:
            """Generate text based on prompt"""
            try:
                # Simulate text generation (replace with actual model)
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
                    
                    # Add some randomness
                    if temperature > 0.5:
                        continuation += " " + " ".join([
                            "The", "future", "looks", "promising", "for", "AI", "applications."
                        ])
                    
                    generated_texts.append(continuation[:max_length])
                
                # Create analysis
                analysis = f"""
                **Text Generation Analysis:**
                - **Prompt Length**: {len(prompt)} characters
                - **Generated Length**: {len(generated_texts[0])} characters
                - **Temperature**: {temperature} (creativity level)
                - **Top-p**: {top_p} (nucleus sampling)
                - **Samples Generated**: {num_samples}
                
                **Generation Parameters:**
                - Max Length: {max_length}
                - Temperature: {temperature}
                - Top-p: {top_p}
                """
                
                return analysis, generated_texts
                
            except Exception as e:
                return f"Error generating text: {str(e)}", []
        
        def analyze_text(text: str) -> str:
            """Analyze text characteristics"""
            try:
                words = text.split()
                sentences = text.split('.')
                characters = len(text)
                
                analysis = f"""
                **Text Analysis:**
                - **Characters**: {characters}
                - **Words**: {len(words)}
                - **Sentences**: {len(sentences)}
                - **Average Word Length**: {np.mean([len(word) for word in words]):.2f}
                - **Average Sentence Length**: {np.mean([len(sent.split()) for sent in sentences if sent.strip()]):.2f}
                
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
                
            except Exception as e:
                return f"Error analyzing text: {str(e)}"
        
        # Create interface
        with gr.Blocks(title="Text Generation Demo", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🤖 Interactive Text Generation Demo")
            gr.Markdown("Generate and analyze text using AI models")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input Parameters")
                    
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your text prompt here...",
                        lines=3,
                        value="The future of artificial intelligence"
                    )
                    
                    max_length = gr.Slider(
                        minimum=50, maximum=500, value=100, step=10,
                        label="Max Length"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                        label="Temperature (Creativity)"
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.9, step=0.1,
                        label="Top-p (Nucleus Sampling)"
                    )
                    
                    num_samples = gr.Slider(
                        minimum=1, maximum=5, value=3, step=1,
                        label="Number of Samples"
                    )
                    
                    generate_btn = gr.Button("🚀 Generate Text", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Generated Text")
                    
                    analysis_output = gr.Markdown(label="Generation Analysis")
                    
                    generated_outputs = []
                    for i in range(5):
                        output = gr.Textbox(
                            label=f"Sample {i+1}",
                            lines=4,
                            visible=i < 3
                        )
                        generated_outputs.append(output)
            
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
            
            # Event handlers
            generate_btn.click(
                fn=generate_text,
                inputs=[prompt_input, max_length, temperature, top_p, num_samples],
                outputs=[analysis_output] + generated_outputs
            )
            
            analyze_btn.click(
                fn=analyze_text,
                inputs=[text_for_analysis],
                outputs=[analysis_result]
            )
        
        return interface
    
    def create_image_generation_demo(self) -> gr.Interface:
        """Create interactive image generation demo"""
        
        def generate_image(prompt: str, negative_prompt: str = "", 
                          num_steps: int = 50, guidance_scale: float = 7.5,
                          seed: int = -1, width: int = 512, height: int = 512) -> Tuple[Image.Image, str]:
            """Generate image based on prompt"""
            try:
                # Simulate image generation (replace with actual diffusion model)
                if seed == -1:
                    seed = np.random.randint(0, 1000000)
                
                # Create a simple generated image (replace with actual model)
                img = Image.new('RGB', (width, height), color='white')
                draw = ImageDraw.Draw(img)
                
                # Add some visual elements based on prompt
                if 'mountain' in prompt.lower():
                    # Draw mountains
                    points = [(0, height), (width//4, height//2), (width//2, height//3), 
                             (3*width//4, height//2), (width, height)]
                    draw.polygon(points, fill='gray')
                
                if 'sky' in prompt.lower() or 'blue' in prompt.lower():
                    # Draw sky
                    draw.rectangle([0, 0, width, height//2], fill='lightblue')
                
                if 'sun' in prompt.lower():
                    # Draw sun
                    draw.ellipse([width-100, 50, width-50, 100], fill='yellow')
                
                # Add text overlay
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                draw.text((10, 10), f"Generated: {prompt[:30]}...", fill='black', font=font)
                draw.text((10, height-30), f"Seed: {seed}", fill='black', font=font)
                
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
                
                return img, info
                
            except Exception as e:
                error_img = Image.new('RGB', (width, height), color='red')
                return error_img, f"Error generating image: {str(e)}"
        
        def analyze_image(image: Image.Image) -> str:
            """Analyze image characteristics"""
            try:
                # Convert to numpy array
                img_array = np.array(image)
                
                analysis = f"""
                **Image Analysis:**
                - **Dimensions**: {image.size[0]}x{image.size[1]}
                - **Mode**: {image.mode}
                - **Format**: {image.format}
                
                **Color Analysis:**
                - **Mean RGB**: ({img_array.mean(axis=(0,1))[0]:.1f}, {img_array.mean(axis=(0,1))[1]:.1f}, {img_array.mean(axis=(0,1))[2]:.1f})
                - **Standard Deviation**: ({img_array.std(axis=(0,1))[0]:.1f}, {img_array.std(axis=(0,1))[1]:.1f}, {img_array.std(axis=(0,1))[2]:.1f})
                - **Brightness**: {img_array.mean():.1f}
                - **Contrast**: {img_array.std():.1f}
                """
                
                return analysis
                
            except Exception as e:
                return f"Error analyzing image: {str(e)}"
        
        # Create interface
        with gr.Blocks(title="Image Generation Demo", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎨 Interactive Image Generation Demo")
            gr.Markdown("Generate and analyze images using diffusion models")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Generation Parameters")
                    
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to generate...",
                        lines=3,
                        value="A serene landscape with mountains and a lake"
                    )
                    
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="What to avoid in the image...",
                        lines=2,
                        value="blurry, low quality, distorted"
                    )
                    
                    with gr.Row():
                        num_steps = gr.Slider(
                            minimum=10, maximum=100, value=50, step=5,
                            label="Number of Steps"
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
                    
                    seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1
                    )
                    
                    generate_btn = gr.Button("🎨 Generate Image", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Generated Image")
                    image_output = gr.Image(label="Generated Image")
                    info_output = gr.Markdown(label="Generation Info")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Image Analysis")
                    analyze_btn = gr.Button("📊 Analyze Image")
                    analysis_output = gr.Markdown(label="Analysis Results")
            
            # Event handlers
            generate_btn.click(
                fn=generate_image,
                inputs=[prompt_input, negative_prompt, num_steps, guidance_scale, seed, width, height],
                outputs=[image_output, info_output]
            )
            
            analyze_btn.click(
                fn=analyze_image,
                inputs=[image_output],
                outputs=[analysis_output]
            )
        
        return interface
    
    def create_audio_processing_demo(self) -> gr.Interface:
        """Create interactive audio processing demo"""
        
        def process_audio(audio_input, operation: str, parameters: Dict) -> Tuple[np.ndarray, str]:
            """Process audio with various operations"""
            try:
                if audio_input is None:
                    return None, "No audio input provided"
                
                audio_data, sample_rate = audio_input
                
                if operation == "noise_reduction":
                    # Simulate noise reduction
                    processed_audio = audio_data * 0.8  # Reduce volume
                    info = f"Applied noise reduction (volume reduced by 20%)"
                
                elif operation == "equalizer":
                    # Simulate equalizer
                    processed_audio = audio_data * np.random.uniform(0.5, 1.5, len(audio_data))
                    info = f"Applied equalizer with random frequency adjustments"
                
                elif operation == "reverb":
                    # Simulate reverb
                    delay = int(sample_rate * 0.1)  # 100ms delay
                    processed_audio = audio_data.copy()
                    processed_audio[delay:] += audio_data[:-delay] * 0.3
                    info = f"Applied reverb effect (100ms delay, 30% mix)"
                
                elif operation == "pitch_shift":
                    # Simulate pitch shift
                    processed_audio = np.interp(
                        np.arange(len(audio_data)),
                        np.arange(len(audio_data)) * 1.1,  # Pitch up by 10%
                        audio_data
                    )
                    info = f"Applied pitch shift (pitch increased by 10%)"
                
                else:
                    processed_audio = audio_data
                    info = f"No processing applied"
                
                return (processed_audio, sample_rate), info
                
            except Exception as e:
                return None, f"Error processing audio: {str(e)}"
        
        def analyze_audio(audio_input) -> str:
            """Analyze audio characteristics"""
            try:
                if audio_input is None:
                    return "No audio input provided"
                
                audio_data, sample_rate = audio_input
                
                # Calculate audio features
                duration = len(audio_data) / sample_rate
                rms_energy = np.sqrt(np.mean(audio_data**2))
                zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
                
                # Spectral features
                fft = np.fft.fft(audio_data)
                magnitude = np.abs(fft)
                spectral_centroid = np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude)
                
                analysis = f"""
                **Audio Analysis:**
                - **Duration**: {duration:.2f} seconds
                - **Sample Rate**: {sample_rate} Hz
                - **RMS Energy**: {rms_energy:.4f}
                - **Zero Crossings**: {zero_crossings}
                - **Spectral Centroid**: {spectral_centroid:.2f}
                - **Max Amplitude**: {np.max(np.abs(audio_data)):.4f}
                - **Min Amplitude**: {np.min(np.abs(audio_data)):.4f}
                """
                
                return analysis
                
            except Exception as e:
                return f"Error analyzing audio: {str(e)}"
        
        # Create interface
        with gr.Blocks(title="Audio Processing Demo", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎵 Interactive Audio Processing Demo")
            gr.Markdown("Process and analyze audio with various effects")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Audio Input")
                    audio_input = gr.Audio(
                        label="Input Audio",
                        type="numpy"
                    )
                    
                    gr.Markdown("### Processing Options")
                    operation = gr.Dropdown(
                        choices=["noise_reduction", "equalizer", "reverb", "pitch_shift"],
                        label="Operation",
                        value="noise_reduction"
                    )
                    
                    process_btn = gr.Button("🎵 Process Audio", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Processed Audio")
                    audio_output = gr.Audio(
                        label="Processed Audio",
                        type="numpy"
                    )
                    info_output = gr.Markdown(label="Processing Info")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Audio Analysis")
                    analyze_btn = gr.Button("📊 Analyze Audio")
                    analysis_output = gr.Markdown(label="Analysis Results")
            
            # Event handlers
            process_btn.click(
                fn=process_audio,
                inputs=[audio_input, operation, {}],
                outputs=[audio_output, info_output]
            )
            
            analyze_btn.click(
                fn=analyze_audio,
                inputs=[audio_input],
                outputs=[analysis_output]
            )
        
        return interface
    
    def create_training_visualization_demo(self) -> gr.Interface:
        """Create interactive training visualization demo"""
        
        def update_training_plot(epochs: int, learning_rate: float, batch_size: int) -> Tuple[go.Figure, str]:
            """Update training visualization"""
            try:
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
                
                # Create summary
                summary = f"""
                **Training Summary:**
                - **Total Epochs**: {epochs}
                - **Initial Learning Rate**: {learning_rate:.6f}
                - **Batch Size**: {batch_size}
                - **Final Train Loss**: {train_loss[-1]:.4f}
                - **Final Val Loss**: {val_loss[-1]:.4f}
                - **Final Train Acc**: {train_acc[-1]:.4f}
                - **Final Val Acc**: {val_acc[-1]:.4f}
                - **Final Learning Rate**: {lr_schedule[-1]:.6f}
                """
                
                return fig, summary
                
            except Exception as e:
                return go.Figure(), f"Error creating visualization: {str(e)}"
        
        def create_performance_metrics(epochs: int) -> Tuple[go.Figure, str]:
            """Create performance metrics visualization"""
            try:
                x = list(range(1, epochs + 1))
                
                # Generate performance metrics
                gpu_util = [np.random.uniform(60, 95) for _ in x]
                memory_util = [np.random.uniform(40, 85) for _ in x]
                throughput = [np.random.uniform(100, 500) for _ in x]
                latency = [np.random.uniform(10, 50) for _ in x]
                
                # Create subplots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('GPU Utilization', 'Memory Usage', 'Throughput', 'Latency'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Add traces
                fig.add_trace(
                    go.Scatter(x=x, y=gpu_util, name="GPU %", line=dict(color='red'), fill='tonexty'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=x, y=memory_util, name="Memory %", line=dict(color='blue'), fill='tonexty'),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(x=x, y=throughput, name="Samples/sec", line=dict(color='green')),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=x, y=latency, name="ms", line=dict(color='orange')),
                    row=2, col=2
                )
                
                # Update layout
                fig.update_layout(
                    height=600,
                    title_text="Performance Metrics",
                    showlegend=True
                )
                
                # Update axes
                for i in range(1, 3):
                    for j in range(1, 3):
                        fig.update_xaxes(title_text="Epoch", row=i, col=j)
                
                fig.update_yaxes(title_text="GPU %", row=1, col=1)
                fig.update_yaxes(title_text="Memory %", row=1, col=2)
                fig.update_yaxes(title_text="Samples/sec", row=2, col=1)
                fig.update_yaxes(title_text="Latency (ms)", row=2, col=2)
                
                # Create summary
                summary = f"""
                **Performance Summary:**
                - **Average GPU Utilization**: {np.mean(gpu_util):.1f}%
                - **Average Memory Usage**: {np.mean(memory_util):.1f}%
                - **Average Throughput**: {np.mean(throughput):.1f} samples/sec
                - **Average Latency**: {np.mean(latency):.1f} ms
                - **Peak GPU Usage**: {np.max(gpu_util):.1f}%
                - **Peak Memory Usage**: {np.max(memory_util):.1f}%
                """
                
                return fig, summary
                
            except Exception as e:
                return go.Figure(), f"Error creating performance metrics: {str(e)}"
        
        # Create interface
        with gr.Blocks(title="Training Visualization Demo", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 📊 Interactive Training Visualization Demo")
            gr.Markdown("Visualize training progress and performance metrics")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Training Parameters")
                    
                    epochs = gr.Slider(
                        minimum=10, maximum=200, value=100, step=10,
                        label="Number of Epochs"
                    )
                    
                    learning_rate = gr.Slider(
                        minimum=1e-5, maximum=1e-2, value=1e-3, step=1e-5,
                        label="Initial Learning Rate"
                    )
                    
                    batch_size = gr.Slider(
                        minimum=16, maximum=256, value=64, step=16,
                        label="Batch Size"
                    )
                    
                    update_btn = gr.Button("📈 Update Training Plot", variant="primary")
                    metrics_btn = gr.Button("⚡ Show Performance Metrics", variant="secondary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Training Visualization")
                    plot_output = gr.Plot(label="Training Progress")
                    summary_output = gr.Markdown(label="Training Summary")
            
            # Event handlers
            update_btn.click(
                fn=update_training_plot,
                inputs=[epochs, learning_rate, batch_size],
                outputs=[plot_output, summary_output]
            )
            
            metrics_btn.click(
                fn=create_performance_metrics,
                inputs=[epochs],
                outputs=[plot_output, summary_output]
            )
        
        return interface
    
    def create_radio_control_demo(self) -> gr.Interface:
        """Create interactive radio control demo"""
        
        if not self.radio:
            return gr.Interface(
                fn=lambda: "Radio integration not available",
                inputs=[],
                outputs=gr.Textbox(),
                title="Radio Control Demo",
                description="Radio integration is not enabled"
            )
        
        def search_and_play(query: str, country: str, volume: float) -> Tuple[str, List[str]]:
            """Search and play radio stations"""
            try:
                stations = self.radio.search_radio_stations(query, country, limit=5)
                
                if stations:
                    station_list = []
                    for i, station in enumerate(stations, 1):
                        station_info = f"{i}. {station.get('name', 'Unknown')}"
                        if 'country' in station:
                            station_info += f" ({station['country']})"
                        station_list.append(station_info)
                    
                    # Try to play the first station
                    success = self.radio.play_station(stations[0]['url'], volume)
                    status = f"Playing: {stations[0]['name']}" if success else "Failed to play"
                    
                    return status, station_list
                else:
                    return "No stations found", []
                    
            except Exception as e:
                return f"Error: {str(e)}", []
        
        def stop_radio() -> str:
            """Stop radio playback"""
            try:
                self.radio.stop_playback()
                return "Playback stopped"
            except Exception as e:
                return f"Error stopping playback: {str(e)}"
        
        def get_radio_status() -> str:
            """Get current radio status"""
            try:
                status = self.trainer.get_radio_status()
                if status['enabled']:
                    return f"""
                    **Radio Status:**
                    - **Playing**: {status['is_playing']}
                    - **Current Station**: {status['current_station'] or 'None'}
                    - **Volume**: {status['volume']:.2f}
                    - **Track Info**: {status['track_info']}
                    """
                else:
                    return "Radio integration not enabled"
            except Exception as e:
                return f"Error getting status: {str(e)}"
        
        def set_volume(volume: float) -> str:
            """Set radio volume"""
            try:
                self.radio.set_volume(volume)
                return f"Volume set to {volume:.2f}"
            except Exception as e:
                return f"Error setting volume: {str(e)}"
        
        # Create interface
        with gr.Blocks(title="Radio Control Demo", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎵 Interactive Radio Control Demo")
            gr.Markdown("Control radio playback and search for stations")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Radio Control")
                    
                    query_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter genre, artist, or station name...",
                        value="jazz"
                    )
                    
                    country_input = gr.Textbox(
                        label="Country (optional)",
                        placeholder="US, Germany, UK, etc.",
                        value=""
                    )
                    
                    volume_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                        label="Volume"
                    )
                    
                    with gr.Row():
                        search_btn = gr.Button("🔍 Search & Play", variant="primary")
                        stop_btn = gr.Button("⏹️ Stop", variant="secondary")
                    
                    status_btn = gr.Button("📊 Get Status")
                    volume_btn = gr.Button("🔊 Set Volume")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Radio Status")
                    status_output = gr.Markdown(label="Status")
                    
                    gr.Markdown("### Found Stations")
                    stations_output = gr.Textbox(
                        label="Available Stations",
                        lines=10,
                        interactive=False
                    )
            
            # Event handlers
            search_btn.click(
                fn=search_and_play,
                inputs=[query_input, country_input, volume_slider],
                outputs=[status_output, stations_output]
            )
            
            stop_btn.click(
                fn=stop_radio,
                inputs=[],
                outputs=[status_output]
            )
            
            status_btn.click(
                fn=get_radio_status,
                inputs=[],
                outputs=[status_output]
            )
            
            volume_btn.click(
                fn=set_volume,
                inputs=[volume_slider],
                outputs=[status_output]
            )
        
        return interface
    
    def create_comprehensive_demo(self) -> gr.Interface:
        """Create comprehensive demo with all features"""
        
        # Create all individual demos
        text_demo = self.create_text_generation_demo()
        image_demo = self.create_image_generation_demo()
        audio_demo = self.create_audio_processing_demo()
        training_demo = self.create_training_visualization_demo()
        radio_demo = self.create_radio_control_demo()
        
        # Create comprehensive interface
        with gr.Blocks(title="Comprehensive AI Demo Suite", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🤖 Comprehensive AI Demo Suite")
            gr.Markdown("Interactive demos for all AI capabilities")
            
            with gr.Tabs():
                with gr.TabItem("📝 Text Generation"):
                    text_demo.render()
                
                with gr.TabItem("🎨 Image Generation"):
                    image_demo.render()
                
                with gr.TabItem("🎵 Audio Processing"):
                    audio_demo.render()
                
                with gr.TabItem("📊 Training Visualization"):
                    training_demo.render()
                
                with gr.TabItem("🎵 Radio Control"):
                    radio_demo.render()
        
        return interface
    
    def launch_demos(self, port: int = 7860, share: bool = False):
        """Launch all demos"""
        print("🚀 Launching Interactive Gradio Demos...")
        
        # Create comprehensive demo
        demo = self.create_comprehensive_demo()
        
        # Launch the demo
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            show_error=True,
            quiet=False
        )


def main():
    """Main function to run the interactive demos"""
    print("🎯 Starting Interactive Gradio Demos...")
    
    # Create demo instance
    demos = InteractiveGradioDemos()
    
    # Launch demos
    demos.launch_demos(port=7860, share=False)


match __name__:
    case "__main__":
    main() 