"""
Interactive Gradio demos for Blaze AI model inference and visualization.
"""
from __future__ import annotations

import asyncio
import gradio as gr
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time

from ..core.interfaces import CoreConfig
from ..utils.logging import get_logger
from ..utils.performance import get_performance_profiler
from ..utils.initialization import get_loss_function, get_optimizer
from ..engines.llm import LLMEngine
from ..engines.diffusion import DiffusionEngine
from .validation import (
    get_text_generation_validator,
    get_image_generation_validator,
    get_training_validator,
    get_gradio_error_handler,
    get_safe_gradio_executor
)

logger = get_logger(__name__)

class InteractiveModelDemos:
    """Interactive demos for model inference and visualization."""
    
    def __init__(self, config: Optional[CoreConfig] = None):
        self.config = config or CoreConfig()
        self.logger = get_logger(__name__)
        self.performance_profiler = get_performance_profiler()
        
        # Initialize validators and error handlers
        self.text_validator = get_text_generation_validator()
        self.image_validator = get_image_generation_validator()
        self.training_validator = get_training_validator()
        self.error_handler = get_gradio_error_handler()
        self.safe_executor = get_safe_gradio_executor()
        
        # Initialize engines
        self.llm_engine = None
        self.diffusion_engine = None
        
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize AI engines."""
        try:
            # Initialize LLM engine
            llm_config = {
                "model_name": "gpt2",
                "device": "auto",
                "precision": "float16",
                "enable_amp": True,
                "gradient_accumulation_steps": 4,
                "mixed_precision": True
            }
            self.llm_engine = LLMEngine("llm", llm_config)
            
            # Initialize Diffusion engine
            diffusion_config = {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "device": "auto",
                "precision": "float16",
                "enable_xformers": True,
                "mixed_precision": True
            }
            self.diffusion_engine = DiffusionEngine("diffusion", diffusion_config)
            
            self.logger.info("AI engines initialized for demos")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize engines: {e}")
            raise
    
    def create_text_generation_demo(self) -> gr.Interface:
        """Create interactive text generation demo."""
        
                       def generate_text_with_params(prompt: str, max_length: int, temperature: float, 
                                           top_p: float, do_sample: bool, num_return_sequences: int) -> Tuple[str, Dict]:
                   """Generate text with advanced parameters."""
                   try:
                       # Validate inputs
                       is_valid, error_msg = self.text_validator.validate_text_generation_params(
                           prompt, max_length, temperature, top_p, do_sample, num_return_sequences
                       )
                       if not is_valid:
                           return self.error_handler.handle_validation_error(error_msg)
                       
                       # Execute with safe executor
                       result = asyncio.run(self.safe_executor.execute_with_timeout(
                           "text_generation",
                           self._execute_text_generation,
                           prompt, max_length, temperature, top_p, do_sample, num_return_sequences
                       ))
                       
                       if isinstance(result, tuple) and len(result) == 2 and "error" in result[1]:
                           return result
                       
                       generated_text = result.get("text", "")
                       
                       # Create statistics
                       stats = {
                           "prompt_length": len(prompt),
                           "generated_length": len(generated_text),
                           "temperature": temperature,
                           "top_p": top_p,
                           "do_sample": do_sample,
                           "num_return_sequences": num_return_sequences,
                           "generation_time": time.time()
                       }
                       
                       return generated_text, stats
                       
                   except Exception as e:
                       return self.error_handler.handle_system_error(e, "text generation")
               
               async def _execute_text_generation(self, prompt: str, max_length: int, temperature: float, 
                                                top_p: float, do_sample: bool, num_return_sequences: int) -> Dict:
                   """Execute text generation with performance profiling."""
                   with self.performance_profiler.profile("text_generation"):
                       result = await self.llm_engine.execute("generate_text", {
                           "prompt": prompt,
                           "max_length": max_length,
                           "temperature": temperature,
                           "top_p": top_p,
                           "do_sample": do_sample,
                           "num_return_sequences": num_return_sequences
                       })
                   return result
        
        # Create interface
        interface = gr.Interface(
            fn=generate_text_with_params,
            inputs=[
                gr.Textbox(label="Prompt", placeholder="Enter your text prompt here...", lines=3),
                gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max Length"),
                gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
                gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p"),
                gr.Checkbox(label="Enable Sampling", value=True),
                gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Number of Sequences")
            ],
            outputs=[
                gr.Textbox(label="Generated Text", lines=10),
                gr.JSON(label="Generation Statistics")
            ],
            title="🚀 Advanced Text Generation Demo",
            description="Generate creative text with advanced parameters and real-time statistics.",
            examples=[
                ["The future of artificial intelligence", 150, 0.8, 0.9, True, 1],
                ["Once upon a time in a magical forest", 200, 0.9, 0.95, True, 1],
                ["The benefits of renewable energy include", 100, 0.6, 0.8, True, 1]
            ],
            cache_examples=True
        )
        
        return interface
    
    def create_image_generation_demo(self) -> gr.Interface:
        """Create interactive image generation demo."""
        
                       def generate_image_with_params(prompt: str, negative_prompt: str, num_steps: int,
                                            guidance_scale: float, width: int, height: int, 
                                            seed: Optional[int], num_images: int) -> Tuple[List[Image.Image], Dict]:
                   """Generate images with advanced parameters."""
                   try:
                       # Validate inputs
                       is_valid, error_msg = self.image_validator.validate_image_generation_params(
                           prompt, negative_prompt, num_steps, guidance_scale, width, height, seed, num_images
                       )
                       if not is_valid:
                           return [], self.error_handler.handle_validation_error(error_msg)[1]
                       
                       # Execute with safe executor
                       result = asyncio.run(self.safe_executor.execute_with_timeout(
                           "image_generation",
                           self._execute_image_generation,
                           prompt, negative_prompt, num_steps, guidance_scale, width, height, seed, num_images
                       ))
                       
                       if isinstance(result, tuple) and len(result) == 2 and "error" in result[1]:
                           return [], result[1]
                       
                       return result
                       
                   except Exception as e:
                       return [], self.error_handler.handle_system_error(e, "image generation")[1]
               
               async def _execute_image_generation(self, prompt: str, negative_prompt: str, num_steps: int,
                                                guidance_scale: float, width: int, height: int, 
                                                seed: Optional[int], num_images: int) -> Tuple[List[Image.Image], Dict]:
                   """Execute image generation with performance profiling."""
                   images = []
                   generation_info = {
                       "prompt": prompt,
                       "negative_prompt": negative_prompt,
                       "steps": num_steps,
                       "guidance_scale": guidance_scale,
                       "dimensions": f"{width}x{height}",
                       "seed": seed,
                       "num_images": num_images,
                       "generation_time": time.time()
                   }
                   
                   with self.performance_profiler.profile("image_generation"):
                       # Generate multiple images
                       for i in range(num_images):
                           current_seed = seed + i if seed is not None else None
                           
                           result = await self.diffusion_engine.execute("generate_image", {
                               "prompt": prompt,
                               "negative_prompt": negative_prompt,
                               "num_inference_steps": num_steps,
                               "guidance_scale": guidance_scale,
                               "width": width,
                               "height": height,
                               "seed": current_seed
                           })
                           
                           image_path = result.get("image_path")
                           if image_path and Path(image_path).exists():
                               image = Image.open(image_path)
                               images.append(image)
                   
                   if not images:
                       return [], {"error": "Failed to generate any images."}
                   
                   return images, generation_info
        
        # Create interface
        interface = gr.Interface(
            fn=generate_image_with_params,
            inputs=[
                gr.Textbox(label="Image Prompt", placeholder="A beautiful landscape with mountains and lake...", lines=2),
                gr.Textbox(label="Negative Prompt", placeholder="blurry, low quality, distorted...", lines=2),
                gr.Slider(minimum=10, maximum=100, value=50, step=5, label="Inference Steps"),
                gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.5, label="Guidance Scale"),
                gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width"),
                gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height"),
                gr.Number(label="Seed (optional)", value=None, precision=0),
                gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
            ],
            outputs=[
                gr.Gallery(label="Generated Images", columns=2, rows=2),
                gr.JSON(label="Generation Information")
            ],
            title="🎨 Advanced Image Generation Demo",
            description="Generate stunning images with advanced diffusion parameters and multiple outputs.",
            examples=[
                ["A serene mountain landscape at sunset", "blurry, low quality", 50, 7.5, 512, 512, None, 1],
                ["A futuristic city with flying cars", "dark, scary", 75, 8.0, 768, 512, None, 1],
                ["A cute cat playing with yarn", "realistic, photograph", 30, 6.0, 512, 512, None, 1]
            ],
            cache_examples=True
        )
        
        return interface
    
    def create_model_comparison_demo(self) -> gr.Interface:
        """Create demo for comparing different model configurations."""
        
        def compare_models(prompt: str, model_configs: str) -> Tuple[str, Dict]:
            """Compare different model configurations."""
            try:
                if not prompt.strip():
                    return "Please enter a valid prompt.", {}
                
                # Parse model configurations
                try:
                    configs = json.loads(model_configs)
                except json.JSONDecodeError:
                    return "Invalid JSON format for model configurations.", {}
                
                results = {}
                
                for config_name, config in configs.items():
                    try:
                        # Generate text with specific configuration
                        result = asyncio.run(self.llm_engine.execute("generate_text", {
                            "prompt": prompt,
                            "max_length": config.get("max_length", 100),
                            "temperature": config.get("temperature", 0.7),
                            "top_p": config.get("top_p", 0.9),
                            "do_sample": config.get("do_sample", True)
                        }))
                        
                        results[config_name] = {
                            "text": result.get("text", ""),
                            "config": config,
                            "success": True
                        }
                        
                    except Exception as e:
                        results[config_name] = {
                            "error": str(e),
                            "config": config,
                            "success": False
                        }
                
                # Format output
                output_text = ""
                for config_name, result in results.items():
                    output_text += f"=== {config_name} ===\n"
                    if result["success"]:
                        output_text += f"Text: {result['text']}\n"
                        output_text += f"Config: {result['config']}\n"
                    else:
                        output_text += f"Error: {result['error']}\n"
                    output_text += "\n"
                
                return output_text, {"comparison_results": results}
                
            except Exception as e:
                self.logger.error(f"Model comparison failed: {e}")
                return f"Error: {str(e)}", {"error": str(e)}
        
        # Create interface
        interface = gr.Interface(
            fn=compare_models,
            inputs=[
                gr.Textbox(label="Prompt", placeholder="Enter a prompt to compare models...", lines=3),
                gr.Textbox(
                    label="Model Configurations (JSON)",
                    placeholder='{"Creative": {"temperature": 0.9, "top_p": 0.95}, "Conservative": {"temperature": 0.3, "top_p": 0.8}}',
                    lines=5
                )
            ],
            outputs=[
                gr.Textbox(label="Comparison Results", lines=15),
                gr.JSON(label="Detailed Results")
            ],
            title="🔍 Model Configuration Comparison Demo",
            description="Compare different model configurations and see how they affect text generation.",
            examples=[
                ["The future of technology", '{"Creative": {"temperature": 0.9, "top_p": 0.95, "max_length": 150}, "Conservative": {"temperature": 0.3, "top_p": 0.8, "max_length": 100}}']
            ]
        )
        
        return interface
    
    def create_training_visualization_demo(self) -> gr.Interface:
        """Create demo for visualizing training progress."""
        
        def visualize_training(training_data: str, epochs: int, batch_size: int,
                             learning_rate: float, model_type: str) -> Tuple[Any, Dict]:
            """Visualize training progress with simulated data."""
            try:
                # Parse training data
                try:
                    data = json.loads(training_data)
                except json.JSONDecodeError:
                    return None, {"error": "Invalid JSON format for training data."}
                
                # Simulate training progress
                losses = []
                accuracies = []
                learning_rates = []
                
                initial_loss = 2.0
                for epoch in range(epochs):
                    # Simulate decreasing loss
                    loss = initial_loss * (0.9 ** epoch) + np.random.normal(0, 0.05)
                    losses.append(loss)
                    
                    # Simulate increasing accuracy
                    accuracy = 0.3 + 0.6 * (1 - 0.9 ** epoch) + np.random.normal(0, 0.02)
                    accuracies.append(accuracy)
                    
                    # Simulate learning rate schedule
                    lr = learning_rate * (0.95 ** epoch)
                    learning_rates.append(lr)
                
                # Create visualization
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                
                # Loss plot
                ax1.plot(range(1, epochs + 1), losses, 'b-', linewidth=2, marker='o')
                ax1.set_title('Training Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.grid(True, alpha=0.3)
                
                # Accuracy plot
                ax2.plot(range(1, epochs + 1), accuracies, 'g-', linewidth=2, marker='s')
                ax2.set_title('Training Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.grid(True, alpha=0.3)
                
                # Learning rate plot
                ax3.plot(range(1, epochs + 1), learning_rates, 'r-', linewidth=2, marker='^')
                ax3.set_title('Learning Rate Schedule')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Learning Rate')
                ax3.grid(True, alpha=0.3)
                
                # Combined plot
                ax4_twin = ax4.twinx()
                line1 = ax4.plot(range(1, epochs + 1), losses, 'b-', label='Loss', linewidth=2)
                line2 = ax4_twin.plot(range(1, epochs + 1), accuracies, 'g-', label='Accuracy', linewidth=2)
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Loss', color='b')
                ax4_twin.set_ylabel('Accuracy', color='g')
                ax4.grid(True, alpha=0.3)
                
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax4.legend(lines, labels, loc='upper right')
                
                plt.tight_layout()
                
                # Training statistics
                stats = {
                    "model_type": model_type,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "final_loss": losses[-1],
                    "final_accuracy": accuracies[-1],
                    "data_samples": len(data),
                    "training_time": time.time()
                }
                
                return fig, stats
                
            except Exception as e:
                self.logger.error(f"Training visualization failed: {e}")
                return None, {"error": str(e)}
        
        # Create interface
        interface = gr.Interface(
            fn=visualize_training,
            inputs=[
                gr.Textbox(
                    label="Training Data (JSON)",
                    placeholder='[{"text": "sample text 1"}, {"text": "sample text 2"}]',
                    lines=5
                ),
                gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Epochs"),
                gr.Slider(minimum=1, maximum=32, value=4, step=1, label="Batch Size"),
                gr.Number(label="Learning Rate", value=1e-5, precision=1e-8),
                gr.Dropdown(choices=["llm", "diffusion"], value="llm", label="Model Type")
            ],
            outputs=[
                gr.Plot(label="Training Visualization"),
                gr.JSON(label="Training Statistics")
            ],
            title="📊 Training Progress Visualization Demo",
            description="Visualize training progress with interactive plots and statistics.",
            examples=[
                ['[{"text": "AI is transforming the world"}, {"text": "Machine learning advances rapidly"}]', 15, 8, 1e-5, "llm"]
            ]
        )
        
        return interface
    
    def create_performance_analysis_demo(self) -> gr.Interface:
        """Create demo for performance analysis and benchmarking."""
        
        def analyze_performance(operation_type: str, num_iterations: int, 
                              batch_size: int, model_config: str) -> Tuple[Any, Dict]:
            """Analyze model performance with different configurations."""
            try:
                # Parse model configuration
                try:
                    config = json.loads(model_config)
                except json.JSONDecodeError:
                    return None, {"error": "Invalid JSON format for model configuration."}
                
                # Simulate performance metrics
                times = []
                memory_usage = []
                throughput = []
                
                for i in range(num_iterations):
                    # Simulate operation time
                    base_time = 0.1 if operation_type == "text_generation" else 2.0
                    time_taken = base_time * (1 + np.random.normal(0, 0.1))
                    times.append(time_taken)
                    
                    # Simulate memory usage
                    base_memory = 512 if operation_type == "text_generation" else 2048
                    memory = base_memory * (1 + np.random.normal(0, 0.05))
                    memory_usage.append(memory)
                    
                    # Calculate throughput
                    throughput.append(batch_size / time_taken)
                
                # Create performance visualization
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                
                # Response time distribution
                ax1.hist(times, bins=20, alpha=0.7, color='blue', edgecolor='black')
                ax1.set_title('Response Time Distribution')
                ax1.set_xlabel('Time (seconds)')
                ax1.set_ylabel('Frequency')
                ax1.grid(True, alpha=0.3)
                
                # Memory usage over time
                ax2.plot(range(1, num_iterations + 1), memory_usage, 'g-', linewidth=2)
                ax2.set_title('Memory Usage Over Time')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Memory (MB)')
                ax2.grid(True, alpha=0.3)
                
                # Throughput analysis
                ax3.plot(range(1, num_iterations + 1), throughput, 'r-', linewidth=2)
                ax3.set_title('Throughput Over Time')
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Throughput (ops/sec)')
                ax3.grid(True, alpha=0.3)
                
                # Performance summary
                metrics = ['Avg Time', 'Avg Memory', 'Avg Throughput', 'Std Time']
                values = [np.mean(times), np.mean(memory_usage), np.mean(throughput), np.std(times)]
                colors = ['blue', 'green', 'red', 'orange']
                
                bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
                ax4.set_title('Performance Summary')
                ax4.set_ylabel('Value')
                ax4.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.2f}', ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Performance statistics
                stats = {
                    "operation_type": operation_type,
                    "num_iterations": num_iterations,
                    "batch_size": batch_size,
                    "model_config": config,
                    "avg_response_time": np.mean(times),
                    "avg_memory_usage": np.mean(memory_usage),
                    "avg_throughput": np.mean(throughput),
                    "std_response_time": np.std(times),
                    "min_response_time": np.min(times),
                    "max_response_time": np.max(times)
                }
                
                return fig, stats
                
            except Exception as e:
                self.logger.error(f"Performance analysis failed: {e}")
                return None, {"error": str(e)}
        
        # Create interface
        interface = gr.Interface(
            fn=analyze_performance,
            inputs=[
                gr.Dropdown(
                    choices=["text_generation", "image_generation"],
                    value="text_generation",
                    label="Operation Type"
                ),
                gr.Slider(minimum=10, maximum=100, value=50, step=10, label="Number of Iterations"),
                gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Batch Size"),
                gr.Textbox(
                    label="Model Configuration (JSON)",
                    placeholder='{"temperature": 0.7, "max_length": 100}',
                    lines=3
                )
            ],
            outputs=[
                gr.Plot(label="Performance Analysis"),
                gr.JSON(label="Performance Statistics")
            ],
            title="⚡ Performance Analysis Demo",
            description="Analyze model performance with detailed metrics and visualizations.",
            examples=[
                ["text_generation", 30, 8, '{"temperature": 0.7, "max_length": 100}'],
                ["image_generation", 20, 2, '{"guidance_scale": 7.5, "num_steps": 50}']
            ]
        )
        
        return interface
    
    def create_error_analysis_demo(self) -> gr.Interface:
        """Create demo for error analysis and debugging."""
        
        def analyze_errors(error_logs: str, error_type: str, 
                          severity_threshold: float) -> Tuple[Any, Dict]:
            """Analyze error patterns and provide debugging insights."""
            try:
                # Parse error logs
                try:
                    logs = json.loads(error_logs)
                except json.JSONDecodeError:
                    return None, {"error": "Invalid JSON format for error logs."}
                
                # Simulate error analysis
                error_types = []
                severities = []
                timestamps = []
                messages = []
                
                for log in logs:
                    error_types.append(log.get("type", "unknown"))
                    severities.append(log.get("severity", 0.5))
                    timestamps.append(log.get("timestamp", time.time()))
                    messages.append(log.get("message", ""))
                
                # Create error analysis visualization
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                
                # Error type distribution
                unique_types, type_counts = np.unique(error_types, return_counts=True)
                ax1.pie(type_counts, labels=unique_types, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Error Type Distribution')
                
                # Severity distribution
                ax2.hist(severities, bins=20, alpha=0.7, color='red', edgecolor='black')
                ax2.axvline(severity_threshold, color='orange', linestyle='--', label=f'Threshold ({severity_threshold})')
                ax2.set_title('Error Severity Distribution')
                ax2.set_xlabel('Severity')
                ax2.set_ylabel('Frequency')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Errors over time
                ax3.scatter(timestamps, severities, alpha=0.6, c=severities, cmap='Reds')
                ax3.set_title('Errors Over Time')
                ax3.set_xlabel('Timestamp')
                ax3.set_ylabel('Severity')
                ax3.grid(True, alpha=0.3)
                
                # Error correlation matrix (simulated)
                error_correlation = np.random.rand(4, 4)
                np.fill_diagonal(error_correlation, 1.0)
                im = ax4.imshow(error_correlation, cmap='Reds', vmin=0, vmax=1)
                ax4.set_title('Error Correlation Matrix')
                ax4.set_xticks(range(4))
                ax4.set_yticks(range(4))
                ax4.set_xticklabels(['Type A', 'Type B', 'Type C', 'Type D'])
                ax4.set_yticklabels(['Type A', 'Type B', 'Type C', 'Type D'])
                
                # Add colorbar
                plt.colorbar(im, ax=ax4)
                
                plt.tight_layout()
                
                # Error statistics
                stats = {
                    "total_errors": len(logs),
                    "unique_error_types": len(unique_types),
                    "avg_severity": np.mean(severities),
                    "max_severity": np.max(severities),
                    "errors_above_threshold": sum(1 for s in severities if s > severity_threshold),
                    "most_common_error": unique_types[np.argmax(type_counts)],
                    "error_rate": len(logs) / 1000,  # Assuming 1000 total operations
                    "analysis_time": time.time()
                }
                
                return fig, stats
                
            except Exception as e:
                self.logger.error(f"Error analysis failed: {e}")
                return None, {"error": str(e)}
        
        # Create interface
        interface = gr.Interface(
            fn=analyze_errors,
            inputs=[
                gr.Textbox(
                    label="Error Logs (JSON)",
                    placeholder='[{"type": "timeout", "severity": 0.8, "message": "Request timeout"}]',
                    lines=8
                ),
                gr.Dropdown(
                    choices=["all", "timeout", "memory", "validation", "network"],
                    value="all",
                    label="Error Type Filter"
                ),
                gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Severity Threshold")
            ],
            outputs=[
                gr.Plot(label="Error Analysis"),
                gr.JSON(label="Error Statistics")
            ],
            title="🐛 Error Analysis Demo",
            description="Analyze error patterns and get debugging insights.",
            examples=[
                ['[{"type": "timeout", "severity": 0.8, "message": "Request timeout"}, {"type": "memory", "severity": 0.6, "message": "Memory limit exceeded"}]', "all", 0.5]
            ]
        )
        
        return interface

# Factory functions
def create_text_generation_demo(config: Optional[CoreConfig] = None) -> gr.Interface:
    """Create text generation demo."""
    demos = InteractiveModelDemos(config)
    return demos.create_text_generation_demo()

def create_image_generation_demo(config: Optional[CoreConfig] = None) -> gr.Interface:
    """Create image generation demo."""
    demos = InteractiveModelDemos(config)
    return demos.create_image_generation_demo()

def create_model_comparison_demo(config: Optional[CoreConfig] = None) -> gr.Interface:
    """Create model comparison demo."""
    demos = InteractiveModelDemos(config)
    return demos.create_model_comparison_demo()

def create_training_visualization_demo(config: Optional[CoreConfig] = None) -> gr.Interface:
    """Create training visualization demo."""
    demos = InteractiveModelDemos(config)
    return demos.create_training_visualization_demo()

def create_performance_analysis_demo(config: Optional[CoreConfig] = None) -> gr.Interface:
    """Create performance analysis demo."""
    demos = InteractiveModelDemos(config)
    return demos.create_performance_analysis_demo()

def create_error_analysis_demo(config: Optional[CoreConfig] = None) -> gr.Interface:
    """Create error analysis demo."""
    demos = InteractiveModelDemos(config)
    return demos.create_error_analysis_demo()

__all__ = [
    "InteractiveModelDemos",
    "create_text_generation_demo",
    "create_image_generation_demo", 
    "create_model_comparison_demo",
    "create_training_visualization_demo",
    "create_performance_analysis_demo",
    "create_error_analysis_demo"
]
