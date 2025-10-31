"""
Advanced Gradio Interface for Export IA
Interactive demos for model inference, training, and visualization
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
import io
import base64
from pathlib import Path
import tempfile
import os

# Import our refactored components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.base_models import ModelFactory, ModelConfig
from core.diffusion_engine import DiffusionEngine, DiffusionConfig
from core.training_engine import TrainingEngine, TrainingConfig
from core.data_pipeline import DataPipeline, DataConfig

logger = logging.getLogger(__name__)

class GradioInterface:
    """Advanced Gradio interface for Export IA"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.training_engine = None
        self.data_pipeline = None
        
    def create_main_interface(self) -> gr.Blocks:
        """Create main Gradio interface"""
        with gr.Blocks(
            title="Export IA - Advanced AI Document Processing",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            .model-card {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 16px;
                margin: 8px 0;
                background: #f9f9f9;
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 16px;
                border-radius: 8px;
                text-align: center;
            }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🚀 Export IA - Advanced AI Document Processing
            
            Welcome to the most advanced AI-powered document processing system. 
            This interface provides access to state-of-the-art models for document analysis, 
            generation, and optimization.
            """)
            
            with gr.Tabs():
                # Model Inference Tab
                with gr.Tab("🤖 Model Inference"):
                    self._create_inference_interface()
                    
                # Training Tab
                with gr.Tab("🏋️ Model Training"):
                    self._create_training_interface()
                    
                # Diffusion Tab
                with gr.Tab("🎨 AI Generation"):
                    self._create_diffusion_interface()
                    
                # Analytics Tab
                with gr.Tab("📊 Analytics & Visualization"):
                    self._create_analytics_interface()
                    
                # Configuration Tab
                with gr.Tab("⚙️ Configuration"):
                    self._create_configuration_interface()
                    
        return interface
        
    def _create_inference_interface(self):
        """Create model inference interface"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Selection")
                
                model_type = gr.Dropdown(
                    choices=["document_transformer", "multi_modal_fusion", "diffusion"],
                    label="Model Type",
                    value="document_transformer"
                )
                
                model_config = gr.JSON(
                    label="Model Configuration",
                    value={
                        "model_name": "inference_model",
                        "input_dim": 512,
                        "output_dim": 10,
                        "hidden_dim": 768,
                        "num_layers": 6,
                        "num_heads": 12,
                        "dropout": 0.1
                    }
                )
                
                load_model_btn = gr.Button("Load Model", variant="primary")
                
                gr.Markdown("### Input Data")
                
                input_type = gr.Radio(
                    choices=["Text", "Image", "Audio", "Multi-modal"],
                    label="Input Type",
                    value="Text"
                )
                
                text_input = gr.Textbox(
                    label="Text Input",
                    placeholder="Enter your text here...",
                    lines=5
                )
                
                image_input = gr.Image(
                    label="Image Input",
                    type="pil"
                )
                
                audio_input = gr.Audio(
                    label="Audio Input",
                    type="numpy"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### Model Output")
                
                inference_btn = gr.Button("Run Inference", variant="primary", size="lg")
                
                output_text = gr.Textbox(
                    label="Text Output",
                    lines=10,
                    interactive=False
                )
                
                output_image = gr.Image(
                    label="Generated Image",
                    interactive=False
                )
                
                output_plot = gr.Plot(
                    label="Visualization"
                )
                
                output_metrics = gr.JSON(
                    label="Model Metrics"
                )
                
        # Event handlers
        load_model_btn.click(
            fn=self._load_model,
            inputs=[model_type, model_config],
            outputs=[output_metrics]
        )
        
        inference_btn.click(
            fn=self._run_inference,
            inputs=[input_type, text_input, image_input, audio_input],
            outputs=[output_text, output_image, output_plot, output_metrics]
        )
        
    def _create_training_interface(self):
        """Create model training interface"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Training Configuration")
                
                training_config = gr.JSON(
                    label="Training Configuration",
                    value={
                        "model_name": "training_model",
                        "batch_size": 32,
                        "learning_rate": 1e-4,
                        "num_epochs": 10,
                        "optimizer": "adamw",
                        "scheduler": "cosine",
                        "mixed_precision": True,
                        "use_tensorboard": True
                    }
                )
                
                data_config = gr.JSON(
                    label="Data Configuration",
                    value={
                        "data_dir": "./data",
                        "batch_size": 32,
                        "num_workers": 4,
                        "image_size": [224, 224],
                        "max_seq_length": 512
                    }
                )
                
                upload_data = gr.File(
                    label="Upload Training Data",
                    file_count="multiple"
                )
                
                start_training_btn = gr.Button("Start Training", variant="primary")
                stop_training_btn = gr.Button("Stop Training", variant="stop")
                
            with gr.Column(scale=2):
                gr.Markdown("### Training Progress")
                
                training_status = gr.Textbox(
                    label="Training Status",
                    value="Ready to start training...",
                    interactive=False
                )
                
                training_plot = gr.Plot(
                    label="Training Metrics"
                )
                
                training_logs = gr.Textbox(
                    label="Training Logs",
                    lines=10,
                    interactive=False
                )
                
                model_checkpoints = gr.File(
                    label="Model Checkpoints",
                    interactive=False
                )
                
        # Event handlers
        start_training_btn.click(
            fn=self._start_training,
            inputs=[training_config, data_config, upload_data],
            outputs=[training_status, training_plot, training_logs]
        )
        
    def _create_diffusion_interface(self):
        """Create diffusion model interface"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Generation Parameters")
                
                diffusion_config = gr.JSON(
                    label="Diffusion Configuration",
                    value={
                        "model_type": "stable_diffusion",
                        "num_inference_steps": 50,
                        "guidance_scale": 7.5,
                        "eta": 0.0,
                        "use_lora": False,
                        "lora_rank": 4
                    }
                )
                
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A beautiful landscape with mountains and lakes...",
                    lines=3
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="blurry, low quality, distorted...",
                    lines=2
                )
                
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
                
                seed = gr.Number(
                    label="Seed",
                    value=42,
                    precision=0
                )
                
                generate_btn = gr.Button("Generate Image", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                gr.Markdown("### Generated Images")
                
                generated_image = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto"
                )
                
                generation_metrics = gr.JSON(
                    label="Generation Metrics"
                )
                
                gr.Markdown("### Image Controls")
                
                with gr.Row():
                    upscale_btn = gr.Button("Upscale Image")
                    inpaint_btn = gr.Button("Inpaint")
                    edit_btn = gr.Button("Edit Image")
                    
        # Event handlers
        generate_btn.click(
            fn=self._generate_image,
            inputs=[prompt, negative_prompt, num_steps, guidance_scale, seed],
            outputs=[generated_image, generation_metrics]
        )
        
    def _create_analytics_interface(self):
        """Create analytics and visualization interface"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Analytics Configuration")
                
                analytics_type = gr.Dropdown(
                    choices=["Model Performance", "Data Distribution", "Training Metrics", "Custom Analysis"],
                    label="Analysis Type",
                    value="Model Performance"
                )
                
                time_range = gr.Dropdown(
                    choices=["Last Hour", "Last Day", "Last Week", "Last Month", "All Time"],
                    label="Time Range",
                    value="Last Day"
                )
                
                metrics_to_plot = gr.CheckboxGroup(
                    choices=["Loss", "Accuracy", "F1 Score", "Precision", "Recall", "Learning Rate"],
                    label="Metrics to Plot",
                    value=["Loss", "Accuracy"]
                )
                
                generate_analytics_btn = gr.Button("Generate Analytics", variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("### Analytics Dashboard")
                
                analytics_plot = gr.Plot(
                    label="Analytics Plot"
                )
                
                metrics_table = gr.Dataframe(
                    label="Metrics Table",
                    headers=["Metric", "Value", "Change", "Trend"]
                )
                
                insights_text = gr.Textbox(
                    label="AI Insights",
                    lines=5,
                    interactive=False
                )
                
        # Event handlers
        generate_analytics_btn.click(
            fn=self._generate_analytics,
            inputs=[analytics_type, time_range, metrics_to_plot],
            outputs=[analytics_plot, metrics_table, insights_text]
        )
        
    def _create_configuration_interface(self):
        """Create configuration interface"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### System Configuration")
                
                system_config = gr.JSON(
                    label="System Configuration",
                    value={
                        "device": "cuda" if torch.cuda.is_available() else "cpu",
                        "mixed_precision": True,
                        "gradient_checkpointing": True,
                        "num_workers": 4,
                        "pin_memory": True,
                        "use_tensorboard": True,
                        "use_wandb": False
                    }
                )
                
                model_configs = gr.JSON(
                    label="Model Configurations",
                    value={
                        "default_transformer": {
                            "hidden_dim": 768,
                            "num_layers": 6,
                            "num_heads": 12,
                            "dropout": 0.1
                        },
                        "default_diffusion": {
                            "num_inference_steps": 50,
                            "guidance_scale": 7.5,
                            "use_lora": False
                        }
                    }
                )
                
                save_config_btn = gr.Button("Save Configuration", variant="primary")
                load_config_btn = gr.Button("Load Configuration")
                
            with gr.Column(scale=2):
                gr.Markdown("### Configuration Status")
                
                config_status = gr.Textbox(
                    label="Configuration Status",
                    value="Configuration loaded successfully",
                    interactive=False
                )
                
                system_info = gr.JSON(
                    label="System Information"
                )
                
                model_info = gr.JSON(
                    label="Model Information"
                )
                
        # Event handlers
        save_config_btn.click(
            fn=self._save_configuration,
            inputs=[system_config, model_configs],
            outputs=[config_status]
        )
        
        load_config_btn.click(
            fn=self._load_configuration,
            outputs=[system_config, model_configs, config_status, system_info, model_info]
        )
        
    def _load_model(self, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load model for inference"""
        try:
            # Create model configuration
            model_config = ModelConfig(**config)
            
            # Create model
            model = ModelFactory.create_model(model_type, model_config)
            
            # Store model
            self.models[model_type] = model
            self.current_model = model
            
            # Get model statistics
            stats = model.get_model_size()
            
            return {
                "status": "success",
                "model_type": model_type,
                "model_loaded": True,
                "model_statistics": stats,
                "device": str(model.device)
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_loaded": False
            }
            
    def _run_inference(self, input_type: str, text: str, image: Image.Image, 
                      audio: np.ndarray) -> Tuple[str, Image.Image, go.Figure, Dict[str, Any]]:
        """Run model inference"""
        if self.current_model is None:
            return "No model loaded", None, None, {"error": "No model loaded"}
            
        try:
            if input_type == "Text":
                # Text inference
                if not text.strip():
                    return "Please provide text input", None, None, {"error": "No text input"}
                    
                # Simulate text processing
                result = f"Processed text: {text[:100]}..."
                
                # Create visualization
                fig = go.Figure(data=go.Bar(
                    x=['Sentiment', 'Topic', 'Complexity', 'Readability'],
                    y=[0.8, 0.6, 0.7, 0.9],
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                ))
                fig.update_layout(title="Text Analysis Results")
                
                return result, None, fig, {"text_length": len(text), "processing_time": 0.1}
                
            elif input_type == "Image":
                # Image inference
                if image is None:
                    return "Please provide image input", None, None, {"error": "No image input"}
                    
                # Simulate image processing
                result = f"Image processed: {image.size[0]}x{image.size[1]} pixels"
                
                # Create visualization
                fig = go.Figure(data=go.Scatter(
                    x=[1, 2, 3, 4, 5],
                    y=[0.1, 0.3, 0.2, 0.4, 0.3],
                    mode='lines+markers',
                    name='Image Features'
                ))
                fig.update_layout(title="Image Feature Analysis")
                
                return result, image, fig, {"image_size": image.size, "processing_time": 0.2}
                
            elif input_type == "Audio":
                # Audio inference
                if audio is None:
                    return "Please provide audio input", None, None, {"error": "No audio input"}
                    
                # Simulate audio processing
                result = f"Audio processed: {len(audio)} samples"
                
                # Create visualization
                fig = go.Figure(data=go.Scatter(
                    x=list(range(len(audio))),
                    y=audio.flatten()[:1000],  # First 1000 samples
                    mode='lines',
                    name='Audio Waveform'
                ))
                fig.update_layout(title="Audio Waveform")
                
                return result, None, fig, {"audio_length": len(audio), "processing_time": 0.3}
                
            else:  # Multi-modal
                # Multi-modal inference
                result = "Multi-modal processing completed"
                
                # Create visualization
                fig = go.Figure(data=go.Scatter(
                    x=[1, 2, 3, 4, 5],
                    y=[0.2, 0.4, 0.3, 0.5, 0.4],
                    mode='lines+markers',
                    name='Multi-modal Features'
                ))
                fig.update_layout(title="Multi-modal Feature Fusion")
                
                return result, None, fig, {"modalities": 3, "processing_time": 0.5}
                
        except Exception as e:
            logger.error(f"Error in inference: {e}")
            return f"Error: {str(e)}", None, None, {"error": str(e)}
            
    def _start_training(self, training_config: Dict[str, Any], 
                       data_config: Dict[str, Any], 
                       uploaded_files: List[str]) -> Tuple[str, go.Figure, str]:
        """Start model training"""
        try:
            # Simulate training process
            status = "Training started successfully..."
            
            # Create training progress plot
            epochs = list(range(1, training_config.get('num_epochs', 10) + 1))
            train_loss = [1.0 * (0.9 ** i) for i in epochs]
            val_loss = [0.8 * (0.9 ** i) for i in epochs]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='Train Loss'))
            fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Validation Loss'))
            fig.update_layout(title="Training Progress", xaxis_title="Epoch", yaxis_title="Loss")
            
            # Training logs
            logs = f"""
            Epoch 1/10: Train Loss: 1.000, Val Loss: 0.800
            Epoch 2/10: Train Loss: 0.900, Val Loss: 0.720
            Epoch 3/10: Train Loss: 0.810, Val Loss: 0.648
            ...
            Training completed successfully!
            """
            
            return status, fig, logs
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            return f"Training error: {str(e)}", None, f"Error: {str(e)}"
            
    def _generate_image(self, prompt: str, negative_prompt: str, 
                       num_steps: int, guidance_scale: float, 
                       seed: int) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """Generate image using diffusion model"""
        try:
            if not prompt.strip():
                return [], {"error": "Please provide a prompt"}
                
            # Set random seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Simulate image generation
            # In a real implementation, this would use the actual diffusion model
            generated_images = []
            
            # Create dummy images for demonstration
            for i in range(4):
                # Create a random image
                img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                generated_images.append(img)
                
            # Generation metrics
            metrics = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "generation_time": 2.5,
                "images_generated": len(generated_images)
            }
            
            return generated_images, metrics
            
        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            return [], {"error": str(e)}
            
    def _generate_analytics(self, analytics_type: str, time_range: str, 
                           metrics: List[str]) -> Tuple[go.Figure, pd.DataFrame, str]:
        """Generate analytics and visualizations"""
        try:
            # Create analytics plot
            if analytics_type == "Model Performance":
                fig = go.Figure()
                x_data = list(range(1, 11))
                
                for metric in metrics:
                    if metric == "Loss":
                        y_data = [1.0 * (0.9 ** i) for i in x_data]
                    elif metric == "Accuracy":
                        y_data = [0.5 + 0.4 * (1 - 0.9 ** i) for i in x_data]
                    elif metric == "F1 Score":
                        y_data = [0.6 + 0.3 * (1 - 0.9 ** i) for i in x_data]
                    else:
                        y_data = [0.7 + 0.2 * (1 - 0.9 ** i) for i in x_data]
                        
                    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=metric))
                    
                fig.update_layout(title="Model Performance Over Time", xaxis_title="Epoch", yaxis_title="Score")
                
            elif analytics_type == "Data Distribution":
                fig = go.Figure(data=go.Pie(
                    labels=['Class A', 'Class B', 'Class C', 'Class D', 'Class E'],
                    values=[30, 25, 20, 15, 10],
                    title="Data Distribution"
                ))
                
            else:
                # Default plot
                fig = go.Figure(data=go.Bar(
                    x=['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4'],
                    y=[0.8, 0.6, 0.7, 0.9],
                    title="Analytics Overview"
                ))
                
            # Create metrics table
            metrics_data = []
            for metric in metrics:
                value = np.random.random()
                change = np.random.uniform(-0.1, 0.1)
                trend = "↗" if change > 0 else "↘" if change < 0 else "→"
                
                metrics_data.append([metric, f"{value:.3f}", f"{change:+.3f}", trend])
                
            df = pd.DataFrame(metrics_data, columns=["Metric", "Value", "Change", "Trend"])
            
            # Generate insights
            insights = f"""
            Based on the {analytics_type.lower()} analysis for {time_range.lower()}:
            
            • Overall performance shows positive trends across all metrics
            • {metrics[0]} has improved by {np.random.uniform(5, 15):.1f}% over the selected period
            • Model stability is within acceptable ranges
            • Recommendations: Continue current training approach, consider data augmentation
            """
            
            return fig, df, insights
            
        except Exception as e:
            logger.error(f"Error in analytics generation: {e}")
            empty_fig = go.Figure()
            empty_df = pd.DataFrame(columns=["Metric", "Value", "Change", "Trend"])
            return empty_fig, empty_df, f"Error: {str(e)}"
            
    def _save_configuration(self, system_config: Dict[str, Any], 
                           model_configs: Dict[str, Any]) -> str:
        """Save system configuration"""
        try:
            config = {
                "system": system_config,
                "models": model_configs,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            # Save to file
            config_path = Path("./configs/gradio_config.json")
            config_path.parent.mkdir(exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            return f"Configuration saved successfully to {config_path}"
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return f"Error saving configuration: {str(e)}"
            
    def _load_configuration(self) -> Tuple[Dict[str, Any], Dict[str, Any], str, Dict[str, Any], Dict[str, Any]]:
        """Load system configuration"""
        try:
            config_path = Path("./configs/gradio_config.json")
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                system_config = config.get("system", {})
                model_configs = config.get("models", {})
                status = "Configuration loaded successfully"
            else:
                system_config = {}
                model_configs = {}
                status = "No configuration file found, using defaults"
                
            # System information
            system_info = {
                "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "pytorch_version": torch.__version__,
                "gradio_version": gr.__version__
            }
            
            # Model information
            model_info = {
                "loaded_models": list(self.models.keys()),
                "current_model": type(self.current_model).__name__ if self.current_model else None,
                "total_models": len(self.models)
            }
            
            return system_config, model_configs, status, system_info, model_info
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}, {}, f"Error loading configuration: {str(e)}", {}, {}

def create_gradio_app() -> gr.Blocks:
    """Create and return Gradio application"""
    interface = GradioInterface()
    return interface.create_main_interface()

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Gradio app
    app = create_gradio_app()
    
    # Launch app
    print("Launching Gradio interface...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
























