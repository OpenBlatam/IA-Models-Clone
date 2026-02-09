"""
Advanced Gradio Apps - Enhanced Gradio Interfaces
==================================================

Advanced Gradio interfaces with:
- Multiple input types
- Real-time visualization
- Model comparison
- Batch processing
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
import torch
import torch.nn as nn
import gradio as gr
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def create_model_comparison_app(
    models: Dict[str, nn.Module],
    inference_fn: Optional[Callable] = None,
    input_type: str = 'text',
    title: str = "Model Comparison"
) -> gr.Blocks:
    """
    Create a Gradio app for comparing multiple models.
    
    Args:
        models: Dictionary of model_name -> model
        inference_fn: Custom inference function
        input_type: Type of input ('text', 'image')
        title: App title
        
    Returns:
        Gradio Blocks interface
    """
    def compare_models(input_data: Any) -> Dict[str, str]:
        """Compare outputs from all models."""
        results = {}
        
        for name, model in models.items():
            try:
                if inference_fn:
                    output = inference_fn(model, input_data)
                else:
                    # Default inference
                    if input_type == 'text':
                        # Simple text processing
                        output = f"Output from {name}: {input_data[:50]}..."
                    else:
                        output = f"Output from {name}: processed"
                
                results[name] = str(output)
            except Exception as e:
                results[name] = f"Error: {str(e)}"
        
        return results
    
    if input_type == 'text':
        input_component = gr.Textbox(label="Input Text", lines=5)
    else:
        input_component = gr.Image(type="pil", label="Input Image")
    
    interface = gr.Interface(
        fn=compare_models,
        inputs=input_component,
        outputs=gr.JSON(label="Model Outputs"),
        title=title,
        description="Compare outputs from multiple models"
    )
    
    return interface


def create_interactive_training_app(
    model: nn.Module,
    train_fn: Callable,
    config: Dict[str, Any]
) -> gr.Blocks:
    """
    Create an interactive app for training visualization.
    
    Args:
        model: PyTorch model
        train_fn: Training function
        config: Training configuration
        
    Returns:
        Gradio Blocks interface
    """
    def train_model(
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
        progress=gr.Progress()
    ) -> Tuple[str, Dict[str, List[float]]]:
        """Train model with interactive parameters."""
        config['learning_rate'] = learning_rate
        config['batch_size'] = batch_size
        config['num_epochs'] = num_epochs
        
        try:
            history = train_fn(model, config, progress=progress)
            
            # Format results
            results_text = f"Training completed!\n"
            results_text += f"Final train loss: {history['train_loss'][-1]:.4f}\n"
            if 'val_loss' in history:
                results_text += f"Final val loss: {history['val_loss'][-1]:.4f}\n"
            
            return results_text, history
        except Exception as e:
            return f"Error: {str(e)}", {}
    
    with gr.Blocks(title="Interactive Training") as app:
        gr.Markdown("# Interactive Model Training")
        
        with gr.Row():
            with gr.Column():
                lr = gr.Slider(1e-5, 1e-2, value=1e-4, label="Learning Rate", log=True)
                bs = gr.Slider(8, 128, value=32, step=8, label="Batch Size")
                epochs = gr.Slider(1, 50, value=10, step=1, label="Epochs")
                train_btn = gr.Button("Train Model", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(label="Training Results", lines=10)
                output_plot = gr.Plot(label="Training Curves")
        
        train_btn.click(
            fn=train_model,
            inputs=[lr, bs, epochs],
            outputs=[output_text, output_plot]
        )
    
    return app


def create_batch_inference_app(
    model: nn.Module,
    inference_fn: Callable,
    max_batch_size: int = 10
) -> gr.Blocks:
    """
    Create an app for batch inference.
    
    Args:
        model: PyTorch model
        inference_fn: Inference function
        max_batch_size: Maximum batch size
        
    Returns:
        Gradio Blocks interface
    """
    def process_batch(inputs: List[str]) -> List[str]:
        """Process batch of inputs."""
        results = []
        
        for input_data in inputs:
            try:
                output = inference_fn(model, input_data)
                results.append(str(output))
            except Exception as e:
                results.append(f"Error: {str(e)}")
        
        return results
    
    interface = gr.Interface(
        fn=process_batch,
        inputs=gr.Dataframe(
            headers=["Input"],
            label="Batch Inputs",
            row_count=(1, max_batch_size)
        ),
        outputs=gr.Dataframe(
            headers=["Output"],
            label="Batch Outputs"
        ),
        title="Batch Inference",
        description=f"Process up to {max_batch_size} inputs at once"
    )
    
    return interface



