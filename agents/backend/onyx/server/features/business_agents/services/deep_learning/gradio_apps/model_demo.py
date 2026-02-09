"""
Gradio Interface - Interactive Model Demo
==========================================

User-friendly Gradio interface for model inference and visualization.
"""

import gradio as gr
import torch
import numpy as np
from typing import Callable, Optional, Dict, Any
import logging

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class ModelDemo:
    """Gradio demo for model inference."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        inference_fn: Callable,
        device: torch.device,
        title: str = "Model Demo",
        description: str = "Interactive model inference"
    ):
        """
        Initialize model demo.
        
        Args:
            model: Model for inference
            inference_fn: Function to perform inference
            device: Target device
            title: Demo title
            description: Demo description
        """
        self.model = model
        self.inference_fn = inference_fn
        self.device = device
        self.title = title
        self.description = description
        self.model.eval()
    
    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Perform prediction with error handling.
        
        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments
        
        Returns:
            Prediction results
        """
        try:
            result = self.inference_fn(self.model, *args, device=self.device, **kwargs)
            return result
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return {"error": str(e)}
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        # This is a template - customize based on your model type
        with gr.Blocks(title=self.title) as demo:
            gr.Markdown(f"# {self.title}\n{self.description}")
            
            with gr.Row():
                input_component = gr.Textbox(
                    label="Input",
                    placeholder="Enter your input here..."
                )
            
            with gr.Row():
                submit_btn = gr.Button("Predict", variant="primary")
                clear_btn = gr.Button("Clear")
            
            with gr.Row():
                output_component = gr.JSON(label="Output")
            
            submit_btn.click(
                fn=self.predict,
                inputs=[input_component],
                outputs=[output_component]
            )
            
            clear_btn.click(
                fn=lambda: ("", {}),
                outputs=[input_component, output_component]
            )
        
        return demo


def create_model_demo(
    model: torch.nn.Module,
    inference_fn: Callable,
    device: torch.device,
    **kwargs
) -> gr.Blocks:
    """
    Create Gradio demo for model.
    
    Args:
        model: Model for inference
        inference_fn: Inference function
        device: Target device
        **kwargs: Additional demo arguments
    
    Returns:
        Gradio Blocks interface
    """
    demo = ModelDemo(model, inference_fn, device, **kwargs)
    return demo.create_interface()



