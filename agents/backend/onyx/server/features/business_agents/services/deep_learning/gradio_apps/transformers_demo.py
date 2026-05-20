"""
Gradio Demo for Transformers Models
====================================

Interactive interface for HuggingFace transformer models.
"""

import gradio as gr
import torch
from typing import Optional, Dict, Any
import logging

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

try:
    from ..models.transformers_models import HuggingFaceModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    HuggingFaceModel = None


def create_transformers_demo(
    model: HuggingFaceModel,
    title: str = "Transformer Model Demo",
    description: str = "Interactive inference with HuggingFace transformers"
) -> gr.Blocks:
    """
    Create Gradio demo for transformer model.
    
    Args:
        model: HuggingFace model instance
        title: Demo title
        description: Demo description
    
    Returns:
        Gradio Blocks interface
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers models not available")
    
    def predict(text: str, max_length: int = 512) -> Dict[str, Any]:
        """Perform prediction."""
        try:
            if not text.strip():
                return {"error": "Please enter some text"}
            
            result = model.predict(text, max_length=max_length, return_probs=True)
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}
    
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\n{description}")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter your text here...",
                    lines=5
                )
                max_length_slider = gr.Slider(
                    minimum=64,
                    maximum=512,
                    value=512,
                    step=64,
                    label="Max Length"
                )
                predict_btn = gr.Button("Predict", variant="primary")
            
            with gr.Column():
                output_json = gr.JSON(label="Output")
                output_text = gr.Textbox(
                    label="Formatted Output",
                    lines=10
                )
        
        predict_btn.click(
            fn=predict,
            inputs=[text_input, max_length_slider],
            outputs=[output_json, output_text]
        )
        
        gr.Examples(
            examples=[
                ["This is a great movie!"],
                ["I love this product."],
                ["The service was terrible."]
            ],
            inputs=[text_input]
        )
    
    return demo



