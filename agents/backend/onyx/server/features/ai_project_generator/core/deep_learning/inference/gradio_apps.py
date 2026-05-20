"""
Gradio Apps - Interactive Demos for Model Inference
====================================================

Creates user-friendly Gradio interfaces for model inference and visualization.
"""

import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
import torch
import torch.nn as nn
import gradio as gr
import numpy as np
from PIL import Image

from .inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


def create_gradio_app(
    model: nn.Module,
    inference_fn: Optional[Callable] = None,
    input_components: Optional[List] = None,
    output_components: Optional[List] = None,
    title: str = "Model Inference Demo",
    description: str = "Interactive demo for model inference",
    examples: Optional[List] = None,
    device: Optional[torch.device] = None
) -> gr.Blocks:
    """
    Create a Gradio app for model inference.
    
    Args:
        model: PyTorch model
        inference_fn: Custom inference function (uses InferenceEngine if None)
        input_components: Gradio input components
        output_components: Gradio output components
        title: App title
        description: App description
        examples: Example inputs
        device: Device to run inference on
        
    Returns:
        Gradio Blocks interface
    """
    # Create inference engine if no custom function provided
    if inference_fn is None:
        engine = InferenceEngine(model, device=device)
        
        def default_inference_fn(*args):
            """Default inference function."""
            try:
                # Convert Gradio inputs to model inputs
                inputs = _prepare_gradio_inputs(args, model)
                predictions = engine.predict(inputs, return_probabilities=True)
                return _format_gradio_outputs(predictions)
            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                return f"Error: {str(e)}"
        
        inference_fn = default_inference_fn
    
    # Default input components if not provided
    if input_components is None:
        input_components = [gr.Textbox(label="Input Text", lines=3)]
    
    # Default output components if not provided
    if output_components is None:
        output_components = [gr.Textbox(label="Output")]
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=inference_fn,
        inputs=input_components,
        outputs=output_components,
        title=title,
        description=description,
        examples=examples
    )
    
    return interface


def create_text_classification_app(
    model: nn.Module,
    tokenizer: Optional[Callable] = None,
    class_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None
) -> gr.Blocks:
    """
    Create a Gradio app for text classification.
    
    Args:
        model: Text classification model
        tokenizer: Tokenizer function
        class_names: List of class names
        device: Device to run inference on
        
    Returns:
        Gradio Blocks interface
    """
    engine = InferenceEngine(model, device=device)
    
    def classify_text(text: str) -> str:
        """Classify input text."""
        try:
            # Tokenize if tokenizer provided
            if tokenizer:
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            else:
                # Fallback: simple encoding
                inputs = {'input_ids': torch.tensor([[hash(text) % 10000]])}
            
            # Run inference
            predictions = engine.predict(inputs, return_probabilities=True, top_k=5)
            
            # Format output
            if isinstance(predictions, dict) and 'probabilities' in predictions:
                probs = predictions['probabilities'][0]
                if class_names:
                    results = []
                    for idx, prob in enumerate(probs):
                        class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
                        results.append(f"{class_name}: {prob:.2%}")
                    return "\n".join(results)
                else:
                    return f"Prediction: {probs.argmax().item()} (confidence: {probs.max():.2%})"
            else:
                return str(predictions)
                
        except Exception as e:
            logger.error(f"Classification error: {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    interface = gr.Interface(
        fn=classify_text,
        inputs=gr.Textbox(label="Input Text", lines=5, placeholder="Enter text to classify..."),
        outputs=gr.Textbox(label="Classification Results", lines=10),
        title="Text Classification Demo",
        description="Classify text using the trained model",
        examples=None
    )
    
    return interface


def create_image_classification_app(
    model: nn.Module,
    class_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None
) -> gr.Blocks:
    """
    Create a Gradio app for image classification.
    
    Args:
        model: Image classification model
        class_names: List of class names
        device: Device to run inference on
        
    Returns:
        Gradio Blocks interface
    """
    engine = InferenceEngine(model, device=device)
    
    def classify_image(image: Image.Image) -> str:
        """Classify input image."""
        try:
            # Convert PIL image to tensor
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0)
            
            # Run inference
            predictions = engine.predict(image_tensor, return_probabilities=True, top_k=5)
            
            # Format output
            if isinstance(predictions, dict) and 'probabilities' in predictions:
                probs = predictions['probabilities'][0]
                if class_names:
                    results = []
                    top_indices = probs.argsort(descending=True)[:5]
                    for idx in top_indices:
                        class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
                        results.append(f"{class_name}: {probs[idx]:.2%}")
                    return "\n".join(results)
                else:
                    return f"Prediction: {probs.argmax().item()} (confidence: {probs.max():.2%})"
            else:
                return str(predictions)
                
        except Exception as e:
            logger.error(f"Classification error: {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    interface = gr.Interface(
        fn=classify_image,
        inputs=gr.Image(type="pil", label="Input Image"),
        outputs=gr.Textbox(label="Classification Results", lines=10),
        title="Image Classification Demo",
        description="Classify images using the trained model",
        examples=None
    )
    
    return interface


def _prepare_gradio_inputs(args: Tuple, model: nn.Module) -> Dict[str, torch.Tensor]:
    """Prepare Gradio inputs for model."""
    # Simple implementation - can be customized
    if len(args) == 1:
        # Single input
        input_data = args[0]
        if isinstance(input_data, str):
            # Text input
            return {'input_ids': torch.tensor([[hash(input_data) % 10000]])}
        elif isinstance(input_data, Image.Image):
            # Image input
            import torchvision.transforms as transforms
            transform = transforms.ToTensor()
            return {'pixel_values': transform(input_data).unsqueeze(0)}
        else:
            return {'input': torch.tensor([input_data])}
    else:
        # Multiple inputs
        return {f'input_{i}': torch.tensor([arg]) for i, arg in enumerate(args)}


def _format_gradio_outputs(predictions: Any) -> str:
    """Format model predictions for Gradio output."""
    if isinstance(predictions, dict):
        if 'probabilities' in predictions:
            probs = predictions['probabilities']
            if isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy()
            return f"Probabilities: {probs}"
        else:
            return str(predictions)
    elif isinstance(predictions, torch.Tensor):
        return f"Predictions: {predictions.cpu().numpy()}"
    else:
        return str(predictions)



