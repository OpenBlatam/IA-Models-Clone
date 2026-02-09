"""
Modular Model Inference Component for Gradio
"""

from typing import Optional, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not available")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class ModelInferenceComponent:
    """
    Modular component for model inference in Gradio
    """
    
    def __init__(
        self,
        model: Any,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
        device: str = "cuda"
    ):
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio required")
        
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.device = device
        
        # Set model to eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
    
    def predict(
        self,
        input_data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference on input data
        
        Args:
            input_data: Input data (audio, text, features, etc.)
            **kwargs: Additional inference parameters
        
        Returns:
            Prediction results
        """
        try:
            # Preprocess
            if self.preprocess_fn:
                processed_input = self.preprocess_fn(input_data)
            else:
                processed_input = input_data
            
            # Inference
            if hasattr(self.model, '__call__'):
                output = self.model(processed_input, **kwargs)
            else:
                output = self.model.predict(processed_input, **kwargs)
            
            # Postprocess
            if self.postprocess_fn:
                output = self.postprocess_fn(output)
            
            return {
                "success": True,
                "output": output
            }
        
        except Exception as e:
            logger.error(f"Inference error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_interface(
        self,
        input_type: str = "audio",
        output_type: str = "json",
        title: str = "Model Inference"
    ) -> gr.Interface:
        """
        Create Gradio interface
        
        Args:
            input_type: Type of input ("audio", "text", "number", etc.)
            output_type: Type of output ("json", "text", "audio", etc.)
            title: Interface title
        
        Returns:
            Gradio Interface
        """
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio required")
        
        # Define input component
        if input_type == "audio":
            input_component = gr.Audio(type="numpy")
        elif input_type == "text":
            input_component = gr.Textbox(label="Input")
        elif input_type == "number":
            input_component = gr.Number(label="Input")
        else:
            input_component = gr.Textbox(label="Input")
        
        # Define output component
        if output_type == "json":
            output_component = gr.JSON(label="Output")
        elif output_type == "text":
            output_component = gr.Textbox(label="Output")
        elif output_type == "audio":
            output_component = gr.Audio(label="Output")
        else:
            output_component = gr.JSON(label="Output")
        
        # Create interface
        interface = gr.Interface(
            fn=self.predict,
            inputs=input_component,
            outputs=output_component,
            title=title,
            description=f"Inference using {type(self.model).__name__}"
        )
        
        return interface



