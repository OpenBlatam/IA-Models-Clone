"""
Gradio Integration Module
==========================

Integración profesional con Gradio para interfaces interactivas.
Permite crear demos interactivos para modelos de deep learning.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Union
import numpy as np

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logging.warning("Gradio not available. Install with: pip install gradio")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

logger = logging.getLogger(__name__)


class GradioModelInterface:
    """
    Interfaz Gradio profesional para modelos de deep learning.
    
    Soporta:
    - Modelos de clasificación
    - Modelos de regresión
    - Modelos de generación de texto
    - Modelos de generación de imágenes
    - Visualización de resultados
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_fn: Optional[Callable] = None,
        title: str = "Model Inference",
        description: str = "Interactive model inference interface"
    ):
        """
        Inicializar interfaz Gradio.
        
        Args:
            model: Modelo PyTorch (opcional si se usa model_fn)
            model_fn: Función de inferencia personalizada
            title: Título de la interfaz
            description: Descripción de la interfaz
        """
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required. Install with: pip install gradio")
        
        self.model = model
        self.model_fn = model_fn or (lambda x: self._default_predict(x))
        self.title = title
        self.description = description
        self.interface = None
        
        if self.model:
            self.model.eval()
        
        logger.info("GradioModelInterface initialized")
    
    def _default_predict(self, inputs: Any) -> Any:
        """Predicción por defecto."""
        if self.model is None:
            raise ValueError("Model or model_fn must be provided")
        
        # Implementación básica - debe ser sobrescrita
        return "Prediction not implemented"
    
    def create_classification_interface(
        self,
        class_names: List[str],
        input_type: str = "image",
        examples: Optional[List] = None
    ) -> gr.Blocks:
        """
        Crear interfaz para clasificación.
        
        Args:
            class_names: Lista de nombres de clases
            input_type: Tipo de entrada ("image", "text", "audio")
            examples: Ejemplos para la interfaz
            
        Returns:
            Interfaz Gradio
        """
        def predict_fn(input_data):
            try:
                prediction = self.model_fn(input_data)
                
                if isinstance(prediction, (list, np.ndarray)):
                    if len(prediction) == len(class_names):
                        # Probabilidades
                        results = {name: float(prob) for name, prob in zip(class_names, prediction)}
                    else:
                        # Índice de clase
                        class_idx = int(np.argmax(prediction))
                        results = {class_names[class_idx]: float(prediction[class_idx])}
                else:
                    results = {"prediction": str(prediction)}
                
                return results
            except Exception as e:
                logger.error(f"Prediction error: {e}", exc_info=True)
                return {"error": str(e)}
        
        # Crear inputs según tipo
        if input_type == "image":
            input_component = gr.Image(type="pil", label="Input Image")
        elif input_type == "text":
            input_component = gr.Textbox(label="Input Text", lines=3)
        else:
            input_component = gr.File(label="Input File")
        
        with gr.Blocks(title=self.title) as interface:
            gr.Markdown(f"## {self.title}\n{self.description}")
            
            with gr.Row():
                with gr.Column():
                    input_component
                    submit_btn = gr.Button("Predict", variant="primary")
                
                with gr.Column():
                    output = gr.Label(label="Predictions", num_top_classes=len(class_names))
            
            if examples:
                gr.Examples(examples=examples, inputs=input_component)
            
            submit_btn.click(fn=predict_fn, inputs=input_component, outputs=output)
        
        self.interface = interface
        return interface
    
    def create_text_generation_interface(
        self,
        max_length: int = 100,
        temperature: float = 0.7,
        examples: Optional[List[str]] = None
    ) -> gr.Blocks:
        """
        Crear interfaz para generación de texto.
        
        Args:
            max_length: Longitud máxima
            temperature: Temperature para sampling
            examples: Ejemplos de prompts
            
        Returns:
            Interfaz Gradio
        """
        def generate_fn(prompt: str, max_len: int, temp: float):
            try:
                result = self.model_fn(prompt, max_length=max_len, temperature=temp)
                return result
            except Exception as e:
                logger.error(f"Generation error: {e}", exc_info=True)
                return f"Error: {str(e)}"
        
        with gr.Blocks(title=self.title) as interface:
            gr.Markdown(f"## {self.title}\n{self.description}")
            
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3
                    )
                    max_len = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=max_length,
                        step=10,
                        label="Max Length"
                    )
                    temp = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=temperature,
                        step=0.1,
                        label="Temperature"
                    )
                    generate_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    output = gr.Textbox(
                        label="Generated Text",
                        lines=10,
                        interactive=False
                    )
            
            if examples:
                gr.Examples(examples=examples, inputs=prompt)
            
            generate_btn.click(
                fn=generate_fn,
                inputs=[prompt, max_len, temp],
                outputs=output
            )
        
        self.interface = interface
        return interface
    
    def create_image_generation_interface(
        self,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        examples: Optional[List[str]] = None
    ) -> gr.Blocks:
        """
        Crear interfaz para generación de imágenes.
        
        Args:
            num_inference_steps: Número de pasos de inferencia
            guidance_scale: Guidance scale
            examples: Ejemplos de prompts
            
        Returns:
            Interfaz Gradio
        """
        def generate_fn(prompt: str, steps: int, guidance: float):
            try:
                image = self.model_fn(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance
                )
                
                if isinstance(image, np.ndarray):
                    if PIL_AVAILABLE:
                        image = Image.fromarray(image.astype('uint8'))
                    else:
                        return "PIL not available for image conversion"
                
                return image
            except Exception as e:
                logger.error(f"Image generation error: {e}", exc_info=True)
                return None
        
        with gr.Blocks(title=self.title) as interface:
            gr.Markdown(f"## {self.title}\n{self.description}")
            
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to generate...",
                        lines=3
                    )
                    steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=num_inference_steps,
                        step=5,
                        label="Inference Steps"
                    )
                    guidance = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=guidance_scale,
                        step=0.5,
                        label="Guidance Scale"
                    )
                    generate_btn = gr.Button("Generate Image", variant="primary")
                
                with gr.Column():
                    output = gr.Image(label="Generated Image", type="pil")
            
            if examples:
                gr.Examples(examples=examples, inputs=prompt)
            
            generate_btn.click(
                fn=generate_fn,
                inputs=[prompt, steps, guidance],
                outputs=output
            )
        
        self.interface = interface
        return interface
    
    def launch(
        self,
        share: bool = False,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        **kwargs
    ):
        """
        Lanzar interfaz Gradio.
        
        Args:
            share: Compartir públicamente
            server_name: Nombre del servidor
            server_port: Puerto del servidor
            **kwargs: Argumentos adicionales para launch()
        """
        if self.interface is None:
            raise ValueError("Interface not created. Call create_*_interface first.")
        
        self.interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            **kwargs
        )
        logger.info(f"Gradio interface launched on {server_name}:{server_port}")


def create_model_comparison_interface(
    models: Dict[str, Callable],
    input_component: gr.components.Component,
    output_component: gr.components.Component
) -> gr.Blocks:
    """
    Crear interfaz para comparar múltiples modelos.
    
    Args:
        models: Dict con nombre -> función de predicción
        input_component: Componente de entrada
        output_component: Componente de salida
        
    Returns:
        Interfaz Gradio
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is required")
    
    def compare_models(input_data):
        results = {}
        for name, model_fn in models.items():
            try:
                prediction = model_fn(input_data)
                results[name] = prediction
            except Exception as e:
                results[name] = f"Error: {str(e)}"
        return results
    
    with gr.Blocks(title="Model Comparison") as interface:
        gr.Markdown("## Model Comparison\nCompare predictions from multiple models")
        
        input_component
        compare_btn = gr.Button("Compare Models", variant="primary")
        
        with gr.Row():
            for name in models.keys():
                with gr.Column():
                    gr.Markdown(f"### {name}")
                    output_component
        
        compare_btn.click(fn=compare_models, inputs=input_component, outputs=output_component)
    
    return interface

