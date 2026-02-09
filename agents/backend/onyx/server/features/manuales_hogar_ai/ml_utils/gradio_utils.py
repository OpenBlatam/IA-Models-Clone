"""
Gradio Utils - Utilidades de Gradio
===================================

Utilidades para crear interfaces Gradio para modelos ML/DL.
"""

import logging
from typing import Optional, Callable, Dict, Any, List, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Intentar importar gradio
try:
    import gradio as gr
    _has_gradio = True
except ImportError:
    _has_gradio = False
    logger.warning("gradio library not available")


class GradioInterfaceBuilder:
    """
    Builder para crear interfaces Gradio.
    """
    
    def __init__(self, title: str = "ML Model Interface", description: str = ""):
        """
        Inicializar builder.
        
        Args:
            title: Título de la interfaz
            description: Descripción
        """
        if not _has_gradio:
            raise ImportError("gradio library is required")
        
        self.title = title
        self.description = description
        self.components: List[Any] = []
        self.functions: List[Callable] = []
    
    def add_text_input(
        self,
        label: str,
        placeholder: Optional[str] = None,
        default: Optional[str] = None
    ) -> 'GradioInterfaceBuilder':
        """
        Agregar input de texto.
        
        Args:
            label: Etiqueta
            placeholder: Placeholder
            default: Valor por defecto
            
        Returns:
            Self para chaining
        """
        component = gr.Textbox(
            label=label,
            placeholder=placeholder,
            value=default
        )
        self.components.append(component)
        return self
    
    def add_number_input(
        self,
        label: str,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        default: Optional[float] = None
    ) -> 'GradioInterfaceBuilder':
        """
        Agregar input numérico.
        
        Args:
            label: Etiqueta
            minimum: Valor mínimo
            maximum: Valor máximo
            default: Valor por defecto
            
        Returns:
            Self para chaining
        """
        component = gr.Number(
            label=label,
            minimum=minimum,
            maximum=maximum,
            value=default
        )
        self.components.append(component)
        return self
    
    def add_image_input(
        self,
        label: str = "Input Image",
        image_type: str = "pil"
    ) -> 'GradioInterfaceBuilder':
        """
        Agregar input de imagen.
        
        Args:
            label: Etiqueta
            image_type: Tipo de imagen (pil, numpy, etc.)
            
        Returns:
            Self para chaining
        """
        component = gr.Image(label=label, type=image_type)
        self.components.append(component)
        return self
    
    def add_slider(
        self,
        label: str,
        minimum: float,
        maximum: float,
        default: Optional[float] = None,
        step: Optional[float] = None
    ) -> 'GradioInterfaceBuilder':
        """
        Agregar slider.
        
        Args:
            label: Etiqueta
            minimum: Valor mínimo
            maximum: Valor máximo
            default: Valor por defecto
            step: Paso
            
        Returns:
            Self para chaining
        """
        component = gr.Slider(
            label=label,
            minimum=minimum,
            maximum=maximum,
            value=default,
            step=step
        )
        self.components.append(component)
        return self
    
    def add_dropdown(
        self,
        label: str,
        choices: List[str],
        default: Optional[str] = None
    ) -> 'GradioInterfaceBuilder':
        """
        Agregar dropdown.
        
        Args:
            label: Etiqueta
            choices: Opciones
            default: Valor por defecto
            
        Returns:
            Self para chaining
        """
        component = gr.Dropdown(
            label=label,
            choices=choices,
            value=default
        )
        self.components.append(component)
        return self
    
    def set_function(self, func: Callable) -> 'GradioInterfaceBuilder':
        """
        Establecer función de procesamiento.
        
        Args:
            func: Función a ejecutar
            
        Returns:
            Self para chaining
        """
        self.functions.append(func)
        return self
    
    def build(
        self,
        outputs: Optional[List[Any]] = None,
        examples: Optional[List[Any]] = None,
        theme: Optional[str] = None
    ) -> gr.Blocks:
        """
        Construir interfaz Gradio.
        
        Args:
            outputs: Componentes de salida
            examples: Ejemplos
            theme: Tema
            
        Returns:
            Interfaz Gradio
        """
        if not self.functions:
            raise ValueError("No function set")
        
        func = self.functions[0]
        
        if outputs is None:
            outputs = [gr.Textbox(label="Output")]
        
        interface = gr.Interface(
            fn=func,
            inputs=self.components,
            outputs=outputs,
            title=self.title,
            description=self.description,
            examples=examples,
            theme=theme
        )
        
        return interface
    
    def build_advanced(
        self,
        outputs: Optional[List[Any]] = None,
        examples: Optional[List[Any]] = None,
        theme: Optional[str] = None
    ) -> gr.Blocks:
        """
        Construir interfaz Gradio avanzada con Blocks.
        
        Args:
            outputs: Componentes de salida
            examples: Ejemplos
            theme: Tema
            
        Returns:
            Blocks de Gradio
        """
        if not self.functions:
            raise ValueError("No function set")
        
        func = self.functions[0]
        
        if outputs is None:
            outputs = [gr.Textbox(label="Output")]
        
        with gr.Blocks(title=self.title, theme=theme) as blocks:
            gr.Markdown(f"# {self.title}")
            if self.description:
                gr.Markdown(self.description)
            
            with gr.Row():
                with gr.Column():
                    for component in self.components:
                        component.render()
                    
                    if examples:
                        gr.Examples(examples=examples)
                
                with gr.Column():
                    for output in outputs:
                        output.render()
            
            # Botón de procesamiento
            process_btn = gr.Button("Process", variant="primary")
            process_btn.click(
                fn=func,
                inputs=self.components,
                outputs=outputs
            )
        
        return blocks


def create_model_demo(
    predict_fn: Callable,
    input_type: str = "text",
    output_type: str = "text",
    title: str = "Model Demo",
    description: str = "",
    examples: Optional[List[Any]] = None
) -> gr.Interface:
    """
    Crear demo simple de modelo.
    
    Args:
        predict_fn: Función de predicción
        input_type: Tipo de input (text, image, audio)
        output_type: Tipo de output (text, image, audio)
        title: Título
        description: Descripción
        examples: Ejemplos
        
    Returns:
        Interfaz Gradio
    """
    if not _has_gradio:
        raise ImportError("gradio library is required")
    
    # Inputs
    if input_type == "text":
        inputs = gr.Textbox(label="Input Text")
    elif input_type == "image":
        inputs = gr.Image(label="Input Image")
    else:
        inputs = gr.Textbox(label="Input")
    
    # Outputs
    if output_type == "text":
        outputs = gr.Textbox(label="Output")
    elif output_type == "image":
        outputs = gr.Image(label="Output Image")
    else:
        outputs = gr.Textbox(label="Output")
    
    interface = gr.Interface(
        fn=predict_fn,
        inputs=inputs,
        outputs=outputs,
        title=title,
        description=description,
        examples=examples
    )
    
    return interface


def create_comparison_demo(
    models: Dict[str, Callable],
    input_type: str = "text",
    title: str = "Model Comparison",
    description: str = ""
) -> gr.Blocks:
    """
    Crear demo de comparación de modelos.
    
    Args:
        models: Diccionario de modelos {name: predict_fn}
        input_type: Tipo de input
        title: Título
        description: Descripción
        
    Returns:
        Blocks de Gradio
    """
    if not _has_gradio:
        raise ImportError("gradio library is required")
    
    with gr.Blocks(title=title) as blocks:
        gr.Markdown(f"# {title}")
        if description:
            gr.Markdown(description)
        
        # Input compartido
        if input_type == "text":
            shared_input = gr.Textbox(label="Input Text")
        elif input_type == "image":
            shared_input = gr.Image(label="Input Image")
        else:
            shared_input = gr.Textbox(label="Input")
        
        # Outputs para cada modelo
        with gr.Row():
            for model_name, predict_fn in models.items():
                with gr.Column():
                    gr.Markdown(f"### {model_name}")
                    output = gr.Textbox(label="Output")
                    btn = gr.Button(f"Run {model_name}")
                    btn.click(
                        fn=predict_fn,
                        inputs=shared_input,
                        outputs=output
                    )
    
    return blocks




