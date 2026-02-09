"""
Gradio Integration - Integración con Gradio para demos interactivos
====================================================================
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GradioComponent:
    """Componente de Gradio"""
    component_type: str
    label: str
    default_value: Any = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GradioInterface:
    """Interfaz de Gradio"""
    name: str
    description: str
    inputs: List[GradioComponent]
    outputs: List[GradioComponent]
    function: Callable
    examples: Optional[List[List[Any]]] = None
    theme: str = "default"
    share: bool = False


class GradioManager:
    """Gestor de interfaces Gradio"""
    
    def __init__(self):
        self.interfaces: Dict[str, GradioInterface] = {}
        self.apps: Dict[str, Any] = {}
    
    def create_interface(
        self,
        name: str,
        description: str,
        inputs: List[GradioComponent],
        outputs: List[GradioComponent],
        function: Callable,
        examples: Optional[List[List[Any]]] = None,
        theme: str = "default",
        share: bool = False
    ) -> GradioInterface:
        """Crea una interfaz de Gradio"""
        interface = GradioInterface(
            name=name,
            description=description,
            inputs=inputs,
            outputs=outputs,
            function=function,
            examples=examples,
            theme=theme,
            share=share
        )
        
        self.interfaces[name] = interface
        logger.info(f"Interfaz Gradio {name} creada")
        return interface
    
    def build_app(self, interface_name: str) -> Any:
        """Construye una app de Gradio"""
        try:
            import gradio as gr
            
            if interface_name not in self.interfaces:
                raise ValueError(f"Interfaz {interface_name} no encontrada")
            
            interface = self.interfaces[interface_name]
            
            # Convertir componentes a objetos Gradio
            gradio_inputs = []
            for inp in interface.inputs:
                comp = self._create_gradio_component(inp)
                gradio_inputs.append(comp)
            
            gradio_outputs = []
            for out in interface.outputs:
                comp = self._create_gradio_component(out)
                gradio_outputs.append(comp)
            
            # Crear interfaz
            app = gr.Interface(
                fn=interface.function,
                inputs=gradio_inputs,
                outputs=gradio_outputs,
                title=interface.name,
                description=interface.description,
                examples=interface.examples,
                theme=interface.theme
            )
            
            self.apps[interface_name] = app
            logger.info(f"App Gradio {interface_name} construida")
            return app
        except ImportError:
            logger.error("Gradio no instalado")
            raise
    
    def _create_gradio_component(self, component: GradioComponent) -> Any:
        """Crea un componente de Gradio"""
        try:
            import gradio as gr
            
            component_map = {
                "textbox": gr.Textbox,
                "number": gr.Number,
                "slider": gr.Slider,
                "dropdown": gr.Dropdown,
                "checkbox": gr.Checkbox,
                "radio": gr.Radio,
                "image": gr.Image,
                "audio": gr.Audio,
                "video": gr.Video,
                "file": gr.File,
                "dataframe": gr.Dataframe,
                "json": gr.JSON,
                "html": gr.HTML,
                "markdown": gr.Markdown
            }
            
            comp_class = component_map.get(component.component_type, gr.Textbox)
            default_kwargs = {"label": component.label}
            
            if component.default_value is not None:
                default_kwargs["value"] = component.default_value
            
            default_kwargs.update(component.kwargs)
            
            return comp_class(**default_kwargs)
        except ImportError:
            logger.error("Gradio no instalado")
            return None
    
    def launch(
        self,
        interface_name: str,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False
    ):
        """Lanza una interfaz de Gradio"""
        if interface_name not in self.apps:
            self.build_app(interface_name)
        
        app = self.apps[interface_name]
        app.launch(
            server_name=server_name,
            server_port=server_port,
            share=share
        )
    
    def create_code_improvement_demo(
        self,
        improvement_function: Callable
    ) -> str:
        """Crea un demo de mejora de código"""
        try:
            import gradio as gr
            
            def demo_function(code: str, language: str = "python") -> str:
                """Función del demo"""
                try:
                    result = improvement_function(code, language)
                    return result
                except Exception as e:
                    return f"Error: {str(e)}"
            
            interface = gr.Interface(
                fn=demo_function,
                inputs=[
                    gr.Textbox(
                        label="Código",
                        placeholder="Pega tu código aquí...",
                        lines=10
                    ),
                    gr.Dropdown(
                        label="Lenguaje",
                        choices=["python", "javascript", "typescript", "java", "cpp"],
                        value="python"
                    )
                ],
                outputs=gr.Textbox(
                    label="Código Mejorado",
                    lines=15
                ),
                title="Code Improvement Demo",
                description="Mejora tu código usando IA basada en research papers"
            )
            
            return interface
        except ImportError:
            logger.error("Gradio no instalado")
            return None




