"""
Interface Generator - Generador de interfaces Gradio
======================================================

Genera interfaces interactivas con Gradio para diferentes tipos de modelos.
Incluye validación de inputs, manejo de errores robusto, y visualizaciones.
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class InterfaceGenerator:
    """Generador de interfaces Gradio"""
    
    def __init__(self):
        """Inicializa el generador de interfaces"""
        pass
    
    def generate(
        self,
        services_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera interfaces Gradio.
        
        Args:
            services_dir: Directorio donde generar las interfaces
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        services_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar interfaces según el tipo
        if keywords.get("is_diffusion"):
            self._generate_diffusion_interface(services_dir, keywords, project_info)
        
        if keywords.get("is_transformer") or keywords.get("is_llm"):
            self._generate_text_generation_interface(services_dir, keywords, project_info)
        
        if keywords.get("ai_type") == "classification":
            self._generate_classification_interface(services_dir, keywords, project_info)
        
        # Generar factory e interfaz general
        self._generate_interface_factory(services_dir, keywords, project_info)
        self._generate_general_interface(services_dir, keywords, project_info)
        # Generar utilidades de validación
        self._generate_validation_utils(services_dir, keywords, project_info)
    
    def _generate_interface_factory(
        self,
        services_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera factory para crear interfaces"""
        
        factory_content = f'''"""
Gradio Interface Factory - Factory para crear interfaces
{'=' * 60}

Crea interfaces Gradio según el tipo de modelo.
"""

import gradio as gr
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_gradio_interface(model, task_type: str = "general", **kwargs):
    """
    Crea una interfaz Gradio según el tipo de tarea.
    
    Args:
        model: Modelo a usar
        task_type: Tipo de tarea (diffusion, generation, classification, etc.)
        **kwargs: Argumentos adicionales
        
    Returns:
        Interfaz Gradio
    """
    if task_type == "diffusion":
        from .diffusion_interface import create_diffusion_interface
        return create_diffusion_interface(model, **kwargs)
    
    elif task_type in ["generation", "llm", "transformer"]:
        from .text_generation_interface import create_text_generation_interface
        return create_text_generation_interface(model, **kwargs)
    
    elif task_type == "classification":
        from .classification_interface import create_classification_interface
        return create_classification_interface(model, **kwargs)
    
    else:
        from .general_interface import create_general_interface
        return create_general_interface(model, **kwargs)


def launch_interface(
    interface: gr.Interface,
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
):
    """
    Lanza la interfaz Gradio.
    
    Args:
        interface: Interfaz Gradio
        share: Si crear enlace público
        server_name: Nombre del servidor
        server_port: Puerto del servidor
    """
    interface.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
    )
'''
        
        (services_dir / "gradio_interface.py").write_text(factory_content, encoding="utf-8")
    
    def _generate_diffusion_interface(
        self,
        services_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera interfaz para diffusion models"""
        
        interface_content = '''"""
Diffusion Interface - Interfaz Gradio para modelos de difusión
================================================================
"""

import gradio as gr
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_diffusion_interface(model, **kwargs):
    """Crea interfaz para modelos de difusión con validación mejorada"""
    from .validation_utils import validate_and_sanitize_input
    
    def generate_image(prompt: str, negative_prompt: str, num_steps: int, guidance_scale: float):
        try:
            # Validar y sanitizar inputs
            prompt, error = validate_and_sanitize_input(
                prompt,
                input_type="prompt",
                max_length=500
            )
            if error:
                return None, f"Error en prompt: {error}"
            
            if negative_prompt:
                negative_prompt, error = validate_and_sanitize_input(
                    negative_prompt,
                    input_type="text",
                    max_length=500
                )
                if error:
                    logger.warning(f"Error en negative prompt: {error}")
                    negative_prompt = None
            
            num_steps, error = validate_and_sanitize_input(
                num_steps,
                input_type="number",
                min_value=10,
                max_value=100
            )
            if error:
                return None, f"Error en num_steps: {error}"
            
            guidance_scale, error = validate_and_sanitize_input(
                guidance_scale,
                input_type="number",
                min_value=1.0,
                max_value=20.0
            )
            if error:
                return None, f"Error en guidance_scale: {error}"
            
            # Generar imagen
            images = model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=int(num_steps),
                guidance_scale=float(guidance_scale),
            )
            return images[0] if images else None
        
        except Exception as e:
            logger.error(f"Error generando imagen: {e}", exc_info=True)
            return None, f"Error generando imagen: {str(e)}"
    
    interface = gr.Interface(
        fn=generate_image,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="Describe la imagen que quieres generar..."),
            gr.Textbox(label="Negative Prompt (opcional)", placeholder="Qué evitar en la imagen..."),
            gr.Slider(minimum=20, maximum=100, value=50, step=5, label="Inference Steps"),
            gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.5, label="Guidance Scale"),
        ],
        outputs=gr.Image(label="Generated Image"),
        title="Image Generation - Diffusion Model",
        description="Genera imágenes usando modelos de difusión",
    )
    
    return interface
'''
        
        (services_dir / "diffusion_interface.py").write_text(interface_content, encoding="utf-8")
    
    def _generate_text_generation_interface(
        self,
        services_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera interfaz para generación de texto"""
        
        interface_content = '''"""
Text Generation Interface - Interfaz Gradio para generación de texto
======================================================================
"""

import gradio as gr
import logging

logger = logging.getLogger(__name__)


def create_text_generation_interface(model, **kwargs):
    """Crea interfaz para generación de texto con validación mejorada"""
    from .validation_utils import validate_and_sanitize_input
    
    def generate_text(prompt: str, max_length: int, temperature: float):
        try:
            # Validar inputs
            prompt, error = validate_and_sanitize_input(
                prompt,
                input_type="prompt",
                max_length=1000
            )
            if error:
                return f"Error en prompt: {error}"
            
            max_length, error = validate_and_sanitize_input(
                max_length,
                input_type="number",
                min_value=10,
                max_value=2048
            )
            if error:
                return f"Error en max_length: {error}"
            
            temperature, error = validate_and_sanitize_input(
                temperature,
                input_type="number",
                min_value=0.1,
                max_value=2.0
            )
            if error:
                return f"Error en temperature: {error}"
            
            # Generar texto
            result = model.predict(prompt, max_length=int(max_length))
            return result.get("generated_text", "Error generando texto")
        
        except Exception as e:
            logger.error(f"Error generando texto: {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    interface = gr.Interface(
        fn=generate_text,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="Escribe tu prompt aquí..."),
            gr.Slider(minimum=50, maximum=512, value=100, step=10, label="Max Length"),
            gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        ],
        outputs=gr.Textbox(label="Generated Text"),
        title="Text Generation - LLM",
        description="Genera texto usando modelos de lenguaje",
    )
    
    return interface
'''
        
        (services_dir / "text_generation_interface.py").write_text(interface_content, encoding="utf-8")
    
    def _generate_classification_interface(
        self,
        services_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera interfaz para clasificación"""
        
        interface_content = '''"""
Classification Interface - Interfaz Gradio para clasificación
================================================================
"""

import gradio as gr
import logging

logger = logging.getLogger(__name__)


def create_classification_interface(model, **kwargs):
    """Crea interfaz para clasificación"""
    
    def classify_text(text: str):
        try:
            result = model.predict(text)
            return f"Clase: {result.get('predicted_class', 'N/A')}, Confianza: {result.get('confidence', 0):.2%}"
        except Exception as e:
            logger.error(f"Error clasificando: {e}")
            return f"Error: {str(e)}"
    
    interface = gr.Interface(
        fn=classify_text,
        inputs=gr.Textbox(label="Text", placeholder="Texto a clasificar..."),
        outputs=gr.Textbox(label="Classification Result"),
        title="Text Classification",
        description="Clasifica texto usando modelos de IA",
    )
    
    return interface
'''
        
        (services_dir / "classification_interface.py").write_text(interface_content, encoding="utf-8")
    
    def _generate_general_interface(
        self,
        services_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera interfaz general"""
        
        interface_content = '''"""
General Interface - Interfaz Gradio general
=============================================

Interfaz genérica para cualquier tipo de modelo.
"""

import gradio as gr
import logging

logger = logging.getLogger(__name__)


def create_general_interface(model, **kwargs):
    """Crea interfaz general"""
    
    def process(input_text: str):
        try:
            if hasattr(model, 'predict'):
                result = model.predict(input_text)
            elif hasattr(model, 'generate'):
                result = model.generate(input_text)
            else:
                result = f"Modelo procesado: {input_text}"
            return str(result)
        except Exception as e:
            logger.error(f"Error procesando: {e}")
            return f"Error: {str(e)}"
    
    interface = gr.Interface(
        fn=process,
        inputs=gr.Textbox(label="Input", placeholder="Ingresa tu input aquí..."),
        outputs=gr.Textbox(label="Output"),
        title="AI Model Interface",
        description="Interfaz general para modelos de IA",
    )
    
    return interface
'''
        
        (services_dir / "general_interface.py").write_text(interface_content, encoding="utf-8")

