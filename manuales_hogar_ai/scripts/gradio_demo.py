"""
Demo Gradio
===========

Demo interactivo con Gradio para el sistema de manuales.
"""

import gradio as gr
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.manual_generator_model import ManualGeneratorModel
from ml.image_generation.image_generator import ImageGenerator
from ml.embeddings.embedding_service import EmbeddingService
from ml.config.ml_config import get_ml_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar servicios
config = get_ml_config()

# Modelos (cargar bajo demanda)
_generator_model = None
_image_generator = None
_embedding_service = None


def get_generator_model():
    """Obtener modelo generador (lazy loading)."""
    global _generator_model
    if _generator_model is None:
        _generator_model = ManualGeneratorModel(
            model_name=config.generation_model,
            use_lora=config.use_lora,
            device=config.device
        )
    return _generator_model


def get_image_generator():
    """Obtener generador de imágenes (lazy loading)."""
    global _image_generator
    if _image_generator is None:
        _image_generator = ImageGenerator(
            model_name=config.image_model,
            use_xl=config.use_sd_xl,
            device=config.device
        )
    return _image_generator


def get_embedding_service():
    """Obtener servicio de embeddings (lazy loading)."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(
            model_name=config.embedding_model,
            device=config.device
        )
    return _embedding_service


def generate_manual(
    problem: str,
    category: str,
    use_local: bool
) -> str:
    """Generar manual."""
    try:
        if use_local:
            model = get_generator_model()
            manual = model.generate_manual(problem, category)
        else:
            # Usar OpenRouter (implementar si es necesario)
            return "Usa el endpoint de API para OpenRouter"
        
        return manual
    except Exception as e:
        return f"Error: {str(e)}"


def generate_illustration(
    step_description: str,
    category: str,
    style: str
):
    """Generar ilustración."""
    try:
        generator = get_image_generator()
        image = generator.generate_manual_illustration(
            step_description=step_description,
            category=category,
            style=style
        )
        return image
    except Exception as e:
        return None


def calculate_similarity(text1: str, text2: str) -> str:
    """Calcular similitud."""
    try:
        service = get_embedding_service()
        sim = service.similarity(text1, text2)
        return f"Similitud: {sim:.2%}"
    except Exception as e:
        return f"Error: {str(e)}"


# Crear interfaz Gradio
with gr.Blocks(title="Manuales Hogar AI - Demo") as demo:
    gr.Markdown("# 🏠 Manuales Hogar AI - Demo Interactivo")
    gr.Markdown("Genera manuales paso a paso tipo LEGO para oficios populares")
    
    with gr.Tabs():
        with gr.Tab("Generar Manual"):
            with gr.Row():
                with gr.Column():
                    problem_input = gr.Textbox(
                        label="Descripción del Problema",
                        placeholder="Ej: Tengo una fuga de agua en el grifo de la cocina",
                        lines=5
                    )
                    category_input = gr.Dropdown(
                        label="Categoría",
                        choices=[
                            "plomeria", "techos", "carpinteria", "electricidad",
                            "albanileria", "pintura", "herreria", "jardineria", "general"
                        ],
                        value="general"
                    )
                    use_local = gr.Checkbox(
                        label="Usar Modelo Local",
                        value=False
                    )
                    generate_btn = gr.Button("Generar Manual", variant="primary")
                
                with gr.Column():
                    manual_output = gr.Textbox(
                        label="Manual Generado",
                        lines=20,
                        interactive=False
                    )
            
            generate_btn.click(
                fn=generate_manual,
                inputs=[problem_input, category_input, use_local],
                outputs=manual_output
            )
        
        with gr.Tab("Generar Ilustración"):
            with gr.Row():
                with gr.Column():
                    step_input = gr.Textbox(
                        label="Descripción del Paso",
                        placeholder="Ej: Cerrar la llave de paso principal",
                        lines=3
                    )
                    category_ill = gr.Dropdown(
                        label="Categoría",
                        choices=[
                            "plomeria", "techos", "carpinteria", "electricidad",
                            "albanileria", "pintura", "herreria", "jardineria", "general"
                        ],
                        value="general"
                    )
                    style_input = gr.Dropdown(
                        label="Estilo",
                        choices=["lego_instruction", "technical_diagram", "realistic"],
                        value="lego_instruction"
                    )
                    generate_ill_btn = gr.Button("Generar Ilustración", variant="primary")
                
                with gr.Column():
                    image_output = gr.Image(label="Ilustración Generada")
            
            generate_ill_btn.click(
                fn=generate_illustration,
                inputs=[step_input, category_ill, style_input],
                outputs=image_output
            )
        
        with gr.Tab("Similitud Semántica"):
            with gr.Row():
                with gr.Column():
                    text1_input = gr.Textbox(
                        label="Texto 1",
                        lines=3
                    )
                    text2_input = gr.Textbox(
                        label="Texto 2",
                        lines=3
                    )
                    similarity_btn = gr.Button("Calcular Similitud", variant="primary")
                
                with gr.Column():
                    similarity_output = gr.Textbox(
                        label="Resultado",
                        lines=2
                    )
            
            similarity_btn.click(
                fn=calculate_similarity,
                inputs=[text1_input, text2_input],
                outputs=similarity_output
            )
    
    gr.Markdown("### 💡 Notas")
    gr.Markdown("""
    - Los modelos se cargan bajo demanda (lazy loading)
    - La primera generación puede tardar más
    - Se requiere GPU para mejor rendimiento
    - Los modelos locales son opcionales
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )




