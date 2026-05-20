"""
Gradio Interface - Interfaz interactiva con Gradio
==================================================
Interfaz de usuario para demostración y visualización de modelos
"""

import logging
from typing import Dict, List, Any, Optional
import os

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logging.warning("Gradio library not available")

logger = logging.getLogger(__name__)


class GradioInterface:
    """Interfaz Gradio para el sistema"""
    
    def __init__(self, prototype_generator=None, llm_system=None, diffusion_system=None):
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio library is required")
        
        self.prototype_generator = prototype_generator
        self.llm_system = llm_system
        self.diffusion_system = diffusion_system
        self.app = None
    
    def create_interface(self) -> gr.Blocks:
        """Crea interfaz Gradio"""
        with gr.Blocks(title="3D Prototype AI", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🏭 3D Prototype AI - Sistema de Generación de Prototipos")
            
            with gr.Tabs():
                # Tab 1: Generación de Prototipos
                with gr.Tab("🎨 Generar Prototipo"):
                    with gr.Row():
                        with gr.Column():
                            product_input = gr.Textbox(
                                label="Descripción del Producto",
                                placeholder="Ej: Quiero hacer una licuadora potente",
                                lines=3
                            )
                            product_type = gr.Dropdown(
                                label="Tipo de Producto",
                                choices=["licuadora", "estufa", "maquina", "electrodomestico", "herramienta", "mueble", "dispositivo", "otro"],
                                value="licuadora"
                            )
                            budget = gr.Slider(
                                label="Presupuesto (USD)",
                                minimum=50,
                                maximum=1000,
                                value=150,
                                step=10
                            )
                            generate_btn = gr.Button("Generar Prototipo", variant="primary")
                        
                        with gr.Column():
                            output_json = gr.JSON(label="Resultado")
                            output_markdown = gr.Markdown(label="Resumen")
                    
                    generate_btn.click(
                        fn=self._generate_prototype,
                        inputs=[product_input, product_type, budget],
                        outputs=[output_json, output_markdown]
                    )
                
                # Tab 2: Generación con LLM
                with gr.Tab("🤖 LLM Avanzado"):
                    with gr.Row():
                        with gr.Column():
                            llm_prompt = gr.Textbox(
                                label="Prompt para LLM",
                                placeholder="Describe el producto en detalle...",
                                lines=5
                            )
                            llm_model = gr.Dropdown(
                                label="Modelo LLM",
                                choices=["gpt2", "distilgpt2"],
                                value="gpt2"
                            )
                            llm_generate_btn = gr.Button("Generar con LLM", variant="primary")
                        
                        with gr.Column():
                            llm_output = gr.Textbox(label="Texto Generado", lines=10)
                    
                    llm_generate_btn.click(
                        fn=self._generate_with_llm,
                        inputs=[llm_prompt, llm_model],
                        outputs=[llm_output]
                    )
                
                # Tab 3: Generación de Imágenes
                with gr.Tab("🎨 Generar Imagen"):
                    with gr.Row():
                        with gr.Column():
                            image_prompt = gr.Textbox(
                                label="Prompt para Imagen",
                                placeholder="3D model of a blender, high quality, detailed",
                                lines=3
                            )
                            negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="blurry, low quality, distorted",
                                lines=2
                            )
                            num_images = gr.Slider(
                                label="Número de Imágenes",
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1
                            )
                            image_generate_btn = gr.Button("Generar Imagen", variant="primary")
                        
                        with gr.Column():
                            image_output = gr.Gallery(label="Imágenes Generadas")
                    
                    image_generate_btn.click(
                        fn=self._generate_image,
                        inputs=[image_prompt, negative_prompt, num_images],
                        outputs=[image_output]
                    )
            
            self.app = app
            return app
    
    def _generate_prototype(self, description: str, product_type: str, budget: float):
        """Genera prototipo"""
        if not self.prototype_generator:
            return {"error": "Prototype generator not available"}, "Error: Generador no disponible"
        
        try:
            from ..models.schemas import PrototypeRequest, ProductType
            
            request = PrototypeRequest(
                product_description=description,
                product_type=ProductType(product_type),
                budget=budget
            )
            
            import asyncio
            response = asyncio.run(self.prototype_generator.generate_prototype(request))
            
            # Formatear markdown
            markdown = f"""
# {response.product_name}

## Especificaciones
{response.specifications}

## Costo Total Estimado
${response.total_cost_estimate}

## Materiales Necesarios
{len(response.materials)} materiales

## Dificultad
{response.difficulty_level}
"""
            
            return response.dict(), markdown
        except Exception as e:
            logger.error(f"Error generating prototype: {e}")
            return {"error": str(e)}, f"Error: {str(e)}"
    
    def _generate_with_llm(self, prompt: str, model_id: str):
        """Genera texto con LLM"""
        if not self.llm_system:
            return "LLM system not available"
        
        try:
            result = self.llm_system.generate_text(model_id, prompt)
            return result["generated_text"]
        except Exception as e:
            logger.error(f"Error generating with LLM: {e}")
            return f"Error: {str(e)}"
    
    def _generate_image(self, prompt: str, negative_prompt: str, num_images: int):
        """Genera imagen"""
        if not self.diffusion_system:
            return []
        
        try:
            result = self.diffusion_system.generate_image(
                "default",
                prompt,
                negative_prompt if negative_prompt else None,
                num_images
            )
            return result["images"]
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return []
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
        """Lanza la interfaz"""
        if not self.app:
            self.create_interface()
        
        self.app.launch(share=share, server_name=server_name, server_port=server_port)




