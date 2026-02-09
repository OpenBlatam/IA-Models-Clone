"""
Gradio Demo - Demo Interactivo
===============================

Interfaz Gradio para demostración de capacidades de IA.
"""

import gradio as gr
import logging
from typing import Optional
import torch

from .content_analyzer import ContentAnalyzer
from .text_generator import AdvancedTextGenerator
from .image_generator import ImageGenerator

logger = logging.getLogger(__name__)


class GradioDemo:
    """Demo interactivo con Gradio"""
    
    def __init__(
        self,
        enable_image_gen: bool = True,
        device: Optional[str] = None
    ):
        """
        Inicializar demo
        
        Args:
            enable_image_gen: Habilitar generación de imágenes
            device: Dispositivo
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Inicializar componentes
        self.content_analyzer = ContentAnalyzer(device=self.device)
        self.text_generator = AdvancedTextGenerator(device=self.device)
        
        if enable_image_gen and self.device == "cuda":
            try:
                self.image_generator = ImageGenerator(device=self.device)
            except Exception as e:
                logger.warning(f"No se pudo cargar generador de imágenes: {e}")
                self.image_generator = None
        else:
            self.image_generator = None
    
    def analyze_content_interface(self, text: str, platform: str) -> str:
        """Interfaz para análisis de contenido"""
        try:
            analysis = self.content_analyzer.analyze_content_quality(text, platform)
            
            result = f"""
**Análisis de Contenido:**
- Longitud: {analysis['length']} caracteres
- Palabras: {analysis['word_count']}
- Sentimiento: {analysis['sentiment']['label']} (confianza: {analysis['sentiment']['score']:.2f})
- Dentro del límite: {'✅' if analysis['within_limit'] else '❌'}
- Uso: {analysis['usage_percentage']:.1f}% del límite
- Hashtags: {'✅' if analysis['has_hashtags'] else '❌'}
- Menciones: {'✅' if analysis['has_mentions'] else '❌'}
- Enlaces: {'✅' if analysis['has_links'] else '❌'}
"""
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_text_interface(
        self,
        topic: str,
        platform: str,
        tone: str,
        length: str
    ) -> str:
        """Interfaz para generación de texto"""
        try:
            generated = self.text_generator.generate_post(
                topic=topic,
                platform=platform,
                tone=tone,
                length=length
            )
            return generated
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_image_interface(self, prompt: str) -> Optional:
        """Interfaz para generación de imágenes"""
        if not self.image_generator:
            return None
        
        try:
            image = self.image_generator.generate(prompt)
            return image
        except Exception as e:
            logger.error(f"Error generando imagen: {e}")
            return None
    
    def create_interface(self) -> gr.Blocks:
        """Crear interfaz Gradio completa"""
        with gr.Blocks(title="Community Manager AI - Demo") as demo:
            gr.Markdown("# 🚀 Community Manager AI - Demo Interactivo")
            
            with gr.Tabs():
                # Tab 1: Análisis de Contenido
                with gr.Tab("📊 Análisis de Contenido"):
                    gr.Markdown("### Analiza la calidad de tu contenido")
                    content_input = gr.Textbox(
                        label="Contenido",
                        placeholder="Escribe tu post aquí...",
                        lines=5
                    )
                    platform_select = gr.Dropdown(
                        choices=["twitter", "facebook", "instagram", "linkedin"],
                        label="Plataforma",
                        value="twitter"
                    )
                    analyze_btn = gr.Button("Analizar", variant="primary")
                    analysis_output = gr.Markdown()
                    
                    analyze_btn.click(
                        self.analyze_content_interface,
                        inputs=[content_input, platform_select],
                        outputs=analysis_output
                    )
                
                # Tab 2: Generación de Texto
                with gr.Tab("✍️ Generación de Texto"):
                    gr.Markdown("### Genera contenido con IA")
                    topic_input = gr.Textbox(label="Tema", placeholder="Ej: Tecnología")
                    platform_gen = gr.Dropdown(
                        choices=["twitter", "facebook", "instagram", "linkedin"],
                        label="Plataforma",
                        value="twitter"
                    )
                    tone_select = gr.Dropdown(
                        choices=["professional", "casual", "funny", "inspirational"],
                        label="Tono",
                        value="professional"
                    )
                    length_select = gr.Dropdown(
                        choices=["short", "medium", "long"],
                        label="Longitud",
                        value="medium"
                    )
                    generate_btn = gr.Button("Generar", variant="primary")
                    generated_output = gr.Textbox(label="Post Generado", lines=5)
                    
                    generate_btn.click(
                        self.generate_text_interface,
                        inputs=[topic_input, platform_gen, tone_select, length_select],
                        outputs=generated_output
                    )
                
                # Tab 3: Generación de Imágenes
                if self.image_generator:
                    with gr.Tab("🎨 Generación de Imágenes"):
                        gr.Markdown("### Genera imágenes para memes y contenido")
                        image_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Ej: funny cat meme, colorful, high quality",
                            lines=3
                        )
                        image_btn = gr.Button("Generar Imagen", variant="primary")
                        image_output = gr.Image(type="pil")
                        
                        image_btn.click(
                            self.generate_image_interface,
                            inputs=image_prompt,
                            outputs=image_output
                        )
        
        return demo
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
        """Lanzar demo"""
        demo = self.create_interface()
        demo.launch(share=share, server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    demo = GradioDemo()
    demo.launch()




