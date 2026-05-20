"""
Advanced Gradio Demo - Demo Gradio Avanzado
===========================================

Demo interactivo avanzado con múltiples funcionalidades.
"""

import logging
import gradio as gr
import torch
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class AdvancedGradioDemo:
    """Demo Gradio avanzado"""
    
    def __init__(
        self,
        content_analyzer: Optional[Any] = None,
        text_generator: Optional[Any] = None,
        image_generator: Optional[Any] = None,
        sentiment_analyzer: Optional[Any] = None
    ):
        """
        Inicializar demo
        
        Args:
            content_analyzer: Analizador de contenido
            text_generator: Generador de texto
            image_generator: Generador de imágenes
            sentiment_analyzer: Analizador de sentimiento
        """
        self.content_analyzer = content_analyzer
        self.text_generator = text_generator
        self.image_generator = image_generator
        self.sentiment_analyzer = sentiment_analyzer
    
    def analyze_content(self, text: str) -> Dict[str, Any]:
        """Analizar contenido"""
        if not self.content_analyzer:
            return {"error": "Content analyzer no disponible"}
        
        try:
            result = self.content_analyzer.analyze(text)
            return {
                "sentiment": result.get("sentiment", "N/A"),
                "topics": result.get("topics", []),
                "keywords": result.get("keywords", [])
            }
        except Exception as e:
            return {"error": str(e)}
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generar texto"""
        if not self.text_generator:
            return "[Text generator no disponible]"
        
        try:
            generated = self.text_generator.generate(prompt, max_length=max_length)
            return generated
        except Exception as e:
            return f"[Error: {str(e)}]"
    
    def generate_image(self, prompt: str, num_inference_steps: int = 50) -> Image.Image:
        """Generar imagen"""
        if not self.image_generator:
            return None
        
        try:
            image = self.image_generator.generate(prompt, num_inference_steps=num_inference_steps)
            return image
        except Exception as e:
            logger.error(f"Error generando imagen: {e}")
            return None
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analizar sentimiento"""
        if not self.sentiment_analyzer:
            return {"error": "Sentiment analyzer no disponible"}
        
        try:
            result = self.sentiment_analyzer.analyze(text)
            return {
                "label": result.get("label", "N/A"),
                "score": result.get("score", 0.0)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def create_interface(self) -> gr.Blocks:
        """Crear interfaz Gradio"""
        with gr.Blocks(title="Community Manager AI - Advanced Demo") as demo:
            gr.Markdown("# 🤖 Community Manager AI - Advanced Demo")
            
            with gr.Tabs():
                # Tab 1: Content Analysis
                with gr.Tab("Content Analysis"):
                    with gr.Row():
                        with gr.Column():
                            content_input = gr.Textbox(
                                label="Text to Analyze",
                                placeholder="Enter text here...",
                                lines=5
                            )
                            analyze_btn = gr.Button("Analyze", variant="primary")
                        
                        with gr.Column():
                            sentiment_output = gr.Textbox(label="Sentiment")
                            topics_output = gr.JSON(label="Topics")
                            keywords_output = gr.JSON(label="Keywords")
                    
                    analyze_btn.click(
                        self.analyze_content,
                        inputs=content_input,
                        outputs=[sentiment_output, topics_output, keywords_output]
                    )
                
                # Tab 2: Text Generation
                with gr.Tab("Text Generation"):
                    with gr.Row():
                        with gr.Column():
                            prompt_input = gr.Textbox(
                                label="Prompt",
                                placeholder="Enter prompt...",
                                lines=3
                            )
                            max_length_slider = gr.Slider(
                                minimum=10,
                                maximum=500,
                                value=100,
                                step=10,
                                label="Max Length"
                            )
                            generate_btn = gr.Button("Generate", variant="primary")
                        
                        with gr.Column():
                            generated_output = gr.Textbox(
                                label="Generated Text",
                                lines=10
                            )
                    
                    generate_btn.click(
                        self.generate_text,
                        inputs=[prompt_input, max_length_slider],
                        outputs=generated_output
                    )
                
                # Tab 3: Image Generation
                with gr.Tab("Image Generation"):
                    with gr.Row():
                        with gr.Column():
                            image_prompt = gr.Textbox(
                                label="Image Prompt",
                                placeholder="Describe the image...",
                                lines=3
                            )
                            steps_slider = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=10,
                                label="Inference Steps"
                            )
                            image_btn = gr.Button("Generate Image", variant="primary")
                        
                        with gr.Column():
                            image_output = gr.Image(label="Generated Image")
                    
                    image_btn.click(
                        self.generate_image,
                        inputs=[image_prompt, steps_slider],
                        outputs=image_output
                    )
                
                # Tab 4: Sentiment Analysis
                with gr.Tab("Sentiment Analysis"):
                    with gr.Row():
                        with gr.Column():
                            sentiment_input = gr.Textbox(
                                label="Text",
                                placeholder="Enter text for sentiment analysis...",
                                lines=5
                            )
                            sentiment_btn = gr.Button("Analyze Sentiment", variant="primary")
                        
                        with gr.Column():
                            sentiment_label = gr.Textbox(label="Sentiment Label")
                            sentiment_score = gr.Number(label="Confidence Score")
                    
                    sentiment_btn.click(
                        self.analyze_sentiment,
                        inputs=sentiment_input,
                        outputs=[sentiment_label, sentiment_score]
                    )
            
            gr.Markdown("### 💡 Tips")
            gr.Markdown("""
            - Use clear and specific prompts for better results
            - Adjust max_length for text generation based on your needs
            - More inference steps = better quality but slower generation
            """)
        
        return demo
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
        """Lanzar demo"""
        demo = self.create_interface()
        demo.launch(share=share, server_name=server_name, server_port=server_port)




