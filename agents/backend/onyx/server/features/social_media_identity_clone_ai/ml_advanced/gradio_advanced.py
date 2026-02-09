"""
Demo avanzado de Gradio con más funcionalidades
"""

import gradio as gr
import logging
from typing import Optional, List
import torch

from .transformer_service import get_transformer_service
from .diffusion_service import get_diffusion_service
from .lora_finetuning import get_lora_finetuner
from .training.evaluator import Evaluator
from .utils.profiler import PerformanceProfiler

logger = logging.getLogger(__name__)


def analyze_text_advanced(text: str, show_features: bool = True) -> str:
    """Análisis avanzado de texto"""
    try:
        transformer = get_transformer_service()
        result = transformer.analyze_text_style(text)
        
        output = f"""
## Análisis de Estilo

**Estilo**: {result['style']}  
**Confianza**: {result['confidence']:.2%}  
**Score de Sentimiento**: {result.get('sentiment_score', 0):.2f}
"""
        
        if show_features:
            features = result['features']
            output += "\n### Características:\n"
            output += f"- Longitud: {features.get('length', 0)} caracteres\n"
            output += f"- Palabras: {features.get('word_count', 0)}\n"
            output += f"- Emojis: {'Sí' if features.get('has_emojis') else 'No'}\n"
            output += f"- Hashtags: {'Sí' if features.get('has_hashtags') else 'No'}\n"
            output += f"- Preguntas: {'Sí' if features.get('has_questions') else 'No'}\n"
            output += f"- Exclamaciones: {'Sí' if features.get('has_exclamations') else 'No'}\n"
            output += f"- Longitud promedio de palabras: {features.get('avg_word_length', 0):.2f}\n"
        
        return output
    except Exception as e:
        return f"Error: {str(e)}"


def generate_image_advanced(
    prompt: str,
    negative_prompt: Optional[str],
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None
) -> Optional[gr.Image]:
    """Generación avanzada de imágenes"""
    try:
        diffusion = get_diffusion_service()
        
        if not diffusion.pipeline:
            return None
        
        image = diffusion.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed if seed else None
        )
        
        return image
    except Exception as e:
        logger.error(f"Error generando imagen: {e}")
        return None


def find_similar_content(query: str, candidates: str, top_k: int = 3) -> str:
    """Encuentra contenido similar"""
    try:
        transformer = get_transformer_service()
        
        candidate_list = [c.strip() for c in candidates.split("\n") if c.strip()]
        
        if not candidate_list:
            return "No hay candidatos para comparar"
        
        similar = transformer.find_similar_content(
            query_text=query,
            candidate_texts=candidate_list,
            top_k=min(top_k, len(candidate_list))
        )
        
        output = "## Contenido Similar\n\n"
        for i, item in enumerate(similar, 1):
            output += f"**{i}. Similaridad: {item['similarity']:.2%}**\n"
            output += f"{item['text']}\n\n"
        
        return output
    except Exception as e:
        return f"Error: {str(e)}"


def profile_model_performance(model_name: str) -> str:
    """Profilea performance del modelo"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        profiler = PerformanceProfiler(device=device)
        
        # Simular profiling
        output = f"## Performance Profile\n\n"
        output += f"**Device**: {device}\n"
        output += f"**Model**: {model_name}\n\n"
        
        if device == "cuda":
            memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            output += f"**GPU Memory**: {memory_mb:.2f} MB\n"
        
        output += "\n*Ejecuta profiling completo desde la API para métricas detalladas*"
        
        return output
    except Exception as e:
        return f"Error: {str(e)}"


def create_advanced_gradio_interface():
    """Crea interfaz avanzada de Gradio"""
    
    with gr.Blocks(title="Social Media Identity Clone AI - Advanced", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Social Media Identity Clone AI - Demo Avanzado")
        gr.Markdown("Sistema completo con Deep Learning avanzado")
        
        with gr.Tabs():
            # Tab 1: Análisis Avanzado
            with gr.Tab("📊 Análisis Avanzado"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Texto a analizar",
                            placeholder="Ingresa texto para análisis completo...",
                            lines=5
                        )
                        show_features = gr.Checkbox(
                            label="Mostrar características detalladas",
                            value=True
                        )
                        analyze_btn = gr.Button("Analizar", variant="primary", size="lg")
                    
                    with gr.Column():
                        text_output = gr.Markdown(label="Resultado del Análisis")
                
                analyze_btn.click(
                    fn=analyze_text_advanced,
                    inputs=[text_input, show_features],
                    outputs=text_output
                )
            
            # Tab 2: Generación de Imágenes Avanzada
            with gr.Tab("🎨 Generación de Imágenes"):
                with gr.Row():
                    with gr.Column():
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe la imagen...",
                            lines=3
                        )
                        negative_prompt_input = gr.Textbox(
                            label="Negative Prompt (opcional)",
                            placeholder="Lo que NO quieres en la imagen...",
                            lines=2
                        )
                        
                        with gr.Row():
                            num_steps = gr.Slider(
                                label="Inference Steps",
                                minimum=20,
                                maximum=100,
                                value=50,
                                step=10
                            )
                            guidance_scale = gr.Slider(
                                label="Guidance Scale",
                                minimum=1.0,
                                maximum=20.0,
                                value=7.5,
                                step=0.5
                            )
                        
                        seed_input = gr.Number(
                            label="Seed (opcional)",
                            value=None,
                            precision=0
                        )
                        generate_btn = gr.Button("Generar Imagen", variant="primary", size="lg")
                    
                    with gr.Column():
                        image_output = gr.Image(label="Imagen Generada")
                
                generate_btn.click(
                    fn=generate_image_advanced,
                    inputs=[prompt_input, negative_prompt_input, num_steps, guidance_scale, seed_input],
                    outputs=image_output
                )
            
            # Tab 3: Búsqueda Semántica
            with gr.Tab("🔍 Búsqueda Semántica"):
                with gr.Row():
                    with gr.Column():
                        query_input = gr.Textbox(
                            label="Texto de búsqueda",
                            placeholder="¿Qué contenido buscas?",
                            lines=2
                        )
                        candidates_input = gr.Textbox(
                            label="Candidatos (uno por línea)",
                            placeholder="Texto 1\nTexto 2\nTexto 3",
                            lines=10
                        )
                        top_k_slider = gr.Slider(
                            label="Top K",
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1
                        )
                        search_btn = gr.Button("Buscar Similar", variant="primary")
                    
                    with gr.Column():
                        search_output = gr.Markdown(label="Resultados")
                
                search_btn.click(
                    fn=find_similar_content,
                    inputs=[query_input, candidates_input, top_k_slider],
                    outputs=search_output
                )
            
            # Tab 4: Performance
            with gr.Tab("⚡ Performance"):
                with gr.Row():
                    with gr.Column():
                        model_name_input = gr.Textbox(
                            label="Nombre del Modelo",
                            value="gpt2",
                            placeholder="gpt2, distilbert, etc."
                        )
                        profile_btn = gr.Button("Profilear Modelo", variant="primary")
                    
                    with gr.Column():
                        profile_output = gr.Markdown(label="Métricas de Performance")
                
                profile_btn.click(
                    fn=profile_model_performance,
                    inputs=model_name_input,
                    outputs=profile_output
                )
        
        gr.Markdown("---")
        gr.Markdown("### ℹ️ Información")
        gr.Markdown("""
        - **Análisis Avanzado**: Análisis completo de estilo y características
        - **Generación de Imágenes**: Control completo sobre parámetros de generación
        - **Búsqueda Semántica**: Encuentra contenido similar usando embeddings
        - **Performance**: Profilea modelos para optimización
        """)
    
    return demo


def launch_advanced_demo(
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860
):
    """Lanza demo avanzado"""
    demo = create_advanced_gradio_interface()
    demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port
    )


if __name__ == "__main__":
    launch_advanced_demo()




