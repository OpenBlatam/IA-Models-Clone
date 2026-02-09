"""
Demo interactivo con Gradio para el sistema
"""

import gradio as gr
import logging
from typing import Optional

from .transformer_service import get_transformer_service
from .diffusion_service import get_diffusion_service
from .lora_finetuning import get_lora_finetuner

logger = logging.getLogger(__name__)


def analyze_text_style(text: str) -> str:
    """Analiza estilo de texto"""
    try:
        transformer_service = get_transformer_service()
        result = transformer_service.analyze_text_style(text)
        
        return f"""
**Estilo**: {result['style']}
**Confianza**: {result['confidence']:.2%}
**Score de Sentimiento**: {result.get('sentiment_score', 0):.2f}

**Características**:
- Longitud: {result['features'].get('length', 0)} caracteres
- Palabras: {result['features'].get('word_count', 0)}
- Tiene emojis: {'Sí' if result['features'].get('has_emojis') else 'No'}
- Tiene hashtags: {'Sí' if result['features'].get('has_hashtags') else 'No'}
"""
    except Exception as e:
        return f"Error: {str(e)}"


def generate_image_from_prompt(prompt: str, seed: Optional[int]) -> Optional[gr.Image]:
    """Genera imagen desde prompt"""
    try:
        diffusion_service = get_diffusion_service()
        image = diffusion_service.generate_image(
            prompt=prompt,
            seed=seed if seed else None
        )
        return image
    except Exception as e:
        logger.error(f"Error generando imagen: {e}")
        return None


def create_gradio_interface():
    """Crea interfaz de Gradio"""
    
    with gr.Blocks(title="Social Media Identity Clone AI") as demo:
        gr.Markdown("# 🎨 Social Media Identity Clone AI - Demo Interactivo")
        
        with gr.Tabs():
            # Tab 1: Análisis de Texto
            with gr.Tab("Análisis de Texto"):
                gr.Markdown("### Analiza el estilo y características de texto")
                text_input = gr.Textbox(
                    label="Texto a analizar",
                    placeholder="Ingresa texto para analizar su estilo...",
                    lines=5
                )
                analyze_btn = gr.Button("Analizar", variant="primary")
                text_output = gr.Markdown(label="Resultado del Análisis")
                
                analyze_btn.click(
                    fn=analyze_text_style,
                    inputs=text_input,
                    outputs=text_output
                )
            
            # Tab 2: Generación de Imágenes
            with gr.Tab("Generación de Imágenes"):
                gr.Markdown("### Genera imágenes usando modelos de difusión")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe la imagen que quieres generar...",
                    lines=3
                )
                seed_input = gr.Number(
                    label="Seed (opcional)",
                    value=None,
                    precision=0
                )
                generate_btn = gr.Button("Generar Imagen", variant="primary")
                image_output = gr.Image(label="Imagen Generada")
                
                generate_btn.click(
                    fn=generate_image_from_prompt,
                    inputs=[prompt_input, seed_input],
                    outputs=image_output
                )
            
            # Tab 3: Fine-tuning
            with gr.Tab("Fine-tuning (Avanzado)"):
                gr.Markdown("### Fine-tuning con LoRA")
                gr.Markdown("""
                **Nota**: El fine-tuning requiere configuración adicional.
                
                Para usar esta funcionalidad:
                1. Prepara tu dataset de textos
                2. Configura los parámetros de entrenamiento
                3. Ejecuta el entrenamiento desde la API
                
                Ver documentación para más detalles.
                """)
        
        gr.Markdown("---")
        gr.Markdown("### ℹ️ Información")
        gr.Markdown("""
        - **Análisis de Texto**: Usa modelos transformer para analizar estilo y sentimiento
        - **Generación de Imágenes**: Usa Stable Diffusion para generar imágenes
        - **Fine-tuning**: Personaliza modelos con LoRA para tu identidad específica
        """)
    
    return demo


def launch_demo(share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
    """Lanza demo de Gradio"""
    demo = create_gradio_interface()
    demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port
    )


if __name__ == "__main__":
    launch_demo()




