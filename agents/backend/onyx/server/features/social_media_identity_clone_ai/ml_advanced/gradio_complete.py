"""
Demo completo de Gradio con todas las funcionalidades
"""

import gradio as gr
import logging
from typing import Optional, List
import torch
import numpy as np

from .transformer_service import get_transformer_service
from .diffusion_service import get_diffusion_service
from .lora_finetuning import get_lora_finetuner
from .data.dataset_analysis import DatasetAnalyzer
from .data.data_augmentation import TextAugmenter
from .evaluation.advanced_metrics import AdvancedMetrics
from .visualization.visualizer import TrainingVisualizer

logger = logging.getLogger(__name__)


def analyze_dataset_complete(texts: str) -> str:
    """Análisis completo de dataset"""
    try:
        analyzer = DatasetAnalyzer()
        
        text_list = [t.strip() for t in texts.split("\n") if t.strip()]
        if not text_list:
            return "No hay textos para analizar"
        
        analysis = analyzer.analyze_text_dataset(text_list)
        
        output = "## Análisis de Dataset\n\n"
        output += f"**Tamaño del dataset**: {analysis['dataset_size']}\n\n"
        
        output += "### Longitud de Textos\n"
        length = analysis['text_length']
        output += f"- Media: {length['mean']:.1f} caracteres\n"
        output += f"- Mediana: {length['median']:.1f} caracteres\n"
        output += f"- Min: {length['min']} caracteres\n"
        output += f"- Max: {length['max']} caracteres\n\n"
        
        output += "### Vocabulario\n"
        vocab = analysis['vocabulary']
        output += f"- Tamaño del vocabulario: {vocab['vocab_size']}\n"
        output += f"- Palabras totales: {vocab['total_words']}\n"
        output += f"- Frecuencia promedio: {vocab['avg_word_frequency']:.2f}\n\n"
        
        output += "### Top 10 Palabras\n"
        for word, freq in vocab['top_10_words'].items():
            output += f"- {word}: {freq}\n"
        
        return output
    except Exception as e:
        return f"Error: {str(e)}"


def augment_text(text: str, method: str, num_augmentations: int) -> str:
    """Aumenta texto"""
    try:
        augmenter = TextAugmenter()
        
        if method == "random_deletion":
            augmented = augmenter.random_deletion(text)
        elif method == "random_swap":
            augmented = augmenter.random_swap(text)
        else:
            augmented = text
        
        result = f"**Original**:\n{text}\n\n"
        result += f"**Aumentado ({method})**:\n{augmented}"
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"


def calculate_metrics(predictions: str, references: str) -> str:
    """Calcula métricas avanzadas"""
    try:
        metrics = AdvancedMetrics()
        
        pred_list = [p.strip() for p in predictions.split("\n") if p.strip()]
        ref_list = [r.strip() for r in references.split("\n") if r.strip()]
        
        if len(pred_list) != len(ref_list):
            return "Error: Número de predicciones y referencias debe ser igual"
        
        # BLEU
        refs_list = [[ref] for ref in ref_list]
        bleu = metrics.calculate_bleu(pred_list, refs_list)
        
        # ROUGE
        rouge = metrics.calculate_rouge(pred_list, ref_list)
        
        # Diversity
        diversity = metrics.calculate_diversity(pred_list, n_gram=2)
        
        output = "## Métricas Avanzadas\n\n"
        output += f"**BLEU Score**: {bleu['bleu_mean']:.4f} ± {bleu['bleu_std']:.4f}\n\n"
        output += f"**ROUGE Scores**:\n"
        output += f"- ROUGE-1: {rouge['rouge1']:.4f}\n"
        output += f"- ROUGE-2: {rouge['rouge2']:.4f}\n"
        output += f"- ROUGE-L: {rouge['rougeL']:.4f}\n\n"
        output += f"**Diversity**: {diversity['diversity']:.4f}\n"
        output += f"- Unique n-grams: {diversity['unique_ngrams']}\n"
        output += f"- Total n-grams: {diversity['total_ngrams']}\n"
        
        return output
    except Exception as e:
        return f"Error: {str(e)}"


def create_complete_gradio_interface():
    """Crea interfaz completa de Gradio"""
    
    with gr.Blocks(title="Social Media Identity Clone AI - Complete", theme=gr.themes.Monochrome()) as demo:
        gr.Markdown("# 🎨 Social Media Identity Clone AI - Sistema Completo")
        gr.Markdown("Sistema enterprise con Deep Learning avanzado")
        
        with gr.Tabs():
            # Tab 1: Análisis de Dataset
            with gr.Tab("📊 Análisis de Dataset"):
                with gr.Row():
                    with gr.Column():
                        dataset_input = gr.Textbox(
                            label="Dataset (uno por línea)",
                            placeholder="Texto 1\nTexto 2\nTexto 3",
                            lines=10
                        )
                        analyze_dataset_btn = gr.Button("Analizar Dataset", variant="primary")
                    
                    with gr.Column():
                        dataset_output = gr.Markdown(label="Análisis")
                
                analyze_dataset_btn.click(
                    fn=analyze_dataset_complete,
                    inputs=dataset_input,
                    outputs=dataset_output
                )
            
            # Tab 2: Data Augmentation
            with gr.Tab("🔄 Data Augmentation"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Texto a aumentar",
                            lines=3
                        )
                        method_dropdown = gr.Dropdown(
                            choices=["random_deletion", "random_swap"],
                            value="random_deletion",
                            label="Método"
                        )
                        augment_btn = gr.Button("Aumentar", variant="primary")
                    
                    with gr.Column():
                        augmented_output = gr.Markdown(label="Resultado")
                
                augment_btn.click(
                    fn=augment_text,
                    inputs=[text_input, method_dropdown, gr.Number(value=1, visible=False)],
                    outputs=augmented_output
                )
            
            # Tab 3: Métricas Avanzadas
            with gr.Tab("📈 Métricas Avanzadas"):
                with gr.Row():
                    with gr.Column():
                        predictions_input = gr.Textbox(
                            label="Predicciones (uno por línea)",
                            lines=5
                        )
                        references_input = gr.Textbox(
                            label="Referencias (uno por línea)",
                            lines=5
                        )
                        calculate_metrics_btn = gr.Button("Calcular Métricas", variant="primary")
                    
                    with gr.Column():
                        metrics_output = gr.Markdown(label="Métricas")
                
                calculate_metrics_btn.click(
                    fn=calculate_metrics,
                    inputs=[predictions_input, references_input],
                    outputs=metrics_output
                )
            
            # Tab 4: Análisis de Texto (de gradio_advanced)
            with gr.Tab("🔍 Análisis de Texto"):
                text_analyze = gr.Textbox(label="Texto", lines=5)
                analyze_btn = gr.Button("Analizar")
                text_result = gr.Markdown()
                
                # Reutilizar función de gradio_advanced
                from .gradio_advanced import analyze_text_advanced
                analyze_btn.click(
                    fn=lambda t: analyze_text_advanced(t, True),
                    inputs=text_analyze,
                    outputs=text_result
                )
            
            # Tab 5: Generación de Imágenes (de gradio_advanced)
            with gr.Tab("🎨 Generación de Imágenes"):
                prompt_img = gr.Textbox(label="Prompt", lines=3)
                generate_img_btn = gr.Button("Generar")
                img_result = gr.Image()
                
                from .gradio_advanced import generate_image_advanced
                generate_img_btn.click(
                    fn=lambda p: generate_image_advanced(p, None, 50, 7.5, None),
                    inputs=prompt_img,
                    outputs=img_result
                )
        
        gr.Markdown("---")
        gr.Markdown("### ℹ️ Sistema Completo")
        gr.Markdown("""
        - **Análisis de Dataset**: Estadísticas completas
        - **Data Augmentation**: Aumento de datos
        - **Métricas Avanzadas**: BLEU, ROUGE, Diversity
        - **Análisis de Texto**: Transformers
        - **Generación de Imágenes**: Diffusion models
        """)
    
    return demo


def launch_complete_demo(
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860
):
    """Lanza demo completo"""
    demo = create_complete_gradio_interface()
    demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port
    )


if __name__ == "__main__":
    launch_complete_demo()




