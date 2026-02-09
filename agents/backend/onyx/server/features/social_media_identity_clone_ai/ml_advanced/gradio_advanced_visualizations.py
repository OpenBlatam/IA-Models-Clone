"""
Gradio demo avanzado con visualizaciones interactivas

Mejoras:
- Visualizaciones interactivas con Plotly
- Real-time metrics
- Model comparison
- Training curves
- Embeddings visualization
"""

import gradio as gr
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from .transformer_service import get_transformer_service
from .training.experiment_tracker import ExperimentTracker
from .visualization.interactive_plots import InteractivePlotter

logger = logging.getLogger(__name__)


def visualize_training_curves(
    epochs: int,
    train_loss: str,
    val_loss: str
) -> go.Figure:
    """Visualiza curvas de entrenamiento"""
    try:
        # Parse losses
        train_losses = [float(x) for x in train_loss.split(",") if x.strip()]
        val_losses = [float(x) for x in val_loss.split(",") if x.strip()]
        
        if len(train_losses) != epochs or len(val_losses) != epochs:
            return None
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(1, epochs + 1)),
            y=train_losses,
            mode='lines+markers',
            name='Train Loss',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(1, epochs + 1)),
            y=val_losses,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Training Curves',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error visualizing training curves: {e}")
        return None


def visualize_embeddings(
    texts: str,
    n_components: int = 2
) -> go.Figure:
    """Visualiza embeddings en 2D/3D"""
    try:
        transformer_service = get_transformer_service()
        
        text_list = [t.strip() for t in texts.split("\n") if t.strip()]
        if not text_list:
            return None
        
        # Generar embeddings
        embeddings = transformer_service.generate_embeddings(
            text_list,
            use_cache=True
        )
        
        # Reducir dimensionalidad
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        if n_components == 2:
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = TSNE(n_components=3, random_state=42)
        
        reduced = reducer.fit_transform(embeddings)
        
        # Create plot
        if n_components == 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=reduced[:, 0],
                y=reduced[:, 1],
                mode='markers+text',
                text=text_list,
                textposition="top center",
                marker=dict(
                    size=10,
                    color=range(len(text_list)),
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            fig.update_layout(
                title='Embeddings Visualization (t-SNE 2D)',
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                template='plotly_white'
            )
        else:
            fig = go.Figure(data=[go.Scatter3d(
                x=reduced[:, 0],
                y=reduced[:, 1],
                z=reduced[:, 2],
                mode='markers+text',
                text=text_list,
                marker=dict(
                    size=8,
                    color=range(len(text_list)),
                    colorscale='Viridis',
                    showscale=True
                )
            )])
            fig.update_layout(
                title='Embeddings Visualization (t-SNE 3D)',
                scene=dict(
                    xaxis_title='Component 1',
                    yaxis_title='Component 2',
                    zaxis_title='Component 3'
                ),
                template='plotly_white'
            )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error visualizing embeddings: {e}")
        return None


def visualize_metrics_comparison(
    experiment_names: str,
    metrics: str
) -> go.Figure:
    """Compara métricas entre experimentos"""
    try:
        names = [n.strip() for n in experiment_names.split(",") if n.strip()]
        metric_values = [float(m.strip()) for m in metrics.split(",") if m.strip()]
        
        if len(names) != len(metric_values):
            return None
        
        fig = go.Figure(data=[
            go.Bar(
                x=names,
                y=metric_values,
                marker_color='lightblue',
                text=metric_values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Metrics Comparison',
            xaxis_title='Experiment',
            yaxis_title='Metric Value',
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error visualizing metrics comparison: {e}")
        return None


def create_advanced_demo():
    """Crea demo avanzado de Gradio"""
    
    with gr.Blocks(title="Advanced ML Visualizations") as demo:
        gr.Markdown("# 🚀 Advanced ML Visualizations")
        gr.Markdown("Visualizaciones interactivas para análisis de modelos y entrenamiento")
        
        with gr.Tabs():
            with gr.TabItem("Training Curves"):
                gr.Markdown("### Visualiza curvas de entrenamiento")
                
                with gr.Row():
                    epochs_input = gr.Number(
                        label="Epochs",
                        value=10,
                        precision=0
                    )
                    train_loss_input = gr.Textbox(
                        label="Train Losses (comma-separated)",
                        placeholder="0.5,0.4,0.3,0.25,0.2,0.18,0.15,0.12,0.1,0.08"
                    )
                    val_loss_input = gr.Textbox(
                        label="Validation Losses (comma-separated)",
                        placeholder="0.6,0.5,0.4,0.35,0.3,0.28,0.25,0.22,0.2,0.18"
                    )
                
                plot_btn = gr.Button("Visualize Training Curves")
                training_plot = gr.Plot()
                
                plot_btn.click(
                    fn=visualize_training_curves,
                    inputs=[epochs_input, train_loss_input, val_loss_input],
                    outputs=training_plot
                )
            
            with gr.TabItem("Embeddings"):
                gr.Markdown("### Visualiza embeddings de textos")
                
                texts_input = gr.Textbox(
                    label="Texts (one per line)",
                    lines=10,
                    placeholder="Enter texts here..."
                )
                n_components_input = gr.Radio(
                    choices=[2, 3],
                    value=2,
                    label="Dimensions"
                )
                
                embed_btn = gr.Button("Visualize Embeddings")
                embeddings_plot = gr.Plot()
                
                embed_btn.click(
                    fn=visualize_embeddings,
                    inputs=[texts_input, n_components_input],
                    outputs=embeddings_plot
                )
            
            with gr.TabItem("Metrics Comparison"):
                gr.Markdown("### Compara métricas entre experimentos")
                
                exp_names_input = gr.Textbox(
                    label="Experiment Names (comma-separated)",
                    placeholder="Experiment 1,Experiment 2,Experiment 3"
                )
                metrics_input = gr.Textbox(
                    label="Metric Values (comma-separated)",
                    placeholder="0.85,0.87,0.89"
                )
                
                compare_btn = gr.Button("Compare Metrics")
                comparison_plot = gr.Plot()
                
                compare_btn.click(
                    fn=visualize_metrics_comparison,
                    inputs=[exp_names_input, metrics_input],
                    outputs=comparison_plot
                )
    
    return demo


if __name__ == "__main__":
    demo = create_advanced_demo()
    demo.launch(share=True)




