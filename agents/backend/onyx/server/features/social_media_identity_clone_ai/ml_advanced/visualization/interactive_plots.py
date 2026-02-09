"""
Visualización interactiva avanzada
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


class InteractiveVisualizer:
    """Visualizador interactivo con Plotly"""
    
    def __init__(self):
        pass
    
    def plot_training_interactive(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        learning_rates: Optional[List[float]] = None,
        save_path: Optional[str] = None
    ):
        """Plotea entrenamiento interactivo"""
        fig = make_subplots(
            rows=2 if learning_rates else 1,
            cols=1,
            subplot_titles=("Training Curves", "Learning Rate") if learning_rates else ("Training Curves",),
            vertical_spacing=0.1
        )
        
        # Loss curves
        fig.add_trace(
            go.Scatter(
                y=train_losses,
                mode='lines',
                name='Train Loss',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        if val_losses:
            fig.add_trace(
                go.Scatter(
                    y=val_losses,
                    mode='lines',
                    name='Val Loss',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
        
        # Learning rate
        if learning_rates:
            fig.add_trace(
                go.Scatter(
                    y=learning_rates,
                    mode='lines',
                    name='Learning Rate',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Training Progress",
            height=800 if learning_rates else 400,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_confusion_matrix_interactive(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Plotea confusion matrix interactiva"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='True'
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_embeddings_3d(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """Visualiza embeddings en 3D"""
        from sklearn.manifold import TSNE
        
        if embeddings.shape[1] > 3:
            reducer = TSNE(n_components=3, random_state=42)
            embeddings_3d = reducer.fit_transform(embeddings)
        else:
            embeddings_3d = embeddings
        
        fig = go.Figure()
        
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                fig.add_trace(go.Scatter3d(
                    x=embeddings_3d[mask, 0],
                    y=embeddings_3d[mask, 1],
                    z=embeddings_3d[mask, 2],
                    mode='markers',
                    name=f'Class {label}',
                    marker=dict(size=5)
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=embeddings_3d[:, 0],
                y=embeddings_3d[:, 1],
                z=embeddings_3d[:, 2],
                mode='markers',
                marker=dict(size=5)
            ))
        
        fig.update_layout(title='Embeddings 3D Visualization')
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_metrics_comparison(
        self,
        experiments: Dict[str, Dict[str, List[float]]],
        save_path: Optional[str] = None
    ):
        """Compara métricas de múltiples experimentos"""
        fig = go.Figure()
        
        for exp_name, metrics in experiments.items():
            if "loss" in metrics:
                fig.add_trace(go.Scatter(
                    y=metrics["loss"],
                    mode='lines',
                    name=f'{exp_name} - Loss'
                ))
        
        fig.update_layout(
            title='Experiments Comparison',
            xaxis_title='Epoch',
            yaxis_title='Loss'
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()




