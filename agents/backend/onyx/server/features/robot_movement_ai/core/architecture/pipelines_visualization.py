"""
Visualization Module
====================

Sistema profesional de visualización para deep learning.
Incluye visualización de entrenamiento, modelos, y resultados.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None
    logging.warning("matplotlib/seaborn not available. Visualization disabled.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    logging.warning("plotly not available. Advanced visualization disabled.")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """
    Visualizador profesional de entrenamiento.
    
    Incluye:
    - Gráficos de pérdida y métricas
    - Learning rate schedules
    - Gradient flow
    - Model architecture
    """
    
    def __init__(self, output_dir: str = "./visualizations"):
        """
        Inicializar visualizador.
        
        Args:
            output_dir: Directorio de salida
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for visualization")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TrainingVisualizer initialized: {output_dir}")
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = False
    ) -> str:
        """
        Graficar historial de entrenamiento.
        
        Args:
            history: Dict con métricas (train_loss, val_loss, etc.)
            save_path: Ruta para guardar (None para auto-generar)
            show: Mostrar gráfico
            
        Returns:
            Ruta del archivo guardado
        """
        if save_path is None:
            save_path = self.output_dir / "training_history.png"
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Metrics plot
        metric_keys = [k for k in history.keys() if k not in ['train_loss', 'val_loss', 'epoch', 'learning_rate']]
        if metric_keys:
            for key in metric_keys[:5]:  # Limitar a 5 métricas
                axes[1].plot(history[key], label=key)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Metric Value')
            axes[1].set_title('Training Metrics')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        logger.info(f"Training history plot saved to {save_path}")
        return str(save_path)
    
    def plot_learning_rate_schedule(
        self,
        lr_history: List[float],
        save_path: Optional[str] = None
    ) -> str:
        """
        Graficar schedule de learning rate.
        
        Args:
            lr_history: Historial de learning rates
            save_path: Ruta para guardar
            
        Returns:
            Ruta del archivo guardado
        """
        if save_path is None:
            save_path = self.output_dir / "learning_rate_schedule.png"
        
        plt.figure(figsize=(10, 6))
        plt.plot(lr_history, color='green', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Learning rate schedule plot saved to {save_path}")
        return str(save_path)
    
    def plot_gradient_flow(
        self,
        gradient_history: List[Dict[str, float]],
        save_path: Optional[str] = None
    ) -> str:
        """
        Graficar flujo de gradientes.
        
        Args:
            gradient_history: Historial de estadísticas de gradientes
            save_path: Ruta para guardar
            
        Returns:
            Ruta del archivo guardado
        """
        if save_path is None:
            save_path = self.output_dir / "gradient_flow.png"
        
        if not gradient_history:
            logger.warning("No gradient history provided")
            return ""
        
        steps = range(len(gradient_history))
        norms = [h.get('total_norm', 0) for h in gradient_history]
        max_grads = [h.get('max_grad', 0) for h in gradient_history]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(steps, norms, label='Gradient Norm', color='blue')
        axes[0].axhline(y=10.0, color='red', linestyle='--', label='Max Norm (10.0)')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Gradient Norm')
        axes[0].set_title('Gradient Norm Over Time')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(steps, max_grads, label='Max Gradient', color='orange')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Max Gradient Value')
        axes[1].set_title('Max Gradient Over Time')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gradient flow plot saved to {save_path}")
        return str(save_path)


class ModelVisualizer:
    """
    Visualizador de arquitecturas de modelos.
    
    Incluye:
    - Visualización de arquitectura
    - Feature maps
    - Attention weights
    - Layer activations
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar visualizador de modelo.
        
        Args:
            model: Modelo PyTorch
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self._register_hooks()
        logger.info("ModelVisualizer initialized")
    
    def _register_hooks(self):
        """Registrar hooks para capturar activaciones."""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU)):
                module.register_forward_hook(get_activation(name))
    
    def visualize_architecture(
        self,
        input_shape: Tuple[int, ...],
        save_path: Optional[str] = None
    ) -> str:
        """
        Visualizar arquitectura del modelo.
        
        Args:
            input_shape: Forma de entrada (sin batch)
            save_path: Ruta para guardar
            
        Returns:
            Ruta del archivo guardado
        """
        try:
            from torchviz import make_dot
            
            dummy_input = torch.randn(1, *input_shape)
            output = self.model(dummy_input)
            
            dot = make_dot(output, params=dict(self.model.named_parameters()))
            
            if save_path is None:
                save_path = "model_architecture.png"
            
            dot.render(save_path.replace('.png', ''), format='png')
            logger.info(f"Model architecture saved to {save_path}")
            return save_path
        except ImportError:
            logger.warning("torchviz not available. Install with: pip install torchviz")
            return ""
    
    def plot_feature_maps(
        self,
        input_data: torch.Tensor,
        layer_name: str,
        save_path: Optional[str] = None
    ) -> str:
        """
        Visualizar feature maps de una capa.
        
        Args:
            input_data: Input para el modelo
            layer_name: Nombre de la capa
            save_path: Ruta para guardar
            
        Returns:
            Ruta del archivo guardado
        """
        if not MATPLOTLIB_AVAILABLE:
            return ""
        
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_data)
        
        if layer_name not in self.activations:
            logger.warning(f"Layer {layer_name} not found in activations")
            return ""
        
        features = self.activations[layer_name]
        
        if save_path is None:
            save_path = f"feature_maps_{layer_name}.png"
        
        # Visualizar primeros feature maps
        num_features = min(16, features.shape[1])
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(num_features):
            if features.ndim == 4:  # Conv2d
                axes[i].imshow(features[0, i].cpu().numpy(), cmap='viridis')
            elif features.ndim == 3:  # Conv1d
                axes[i].plot(features[0, i].cpu().numpy())
            axes[i].set_title(f'Feature {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature maps saved to {save_path}")
        return str(save_path)


class InteractiveVisualizer:
    """
    Visualizador interactivo usando Plotly.
    
    Permite visualizaciones interactivas y dinámicas.
    """
    
    def __init__(self, output_dir: str = "./visualizations"):
        """
        Inicializar visualizador interactivo.
        
        Args:
            output_dir: Directorio de salida
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required for interactive visualization")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("InteractiveVisualizer initialized")
    
    def create_interactive_training_plot(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> str:
        """
        Crear gráfico interactivo de entrenamiento.
        
        Args:
            history: Historial de entrenamiento
            save_path: Ruta para guardar HTML
            
        Returns:
            Ruta del archivo guardado
        """
        if save_path is None:
            save_path = self.output_dir / "training_history.html"
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Loss', 'Metrics'),
            vertical_spacing=0.1
        )
        
        # Loss plot
        if 'train_loss' in history:
            fig.add_trace(
                go.Scatter(
                    y=history['train_loss'],
                    mode='lines',
                    name='Train Loss',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(
                    y=history['val_loss'],
                    mode='lines',
                    name='Val Loss',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
        
        # Metrics plot
        metric_keys = [k for k in history.keys() if k not in ['train_loss', 'val_loss', 'epoch']]
        for key in metric_keys[:5]:
            fig.add_trace(
                go.Scatter(
                    y=history[key],
                    mode='lines',
                    name=key
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Metric Value", row=2, col=1)
        fig.update_layout(height=800, title_text="Training History")
        
        fig.write_html(str(save_path))
        logger.info(f"Interactive training plot saved to {save_path}")
        return str(save_path)
    
    def create_confusion_matrix_plot(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Crear matriz de confusión interactiva.
        
        Args:
            confusion_matrix: Matriz de confusión
            class_names: Nombres de clases
            save_path: Ruta para guardar
            
        Returns:
            Ruta del archivo guardado
        """
        if save_path is None:
            save_path = self.output_dir / "confusion_matrix.html"
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(confusion_matrix))]
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=confusion_matrix,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Count")
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=600
        )
        
        fig.write_html(str(save_path))
        logger.info(f"Confusion matrix plot saved to {save_path}")
        return str(save_path)

