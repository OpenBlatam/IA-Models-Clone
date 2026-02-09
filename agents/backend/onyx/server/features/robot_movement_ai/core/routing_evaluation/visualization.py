"""
Evaluation Visualization
========================

Visualizaciones para evaluación de modelos.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # Backend sin GUI
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib no disponible")


class EvaluationVisualizer:
    """
    Visualizador de evaluaciones.
    """
    
    @staticmethod
    def plot_predictions_vs_targets(
        predictions: np.ndarray,
        targets: np.ndarray,
        output_names: List[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plotear predicciones vs targets.
        
        Args:
            predictions: Predicciones
            targets: Targets
            output_names: Nombres de outputs
            save_path: Ruta para guardar (opcional)
            
        Returns:
            Figura
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if output_names is None:
            output_names = [f"Output {i+1}" for i in range(predictions.shape[1])]
        
        n_outputs = predictions.shape[1]
        fig, axes = plt.subplots(1, n_outputs, figsize=(5*n_outputs, 5))
        
        if n_outputs == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, output_names)):
            pred_col = predictions[:, i]
            target_col = targets[:, i]
            
            ax.scatter(target_col, pred_col, alpha=0.5)
            
            # Línea perfecta
            min_val = min(target_col.min(), pred_col.min())
            max_val = max(target_col.max(), pred_col.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
            
            ax.set_xlabel('Target')
            ax.set_ylabel('Prediction')
            ax.set_title(f'{name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_residuals(
        predictions: np.ndarray,
        targets: np.ndarray,
        output_names: List[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plotear residuos.
        
        Args:
            predictions: Predicciones
            targets: Targets
            output_names: Nombres de outputs
            save_path: Ruta para guardar (opcional)
            
        Returns:
            Figura
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if output_names is None:
            output_names = [f"Output {i+1}" for i in range(predictions.shape[1])]
        
        n_outputs = predictions.shape[1]
        fig, axes = plt.subplots(1, n_outputs, figsize=(5*n_outputs, 5))
        
        if n_outputs == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, output_names)):
            residuals = targets[:, i] - predictions[:, i]
            
            ax.scatter(predictions[:, i], residuals, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--')
            
            ax.set_xlabel('Prediction')
            ax.set_ylabel('Residual')
            ax.set_title(f'{name} - Residuals')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_error_distribution(
        predictions: np.ndarray,
        targets: np.ndarray,
        output_names: List[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plotear distribución de errores.
        
        Args:
            predictions: Predicciones
            targets: Targets
            output_names: Nombres de outputs
            save_path: Ruta para guardar (opcional)
            
        Returns:
            Figura
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if output_names is None:
            output_names = [f"Output {i+1}" for i in range(predictions.shape[1])]
        
        n_outputs = predictions.shape[1]
        fig, axes = plt.subplots(1, n_outputs, figsize=(5*n_outputs, 5))
        
        if n_outputs == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, output_names)):
            errors = targets[:, i] - predictions[:, i]
            
            ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='r', linestyle='--', label='Zero Error')
            ax.axvline(x=np.mean(errors), color='g', linestyle='--', label='Mean Error')
            
            ax.set_xlabel('Error')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} - Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def plot_training_curves(
    history: List[Dict[str, float]],
    metrics: List[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plotear curvas de entrenamiento.
    
    Args:
        history: Historial de entrenamiento
        metrics: Métricas a plotear (opcional)
        save_path: Ruta para guardar (opcional)
        
    Returns:
        Figura
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    if metrics is None:
        metrics = ["train_loss", "val_loss"]
    
    epochs = [h.get("epoch", i+1) for i, h in enumerate(history)]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        values = [h.get(metric, 0) for h in history]
        ax.plot(epochs, values, label=metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} over time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plotear matriz de confusión (para clasificación).
    
    Args:
        predictions: Predicciones
        targets: Targets
        save_path: Ruta para guardar (opcional)
        
    Returns:
        Figura
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    # Para regresión, mostrar correlación
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot con densidad
    ax.hexbin(targets.flatten(), predictions.flatten(), gridsize=50, cmap='Blues')
    ax.set_xlabel('Target')
    ax.set_ylabel('Prediction')
    ax.set_title('Prediction vs Target (Density)')
    
    # Línea perfecta
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
    ax.legend()
    
    plt.colorbar(ax.collections[0], ax=ax, label='Density')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

