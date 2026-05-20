"""
Visualization Utilities - Utilidades de visualización
======================================================

Funciones para visualizar modelos, entrenamiento y resultados.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available")


def plot_training_history(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> Optional[bytes]:
    """
    Graficar historial de entrenamiento.
    
    Args:
        train_losses: Lista de pérdidas de entrenamiento
        val_losses: Lista de pérdidas de validación (opcional)
        train_accs: Lista de accuracies de entrenamiento (opcional)
        val_accs: Lista de accuracies de validación (opcional)
        save_path: Ruta para guardar (opcional)
    
    Returns:
        Bytes de la imagen si no se guarda, None si se guarda
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available for plotting")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses:
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracies
    if train_accs:
        axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    if val_accs:
        axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.read()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Optional[bytes]:
    """
    Graficar matriz de confusión.
    
    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones
        class_names: Nombres de clases (opcional)
        save_path: Ruta para guardar (opcional)
    
    Returns:
        Bytes de la imagen si no se guarda, None si se guarda
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available for plotting")
        return None
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.read()


def plot_feature_importance(
    feature_names: List[str],
    importances: List[float],
    top_k: Optional[int] = None,
    save_path: Optional[str] = None
) -> Optional[bytes]:
    """
    Graficar importancia de features.
    
    Args:
        feature_names: Nombres de features
        importances: Importancias
        top_k: Mostrar solo top K (opcional)
        save_path: Ruta para guardar (opcional)
    
    Returns:
        Bytes de la imagen si no se guarda, None si se guarda
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available for plotting")
        return None
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    if top_k:
        indices = indices[:top_k]
    
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_names)), sorted_importances)
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.read()


def plot_learning_curve(
    train_sizes: List[int],
    train_scores: List[float],
    val_scores: List[float],
    save_path: Optional[str] = None
) -> Optional[bytes]:
    """
    Graficar curva de aprendizaje.
    
    Args:
        train_sizes: Tamaños de entrenamiento
        train_scores: Scores de entrenamiento
        val_scores: Scores de validación
        save_path: Ruta para guardar (opcional)
    
    Returns:
        Bytes de la imagen si no se guarda, None si se guarda
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available for plotting")
        return None
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Train Score', linewidth=2)
    plt.plot(train_sizes, val_scores, 'o-', label='Val Score', linewidth=2)
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.read()




