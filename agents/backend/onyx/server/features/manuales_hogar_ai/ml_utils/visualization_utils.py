"""
Visualization Utils - Utilidades de Visualización
===================================================

Utilidades para visualización de modelos y resultados.
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Intentar importar bibliotecas opcionales
try:
    import matplotlib.pyplot as plt
    _has_matplotlib = True
except ImportError:
    _has_matplotlib = False
    logger.warning("matplotlib not available, visualization features will be limited")

try:
    import seaborn as sns
    _has_seaborn = True
except ImportError:
    _has_seaborn = False


class TrainingVisualizer:
    """
    Visualizador de entrenamiento.
    """
    
    def __init__(self):
        """Inicializar visualizador."""
        if not _has_matplotlib:
            raise ImportError("matplotlib is required for TrainingVisualizer")
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plotear historial de entrenamiento.
        
        Args:
            history: Historial con métricas
            save_path: Ruta para guardar (opcional)
            show: Mostrar plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Acc')
        if 'val_acc' in history:
            axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()


class ModelArchitectureVisualizer:
    """
    Visualizador de arquitectura de modelos.
    """
    
    def __init__(self):
        """Inicializar visualizador."""
        if not _has_matplotlib:
            raise ImportError("matplotlib is required for ModelArchitectureVisualizer")
    
    def plot_model_structure(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        save_path: Optional[str] = None
    ):
        """
        Visualizar estructura del modelo.
        
        Args:
            model: Modelo
            input_shape: Forma de entrada
            save_path: Ruta para guardar (opcional)
        """
        try:
            from torchviz import make_dot
            
            # Crear input dummy
            dummy_input = torch.randn(1, *input_shape)
            
            # Forward pass
            output = model(dummy_input)
            
            # Visualizar
            dot = make_dot(output, params=dict(model.named_parameters()))
            
            if save_path:
                dot.render(save_path, format='png')
            else:
                dot.render('model_graph', format='png')
        
        except ImportError:
            logger.warning("torchviz not available, using text summary instead")
            print(model)
    
    def print_model_summary(self, model: nn.Module, input_shape: Tuple[int, ...]):
        """
        Imprimir resumen del modelo.
        
        Args:
            model: Modelo
            input_shape: Forma de entrada
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model: {model.__class__.__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("\nArchitecture:")
        print(model)


class MetricsVisualizer:
    """
    Visualizador de métricas.
    """
    
    def __init__(self):
        """Inicializar visualizador."""
        if not _has_matplotlib:
            raise ImportError("matplotlib is required for MetricsVisualizer")
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plotear matriz de confusión.
        
        Args:
            cm: Matriz de confusión
            class_names: Nombres de clases (opcional)
            save_path: Ruta para guardar (opcional)
        """
        if not _has_matplotlib:
            raise ImportError("matplotlib is required")
        
        if _has_seaborn:
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )
        else:
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        save_path: Optional[str] = None
    ):
        """
        Plotear curva ROC.
        
        Args:
            fpr: False Positive Rate
            tpr: True Positive Rate
            auc: AUC score
            save_path: Ruta para guardar (opcional)
        """
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        ap: float,
        save_path: Optional[str] = None
    ):
        """
        Plotear curva Precision-Recall.
        
        Args:
            precision: Precision
            recall: Recall
            ap: Average Precision
            save_path: Ruta para guardar (opcional)
        """
        plt.plot(recall, precision, label=f'PR curve (AP = {ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

