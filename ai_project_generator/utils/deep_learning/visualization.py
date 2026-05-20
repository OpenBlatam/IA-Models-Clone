"""Visualization Utilities"""

def generate_visualization_code() -> str:
    return '''"""
Visualization Utilities
=======================

Utilidades para visualización de resultados.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
import torch
from sklearn.metrics import confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Visualizador de entrenamiento."""
    
    @staticmethod
    def plot_training_curves(
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: Optional[Dict[str, List[float]]] = None,
        val_metrics: Optional[Dict[str, List[float]]] = None,
        save_path: Optional[str] = None
    ):
        """Plotea curvas de entrenamiento."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        axes[0].plot(train_losses, label='Train Loss', marker='o')
        axes[0].plot(val_losses, label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Metrics
        if train_metrics and val_metrics:
            for metric_name in train_metrics.keys():
                if metric_name in val_metrics:
                    axes[1].plot(train_metrics[metric_name], label=f'Train {metric_name}', marker='o')
                    axes[1].plot(val_metrics[metric_name], label=f'Val {metric_name}', marker='s')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Metric Value')
            axes[1].set_title('Training and Validation Metrics')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: List[int],
        y_pred: List[int],
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Plotea matriz de confusión."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names if class_names else range(len(cm)),
            yticklabels=class_names if class_names else range(len(cm))
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_attention_weights(attention_weights: torch.Tensor, save_path: Optional[str] = None):
        """Visualiza pesos de atención."""
        if attention_weights.dim() > 2:
            attention_weights = attention_weights.mean(dim=0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_weights.cpu().numpy(), cmap='viridis', cbar=True)
        plt.title('Attention Weights')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
'''

