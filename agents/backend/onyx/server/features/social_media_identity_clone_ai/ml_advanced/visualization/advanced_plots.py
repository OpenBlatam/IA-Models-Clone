"""
Visualización avanzada con más gráficos
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
import torch
import logging

logger = logging.getLogger(__name__)

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class AdvancedVisualizer:
    """Visualizador avanzado"""
    
    def __init__(self):
        pass
    
    def plot_loss_landscape(
        self,
        losses: List[float],
        learning_rates: List[float],
        save_path: Optional[str] = None
    ):
        """Plotea landscape de loss vs learning rate"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        ax1.plot(losses, label='Loss', color='blue')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate schedule
        ax2.plot(learning_rates, label='Learning Rate', color='green')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Plotea matriz de confusión"""
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
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_feature_importance(
        self,
        features: List[str],
        importances: np.ndarray,
        top_k: int = 20,
        save_path: Optional[str] = None
    ):
        """Plotea importancia de features"""
        # Ordenar por importancia
        indices = np.argsort(importances)[::-1][:top_k]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_k} Feature Importance')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_training_comparison(
        self,
        experiments: Dict[str, Dict[str, List[float]]],
        save_path: Optional[str] = None
    ):
        """Compara múltiples experimentos"""
        plt.figure(figsize=(12, 6))
        
        for exp_name, metrics in experiments.items():
            if "loss" in metrics:
                plt.plot(metrics["loss"], label=f"{exp_name} - Loss", alpha=0.7)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_gradient_flow(
        self,
        model: torch.nn.Module,
        save_path: Optional[str] = None
    ):
        """Visualiza flujo de gradientes"""
        gradients = []
        layer_names = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients.append(param.grad.norm().item())
                layer_names.append(name)
        
        if not gradients:
            logger.warning("No hay gradientes para visualizar")
            return
        
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(gradients)), gradients)
        plt.yticks(range(len(layer_names)), layer_names)
        plt.xlabel('Gradient Norm')
        plt.title('Gradient Flow')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()




