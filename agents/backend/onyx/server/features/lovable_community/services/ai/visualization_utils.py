"""
Visualization Utilities for Deep Learning

Provides utilities for visualizing:
- Training curves
- Model architecture
- Attention weights
- Embeddings
- Confusion matrices
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available, using matplotlib only")


class TrainingVisualizer:
    """
    Visualize training progress and metrics
    """
    
    @staticmethod
    def plot_training_curves(
        history: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot training curves (loss, accuracy, etc.)
        
        Args:
            history: Dictionary with training history
            save_path: Path to save plot
            show: Whether to show plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        if "train_loss" in history:
            axes[0].plot(history["train_loss"], label="Train Loss")
        if "val_loss" in history:
            axes[0].plot(history["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        if "val_accuracy" in history:
            axes[1].plot(history["val_accuracy"], label="Val Accuracy")
        if "train_accuracy" in history:
            axes[1].plot(history["train_accuracy"], label="Train Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training Accuracy")
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_learning_rate_schedule(
        learning_rates: List[float],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot learning rate schedule
        
        Args:
            learning_rates: List of learning rates
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 4))
        plt.plot(learning_rates)
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    @staticmethod
    def plot_interactive_training_curves(
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create interactive training curves with plotly
        
        Args:
            history: Dictionary with training history
            save_path: Path to save HTML plot
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, falling back to matplotlib")
            TrainingVisualizer.plot_training_curves(history, save_path)
            return
        
        fig = go.Figure()
        
        # Add loss curves
        if "train_loss" in history:
            fig.add_trace(go.Scatter(
                y=history["train_loss"],
                mode='lines',
                name='Train Loss',
                line=dict(color='blue')
            ))
        
        if "val_loss" in history:
            fig.add_trace(go.Scatter(
                y=history["val_loss"],
                mode='lines',
                name='Val Loss',
                line=dict(color='red')
            ))
        
        # Add accuracy curves
        if "val_accuracy" in history:
            fig.add_trace(go.Scatter(
                y=history["val_accuracy"],
                mode='lines',
                name='Val Accuracy',
                line=dict(color='green'),
                yaxis='y2'
            ))
        
        fig.update_layout(
            title="Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis2=dict(
                title="Accuracy",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive plot saved to {save_path}")
        else:
            fig.show()


class ModelVisualizer:
    """
    Visualize model architecture and components
    """
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        normalize: bool = False
    ) -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            save_path: Path to save plot
            normalize: Whether to normalize
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)) if labels else None)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_attention_weights(
        attention_weights: np.ndarray,
        tokens: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize attention weights
        
        Args:
            attention_weights: Attention weight matrix
            tokens: List of tokens
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            attention_weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.title('Attention Weights')
        plt.xlabel('Key')
        plt.ylabel('Query')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    @staticmethod
    def plot_embeddings_2d(
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = "tsne",
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize embeddings in 2D
        
        Args:
            embeddings: Embedding vectors
            labels: Optional labels for coloring
            method: Dimensionality reduction method (tsne, pca, umap)
            save_path: Path to save plot
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "pca":
            reducer = PCA(n_components=2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            scatter = plt.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=labels,
                cmap='viridis',
                alpha=0.6
            )
            plt.colorbar(scatter)
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
        
        plt.title(f'Embeddings Visualization ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Embeddings plot saved to {save_path}")
        else:
            plt.show()


class MetricsVisualizer:
    """
    Visualize evaluation metrics
    """
    
    @staticmethod
    def plot_metrics_comparison(
        metrics_dict: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Compare metrics across different models/versions
        
        Args:
            metrics_dict: Dictionary mapping model names to metrics
            save_path: Path to save plot
        """
        models = list(metrics_dict.keys())
        metric_names = set()
        for metrics in metrics_dict.values():
            metric_names.update(metrics.keys())
        
        metric_names = sorted(list(metric_names))
        
        fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 4))
        
        if len(metric_names) == 1:
            axes = [axes]
        
        for idx, metric_name in enumerate(metric_names):
            values = [metrics_dict[model].get(metric_name, 0) for model in models]
            axes[idx].bar(models, values)
            axes[idx].set_title(metric_name)
            axes[idx].set_ylabel('Score')
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot ROC curve
        
        Args:
            y_true: True binary labels
            y_scores: Predicted scores
            save_path: Path to save plot
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()















