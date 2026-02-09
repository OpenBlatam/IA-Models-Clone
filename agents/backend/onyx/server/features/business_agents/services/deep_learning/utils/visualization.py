"""
Visualization Utilities
======================

Utilities for visualizing training progress and model outputs.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available")

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Visualize training progress."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for visualization")
        
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            metrics: List of metrics to plot (defaults to all)
            save_path: Path to save plot
        """
        if metrics is None:
            metrics = list(history.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric in history:
                axes[idx].plot(history[metric], label=metric)
                axes[idx].set_title(metric)
                axes[idx].set_xlabel('Epoch')
                axes[idx].set_ylabel(metric)
                axes[idx].legend()
                axes[idx].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        elif self.save_dir:
            save_path = self.save_dir / "training_history.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    def plot_loss_comparison(
        self,
        train_loss: List[float],
        val_loss: List[float],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot training vs validation loss.
        
        Args:
            train_loss: Training loss history
            val_loss: Validation loss history
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Train Loss', marker='o')
        plt.plot(val_loss, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.save_dir:
            save_path = self.save_dir / "loss_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
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
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.save_dir:
            save_path = self.save_dir / "confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_learning_curve(
        self,
        learning_rates: List[float],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot learning rate schedule.
        
        Args:
            learning_rates: Learning rate history
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(learning_rates)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif self.save_dir:
            save_path = self.save_dir / "learning_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()


def visualize_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    num_samples: int = 8,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize model predictions.
    
    Args:
        images: Input images
        predictions: Model predictions
        targets: Ground truth (optional)
        num_samples: Number of samples to visualize
        save_path: Path to save plot
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required")
    
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))
    
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        # Input image
        img = images[i].cpu()
        if img.dim() == 3:
            img = img.permute(1, 2, 0)
        axes[0, i].imshow(img.numpy())
        axes[0, i].set_title(f'Input {i+1}')
        axes[0, i].axis('off')
        
        # Prediction
        pred = predictions[i].cpu()
        if pred.dim() > 0:
            pred = pred.argmax() if pred.numel() > 1 else pred.item()
        axes[1, i].text(0.5, 0.5, f'Pred: {pred}', 
                       ha='center', va='center', fontsize=12)
        if targets is not None:
            target = targets[i].cpu().item()
            axes[1, i].text(0.5, 0.3, f'True: {target}',
                           ha='center', va='center', fontsize=12)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()



