"""
Visualization Utilities
Utilities for visualizing training progress and results
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot training history
    
    Args:
        history: Dictionary with training history
        save_path: Path to save plot
        show: Whether to show plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if 'train_loss' in history and history['train_loss']:
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot learning rate
    if 'learning_rates' in history and history['learning_rates']:
        axes[1].plot(history['learning_rates'], label='Learning Rate', marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metrics(
    metrics: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot metrics over time
    
    Args:
        metrics: Dictionary with metrics
        save_path: Path to save plot
        show: Whether to show plot
    """
    num_metrics = len(metrics)
    if num_metrics == 0:
        logger.warning("No metrics to plot")
        return
    
    cols = min(3, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        if idx < len(axes):
            axes[idx].plot(values, marker='o')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric_name)
            axes[idx].set_title(f'{metric_name} over Time')
            axes[idx].grid(True)
    
    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        save_path: Path to save plot
        show: Whether to show plot
    """
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
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_predictions(
    images: np.ndarray,
    predictions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    num_samples: int = 8,
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot predictions on images
    
    Args:
        images: Image array
        predictions: Predictions array
        labels: True labels (optional)
        num_samples: Number of samples to plot
        save_path: Path to save plot
        show: Whether to show plot
    """
    num_samples = min(num_samples, len(images))
    
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Plot image
        if len(images[idx].shape) == 3:
            ax.imshow(images[idx])
        else:
            ax.imshow(images[idx], cmap='gray')
        
        # Add prediction
        pred_text = f"Pred: {predictions[idx]}"
        if labels is not None:
            true_text = f"True: {labels[idx]}"
            ax.set_title(f"{pred_text}\n{true_text}")
        else:
            ax.set_title(pred_text)
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Predictions plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()









