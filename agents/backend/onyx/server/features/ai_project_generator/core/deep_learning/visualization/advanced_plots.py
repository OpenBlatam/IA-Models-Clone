"""
Advanced Visualization Utilities
================================

Advanced plotting and visualization functions.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available")

# Try to import seaborn
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot comprehensive training history.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save figure
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Training History', fontsize=16)
    
    # Loss
    if 'train_loss' in history or 'loss' in history:
        loss_key = 'train_loss' if 'train_loss' in history else 'loss'
        axes[0, 0].plot(history[loss_key], label='Train Loss')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Accuracy
    if 'train_acc' in history or 'accuracy' in history:
        acc_key = 'train_acc' if 'train_acc' in history else 'accuracy'
        axes[0, 1].plot(history[acc_key], label='Train Acc')
        if 'val_acc' in history:
            axes[0, 1].plot(history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Learning Rate
    if 'lr' in history or 'learning_rate' in history:
        lr_key = 'lr' if 'lr' in history else 'learning_rate'
        axes[1, 0].plot(history[lr_key])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].grid(True)
    
    # Additional metrics
    other_metrics = [k for k in history.keys() 
                    if k not in ['train_loss', 'val_loss', 'loss', 
                               'train_acc', 'val_acc', 'accuracy', 
                               'lr', 'learning_rate']]
    if other_metrics:
        for metric in other_metrics[:1]:  # Plot first additional metric
            axes[1, 1].plot(history[metric], label=metric)
            axes[1, 1].set_title('Other Metrics')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history saved to {save_path}")
    
    plt.show()


def visualize_model_architecture(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        save_path: Path to save figure
    """
    try:
        from torchsummary import summary
        summary(model, input_shape)
    except ImportError:
        logger.warning("torchsummary not available, using manual visualization")
        
        # Manual visualization
        layers = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                layers.append({
                    'name': name,
                    'type': type(module).__name__,
                    'params': sum(p.numel() for p in module.parameters())
                })
        
        print("\nModel Architecture:")
        print("=" * 60)
        for layer in layers[:20]:  # Show first 20 layers
            print(f"{layer['name']:30s} {layer['type']:20s} {layer['params']:>10,} params")
        print("=" * 60)


def plot_attention_weights(
    attention_weights: torch.Tensor,
    tokens: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Visualize attention weights.
    
    Args:
        attention_weights: Attention weights tensor [heads, seq_len, seq_len]
        tokens: Token strings (optional)
        save_path: Path to save figure
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available")
        return
    
    if attention_weights.dim() == 3:
        # Average over heads
        attention_weights = attention_weights.mean(dim=0)
    
    attention_weights = attention_weights.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    if tokens:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
    
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title('Attention Weights')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_feature_maps(
    feature_maps: torch.Tensor,
    num_maps: int = 16,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12)
) -> None:
    """
    Visualize feature maps from convolutional layers.
    
    Args:
        feature_maps: Feature maps tensor [batch, channels, height, width]
        num_maps: Number of feature maps to visualize
        save_path: Path to save figure
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available")
        return
    
    if feature_maps.dim() == 4:
        # Take first batch item
        feature_maps = feature_maps[0]
    
    num_channels = min(feature_maps.shape[0], num_maps)
    grid_size = int(np.ceil(np.sqrt(num_channels)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    fig.suptitle('Feature Maps', fontsize=16)
    
    for idx in range(num_channels):
        row = idx // grid_size
        col = idx % grid_size
        
        feature_map = feature_maps[idx].detach().cpu().numpy()
        axes[row, col].imshow(feature_map, cmap='viridis')
        axes[row, col].set_title(f'Channel {idx}')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(num_channels, grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix_advanced(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot advanced confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        normalize: Normalize confusion matrix
        save_path: Path to save figure
        figsize: Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available")
        return
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if SEABORN_AVAILABLE:
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', ax=ax, xticklabels=class_names, yticklabels=class_names)
    else:
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(np.arange(len(class_names) if class_names else cm.shape[1]))
        ax.set_yticks(np.arange(len(class_names) if class_names else cm.shape[0]))
        if class_names:
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = ax.text(j, i, f'{cm[i, j]:.2f}' if normalize else f'{cm[i, j]}',
                             ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()



