"""
Training Visualization

Utilities for plotting training progress and metrics.
"""

import logging
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available for plotting")


class TrainingPlotter:
    """Plot training progress and metrics."""
    
    def __init__(self, figsize: tuple = (12, 6)):
        """
        Initialize training plotter.
        
        Args:
            figsize: Figure size
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for plotting")
        
        self.figsize = figsize
    
    def plot_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = False
    ) -> None:
        """
        Plot training history.
        
        Args:
            history: Dictionary with metric names and values
            save_path: Path to save plot
            show: Whether to show plot
        """
        fig, axes = plt.subplots(1, len(history), figsize=self.figsize)
        
        if len(history) == 1:
            axes = [axes]
        
        for idx, (metric_name, values) in enumerate(history.items()):
            ax = axes[idx] if len(history) > 1 else axes[0]
            ax.plot(values, label=metric_name)
            ax.set_title(metric_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_loss_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> None:
        """
        Plot loss curves.
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses (optional)
            save_path: Path to save plot
            show: Whether to show plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if val_losses:
            ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved loss plot: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_metrics(
        self,
        metrics: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = False
    ) -> None:
        """
        Plot multiple metrics.
        
        Args:
            metrics: Dictionary with metric names and values
            save_path: Path to save plot
            show: Whether to show plot
        """
        num_metrics = len(metrics)
        cols = min(3, num_metrics)
        rows = (num_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        
        if num_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            ax.plot(values, label=metric_name)
            ax.set_title(metric_name, fontsize=12)
            ax.set_xlabel('Step', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved metrics plot: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """Convenience function to plot training history."""
    plotter = TrainingPlotter()
    plotter.plot_history(history, save_path)


def plot_loss_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> None:
    """Convenience function to plot loss curves."""
    plotter = TrainingPlotter()
    plotter.plot_loss_curves(train_losses, val_losses, save_path)


def plot_metrics(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """Convenience function to plot metrics."""
    plotter = TrainingPlotter()
    plotter.plot_metrics(metrics, save_path)



