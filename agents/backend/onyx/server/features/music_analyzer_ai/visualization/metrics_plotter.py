"""
Metrics Plotter
Plot training metrics
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available")


class MetricsPlotter:
    """Plot metrics"""
    
    def __init__(self, figsize: tuple = (10, 6)):
        self.figsize = figsize
    
    def plot_metrics(
        self,
        metrics: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot multiple metrics
        
        Args:
            metrics: Dictionary of metric names to values
            save_path: Optional path to save figure
            show: Whether to show figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return
        
        fig, axes = plt.subplots(
            len(metrics), 1,
            figsize=self.figsize,
            sharex=True
        )
        
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, (name, values) in enumerate(metrics.items()):
            axes[idx].plot(values, label=name)
            axes[idx].set_ylabel(name)
            axes[idx].legend()
            axes[idx].grid(True)
        
        axes[-1].set_xlabel("Step")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_loss_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Plot loss curves"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return
        
        plt.figure(figsize=self.figsize)
        plt.plot(train_losses, label="Train Loss")
        if val_losses:
            plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.title("Training Loss Curves")
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()



