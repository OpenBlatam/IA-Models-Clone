"""
Training Plotter
Plot training progress
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


class TrainingPlotter:
    """Plot training progress"""
    
    def __init__(self, figsize: tuple = (15, 10)):
        self.figsize = figsize
    
    def plot_training_summary(
        self,
        train_metrics: Dict[str, List[float]],
        val_metrics: Optional[Dict[str, List[float]]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Plot comprehensive training summary"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return
        
        num_metrics = len(train_metrics)
        fig, axes = plt.subplots(
            num_metrics, 1,
            figsize=self.figsize,
            sharex=True
        )
        
        if num_metrics == 1:
            axes = [axes]
        
        for idx, (metric_name, train_values) in enumerate(train_metrics.items()):
            axes[idx].plot(train_values, label=f"Train {metric_name}", alpha=0.7)
            
            if val_metrics and metric_name in val_metrics:
                axes[idx].plot(
                    val_metrics[metric_name],
                    label=f"Val {metric_name}",
                    alpha=0.7
                )
            
            axes[idx].set_ylabel(metric_name)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel("Epoch")
        plt.suptitle("Training Progress", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_learning_rate(
        self,
        learning_rates: List[float],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Plot learning rate schedule"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(learning_rates)
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()



