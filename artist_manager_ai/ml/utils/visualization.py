"""
Visualization Utilities
=======================

Utilities for visualizing training progress and model outputs.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Visualizer for training progress."""
    
    def __init__(self, save_dir: str = "plots"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        import os
        self.save_dir = os.path.join(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self._logger = logger
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        if "train_loss" in history and "val_loss" in history:
            axes[0].plot(history["train_loss"], label="Train Loss")
            axes[0].plot(history["val_loss"], label="Val Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training and Validation Loss")
            axes[0].legend()
            axes[0].grid(True)
        
        # Plot metrics
        metric_keys = [k for k in history.keys() if k not in ["train_loss", "val_loss"]]
        if metric_keys:
            for key in metric_keys[:5]:  # Plot up to 5 metrics
                axes[1].plot(history[key], label=key)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Metric Value")
            axes[1].set_title("Training Metrics")
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self._logger.info(f"Plot saved to {save_path}")
        else:
            plt.savefig(f"{self.save_dir}/training_history.png")
        
        plt.close()
    
    def plot_predictions_vs_targets(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot predictions vs targets.
        
        Args:
            predictions: Predicted values
            targets: Target values
            save_path: Path to save plot
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, predictions, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")
        
        plt.xlabel("Targets")
        plt.ylabel("Predictions")
        plt.title("Predictions vs Targets")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(f"{self.save_dir}/predictions_vs_targets.png")
        
        plt.close()




