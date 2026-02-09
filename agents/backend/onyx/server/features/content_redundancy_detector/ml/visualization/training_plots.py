"""
Training Plots
Specialized plotting for training metrics
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrainingPlotter:
    """
    Plot training metrics
    """
    
    @staticmethod
    def plot_loss_history(
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> None:
        """
        Plot loss history
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses (optional)
            save_path: Path to save plot
            show: Whether to display plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', marker='o')
        if val_losses:
            plt.plot(val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_accuracy_history(
        train_accuracies: List[float],
        val_accuracies: Optional[List[float]] = None,
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> None:
        """
        Plot accuracy history
        
        Args:
            train_accuracies: Training accuracies
            val_accuracies: Validation accuracies (optional)
            save_path: Path to save plot
            show: Whether to display plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_accuracies, label='Train Accuracy', marker='o')
        if val_accuracies:
            plt.plot(val_accuracies, label='Val Accuracy', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_learning_rate(
        learning_rates: List[float],
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> None:
        """
        Plot learning rate schedule
        
        Args:
            learning_rates: Learning rate values
            save_path: Path to save plot
            show: Whether to display plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(learning_rates, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()



