"""
Visualization Utilities
Plotting and visualization for training metrics
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """
    Visualize training metrics and results
    """
    
    @staticmethod
    def plot_training_history(
        history: Dict[str, List[float]],
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot training history
        
        Args:
            history: Dictionary with training metrics
            save_path: Path to save plot
            show: Whether to display plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        if 'train_loss' in history and 'val_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss')
            axes[0].plot(history['val_loss'], label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True)
        
        # Plot accuracy
        if 'train_acc' in history and 'val_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Acc')
            axes[1].plot(history['val_acc'], label='Val Acc')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training and Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            save_path: Path to save plot
            show: Whether to display plot
        """
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_learning_curve(
        train_scores: List[float],
        val_scores: List[float],
        train_sizes: List[int],
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot learning curve
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores
            train_sizes: Training set sizes
            save_path: Path to save plot
            show: Whether to display plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(train_sizes, train_mean, 'o-', label='Training Score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        
        ax.plot(train_sizes, val_mean, 'o-', label='Validation Score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Score')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved learning curve to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()



