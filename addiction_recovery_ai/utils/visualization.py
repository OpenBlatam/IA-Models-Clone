"""
Advanced Visualization Tools
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Install: pip install matplotlib")


class ModelVisualizer:
    """Visualize model architecture and predictions"""
    
    def __init__(self):
        """Initialize visualizer"""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for visualization")
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot training curves
        
        Args:
            train_losses: Training losses
            val_losses: Optional validation losses
            save_path: Optional save path
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        
        if val_losses:
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training curves saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: List[float],
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance
        
        Args:
            feature_names: Feature names
            importances: Importance values
            save_path: Optional save path
        """
        plt.figure(figsize=(10, 6))
        
        indices = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_importances = [importances[i] for i in indices]
        
        plt.barh(range(len(sorted_names)), sorted_importances)
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_predictions_vs_actual(
        self,
        predictions: List[float],
        actuals: List[float],
        save_path: Optional[str] = None
    ):
        """
        Plot predictions vs actual values
        
        Args:
            predictions: Predictions
            actuals: Actual values
            save_path: Optional save path
        """
        plt.figure(figsize=(8, 8))
        
        plt.scatter(actuals, predictions, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(min(actuals), min(predictions))
        max_val = max(max(actuals), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Predictions vs Actual')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Predictions plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional label names
            save_path: Optional save path
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        
        if labels:
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels)
            plt.yticks(tick_marks, labels)
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class ProgressVisualizer:
    """Visualize recovery progress"""
    
    def __init__(self):
        """Initialize progress visualizer"""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for visualization")
    
    def plot_progress_timeline(
        self,
        dates: List[str],
        progress_scores: List[float],
        save_path: Optional[str] = None
    ):
        """
        Plot progress timeline
        
        Args:
            dates: List of dates
            progress_scores: Progress scores
            save_path: Optional save path
        """
        plt.figure(figsize=(12, 6))
        
        plt.plot(dates, progress_scores, 'b-', marker='o', linewidth=2, markersize=6)
        plt.fill_between(dates, progress_scores, alpha=0.3)
        
        plt.xlabel('Date')
        plt.ylabel('Progress Score')
        plt.title('Recovery Progress Timeline')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Progress timeline saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_risk_heatmap(
        self,
        risk_scores: List[List[float]],
        dates: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot risk heatmap
        
        Args:
            risk_scores: Risk scores matrix
            dates: List of dates
            save_path: Optional save path
        """
        plt.figure(figsize=(12, 6))
        
        risk_array = np.array(risk_scores)
        plt.imshow(risk_array, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
        plt.colorbar(label='Risk Score')
        
        plt.xlabel('Day')
        plt.ylabel('User')
        plt.title('Relapse Risk Heatmap')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Risk heatmap saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

