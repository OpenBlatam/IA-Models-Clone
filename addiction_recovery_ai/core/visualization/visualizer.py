"""
Visualization Utilities
Model and training visualization
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """
    Visualize training progress
    """
    
    @staticmethod
    def plot_training_history(
        history: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot training history
        
        Args:
            history: Dictionary with training metrics
            save_path: Path to save plot
            show: Show plot
        """
        fig, axes = plt.subplots(1, len(history), figsize=(5 * len(history), 4))
        
        if len(history) == 1:
            axes = [axes]
        
        for idx, (metric_name, values) in enumerate(history.items()):
            axes[idx].plot(values)
            axes[idx].set_title(metric_name)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric_name)
            axes[idx].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_loss_curves(
        train_losses: List[float],
        val_losses: List[float],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot loss curves
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses
            save_path: Path to save plot
            show: Show plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Loss curves saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


class ModelVisualizer:
    """
    Visualize model architecture and outputs
    """
    
    @staticmethod
    def plot_attention_weights(
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot attention weights
        
        Args:
            attention_weights: Attention weights tensor
            tokens: Token strings (optional)
            save_path: Path to save plot
            show: Show plot
        """
        if attention_weights.dim() > 2:
            # Average over heads if multi-head
            attention_weights = attention_weights.mean(dim=0)
        
        attention_np = attention_weights.cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        plt.imshow(attention_np, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.title('Attention Weights')
        
        if tokens:
            plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
            plt.yticks(range(len(tokens)), tokens)
        
        plt.xlabel('Key')
        plt.ylabel('Query')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Attention plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_predictions(
        true_values: List[float],
        predictions: List[float],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot predictions vs true values
        
        Args:
            true_values: True values
            predictions: Predictions
            save_path: Path to save plot
            show: Show plot
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(true_values, predictions, alpha=0.5)
        plt.plot([min(true_values), max(true_values)], 
                [min(true_values), max(true_values)], 
                'r--', label='Perfect Prediction')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predictions vs True Values')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Predictions plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def create_training_visualizer() -> TrainingVisualizer:
    """Factory for training visualizer"""
    return TrainingVisualizer()


def create_model_visualizer() -> ModelVisualizer:
    """Factory for model visualizer"""
    return ModelVisualizer()









