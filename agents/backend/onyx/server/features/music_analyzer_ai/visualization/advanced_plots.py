"""
Advanced Visualization
Advanced plotting and visualization for music analysis
"""

from typing import Dict, Any, Optional, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available")


class AdvancedPlotter:
    """
    Advanced plotting utilities for music analysis
    """
    
    def __init__(self, style: str = "seaborn-v0_8"):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for plotting")
        
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> str:
        """Plot training history"""
        if not MATPLOTLIB_AVAILABLE:
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training History", fontsize=16)
        
        # Loss
        if "train_loss" in history and "val_loss" in history:
            axes[0, 0].plot(history["train_loss"], label="Train Loss")
            axes[0, 0].plot(history["val_loss"], label="Val Loss")
            axes[0, 0].set_title("Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Accuracy
        if "train_acc" in history and "val_acc" in history:
            axes[0, 1].plot(history["train_acc"], label="Train Acc")
            axes[0, 1].plot(history["val_acc"], label="Val Acc")
            axes[0, 1].set_title("Accuracy")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning Rate
        if "learning_rate" in history:
            axes[1, 0].plot(history["learning_rate"])
            axes[1, 0].set_title("Learning Rate")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("LR")
            axes[1, 0].grid(True)
        
        # Gradient Norm
        if "grad_norm" in history:
            axes[1, 1].plot(history["grad_norm"])
            axes[1, 1].set_title("Gradient Norm")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Norm")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(temp_file.name, dpi=150, bbox_inches="tight")
            plt.close()
            return temp_file.name
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Plot confusion matrix"""
        if not MATPLOTLIB_AVAILABLE:
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names or [f"Class {i}" for i in range(len(confusion_matrix))],
            yticklabels=class_names or [f"Class {i}" for i in range(len(confusion_matrix))],
            ax=ax
        )
        
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(temp_file.name, dpi=150, bbox_inches="tight")
            plt.close()
            return temp_file.name
    
    def plot_feature_distribution(
        self,
        features: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> str:
        """Plot feature distributions"""
        if not MATPLOTLIB_AVAILABLE:
            return ""
        
        n_features = len(features)
        cols = 3
        rows = (n_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.suptitle("Feature Distributions", fontsize=16)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, values) in enumerate(features.items()):
            row = idx // cols
            col = idx % cols
            
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.hist(values, bins=50, alpha=0.7, edgecolor="black")
            ax.set_title(name)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(temp_file.name, dpi=150, bbox_inches="tight")
            plt.close()
            return temp_file.name
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Plot attention heatmap"""
        if not MATPLOTLIB_AVAILABLE:
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            attention_weights,
            annot=False,
            cmap="YlOrRd",
            xticklabels=tokens or [f"Token {i}" for i in range(attention_weights.shape[1])],
            yticklabels=tokens or [f"Token {i}" for i in range(attention_weights.shape[0])],
            ax=ax
        )
        
        ax.set_title("Attention Heatmap")
        ax.set_ylabel("Query")
        ax.set_xlabel("Key")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(temp_file.name, dpi=150, bbox_inches="tight")
            plt.close()
            return temp_file.name

