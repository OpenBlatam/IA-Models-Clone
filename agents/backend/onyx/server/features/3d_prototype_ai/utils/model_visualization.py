"""
Model Visualization - Visualización de modelos
================================================
Visualización de arquitecturas, entrenamiento, y resultados
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available")

try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False
    logging.warning("torchviz not available")

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """Sistema de visualización de modelos"""
    
    def __init__(self, output_dir: str = "./storage/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> str:
        """Visualiza historial de entrenamiento"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
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
        
        # Combined metrics
        if len(history) > 0:
            for key, values in history.items():
                if key not in ["train_loss", "val_loss", "train_acc", "val_acc", "learning_rate"]:
                    axes[1, 1].plot(values, label=key)
            axes[1, 1].set_title("Other Metrics")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "training_history.png"
        
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved training history visualization to {save_path}")
        return str(save_path)
    
    def visualize_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Visualiza matriz de confusión"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return ""
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names or range(len(cm)),
            yticklabels=class_names or range(len(cm))
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        if save_path is None:
            save_path = self.output_dir / "confusion_matrix.png"
        
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved confusion matrix to {save_path}")
        return str(save_path)
    
    def visualize_model_architecture(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        save_path: Optional[str] = None
    ) -> str:
        """Visualiza arquitectura del modelo"""
        if not TORCHVIZ_AVAILABLE:
            logger.warning("torchviz not available")
            return ""
        
        try:
            dummy_input = torch.randn(1, *input_shape)
            output = model(dummy_input)
            
            dot = make_dot(output, params=dict(model.named_parameters()))
            
            if save_path is None:
                save_path = self.output_dir / "model_architecture.png"
            
            dot.render(str(save_path).replace(".png", ""), format="png")
            
            logger.info(f"Saved model architecture to {save_path}")
            return str(save_path)
        except Exception as e:
            logger.error(f"Failed to visualize architecture: {e}")
            return ""
    
    def visualize_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        save_path: Optional[str] = None
    ) -> str:
        """Visualiza heatmap de atención"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return ""
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attention_weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="YlOrRd",
            cbar=True
        )
        plt.title("Attention Heatmap")
        plt.xlabel("Key")
        plt.ylabel("Query")
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "attention_heatmap.png"
        
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved attention heatmap to {save_path}")
        return str(save_path)
    
    def visualize_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_k: int = 20,
        save_path: Optional[str] = None
    ) -> str:
        """Visualiza importancia de features"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return ""
        
        # Ordenar por importancia
        indices = np.argsort(importances)[-top_k:][::-1]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importances)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel("Importance")
        plt.title(f"Top {top_k} Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "feature_importance.png"
        
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved feature importance to {save_path}")
        return str(save_path)
    
    def visualize_prediction_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> str:
        """Visualiza distribución de predicciones"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return ""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].hist(y_true, bins=30, alpha=0.5, label="True", color="blue")
        axes[0].hist(y_pred, bins=30, alpha=0.5, label="Predicted", color="red")
        axes[0].set_title("Distribution Comparison")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        
        axes[1].scatter(y_true, y_pred, alpha=0.5)
        axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
        axes[1].set_title("True vs Predicted")
        axes[1].set_xlabel("True")
        axes[1].set_ylabel("Predicted")
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "prediction_distribution.png"
        
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved prediction distribution to {save_path}")
        return str(save_path)




