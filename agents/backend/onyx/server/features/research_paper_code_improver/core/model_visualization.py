"""
Model Visualization System - Sistema de visualización de modelos
=================================================================
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import os

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """Visualizador de modelos"""
    
    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_architecture(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        filename: Optional[str] = None
    ) -> str:
        """Visualiza arquitectura del modelo"""
        try:
            from torchviz import make_dot
            
            # Crear input dummy
            dummy_input = torch.randn(input_shape)
            
            # Forward pass
            output = model(dummy_input)
            
            # Crear gráfico
            dot = make_dot(output, params=dict(model.named_parameters()))
            
            # Guardar
            if filename is None:
                filename = "model_architecture"
            
            filepath = os.path.join(self.output_dir, f"{filename}.png")
            dot.render(filepath, format='png', cleanup=True)
            
            logger.info(f"Arquitectura visualizada: {filepath}")
            return filepath
        except ImportError:
            logger.warning("torchviz no disponible para visualización")
            return ""
    
    def visualize_attention(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        filename: Optional[str] = None
    ) -> str:
        """Visualiza pesos de atención"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                attention_weights,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='Blues',
                cbar=True
            )
            plt.title('Attention Weights')
            plt.xlabel('Key')
            plt.ylabel('Query')
            plt.tight_layout()
            
            if filename is None:
                filename = "attention_weights"
            
            filepath = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Attention visualizado: {filepath}")
            return filepath
        except ImportError:
            logger.warning("matplotlib/seaborn no disponible")
            return ""
    
    def visualize_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: Optional[Dict[str, List[float]]] = None,
        filename: Optional[str] = None
    ) -> str:
        """Visualiza curvas de entrenamiento"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss curves
            axes[0].plot(train_losses, label='Train Loss')
            axes[0].plot(val_losses, label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Curves')
            axes[0].legend()
            axes[0].grid(True)
            
            # Metrics
            if train_metrics:
                for metric_name, values in train_metrics.items():
                    axes[1].plot(values, label=metric_name)
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Metric')
                axes[1].set_title('Training Metrics')
                axes[1].legend()
                axes[1].grid(True)
            
            plt.tight_layout()
            
            if filename is None:
                filename = "training_curves"
            
            filepath = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Curvas de entrenamiento visualizadas: {filepath}")
            return filepath
        except ImportError:
            logger.warning("matplotlib no disponible")
            return ""
    
    def visualize_feature_maps(
        self,
        feature_maps: torch.Tensor,
        num_maps: int = 16,
        filename: Optional[str] = None
    ) -> str:
        """Visualiza feature maps"""
        try:
            import matplotlib.pyplot as plt
            
            # Seleccionar primeros num_maps
            maps = feature_maps[:num_maps].cpu().numpy()
            
            # Crear grid
            rows = int(np.sqrt(num_maps))
            cols = (num_maps + rows - 1) // rows
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
            axes = axes.flatten() if num_maps > 1 else [axes]
            
            for i, ax in enumerate(axes):
                if i < len(maps):
                    ax.imshow(maps[i], cmap='viridis')
                    ax.set_title(f'Feature Map {i+1}')
                    ax.axis('off')
                else:
                    ax.axis('off')
            
            plt.tight_layout()
            
            if filename is None:
                filename = "feature_maps"
            
            filepath = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature maps visualizados: {filepath}")
            return filepath
        except ImportError:
            logger.warning("matplotlib no disponible")
            return ""




