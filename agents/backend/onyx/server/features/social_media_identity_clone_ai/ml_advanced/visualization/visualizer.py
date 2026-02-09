"""
Visualización avanzada para deep learning
"""

import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Visualizador de entrenamiento"""
    
    def __init__(self):
        pass
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        save_path: Optional[str] = None
    ):
        """Plotea curvas de entrenamiento"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        if val_losses:
            plt.plot(val_losses, label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if val_losses:
            plt.subplot(1, 2, 2)
            plt.plot(train_losses, label='Train', color='blue', alpha=0.7)
            plt.plot(val_losses, label='Val', color='red', alpha=0.7)
            plt.fill_between(range(len(train_losses)), train_losses, alpha=0.3)
            plt.fill_between(range(len(val_losses)), val_losses, alpha=0.3)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Gráfico guardado: {save_path}")
        else:
            plt.show()
    
    def plot_attention_weights(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        save_path: Optional[str] = None
    ):
        """Visualiza attention weights"""
        if attention_weights.dim() > 2:
            attention_weights = attention_weights.mean(dim=0)
        
        attention_np = attention_weights.cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        plt.imshow(attention_np, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title('Attention Weights')
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.yticks(range(len(tokens)), tokens)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_embeddings(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        method: str = "tsne"  # "tsne" or "pca"
    ):
        """Visualiza embeddings en 2D"""
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        if embeddings.shape[1] > 2:
            if method == "tsne":
                reducer = TSNE(n_components=2, random_state=42)
            else:
                reducer = PCA(n_components=2)
            
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
        
        if labels:
            for i, label in enumerate(labels):
                plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
        
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(f'Embeddings Visualization ({method.upper()})')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()




