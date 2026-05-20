"""
Visualization - Visualización
==============================

Utilidades para visualización de resultados y métricas.
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Visualizador de entrenamiento"""
    
    def __init__(self, figsize: tuple = (12, 6)):
        """
        Inicializar visualizador
        
        Args:
            figsize: Tamaño de figura
        """
        self.figsize = figsize
        sns.set_style("whitegrid")
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        train_metrics: Optional[Dict[str, List[float]]] = None,
        save_path: Optional[str] = None
    ):
        """
        Graficar curvas de entrenamiento
        
        Args:
            train_losses: Pérdidas de entrenamiento
            val_losses: Pérdidas de validación
            train_metrics: Métricas adicionales
            save_path: Ruta para guardar
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Loss curve
        axes[0].plot(train_losses, label="Train Loss", color="blue")
        if val_losses:
            axes[0].plot(val_losses, label="Val Loss", color="red")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True)
        
        # Metrics
        if train_metrics:
            for metric_name, values in train_metrics.items():
                axes[1].plot(values, label=metric_name)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Metric")
            axes[1].set_title("Training Metrics")
            axes[1].legend()
            axes[1].grid(True)
        else:
            axes[1].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Gráfico guardado: {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Graficar matriz de confusión
        
        Args:
            cm: Matriz de confusión
            class_names: Nombres de clases
            save_path: Ruta para guardar
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Matriz de confusión guardada: {save_path}")
        
        plt.close()
    
    def plot_attention_weights(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        save_path: Optional[str] = None
    ):
        """
        Visualizar pesos de atención
        
        Args:
            attention_weights: Pesos de atención [heads, seq_len, seq_len]
            tokens: Lista de tokens
            save_path: Ruta para guardar
        """
        # Promediar sobre heads
        if attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=0)
        
        attention_weights = attention_weights.cpu().numpy()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attention_weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            cbar_kws={"label": "Attention Weight"}
        )
        plt.xlabel("Key")
        plt.ylabel("Query")
        plt.title("Attention Weights")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Pesos de atención guardados: {save_path}")
        
        plt.close()


class PredictionVisualizer:
    """Visualizador de predicciones"""
    
    @staticmethod
    def visualize_text_generation(
        prompt: str,
        generated: str,
        save_path: Optional[str] = None
    ):
        """
        Visualizar generación de texto
        
        Args:
            prompt: Prompt original
            generated: Texto generado
            save_path: Ruta para guardar
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        
        text = f"Prompt: {prompt}\n\nGenerated: {generated}"
        ax.text(0.1, 0.5, text, fontsize=12, verticalalignment="center",
                wrap=True, family="monospace")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.close()
    
    @staticmethod
    def visualize_sentiment_distribution(
        sentiments: List[str],
        save_path: Optional[str] = None
    ):
        """
        Visualizar distribución de sentimientos
        
        Args:
            sentiments: Lista de sentimientos
            save_path: Ruta para guardar
        """
        from collections import Counter
        
        counts = Counter(sentiments)
        
        plt.figure(figsize=(8, 6))
        plt.bar(counts.keys(), counts.values())
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.title("Sentiment Distribution")
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.close()




