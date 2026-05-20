"""
Metrics Plotter
Plot various metrics and statistics
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MetricsPlotter:
    """
    Plot various metrics
    """
    
    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        show: bool = False,
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
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_feature_importance(
        importance_scores: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 20,
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> None:
        """
        Plot feature importance
        
        Args:
            importance_scores: Importance scores
            feature_names: Feature names
            top_k: Number of top features to show
            save_path: Path to save plot
            show: Whether to display plot
        """
        # Get top k
        top_indices = np.argsort(importance_scores)[-top_k:][::-1]
        top_scores = importance_scores[top_indices]
        
        if feature_names:
            top_names = [feature_names[i] for i in top_indices]
        else:
            top_names = [f"Feature {i}" for i in top_indices]
        
        # Plot
        plt.figure(figsize=(10, max(6, top_k * 0.3)))
        plt.barh(range(len(top_scores)), top_scores)
        plt.yticks(range(len(top_scores)), top_names)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()



