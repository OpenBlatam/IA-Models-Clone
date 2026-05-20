"""
Confusion Matrix Plotter
Plot confusion matrices
"""

from typing import Optional, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    SEABORN_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    SEABORN_AVAILABLE = False
    logger.warning("Matplotlib or Seaborn not available")


class ConfusionMatrixPlotter:
    """Plot confusion matrices"""
    
    def __init__(self, figsize: tuple = (10, 8)):
        self.figsize = figsize
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        normalize: bool = True
    ):
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Optional class names
            save_path: Optional path to save
            show: Whether to show
            normalize: Whether to normalize
        """
        if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
            logger.warning("Matplotlib or Seaborn not available")
            return
        
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            cm = confusion_matrix
            fmt = 'd'
        
        plt.figure(figsize=self.figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()



