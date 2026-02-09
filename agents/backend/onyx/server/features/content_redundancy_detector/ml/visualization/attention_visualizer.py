"""
Attention Visualizer
Visualize attention maps and Grad-CAM
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Visualize attention maps
    """
    
    @staticmethod
    def visualize_gradcam(
        image: Image.Image,
        cam: np.ndarray,
        save_path: Optional[Path] = None,
        show: bool = False,
        alpha: float = 0.4,
    ) -> None:
        """
        Visualize Grad-CAM overlay
        
        Args:
            image: Original image
            cam: Class activation map
            save_path: Path to save visualization
            show: Whether to display
            alpha: Overlay transparency
        """
        # Resize CAM to image size
        from scipy.ndimage import zoom
        cam_resized = zoom(cam, (
            image.size[1] / cam.shape[0],
            image.size[0] / cam.shape[1]
        ))
        
        # Normalize
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        
        # Create overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # CAM heatmap
        im = axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(cam_resized, cmap='jet', alpha=alpha)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()



