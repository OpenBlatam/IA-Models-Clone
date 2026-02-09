"""
Model Visualization

Utilities for visualizing model architectures and attention weights.
"""

import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available for plotting")

try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False
    logger.warning("Torchviz not available for model visualization")


class ModelVisualizer:
    """Visualize model architectures and internals."""
    
    def __init__(self):
        """Initialize model visualizer."""
        pass
    
    def visualize_architecture(
        self,
        model: nn.Module,
        input_size: tuple,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize model architecture.
        
        Args:
            model: Model to visualize
            input_size: Input tensor size
            save_path: Path to save visualization
        """
        if not TORCHVIZ_AVAILABLE:
            logger.warning("Torchviz not available, using text summary")
            # Fallback to text summary
            from core.debugging import get_model_summary
            summary = get_model_summary(model, input_size)
            logger.info(f"Model summary: {summary}")
            return
        
        # Create dummy input
        dummy_input = torch.randn(input_size)
        
        # Forward pass to get output
        try:
            output = model(dummy_input)
            
            # Create visualization
            dot = make_dot(output, params=dict(model.named_parameters()))
            
            if save_path:
                dot.render(save_path, format='png')
                logger.info(f"Saved architecture visualization: {save_path}")
        except Exception as e:
            logger.error(f"Error visualizing architecture: {e}")
    
    def plot_attention_weights(
        self,
        attention_weights: torch.Tensor,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> None:
        """
        Plot attention weights.
        
        Args:
            attention_weights: Attention weights tensor (batch, heads, seq_len, seq_len)
            save_path: Path to save plot
            show: Whether to show plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for plotting")
        
        # Average over heads if multiple heads
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.mean(dim=1)  # Average over heads
        
        # Take first batch item
        if attention_weights.dim() == 3:
            attention_weights = attention_weights[0]
        
        # Convert to numpy
        attn_np = attention_weights.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(attn_np, cmap='viridis', aspect='auto')
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        ax.set_title('Attention Weights', fontsize=14)
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention plot: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def visualize_model_architecture(
    model: nn.Module,
    input_size: tuple,
    save_path: Optional[str] = None
) -> None:
    """Convenience function to visualize architecture."""
    visualizer = ModelVisualizer()
    visualizer.visualize_architecture(model, input_size, save_path)


def plot_attention_weights(
    attention_weights: torch.Tensor,
    save_path: Optional[str] = None
) -> None:
    """Convenience function to plot attention weights."""
    visualizer = ModelVisualizer()
    visualizer.plot_attention_weights(attention_weights, save_path)



