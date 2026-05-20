"""
Modular Visualization Component for Gradio
"""

from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

try:
    import gradio as gr
    import matplotlib.pyplot as plt
    import numpy as np
    GRADIO_AVAILABLE = True
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Gradio or Matplotlib not available")


class VisualizationComponent:
    """
    Modular component for creating visualizations in Gradio
    """
    
    def __init__(self, figsize: tuple = (10, 6), dpi: int = 100):
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio required")
        
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        tokens: Optional[List[str]] = None
    ) -> plt.Figure:
        """
        Plot attention weights
        
        Args:
            attention_weights: Attention matrix [seq_len, seq_len]
            tokens: Optional token labels
        
        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required")
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Average over heads if needed
        if attention_weights.ndim > 2:
            attention_weights = attention_weights.mean(axis=0)
        
        im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title('Attention Weights')
        
        if tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        metrics: Optional[List[str]] = None
    ) -> plt.Figure:
        """
        Plot training history
        
        Args:
            history: Dictionary of metric histories
            metrics: List of metrics to plot
        
        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required")
        
        if metrics is None:
            metrics = [k for k in history.keys() if k != "epoch"]
        
        fig, axes = plt.subplots(
            len(metrics), 1,
            figsize=self.figsize,
            dpi=self.dpi,
            sharex=True
        )
        
        if len(metrics) == 1:
            axes = [axes]
        
        epochs = history.get("epoch", range(len(history[metrics[0]])))
        
        for ax, metric in zip(axes, metrics):
            if metric in history:
                ax.plot(epochs, history[metric], label=metric)
                ax.set_ylabel(metric)
                ax.legend()
                ax.grid(True)
        
        axes[-1].set_xlabel("Epoch")
        plt.tight_layout()
        
        return fig
    
    def plot_feature_distribution(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> plt.Figure:
        """
        Plot feature distributions
        
        Args:
            features: Feature array [num_samples, num_features]
            feature_names: Optional feature names
        
        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required")
        
        num_features = features.shape[1]
        num_cols = min(4, num_features)
        num_rows = (num_features + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(
            num_rows, num_cols,
            figsize=(self.figsize[0] * num_cols, self.figsize[1] * num_rows),
            dpi=self.dpi
        )
        
        if num_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(num_features):
            ax = axes[i]
            ax.hist(features[:, i], bins=30, alpha=0.7)
            ax.set_title(feature_names[i] if feature_names else f"Feature {i}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(num_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_visualization_interface(self) -> gr.Blocks:
        """Create Gradio interface for visualizations"""
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio required")
        
        with gr.Blocks(title="Visualizations") as interface:
            gr.Markdown("# Music Analysis Visualizations")
            
            with gr.Tab("Attention Weights"):
                attention_input = gr.File(label="Upload Attention Weights")
                attention_output = gr.Plot(label="Attention Visualization")
                
                def plot_attention(file):
                    # Load and plot attention weights
                    data = np.load(file.name)
                    fig = self.plot_attention_weights(data)
                    return fig
                
                attention_input.change(
                    fn=plot_attention,
                    inputs=attention_input,
                    outputs=attention_output
                )
            
            with gr.Tab("Training History"):
                history_input = gr.JSON(label="Training History")
                history_output = gr.Plot(label="Training Curves")
                
                def plot_history(history):
                    fig = self.plot_training_history(history)
                    return fig
                
                history_input.change(
                    fn=plot_history,
                    inputs=history_input,
                    outputs=history_output
                )
        
        return interface



