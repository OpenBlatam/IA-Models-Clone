"""
Attention Visualizer Module

Visualizes attention patterns in transformer models.
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("PyTorch not available")


class AttentionVisualizer:
    """
    Visualize attention patterns in transformer models.
    """
    
    @staticmethod
    def extract_attention_weights(model, input_ids, layer_idx: Optional[int] = None):
        """
        Extract attention weights from transformer model.
        
        Args:
            model: Transformer model.
            input_ids: Input token IDs.
            layer_idx: Optional layer index to extract.
        
        Returns:
            Attention weights tensor(s).
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers required for attention visualization")
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
            attentions = outputs.attentions
        
        if layer_idx is not None:
            return attentions[layer_idx]
        return attentions
    
    @staticmethod
    def visualize_attention_pattern(
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Visualize attention pattern.
        
        Args:
            attention_weights: Attention weights tensor.
            tokens: Optional token list for labeling.
        
        Returns:
            Dictionary with attention visualization data.
        """
        # Average over heads
        attention_avg = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]
        
        # Get attention scores
        attention_scores = attention_avg[0].cpu().numpy()
        
        return {
            "attention_matrix": attention_scores.tolist(),
            "shape": list(attention_scores.shape),
            "max_attention": float(attention_scores.max()),
            "min_attention": float(attention_scores.min()),
            "mean_attention": float(attention_scores.mean())
        }



