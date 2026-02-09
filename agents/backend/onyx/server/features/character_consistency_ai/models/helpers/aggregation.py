"""
Embedding Aggregation Utilities
===============================

Utilities for aggregating multiple image embeddings into consistent character embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EmbeddingAggregator:
    """Helper class for aggregating multiple embeddings."""
    
    def __init__(
        self,
        embedding_dim: int,
        aggregator: nn.MultiheadAttention,
        cross_attention: nn.MultiheadAttention,
        fusion_weights: nn.Parameter,
    ):
        """
        Initialize aggregator.
        
        Args:
            embedding_dim: Dimension of embeddings
            aggregator: Self-attention aggregator
            cross_attention: Cross-attention module
            fusion_weights: Learnable fusion weights
        """
        self.embedding_dim = embedding_dim
        self.aggregator = aggregator
        self.cross_attention = cross_attention
        self.fusion_weights = fusion_weights
    
    def aggregate_with_attention(
        self,
        stacked_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate embeddings using self-attention.
        
        Args:
            stacked_embeddings: [batch, num_images, embedding_dim]
            
        Returns:
            Aggregated embedding [batch, embedding_dim]
        """
        # Use mean embedding as query for better global context
        mean_query = stacked_embeddings.mean(dim=1, keepdim=True)  # [batch, 1, embedding_dim]
        
        attn_output, _ = self.aggregator(
            query=mean_query,
            key=stacked_embeddings,
            value=stacked_embeddings,
        )
        
        return attn_output.squeeze(1)  # [batch, embedding_dim]
    
    def aggregate_with_cross_attention(
        self,
        stacked_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate embeddings using cross-attention between images.
        
        Args:
            stacked_embeddings: [batch, num_images, embedding_dim]
            
        Returns:
            Aggregated embedding [batch, embedding_dim]
        """
        num_images = stacked_embeddings.size(1)
        
        if num_images <= 1:
            return self.aggregate_with_attention(stacked_embeddings)
        
        # Use each image to attend to others
        cross_outputs = []
        for i in range(num_images):
            query = stacked_embeddings[:, i:i+1, :]  # [batch, 1, embedding_dim]
            key_value = stacked_embeddings  # [batch, num_images, embedding_dim]
            cross_out, _ = self.cross_attention(
                query=query,
                key=key_value,
                value=key_value,
            )
            cross_outputs.append(cross_out.squeeze(1))  # [batch, embedding_dim]
        
        # Average cross-attention outputs
        cross_aggregated = torch.stack(cross_outputs, dim=0).mean(dim=0)  # [batch, embedding_dim]
        
        return cross_aggregated
    
    def fuse_aggregations(
        self,
        mean_aggregated: torch.Tensor,
        max_aggregated: torch.Tensor,
        attn_aggregated: torch.Tensor,
        cross_aggregated: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Fuse multiple aggregation methods with learned weights.
        
        Args:
            mean_aggregated: Mean aggregation [batch, embedding_dim]
            max_aggregated: Max aggregation [batch, embedding_dim]
            attn_aggregated: Attention aggregation [batch, embedding_dim]
            cross_aggregated: Optional cross-attention aggregation [batch, embedding_dim]
            
        Returns:
            Fused embedding [batch, embedding_dim]
        """
        # Weighted fusion of statistical methods
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = (
            weights[0] * mean_aggregated +
            weights[1] * max_aggregated +
            weights[2] * attn_aggregated
        )
        
        # Blend in cross-attention if available
        if cross_aggregated is not None:
            fused = 0.7 * fused + 0.3 * cross_aggregated
        
        return fused

