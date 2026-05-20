"""
Graph Neural Networks Module
Advanced graph neural network architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class GNNType(Enum):
    """Graph neural network types"""
    GCN = "gcn"  # Graph Convolutional Network
    GAT = "gat"  # Graph Attention Network
    GIN = "gin"  # Graph Isomorphism Network
    SAGE = "sage"  # GraphSAGE
    GNN = "gnn"  # Generic GNN
    TRANSFORMER = "transformer"  # Graph Transformer

@dataclass
class GNNConfig:
    """GNN configuration"""
    gnn_type: GNNType = GNNType.GCN
    input_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 32
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "relu"
    use_batch_norm: bool = True
    use_residual: bool = True
    use_attention: bool = True
    use_edge_features: bool = False
    use_positional_encoding: bool = False

class GraphConvolutionalLayer(nn.Module):
    """Graph Convolutional Layer"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Graph convolution: X' = AXW
        x = torch.matmul(adj, x)  # A * X
        x = self.linear(x)  # W
        x = self.activation(x)
        x = self.dropout_layer(x)
        return x

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer"""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.dropout = dropout
        
        self.w_q = nn.Linear(input_dim, output_dim)
        self.w_k = nn.Linear(input_dim, output_dim)
        self.w_v = nn.Linear(input_dim, output_dim)
        self.w_o = nn.Linear(output_dim, output_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size, num_nodes, _ = x.size()
        
        # Linear projections
        q = self.w_q(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.w_k(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        v = self.w_v(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, heads, nodes, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply adjacency mask
        scores = scores.masked_fill(adj.unsqueeze(1) == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, v)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.output_dim)
        
        # Output projection
        out = self.w_o(out)
        
        return out

class GraphIsomorphismLayer(nn.Module):
    """Graph Isomorphism Network Layer"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Aggregate neighbors
        neighbor_features = torch.matmul(adj, x)
        
        # Apply MLP
        out = self.mlp(neighbor_features)
        out = self.activation(out)
        out = self.dropout_layer(out)
        
        return out

class GraphSAGELayer(nn.Module):
    """GraphSAGE Layer"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.linear = nn.Linear(input_dim * 2, output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Aggregate neighbors
        neighbor_features = torch.matmul(adj, x)
        
        # Concatenate with self features
        combined = torch.cat([x, neighbor_features], dim=-1)
        
        # Apply linear transformation
        out = self.linear(combined)
        out = self.activation(out)
        out = self.dropout_layer(out)
        
        return out

class GraphTransformerLayer(nn.Module):
    """Graph Transformer Layer"""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.dropout = dropout
        
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout_layer(attn_out))
        
        # Feed forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout_layer(ffn_out))
        
        return x

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network"""
    
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Create layers
        for i in range(config.num_layers):
            if i == 0:
                input_dim = config.input_dim
            else:
                input_dim = config.hidden_dim
            
            if i == config.num_layers - 1:
                output_dim = config.output_dim
            else:
                output_dim = config.hidden_dim
            
            # Create layer based on type
            if config.gnn_type == GNNType.GCN:
                layer = GraphConvolutionalLayer(input_dim, output_dim, config.dropout)
            elif config.gnn_type == GNNType.GAT:
                layer = GraphAttentionLayer(input_dim, output_dim, config.num_heads, config.dropout)
            elif config.gnn_type == GNNType.GIN:
                layer = GraphIsomorphismLayer(input_dim, output_dim, config.dropout)
            elif config.gnn_type == GNNType.SAGE:
                layer = GraphSAGELayer(input_dim, output_dim, config.dropout)
            elif config.gnn_type == GNNType.TRANSFORMER:
                layer = GraphTransformerLayer(input_dim, output_dim, config.num_heads, config.dropout)
            else:
                layer = GraphConvolutionalLayer(input_dim, output_dim, config.dropout)
            
            self.layers.append(layer)
            
            # Add batch normalization
            if config.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(output_dim))
            else:
                self.batch_norms.append(nn.Identity())
        
        # Activation function
        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for i, (layer, batch_norm) in enumerate(zip(self.layers, self.batch_norms)):
            # Store residual connection
            residual = x if self.config.use_residual and i > 0 else None
            
            # Forward pass through layer
            x = layer(x, adj)
            
            # Apply batch normalization
            x = batch_norm(x)
            
            # Apply activation
            x = self.activation(x)
            
            # Apply dropout
            x = self.dropout(x)
            
            # Add residual connection
            if residual is not None and x.size() == residual.size():
                x = x + residual
        
        return x

class GraphPooling(nn.Module):
    """Graph pooling operations"""
    
    def __init__(self, pooling_type: str = "mean"):
        super().__init__()
        self.pooling_type = pooling_type
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        if self.pooling_type == "mean":
            if mask is not None:
                x = x * mask.unsqueeze(-1)
                return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            else:
                return x.mean(dim=1)
        elif self.pooling_type == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
            return x.max(dim=1)[0]
        elif self.pooling_type == "sum":
            if mask is not None:
                x = x * mask.unsqueeze(-1)
            return x.sum(dim=1)
        else:
            return x.mean(dim=1)

class GraphClassifier(nn.Module):
    """Graph classification model"""
    
    def __init__(self, config: GNNConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # GNN backbone
        self.gnn = GraphNeuralNetwork(config)
        
        # Pooling
        self.pooling = GraphPooling("mean")
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        # GNN forward pass
        x = self.gnn(x, adj)
        
        # Pooling
        x = self.pooling(x, mask)
        
        # Classification
        x = self.classifier(x)
        
        return x

# Factory functions
def create_gnn(config: GNNConfig) -> GraphNeuralNetwork:
    """Create graph neural network"""
    return GraphNeuralNetwork(config)

def create_graph_classifier(config: GNNConfig, num_classes: int) -> GraphClassifier:
    """Create graph classifier"""
    return GraphClassifier(config, num_classes)

def create_gnn_config(**kwargs) -> GNNConfig:
    """Create GNN configuration"""
    return GNNConfig(**kwargs)

# Example usage
if __name__ == "__main__":
    # Create GNN configuration
    config = create_gnn_config(
        gnn_type=GNNType.GAT,
        input_dim=64,
        hidden_dim=128,
        output_dim=32,
        num_layers=3,
        num_heads=8
    )
    
    # Create GNN
    gnn = create_gnn(config)
    
    # Create graph classifier
    classifier = create_graph_classifier(config, num_classes=10)
    
    # Example forward pass
    x = torch.randn(2, 10, 64)  # [batch, nodes, features]
    adj = torch.randn(2, 10, 10)  # [batch, nodes, nodes]
    adj = (adj > 0.5).float()  # Binary adjacency matrix
    
    output = gnn(x, adj)
    print(f"GNN output shape: {output.shape}")
    
    # Classification
    class_output = classifier(x, adj)
    print(f"Classification output shape: {class_output.shape}")


