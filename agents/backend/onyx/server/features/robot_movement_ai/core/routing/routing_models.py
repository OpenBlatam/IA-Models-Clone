"""
Routing Deep Learning Models
============================

Módulo modular con todos los modelos de deep learning para routing.
Separado del código principal para mejor organización y mantenibilidad.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Deep learning models will be disabled.")


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer for routing."""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GraphAttentionLayer")
            
        self.num_heads = num_heads
        self.out_features = out_features
        self.head_dim = out_features // num_heads
        
        if out_features % num_heads != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by num_heads ({num_heads})")
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.randn(2 * self.head_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos usando Xavier/Glorot uniform."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
        Returns:
            Output features [N, out_features]
        """
        N = h.size(0)
        h = self.W(h)  # [N, out_features]
        h = h.view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        
        # Compute attention scores
        a_input = torch.cat([
            h.repeat(1, 1, N).view(N, self.num_heads, N, self.head_dim),
            h.repeat(N, 1, 1).view(N, self.num_heads, N, self.head_dim)
        ], dim=-1)  # [N, num_heads, N, 2*head_dim]
        
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))  # [N, num_heads, N]
        e = e.masked_fill(adj == 0, float('-inf'))
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)
        
        h_prime = torch.matmul(attention, h)  # [N, num_heads, head_dim]
        h_prime = h_prime.view(N, self.out_features)
        
        return h_prime


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for route optimization."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 3):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GraphNeuralNetwork")
            
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(GraphAttentionLayer(input_dim, output_dim))
        else:
            self.layers.append(GraphAttentionLayer(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GraphAttentionLayer(hidden_dim, hidden_dim))
            self.layers.append(GraphAttentionLayer(hidden_dim, output_dim))
        
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [N, input_dim]
            adjacency: [N, N]
        Returns:
            Node embeddings [N, output_dim]
        """
        h = node_features
        for layer in self.layers:
            h = layer(h, adjacency)
            h = F.relu(h)
        h = self.norm(h)
        return h


class RouteTransformer(nn.Module):
    """Transformer-based route prediction model."""
    
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 512):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RouteTransformer")
            
        self.d_model = d_model
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_projection = nn.Linear(10, d_model)
        self.output_projection = nn.Linear(d_model, 1)
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos del transformer."""
        # Inicialización estándar para transformers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch, seq_len, features]
            mask: Attention mask [batch, seq_len]
        Returns:
            Route scores [batch, seq_len]
        """
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        x = self.transformer(x, src_key_padding_mask=mask)
        scores = self.output_projection(x).squeeze(-1)  # [batch, seq_len]
        return scores


class DeepRouteOptimizer(nn.Module):
    """Deep learning model for route optimization."""
    
    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = None, dropout: float = 0.2):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DeepRouteOptimizer")
            
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos usando Kaiming/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict route quality score."""
        return self.network(x)


class RouteQualityPredictor(nn.Module):
    """Predictor de calidad de rutas usando deep learning."""
    
    def __init__(self, input_dim: int = 20, hidden_dims: List[int] = None, num_outputs: int = 3):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RouteQualityPredictor")
            
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_outputs))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict route quality metrics.
        
        Args:
            x: Route features [batch, input_dim]
        Returns:
            Predictions [batch, num_outputs] (distance, time, cost)
        """
        return self.network(x)


class MultiHeadRouteAttention(nn.Module):
    """Multi-head attention mechanism for route analysis."""
    
    def __init__(self, d_model: int = 128, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MultiHeadRouteAttention")
            
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            mask: [batch, seq_len, seq_len]
        Returns:
            [batch, seq_len, d_model]
        """
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        return output


def create_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create routing models.
    
    Args:
        model_type: Type of model ('gnn', 'transformer', 'deep_optimizer', 'quality_predictor', 'attention')
        **kwargs: Model-specific parameters
    
    Returns:
        Initialized model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to create models")
    
    model_type = model_type.lower()
    
    if model_type == 'gnn':
        return GraphNeuralNetwork(**kwargs)
    elif model_type == 'transformer':
        return RouteTransformer(**kwargs)
    elif model_type == 'deep_optimizer':
        return DeepRouteOptimizer(**kwargs)
    elif model_type == 'quality_predictor':
        return RouteQualityPredictor(**kwargs)
    elif model_type == 'attention':
        return MultiHeadRouteAttention(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get information about a model."""
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": next(model.parameters()).device if list(model.parameters()) else "cpu"
    }

