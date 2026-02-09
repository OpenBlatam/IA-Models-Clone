"""
GNN Utils - Graph Neural Networks Utilities
============================================

Utilidades para Graph Neural Networks.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Intentar importar torch_geometric
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
    _has_torch_geometric = True
except ImportError:
    _has_torch_geometric = False
    logger.warning("torch_geometric not available, GNN features will be limited")


class GCNLayer(nn.Module):
    """
    Capa Graph Convolutional Network.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_torch_geometric: bool = True
    ):
        """
        Inicializar capa GCN.
        
        Args:
            in_channels: Canales de entrada
            out_channels: Canales de salida
            use_torch_geometric: Usar torch_geometric
        """
        super().__init__()
        
        if use_torch_geometric and _has_torch_geometric:
            self.conv = GCNConv(in_channels, out_channels)
            self.use_torch_geometric = True
        else:
            # Implementación manual simplificada
            self.linear = nn.Linear(in_channels, out_channels)
            self.use_torch_geometric = False
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Features de nodos [num_nodes, in_channels]
            edge_index: Índices de aristas [2, num_edges]
            
        Returns:
            Features actualizadas
        """
        if self.use_torch_geometric:
            return self.conv(x, edge_index)
        else:
            # Implementación simplificada
            return F.relu(self.linear(x))


class GATLayer(nn.Module):
    """
    Capa Graph Attention Network.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        use_torch_geometric: bool = True
    ):
        """
        Inicializar capa GAT.
        
        Args:
            in_channels: Canales de entrada
            out_channels: Canales de salida
            heads: Número de heads de atención
            use_torch_geometric: Usar torch_geometric
        """
        super().__init__()
        
        if use_torch_geometric and _has_torch_geometric:
            self.conv = GATConv(in_channels, out_channels, heads=heads)
            self.use_torch_geometric = True
        else:
            self.linear = nn.Linear(in_channels, out_channels)
            self.use_torch_geometric = False
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Features de nodos
            edge_index: Índices de aristas
            
        Returns:
            Features actualizadas
        """
        if self.use_torch_geometric:
            return self.conv(x, edge_index)
        else:
            return F.relu(self.linear(x))


class GraphSAGELayer(nn.Module):
    """
    Capa GraphSAGE.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_torch_geometric: bool = True
    ):
        """
        Inicializar capa GraphSAGE.
        
        Args:
            in_channels: Canales de entrada
            out_channels: Canales de salida
            use_torch_geometric: Usar torch_geometric
        """
        super().__init__()
        
        if use_torch_geometric and _has_torch_geometric:
            self.conv = SAGEConv(in_channels, out_channels)
            self.use_torch_geometric = True
        else:
            self.linear = nn.Linear(in_channels, out_channels)
            self.use_torch_geometric = False
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Features de nodos
            edge_index: Índices de aristas
            
        Returns:
            Features actualizadas
        """
        if self.use_torch_geometric:
            return self.conv(x, edge_index)
        else:
            return F.relu(self.linear(x))


class GraphNeuralNetwork(nn.Module):
    """
    Red neuronal de grafos completa.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        gnn_type: str = "gcn"
    ):
        """
        Inicializar GNN.
        
        Args:
            input_dim: Dimensión de entrada
            hidden_dim: Dimensión oculta
            output_dim: Dimensión de salida
            num_layers: Número de capas
            gnn_type: Tipo de GNN ('gcn', 'gat', 'sage')
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Primera capa
        if gnn_type == "gcn":
            self.layers.append(GCNLayer(input_dim, hidden_dim))
        elif gnn_type == "gat":
            self.layers.append(GATLayer(input_dim, hidden_dim))
        elif gnn_type == "sage":
            self.layers.append(GraphSAGELayer(input_dim, hidden_dim))
        
        # Capas intermedias
        for _ in range(num_layers - 2):
            if gnn_type == "gcn":
                self.layers.append(GCNLayer(hidden_dim, hidden_dim))
            elif gnn_type == "gat":
                self.layers.append(GATLayer(hidden_dim, hidden_dim))
            elif gnn_type == "sage":
                self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
        
        # Capa de salida
        if num_layers > 1:
            if gnn_type == "gcn":
                self.layers.append(GCNLayer(hidden_dim, output_dim))
            elif gnn_type == "gat":
                self.layers.append(GATLayer(hidden_dim, output_dim))
            elif gnn_type == "sage":
                self.layers.append(GraphSAGELayer(hidden_dim, output_dim))
        
        self.fc = nn.Linear(output_dim, output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Features de nodos
            edge_index: Índices de aristas
            batch: Batch assignment (opcional)
            
        Returns:
            Output
        """
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        
        x = self.layers[-1](x, edge_index)
        
        # Pooling global si hay batch
        if batch is not None and _has_torch_geometric:
            x = global_mean_pool(x, batch)
        
        x = self.fc(x)
        return x




