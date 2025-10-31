#!/usr/bin/env python3
"""
Advanced Graph Neural Networks System for Frontier Model Training
Provides comprehensive GNN algorithms, graph processing, and network analysis capabilities.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import networkx as nx
import torch_geometric
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_networkx, from_networkx
import dgl
import dgl.nn as dglnn
from dgl.nn import GraphConv as DGLGraphConv, GATConv as DGLGATConv
import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.layer import GCN, GAT, GraphSAGE
import joblib
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

console = Console()

class GNNTask(Enum):
    """Graph neural network tasks."""
    NODE_CLASSIFICATION = "node_classification"
    LINK_PREDICTION = "link_prediction"
    GRAPH_CLASSIFICATION = "graph_classification"
    GRAPH_REGRESSION = "graph_regression"
    COMMUNITY_DETECTION = "community_detection"
    GRAPH_GENERATION = "graph_generation"
    GRAPH_MATCHING = "graph_matching"
    GRAPH_ISOMORPHISM = "graph_isomorphism"
    SUBGRAPH_MATCHING = "subgraph_matching"
    GRAPH_CLUSTERING = "graph_clustering"
    GRAPH_PARTITIONING = "graph_partitioning"
    CENTRALITY_ANALYSIS = "centrality_analysis"
    INFLUENCE_MAXIMIZATION = "influence_maximization"
    GRAPH_EMBEDDING = "graph_embedding"
    NETWORK_ANALYSIS = "network_analysis"

class GNNModel(Enum):
    """Graph neural network models."""
    # Basic GNN models
    GCN = "gcn"  # Graph Convolutional Network
    GAT = "gat"  # Graph Attention Network
    GRAPH_SAGE = "graph_sage"  # Graph Sample and Aggregate
    GIN = "gin"  # Graph Isomorphism Network
    GCN2 = "gcn2"  # GCN with residual connections
    APPNP = "appnp"  # Approximate Personalized PageRank
    
    # Advanced GNN models
    DIFFPOOL = "diffpool"  # Differentiable Pooling
    SAGPOOL = "sagpool"  # Self-Attention Graph Pooling
    TOPKPOOL = "topkpool"  # Top-K Pooling
    EDGE_CONV = "edge_conv"  # Edge Convolution
    POINT_CONV = "point_conv"  # Point Convolution
    
    # Specialized models
    RGCN = "rgcn"  # Relational Graph Convolutional Network
    HGCN = "hgcn"  # Hypergraph Convolutional Network
    HGT = "hgt"  # Heterogeneous Graph Transformer
    MAGNN = "magnn"  # Multi-attribute Graph Neural Network
    HAN = "han"  # Heterogeneous Attention Network
    
    # Graph generation models
    VAE = "vae"  # Variational Autoencoder for Graphs
    GAN = "gan"  # Generative Adversarial Network for Graphs
    FLOW = "flow"  # Normalizing Flow for Graphs

class GraphType(Enum):
    """Graph types."""
    UNDIRECTED = "undirected"
    DIRECTED = "directed"
    WEIGHTED = "weighted"
    MULTIGRAPH = "multigraph"
    BIPARTITE = "bipartite"
    HETEROGENEOUS = "heterogeneous"
    HYPERGRAPH = "hypergraph"
    TEMPORAL = "temporal"
    DYNAMIC = "dynamic"

class PoolingMethod(Enum):
    """Graph pooling methods."""
    MEAN_POOLING = "mean_pooling"
    MAX_POOLING = "max_pooling"
    SUM_POOLING = "sum_pooling"
    ATTENTION_POOLING = "attention_pooling"
    SET2SET = "set2set"
    SORT_POOLING = "sort_pooling"
    DIFFPOOL = "diffpool"
    SAGPOOL = "sagpool"

class ActivationFunction(Enum):
    """Activation functions."""
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    GELU = "gelu"
    SWISH = "swish"
    TANH = "tanh"
    SIGMOID = "sigmoid"

@dataclass
class GNNConfig:
    """Graph neural network configuration."""
    task: GNNTask = GNNTask.NODE_CLASSIFICATION
    model: GNNModel = GNNModel.GCN
    graph_type: GraphType = GraphType.UNDIRECTED
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    learning_rate: float = 0.01
    num_epochs: int = 200
    batch_size: int = 1
    pooling_method: PoolingMethod = PoolingMethod.MEAN_POOLING
    activation: ActivationFunction = ActivationFunction.RELU
    enable_batch_norm: bool = True
    enable_residual_connections: bool = True
    enable_attention: bool = True
    enable_edge_attributes: bool = True
    enable_node_attributes: bool = True
    enable_graph_attributes: bool = True
    enable_visualization: bool = True
    enable_analysis: bool = True
    device: str = "auto"

@dataclass
class GraphData:
    """Graph data container."""
    graph_id: str
    nodes: List[int]
    edges: List[Tuple[int, int]]
    node_features: Optional[np.ndarray] = None
    edge_features: Optional[np.ndarray] = None
    node_labels: Optional[List[int]] = None
    edge_labels: Optional[List[int]] = None
    graph_label: Optional[int] = None
    metadata: Dict[str, Any] = None

@dataclass
class GNNModelResult:
    """GNN model result."""
    result_id: str
    task: GNNTask
    model: GNNModel
    performance_metrics: Dict[str, float]
    node_embeddings: Optional[np.ndarray] = None
    graph_embeddings: Optional[np.ndarray] = None
    predictions: Optional[List[int]] = None
    model_state: Dict[str, Any] = None
    created_at: datetime = None

class GraphProcessor:
    """Graph data processor."""
    
    def __init__(self, config: GNNConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_graph(self, graph_data: GraphData) -> Data:
        """Process graph data for PyTorch Geometric."""
        console.print("[blue]Processing graph data...[/blue]")
        
        # Convert to PyTorch Geometric format
        edge_index = torch.tensor(graph_data.edges, dtype=torch.long).t().contiguous()
        
        # Node features
        if graph_data.node_features is not None:
            x = torch.tensor(graph_data.node_features, dtype=torch.float)
        else:
            # Create random node features if not provided
            x = torch.randn(len(graph_data.nodes), self.config.hidden_dim)
        
        # Edge features
        edge_attr = None
        if graph_data.edge_features is not None:
            edge_attr = torch.tensor(graph_data.edge_features, dtype=torch.float)
        
        # Node labels
        y = None
        if graph_data.node_labels is not None:
            y = torch.tensor(graph_data.node_labels, dtype=torch.long)
        elif graph_data.graph_label is not None:
            y = torch.tensor([graph_data.graph_label], dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(graph_data.nodes)
        )
        
        console.print("[green]Graph processing completed[/green]")
        return data
    
    def create_sample_graph(self, num_nodes: int = 100, num_edges: int = 200) -> GraphData:
        """Create a sample graph for testing."""
        # Generate random graph
        G = nx.erdos_renyi_graph(num_nodes, num_edges / (num_nodes * (num_nodes - 1) / 2))
        
        # Extract nodes and edges
        nodes = list(G.nodes())
        edges = list(G.edges())
        
        # Generate random node features
        node_features = np.random.randn(num_nodes, self.config.hidden_dim)
        
        # Generate random node labels
        node_labels = np.random.randint(0, 2, num_nodes)
        
        return GraphData(
            graph_id=f"sample_graph_{int(time.time())}",
            nodes=nodes,
            edges=edges,
            node_features=node_features,
            node_labels=node_labels.tolist(),
            metadata={'num_nodes': num_nodes, 'num_edges': len(edges)}
        )

class GNNModelFactory:
    """GNN model factory."""
    
    def __init__(self, config: GNNConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def create_model(self, data: Data) -> nn.Module:
        """Create GNN model."""
        console.print(f"[blue]Creating {self.config.model.value} model for {self.config.task.value}...[/blue]")
        
        try:
            if self.config.model == GNNModel.GCN:
                return self._create_gcn_model(data)
            elif self.config.model == GNNModel.GAT:
                return self._create_gat_model(data)
            elif self.config.model == GNNModel.GRAPH_SAGE:
                return self._create_graphsage_model(data)
            elif self.config.model == GNNModel.GIN:
                return self._create_gin_model(data)
            else:
                return self._create_gcn_model(data)
                
        except Exception as e:
            self.logger.error(f"Model creation failed: {e}")
            return self._create_fallback_model(data)
    
    def _create_gcn_model(self, data: Data) -> nn.Module:
        """Create GCN model."""
        class GCNModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
                super().__init__()
                self.convs = nn.ModuleList()
                self.convs.append(GCNConv(input_dim, hidden_dim))
                
                for _ in range(num_layers - 2):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                
                self.convs.append(GCNConv(hidden_dim, output_dim))
                self.dropout = dropout
            
            def forward(self, x, edge_index, batch=None):
                for i, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if i < len(self.convs) - 1:
                        x = F.relu(x)
                        x = F.dropout(x, p=self.dropout, training=self.training)
                
                if batch is not None:
                    x = global_mean_pool(x, batch)
                
                return x
        
        input_dim = data.x.size(1)
        output_dim = 2 if self.config.task == GNNTask.NODE_CLASSIFICATION else 1
        
        model = GCNModel(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=output_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )
        
        return model.to(self.device)
    
    def _create_gat_model(self, data: Data) -> nn.Module:
        """Create GAT model."""
        class GATModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, heads=8):
                super().__init__()
                self.convs = nn.ModuleList()
                self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
                
                for _ in range(num_layers - 2):
                    self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
                
                self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout))
                self.dropout = dropout
            
            def forward(self, x, edge_index, batch=None):
                for i, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if i < len(self.convs) - 1:
                        x = F.relu(x)
                        x = F.dropout(x, p=self.dropout, training=self.training)
                
                if batch is not None:
                    x = global_mean_pool(x, batch)
                
                return x
        
        input_dim = data.x.size(1)
        output_dim = 2 if self.config.task == GNNTask.NODE_CLASSIFICATION else 1
        
        model = GATModel(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=output_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )
        
        return model.to(self.device)
    
    def _create_graphsage_model(self, data: Data) -> nn.Module:
        """Create GraphSAGE model."""
        class GraphSAGEModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
                super().__init__()
                self.convs = nn.ModuleList()
                self.convs.append(SAGEConv(input_dim, hidden_dim))
                
                for _ in range(num_layers - 2):
                    self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                
                self.convs.append(SAGEConv(hidden_dim, output_dim))
                self.dropout = dropout
            
            def forward(self, x, edge_index, batch=None):
                for i, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if i < len(self.convs) - 1:
                        x = F.relu(x)
                        x = F.dropout(x, p=self.dropout, training=self.training)
                
                if batch is not None:
                    x = global_mean_pool(x, batch)
                
                return x
        
        input_dim = data.x.size(1)
        output_dim = 2 if self.config.task == GNNTask.NODE_CLASSIFICATION else 1
        
        model = GraphSAGEModel(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=output_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )
        
        return model.to(self.device)
    
    def _create_gin_model(self, data: Data) -> nn.Module:
        """Create GIN model."""
        class GINModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
                super().__init__()
                self.convs = nn.ModuleList()
                self.convs.append(GINConv(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )))
                
                for _ in range(num_layers - 2):
                    self.convs.append(GINConv(nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )))
                
                self.convs.append(GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )))
                self.dropout = dropout
            
            def forward(self, x, edge_index, batch=None):
                for i, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if i < len(self.convs) - 1:
                        x = F.relu(x)
                        x = F.dropout(x, p=self.dropout, training=self.training)
                
                if batch is not None:
                    x = global_mean_pool(x, batch)
                
                return x
        
        input_dim = data.x.size(1)
        output_dim = 2 if self.config.task == GNNTask.NODE_CLASSIFICATION else 1
        
        model = GINModel(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=output_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )
        
        return model.to(self.device)
    
    def _create_fallback_model(self, data: Data) -> nn.Module:
        """Create fallback model."""
        return self._create_gcn_model(data)

class GNNTrainer:
    """GNN training engine."""
    
    def __init__(self, config: GNNConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def train_model(self, model: nn.Module, data: Data) -> Dict[str, Any]:
        """Train GNN model."""
        console.print(f"[blue]Training {self.config.model.value} model...[/blue]")
        
        # Move data to device
        data = data.to(self.device)
        
        # Split data for training and validation
        train_mask, val_mask = self._create_masks(data)
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        if self.config.task == GNNTask.NODE_CLASSIFICATION:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # Training loop
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(self.config.num_epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            
            if self.config.task == GNNTask.NODE_CLASSIFICATION:
                loss = criterion(out[train_mask], data.y[train_mask])
                pred = out[train_mask].argmax(dim=1)
                acc = (pred == data.y[train_mask]).float().mean()
            else:
                loss = criterion(out[train_mask], data.y[train_mask].float())
                acc = torch.tensor(0.0)  # No accuracy for regression
            
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                
                if self.config.task == GNNTask.NODE_CLASSIFICATION:
                    val_loss = criterion(val_out[val_mask], data.y[val_mask])
                    val_pred = val_out[val_mask].argmax(dim=1)
                    val_acc = (val_pred == data.y[val_mask]).float().mean()
                else:
                    val_loss = criterion(val_out[val_mask], data.y[val_mask].float())
                    val_acc = torch.tensor(0.0)
            
            # Update history
            training_history['train_loss'].append(loss.item())
            training_history['val_loss'].append(val_loss.item())
            training_history['train_acc'].append(acc.item())
            training_history['val_acc'].append(val_acc.item())
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # Log progress
            if epoch % 50 == 0:
                console.print(f"[blue]Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Acc = {val_acc.item():.4f}[/blue]")
        
        return {
            'training_history': training_history,
            'best_val_acc': best_val_acc.item(),
            'final_train_acc': training_history['train_acc'][-1],
            'final_val_acc': training_history['val_acc'][-1]
        }
    
    def _create_masks(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create training and validation masks."""
        num_nodes = data.num_nodes
        train_size = int(0.8 * num_nodes)
        
        # Random permutation
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[perm[:train_size]] = True
        val_mask[perm[train_size:]] = True
        
        return train_mask, val_mask

class GNNSystem:
    """Main GNN system."""
    
    def __init__(self, config: GNNConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.processor = GraphProcessor(config)
        self.model_factory = GNNModelFactory(config)
        self.trainer = GNNTrainer(config)
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Results storage
        self.gnn_results: Dict[str, GNNModelResult] = {}
    
    def _init_database(self) -> str:
        """Initialize GNN database."""
        db_path = Path("./graph_neural_networks.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gnn_models (
                    model_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    node_embeddings TEXT,
                    graph_embeddings TEXT,
                    predictions TEXT,
                    model_state TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
        
        return str(db_path)
    
    def run_gnn_experiment(self, graph_data: GraphData) -> GNNModelResult:
        """Run complete GNN experiment."""
        console.print(f"[blue]Starting GNN experiment with {self.config.task.value}...[/blue]")
        
        start_time = time.time()
        result_id = f"gnn_exp_{int(time.time())}"
        
        # Process graph data
        data = self.processor.process_graph(graph_data)
        
        # Create model
        model = self.model_factory.create_model(data)
        
        # Train model
        training_result = self.trainer.train_model(model, data)
        
        # Generate embeddings and predictions
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index)
            
            if self.config.task == GNNTask.NODE_CLASSIFICATION:
                predictions = embeddings.argmax(dim=1).cpu().numpy().tolist()
            else:
                predictions = embeddings.cpu().numpy().tolist()
        
        # Evaluate model
        performance_metrics = self._evaluate_model(model, data)
        
        # Create GNN result
        gnn_result = GNNModelResult(
            result_id=result_id,
            task=self.config.task,
            model=self.config.model,
            performance_metrics=performance_metrics,
            node_embeddings=embeddings.cpu().numpy(),
            predictions=predictions,
            model_state={
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
                'num_nodes': data.num_nodes,
                'num_edges': data.edge_index.size(1)
            },
            created_at=datetime.now()
        )
        
        # Store result
        self.gnn_results[result_id] = gnn_result
        
        # Save to database
        self._save_gnn_result(gnn_result)
        
        experiment_time = time.time() - start_time
        console.print(f"[green]GNN experiment completed in {experiment_time:.2f} seconds[/green]")
        console.print(f"[blue]Final accuracy: {performance_metrics.get('accuracy', 0):.4f}[/blue]")
        
        return gnn_result
    
    def _evaluate_model(self, model: nn.Module, data: Data) -> Dict[str, float]:
        """Evaluate GNN model performance."""
        model.eval()
        
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            
            if self.config.task == GNNTask.NODE_CLASSIFICATION:
                pred = out.argmax(dim=1)
                accuracy = (pred == data.y).float().mean().item()
                precision = precision_score(data.y.cpu().numpy(), pred.cpu().numpy(), average='weighted')
                recall = recall_score(data.y.cpu().numpy(), pred.cpu().numpy(), average='weighted')
                f1 = f1_score(data.y.cpu().numpy(), pred.cpu().numpy(), average='weighted')
                
                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
            else:
                mse = F.mse_loss(out.squeeze(), data.y.float()).item()
                mae = F.l1_loss(out.squeeze(), data.y.float()).item()
                
                return {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse)
                }
    
    def _save_gnn_result(self, result: GNNModelResult):
        """Save GNN result to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO gnn_models 
                (model_id, task, model_name, performance_metrics,
                 node_embeddings, graph_embeddings, predictions, model_state, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.result_id,
                result.task.value,
                result.model.value,
                json.dumps(result.performance_metrics),
                json.dumps(result.node_embeddings.tolist()) if result.node_embeddings is not None else None,
                json.dumps(result.graph_embeddings.tolist()) if result.graph_embeddings is not None else None,
                json.dumps(result.predictions) if result.predictions else None,
                json.dumps(result.model_state),
                result.created_at.isoformat()
            ))
    
    def visualize_gnn_results(self, result: GNNModelResult, 
                             output_path: str = None) -> str:
        """Visualize GNN results."""
        if output_path is None:
            output_path = f"gnn_analysis_{result.result_id}.png"
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance metrics
        performance_metrics = result.performance_metrics
        metric_names = list(performance_metrics.keys())
        metric_values = list(performance_metrics.values())
        
        axes[0, 0].bar(metric_names, metric_values)
        axes[0, 0].set_title('Performance Metrics')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Node embeddings (2D projection)
        if result.node_embeddings is not None:
            from sklearn.decomposition import PCA
            embeddings_2d = PCA(n_components=2).fit_transform(result.node_embeddings)
            
            if result.predictions is not None:
                scatter = axes[0, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                           c=result.predictions, cmap='viridis', alpha=0.7)
                axes[0, 1].set_title('Node Embeddings (2D Projection)')
                plt.colorbar(scatter, ax=axes[0, 1])
            else:
                axes[0, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
                axes[0, 1].set_title('Node Embeddings (2D Projection)')
        
        # Model information
        model_state = result.model_state
        info_names = list(model_state.keys())
        info_values = list(model_state.values())
        
        axes[1, 0].bar(info_names, info_values)
        axes[1, 0].set_title('Model Information')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Task and model info
        task_model_info = {
            'Task': len(result.task.value),
            'Model': len(result.model.value),
            'Parameters': result.model_state.get('num_parameters', 0) // 1000000,  # In millions
            'Size (MB)': result.model_state.get('model_size_mb', 0)
        }
        
        info_names = list(task_model_info.keys())
        info_values = list(task_model_info.values())
        
        axes[1, 1].bar(info_names, info_values)
        axes[1, 1].set_title('Task and Model Info')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]GNN visualization saved: {output_path}[/green]")
        return output_path
    
    def get_gnn_summary(self) -> Dict[str, Any]:
        """Get GNN system summary."""
        if not self.gnn_results:
            return {'total_experiments': 0}
        
        total_experiments = len(self.gnn_results)
        
        # Calculate average metrics
        accuracies = [result.performance_metrics.get('accuracy', 0) for result in self.gnn_results.values()]
        f1_scores = [result.performance_metrics.get('f1_score', 0) for result in self.gnn_results.values()]
        
        avg_accuracy = np.mean(accuracies) if accuracies else 0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        
        # Best performing experiment
        best_result = max(self.gnn_results.values(), 
                         key=lambda x: x.performance_metrics.get('accuracy', 0))
        
        return {
            'total_experiments': total_experiments,
            'average_accuracy': avg_accuracy,
            'average_f1_score': avg_f1,
            'best_accuracy': best_result.performance_metrics.get('accuracy', 0),
            'best_experiment_id': best_result.result_id,
            'tasks_used': list(set(result.task.value for result in self.gnn_results.values())),
            'models_used': list(set(result.model.value for result in self.gnn_results.values()))
        }

def main():
    """Main function for GNN CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Graph Neural Networks System")
    parser.add_argument("--task", type=str,
                       choices=["node_classification", "link_prediction", "graph_classification"],
                       default="node_classification", help="GNN task")
    parser.add_argument("--model", type=str,
                       choices=["gcn", "gat", "graph_sage", "gin"],
                       default="gcn", help="GNN model")
    parser.add_argument("--hidden-dim", type=int, default=64,
                       help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.5,
                       help="Dropout rate")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=200,
                       help="Number of epochs")
    parser.add_argument("--num-nodes", type=int, default=100,
                       help="Number of nodes in sample graph")
    parser.add_argument("--num-edges", type=int, default=200,
                       help="Number of edges in sample graph")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create GNN configuration
    config = GNNConfig(
        task=GNNTask(args.task),
        model=GNNModel(args.model),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device
    )
    
    # Create GNN system
    gnn_system = GNNSystem(config)
    
    # Create sample graph data
    processor = GraphProcessor(config)
    sample_graph = processor.create_sample_graph(args.num_nodes, args.num_edges)
    
    # Run GNN experiment
    result = gnn_system.run_gnn_experiment(sample_graph)
    
    # Show results
    console.print(f"[green]GNN experiment completed[/green]")
    console.print(f"[blue]Task: {result.task.value}[/blue]")
    console.print(f"[blue]Model: {result.model.value}[/blue]")
    console.print(f"[blue]Final accuracy: {result.performance_metrics.get('accuracy', 0):.4f}[/blue]")
    
    # Create visualization
    gnn_system.visualize_gnn_results(result)
    
    # Show summary
    summary = gnn_system.get_gnn_summary()
    console.print(f"[blue]Summary: {summary}[/blue]")

if __name__ == "__main__":
    main()
