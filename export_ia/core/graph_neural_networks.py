"""
Graph Neural Networks Engine for Export IA
Advanced GNN architectures with GCN, GAT, GraphSAGE, and graph-based document processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time
import json
import random
from pathlib import Path
from collections import defaultdict, deque
import copy
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GraphConv, ChebConv, ARMAConv, SGConv
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.transforms import NormalizeFeatures, AddSelfLoops
import dgl
import dgl.nn as dglnn
from dgl.nn import GraphConv as DGLGraphConv, GATConv as DGLGATConv
from dgl.nn import SAGEConv as DGLSAGEConv, GINConv as DGLGINConv
import torch_sparse
from torch_sparse import SparseTensor
import torch_scatter
from torch_scatter import scatter_mean, scatter_max, scatter_add

logger = logging.getLogger(__name__)

@dataclass
class GNNConfig:
    """Configuration for Graph Neural Networks"""
    # Model types
    model_type: str = "gcn"  # gcn, gat, sage, gin, transformer, custom
    
    # Graph construction
    graph_type: str = "document"  # document, citation, social, knowledge, custom
    node_features: str = "text"  # text, numerical, categorical, mixed
    edge_features: str = "similarity"  # similarity, distance, categorical, none
    
    # GCN parameters
    gcn_hidden_dim: int = 64
    gcn_num_layers: int = 2
    gcn_dropout: float = 0.2
    gcn_activation: str = "relu"  # relu, tanh, sigmoid, leaky_relu
    
    # GAT parameters
    gat_hidden_dim: int = 64
    gat_num_heads: int = 8
    gat_num_layers: int = 2
    gat_dropout: float = 0.2
    gat_alpha: float = 0.2
    
    # GraphSAGE parameters
    sage_hidden_dim: int = 64
    sage_num_layers: int = 2
    sage_aggregator: str = "mean"  # mean, max, lstm, gcn
    sage_dropout: float = 0.2
    
    # GIN parameters
    gin_hidden_dim: int = 64
    gin_num_layers: int = 2
    gin_eps: float = 0.0
    gin_dropout: float = 0.2
    
    # Transformer parameters
    transformer_hidden_dim: int = 64
    transformer_num_heads: int = 8
    transformer_num_layers: int = 2
    transformer_dropout: float = 0.1
    
    # Graph construction parameters
    similarity_threshold: float = 0.1
    max_neighbors: int = 10
    edge_weight_method: str = "cosine"  # cosine, euclidean, jaccard, custom
    
    # Node features
    node_feature_dim: int = 128
    text_embedding_model: str = "bert"  # bert, word2vec, glove, custom
    feature_normalization: bool = True
    
    # Edge features
    edge_feature_dim: int = 32
    edge_weight_normalization: bool = True
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Task parameters
    task_type: str = "node_classification"  # node_classification, graph_classification, link_prediction, node_regression
    num_classes: int = 2
    output_dim: int = 1
    
    # Evaluation parameters
    evaluation_metrics: List[str] = None  # accuracy, f1, auc, precision, recall
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Performance parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    enable_caching: bool = True

class GraphConstructor:
    """Construct graphs from various data sources"""
    
    def __init__(self, config: GNNConfig):
        self.config = config
        
    def construct_document_graph(self, documents: List[str], 
                                document_features: np.ndarray = None) -> Data:
        """Construct graph from documents"""
        
        num_docs = len(documents)
        
        # Create node features
        if document_features is not None:
            node_features = torch.FloatTensor(document_features)
        else:
            # Use document embeddings
            node_features = self._create_document_embeddings(documents)
            
        # Create adjacency matrix
        adjacency_matrix = self._create_document_adjacency(documents)
        
        # Create edge indices and weights
        edge_indices, edge_weights = self._matrix_to_edge_list(adjacency_matrix)
        
        # Create edge features
        edge_features = self._create_edge_features(edge_indices, edge_weights)
        
        # Create graph data
        graph_data = Data(
            x=node_features,
            edge_index=edge_indices,
            edge_attr=edge_features,
            edge_weight=edge_weights
        )
        
        return graph_data
        
    def construct_citation_graph(self, papers: List[Dict], 
                               paper_features: np.ndarray = None) -> Data:
        """Construct citation graph"""
        
        num_papers = len(papers)
        
        # Create node features
        if paper_features is not None:
            node_features = torch.FloatTensor(paper_features)
        else:
            # Use paper embeddings
            paper_texts = [paper.get('title', '') + ' ' + paper.get('abstract', '') 
                          for paper in papers]
            node_features = self._create_document_embeddings(paper_texts)
            
        # Create citation edges
        edge_indices = []
        edge_weights = []
        
        for i, paper in enumerate(papers):
            citations = paper.get('citations', [])
            for citation in citations:
                if citation < num_papers:
                    edge_indices.append([i, citation])
                    edge_weights.append(1.0)
                    
        if edge_indices:
            edge_indices = torch.LongTensor(edge_indices).t().contiguous()
            edge_weights = torch.FloatTensor(edge_weights)
        else:
            edge_indices = torch.LongTensor([[0], [0]])
            edge_weights = torch.FloatTensor([1.0])
            
        # Create edge features
        edge_features = self._create_edge_features(edge_indices, edge_weights)
        
        # Create graph data
        graph_data = Data(
            x=node_features,
            edge_index=edge_indices,
            edge_attr=edge_features,
            edge_weight=edge_weights
        )
        
        return graph_data
        
    def construct_social_graph(self, users: List[Dict], 
                             user_features: np.ndarray = None) -> Data:
        """Construct social network graph"""
        
        num_users = len(users)
        
        # Create node features
        if user_features is not None:
            node_features = torch.FloatTensor(user_features)
        else:
            # Use user embeddings
            user_texts = [user.get('description', '') for user in users]
            node_features = self._create_document_embeddings(user_texts)
            
        # Create social edges
        edge_indices = []
        edge_weights = []
        
        for i, user in enumerate(users):
            friends = user.get('friends', [])
            for friend in friends:
                if friend < num_users:
                    edge_indices.append([i, friend])
                    edge_weights.append(1.0)
                    
        if edge_indices:
            edge_indices = torch.LongTensor(edge_indices).t().contiguous()
            edge_weights = torch.FloatTensor(edge_weights)
        else:
            edge_indices = torch.LongTensor([[0], [0]])
            edge_weights = torch.FloatTensor([1.0])
            
        # Create edge features
        edge_features = self._create_edge_features(edge_indices, edge_weights)
        
        # Create graph data
        graph_data = Data(
            x=node_features,
            edge_index=edge_indices,
            edge_attr=edge_features,
            edge_weight=edge_weights
        )
        
        return graph_data
        
    def _create_document_embeddings(self, documents: List[str]) -> torch.Tensor:
        """Create document embeddings"""
        
        # Simplified document embedding (in practice, use BERT or other models)
        embeddings = []
        
        for doc in documents:
            # Simple bag-of-words embedding
            words = doc.lower().split()
            embedding = np.zeros(self.config.node_feature_dim)
            
            for word in words:
                # Simple hash-based embedding
                word_hash = hash(word) % self.config.node_feature_dim
                embedding[word_hash] += 1
                
            # Normalize
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
                
            embeddings.append(embedding)
            
        return torch.FloatTensor(embeddings)
        
    def _create_document_adjacency(self, documents: List[str]) -> np.ndarray:
        """Create adjacency matrix for documents"""
        
        num_docs = len(documents)
        adjacency_matrix = np.zeros((num_docs, num_docs))
        
        # Calculate document similarities
        for i in range(num_docs):
            for j in range(i + 1, num_docs):
                similarity = self._calculate_document_similarity(documents[i], documents[j])
                
                if similarity > self.config.similarity_threshold:
                    adjacency_matrix[i, j] = similarity
                    adjacency_matrix[j, i] = similarity
                    
        return adjacency_matrix
        
    def _calculate_document_similarity(self, doc1: str, doc2: str) -> float:
        """Calculate similarity between two documents"""
        
        if self.config.edge_weight_method == "cosine":
            return self._cosine_similarity(doc1, doc2)
        elif self.config.edge_weight_method == "jaccard":
            return self._jaccard_similarity(doc1, doc2)
        else:
            return self._cosine_similarity(doc1, doc2)
            
    def _cosine_similarity(self, doc1: str, doc2: str) -> float:
        """Calculate cosine similarity between documents"""
        
        words1 = set(doc1.lower().split())
        words2 = set(doc2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return 0.0
            
        return len(intersection) / len(union)
        
    def _jaccard_similarity(self, doc1: str, doc2: str) -> float:
        """Calculate Jaccard similarity between documents"""
        
        words1 = set(doc1.lower().split())
        words2 = set(doc2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return 0.0
            
        return len(intersection) / len(union)
        
    def _matrix_to_edge_list(self, adjacency_matrix: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert adjacency matrix to edge list"""
        
        edge_indices = []
        edge_weights = []
        
        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] > 0:
                    edge_indices.append([i, j])
                    edge_weights.append(adjacency_matrix[i, j])
                    
        if edge_indices:
            edge_indices = torch.LongTensor(edge_indices).t().contiguous()
            edge_weights = torch.FloatTensor(edge_weights)
        else:
            edge_indices = torch.LongTensor([[0], [0]])
            edge_weights = torch.FloatTensor([1.0])
            
        return edge_indices, edge_weights
        
    def _create_edge_features(self, edge_indices: torch.Tensor, 
                            edge_weights: torch.Tensor) -> torch.Tensor:
        """Create edge features"""
        
        if self.config.edge_features == "none":
            return None
            
        # Simple edge features based on weights
        edge_features = edge_weights.unsqueeze(1)
        
        # Add more features if needed
        if self.config.edge_feature_dim > 1:
            additional_features = torch.randn(edge_weights.size(0), self.config.edge_feature_dim - 1)
            edge_features = torch.cat([edge_features, additional_features], dim=1)
            
        return edge_features

class GCNModel(nn.Module):
    """Graph Convolutional Network"""
    
    def __init__(self, config: GNNConfig, input_dim: int, output_dim: int):
        super().__init__()
        
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # GCN layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, config.gcn_hidden_dim))
        
        # Hidden layers
        for _ in range(config.gcn_num_layers - 1):
            self.convs.append(GCNConv(config.gcn_hidden_dim, config.gcn_hidden_dim))
            
        # Output layer
        self.convs.append(GCNConv(config.gcn_hidden_dim, output_dim))
        
        # Dropout
        self.dropout = nn.Dropout(config.gcn_dropout)
        
        # Activation
        if config.gcn_activation == "relu":
            self.activation = nn.ReLU()
        elif config.gcn_activation == "tanh":
            self.activation = nn.Tanh()
        elif config.gcn_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif config.gcn_activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        """Forward pass"""
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.activation(x)
            x = self.dropout(x)
            
        # Output layer
        x = self.convs[-1](x, edge_index, edge_weight)
        
        return x

class GATModel(nn.Module):
    """Graph Attention Network"""
    
    def __init__(self, config: GNNConfig, input_dim: int, output_dim: int):
        super().__init__()
        
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # GAT layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, config.gat_hidden_dim, 
                                 heads=config.gat_num_heads, dropout=config.gat_dropout))
        
        # Hidden layers
        for _ in range(config.gat_num_layers - 1):
            self.convs.append(GATConv(config.gat_hidden_dim * config.gat_num_heads, 
                                     config.gat_hidden_dim, heads=config.gat_num_heads, 
                                     dropout=config.gat_dropout))
            
        # Output layer
        self.convs.append(GATConv(config.gat_hidden_dim * config.gat_num_heads, 
                                 output_dim, heads=1, dropout=config.gat_dropout))
        
        # Dropout
        self.dropout = nn.Dropout(config.gat_dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        """Forward pass"""
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = F.elu(x)
            x = self.dropout(x)
            
        # Output layer
        x = self.convs[-1](x, edge_index, edge_weight)
        
        return x

class GraphSAGEModel(nn.Module):
    """GraphSAGE model"""
    
    def __init__(self, config: GNNConfig, input_dim: int, output_dim: int):
        super().__init__()
        
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, config.sage_hidden_dim, 
                                  aggr=config.sage_aggregator))
        
        # Hidden layers
        for _ in range(config.sage_num_layers - 1):
            self.convs.append(SAGEConv(config.sage_hidden_dim, config.sage_hidden_dim, 
                                      aggr=config.sage_aggregator))
            
        # Output layer
        self.convs.append(SAGEConv(config.sage_hidden_dim, output_dim, 
                                  aggr=config.sage_aggregator))
        
        # Dropout
        self.dropout = nn.Dropout(config.sage_dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        """Forward pass"""
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.dropout(x)
            
        # Output layer
        x = self.convs[-1](x, edge_index, edge_weight)
        
        return x

class GINModel(nn.Module):
    """Graph Isomorphism Network"""
    
    def __init__(self, config: GNNConfig, input_dim: int, output_dim: int):
        super().__init__()
        
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # GIN layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(input_dim, config.gin_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gin_hidden_dim, config.gin_hidden_dim)
        ), eps=config.gin_eps))
        
        # Hidden layers
        for _ in range(config.gin_num_layers - 1):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(config.gin_hidden_dim, config.gin_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.gin_hidden_dim, config.gin_hidden_dim)
            ), eps=config.gin_eps))
            
        # Output layer
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(config.gin_hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        ), eps=config.gin_eps))
        
        # Dropout
        self.dropout = nn.Dropout(config.gin_dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        """Forward pass"""
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.dropout(x)
            
        # Output layer
        x = self.convs[-1](x, edge_index, edge_weight)
        
        return x

class TransformerGNNModel(nn.Module):
    """Transformer-based Graph Neural Network"""
    
    def __init__(self, config: GNNConfig, input_dim: int, output_dim: int):
        super().__init__()
        
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Transformer layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(TransformerConv(input_dim, config.transformer_hidden_dim, 
                                         heads=config.transformer_num_heads, 
                                         dropout=config.transformer_dropout))
        
        # Hidden layers
        for _ in range(config.transformer_num_layers - 1):
            self.convs.append(TransformerConv(config.transformer_hidden_dim * config.transformer_num_heads, 
                                             config.transformer_hidden_dim, 
                                             heads=config.transformer_num_heads, 
                                             dropout=config.transformer_dropout))
            
        # Output layer
        self.convs.append(TransformerConv(config.transformer_hidden_dim * config.transformer_num_heads, 
                                         output_dim, heads=1, 
                                         dropout=config.transformer_dropout))
        
        # Dropout
        self.dropout = nn.Dropout(config.transformer_dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        """Forward pass"""
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = F.elu(x)
            x = self.dropout(x)
            
        # Output layer
        x = self.convs[-1](x, edge_index, edge_weight)
        
        return x

class GraphNeuralNetworkEngine:
    """Main Graph Neural Network Engine"""
    
    def __init__(self, config: GNNConfig):
        self.config = config
        self.model = None
        self.graph_constructor = GraphConstructor(config)
        self.device = torch.device(config.device)
        
        # Results storage
        self.results = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        
    def build_model(self, input_dim: int, output_dim: int = None):
        """Build GNN model"""
        
        if output_dim is None:
            output_dim = self.config.output_dim
            
        if self.config.model_type == "gcn":
            self.model = GCNModel(self.config, input_dim, output_dim)
        elif self.config.model_type == "gat":
            self.model = GATModel(self.config, input_dim, output_dim)
        elif self.config.model_type == "sage":
            self.model = GraphSAGEModel(self.config, input_dim, output_dim)
        elif self.config.model_type == "gin":
            self.model = GINModel(self.config, input_dim, output_dim)
        elif self.config.model_type == "transformer":
            self.model = TransformerGNNModel(self.config, input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
        self.model = self.model.to(self.device)
        
    def construct_graph(self, data: Any) -> Data:
        """Construct graph from data"""
        
        if self.config.graph_type == "document":
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                return self.graph_constructor.construct_document_graph(data)
            else:
                raise ValueError("Document graph requires list of strings")
                
        elif self.config.graph_type == "citation":
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                return self.graph_constructor.construct_citation_graph(data)
            else:
                raise ValueError("Citation graph requires list of dictionaries")
                
        elif self.config.graph_type == "social":
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                return self.graph_constructor.construct_social_graph(data)
            else:
                raise ValueError("Social graph requires list of dictionaries")
                
        else:
            raise ValueError(f"Unsupported graph type: {self.config.graph_type}")
            
    def train(self, graph_data: Data, labels: torch.Tensor = None, 
              train_mask: torch.Tensor = None, val_mask: torch.Tensor = None, 
              test_mask: torch.Tensor = None):
        """Train GNN model"""
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Move data to device
        graph_data = graph_data.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if train_mask is not None:
            train_mask = train_mask.to(self.device)
        if val_mask is not None:
            val_mask = val_mask.to(self.device)
        if test_mask is not None:
            test_mask = test_mask.to(self.device)
            
        # Create masks if not provided
        if train_mask is None:
            num_nodes = graph_data.x.size(0)
            train_size = int(num_nodes * self.config.train_ratio)
            val_size = int(num_nodes * self.config.val_ratio)
            
            indices = torch.randperm(num_nodes)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            train_mask[indices[:train_size]] = True
            val_mask[indices[train_size:train_size + val_size]] = True
            test_mask[indices[train_size + val_size:]] = True
            
        # Optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                   lr=self.config.learning_rate, 
                                   weight_decay=self.config.weight_decay)
        
        if self.config.task_type == "node_classification":
            criterion = nn.CrossEntropyLoss()
        elif self.config.task_type == "node_regression":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
            
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
            
            # Calculate loss
            if self.config.task_type == "node_classification":
                loss = criterion(output[train_mask], labels[train_mask])
            elif self.config.task_type == "node_regression":
                loss = criterion(output[train_mask], labels[train_mask])
            else:
                loss = criterion(output[train_mask], labels[train_mask])
                
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Validation
            if val_mask is not None:
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
                    val_loss = criterion(val_output[val_mask], labels[val_mask])
                    
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}")
                
    def predict(self, graph_data: Data) -> torch.Tensor:
        """Make predictions"""
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        self.model.eval()
        
        with torch.no_grad():
            graph_data = graph_data.to(self.device)
            output = self.model(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
            
        return output
        
    def evaluate(self, graph_data: Data, labels: torch.Tensor, 
                test_mask: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance"""
        
        predictions = self.predict(graph_data)
        predictions = predictions[test_mask]
        labels = labels[test_mask]
        
        metrics = {}
        
        if self.config.task_type == "node_classification":
            # Classification metrics
            pred_classes = torch.argmax(predictions, dim=1)
            
            accuracy = accuracy_score(labels.cpu().numpy(), pred_classes.cpu().numpy())
            f1 = f1_score(labels.cpu().numpy(), pred_classes.cpu().numpy(), average='weighted')
            
            metrics['accuracy'] = accuracy
            metrics['f1_score'] = f1
            
            # AUC for binary classification
            if self.config.num_classes == 2:
                auc = roc_auc_score(labels.cpu().numpy(), predictions[:, 1].cpu().numpy())
                metrics['auc'] = auc
                
        elif self.config.task_type == "node_regression":
            # Regression metrics
            mse = F.mse_loss(predictions, labels).item()
            mae = F.l1_loss(predictions, labels).item()
            
            metrics['mse'] = mse
            metrics['mae'] = mae
            metrics['rmse'] = np.sqrt(mse)
            
        return metrics
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        metrics = {
            'model_type': self.config.model_type,
            'graph_type': self.config.graph_type,
            'task_type': self.config.task_type,
            'model_built': self.model is not None,
            'total_training_runs': len(self.results)
        }
        
        return metrics
        
    def save_model(self, filepath: str):
        """Save model"""
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'performance_metrics': self.get_performance_metrics()
        }, filepath)
        
    def load_model(self, filepath: str, input_dim: int, output_dim: int = None):
        """Load model"""
        
        checkpoint = torch.load(filepath)
        
        # Update config
        self.config = GNNConfig(**checkpoint['config'])
        
        # Build model
        self.build_model(input_dim, output_dim)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test GNN engine
    print("Testing Graph Neural Network Engine...")
    
    # Create dummy document data
    documents = [
        "This is a document about machine learning and artificial intelligence.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks for pattern recognition.",
        "Natural language processing is an important AI application.",
        "Computer vision enables machines to understand visual information.",
        "Graph neural networks can process structured data effectively.",
        "Reinforcement learning learns through interaction with environment.",
        "Supervised learning uses labeled training data.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Transfer learning applies knowledge from one domain to another."
    ]
    
    # Create config
    config = GNNConfig(
        model_type="gcn",
        graph_type="document",
        gcn_hidden_dim=32,
        gcn_num_layers=2,
        node_feature_dim=64,
        task_type="node_classification",
        num_classes=2,
        learning_rate=0.01,
        num_epochs=50
    )
    
    # Create engine
    gnn_engine = GraphNeuralNetworkEngine(config)
    
    # Construct graph
    print("Constructing document graph...")
    graph_data = gnn_engine.construct_graph(documents)
    print(f"Graph constructed: {graph_data.x.size(0)} nodes, {graph_data.edge_index.size(1)} edges")
    
    # Build model
    print("Building GNN model...")
    gnn_engine.build_model(input_dim=graph_data.x.size(1), output_dim=config.num_classes)
    
    # Create dummy labels
    labels = torch.randint(0, config.num_classes, (graph_data.x.size(0),))
    
    # Train model
    print("Training GNN model...")
    gnn_engine.train(graph_data, labels)
    
    # Make predictions
    print("Making predictions...")
    predictions = gnn_engine.predict(graph_data)
    print(f"Predictions shape: {predictions.shape}")
    
    # Evaluate model
    print("Evaluating model...")
    num_nodes = graph_data.x.size(0)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[-2:] = True  # Test on last 2 nodes
    
    metrics = gnn_engine.evaluate(graph_data, labels, test_mask)
    print(f"Evaluation metrics: {metrics}")
    
    # Get performance metrics
    performance = gnn_engine.get_performance_metrics()
    print(f"Performance metrics: {performance}")
    
    print("\nGraph Neural Network engine initialized successfully!")
























