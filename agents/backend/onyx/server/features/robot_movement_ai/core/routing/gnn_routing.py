"""
Graph Neural Network Routing Module
===================================

Módulo de enrutamiento basado en Graph Neural Networks (GNN) para
análisis de grafos de rutas y predicción de caminos óptimos.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

logger = logging.getLogger(__name__)

try:
    import torch_geometric
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("torch_geometric no disponible, funcionalidad GNN limitada")


@dataclass
class GraphRouteData:
    """Datos de grafo para GNN."""
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_positions: torch.Tensor
    batch: Optional[torch.Tensor] = None


class GCNRoutePredictor(nn.Module):
    """
    Graph Convolutional Network para predicción de rutas.
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        """
        Inicializar GCN.
        
        Args:
            input_dim: Dimensión de features de nodos
            hidden_dim: Dimensión de capas ocultas
            output_dim: Dimensión de salida
            num_layers: Número de capas GCN
            dropout: Tasa de dropout
        """
        super(GCNRoutePredictor, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Primera capa
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Capas intermedias
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Última capa
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Inicialización
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        for m in self.modules():
            if isinstance(m, GCNConv):
                nn.init.xavier_uniform_(m.lin.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Features de nodos [num_nodes, input_dim]
            edge_index: Índices de aristas [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Embeddings de nodos [num_nodes, output_dim]
        """
        # Capas GCN
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Última capa
        x = self.convs[-1](x, edge_index)
        
        return x


class GATRoutePredictor(nn.Module):
    """
    Graph Attention Network para predicción de rutas con atención.
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        """
        Inicializar GAT.
        
        Args:
            input_dim: Dimensión de features de nodos
            hidden_dim: Dimensión de capas ocultas
            output_dim: Dimensión de salida
            num_layers: Número de capas GAT
            num_heads: Número de heads de atención
            dropout: Tasa de dropout
        """
        super(GATRoutePredictor, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Primera capa
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
        
        # Capas intermedias
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
        
        # Última capa (sin concatenación)
        self.convs.append(GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=dropout, concat=False))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim * num_heads)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norm(x)
            x = F.elu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        return x


class GNNRouteOptimizer:
    """
    Optimizador de rutas usando Graph Neural Networks.
    """
    
    def __init__(
        self,
        model_type: str = "GCN",
        device: Optional[str] = None,
        use_mixed_precision: bool = True
    ):
        """
        Inicializar optimizador GNN.
        
        Args:
            model_type: Tipo de modelo ("GCN", "GAT", "SAGE")
            device: Dispositivo
            use_mixed_precision: Usar precisión mixta
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric no disponible")
        
        # Detectar dispositivo
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        
        # Inicializar modelo
        if model_type == "GCN":
            self.model = GCNRoutePredictor(
                input_dim=10,
                hidden_dim=128,
                output_dim=1,
                num_layers=3
            ).to(self.device)
        elif model_type == "GAT":
            self.model = GATRoutePredictor(
                input_dim=10,
                hidden_dim=128,
                output_dim=1,
                num_layers=3,
                num_heads=4
            ).to(self.device)
        else:
            raise ValueError(f"Modelo no soportado: {model_type}")
        
        self.model_type = model_type
        
        # Optimizador
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        
        # Scaler para mixed precision
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Criterio de pérdida
        self.criterion = nn.MSELoss()
        
        logger.info(f"GNNRouteOptimizer inicializado con modelo {model_type} en {self.device}")
    
    def build_graph_data(
        self,
        nodes: Dict[str, Any],
        edges: Dict[str, Any]
    ) -> Data:
        """
        Construir objeto Data de PyTorch Geometric.
        
        Args:
            nodes: Diccionario de nodos
            edges: Diccionario de aristas
            
        Returns:
            Objeto Data
        """
        # Extraer features de nodos
        node_features = []
        node_positions = []
        node_ids = list(nodes.keys())
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        for node_id in node_ids:
            node = nodes[node_id]
            # Features: [capacity, load, cost, x, y, z, ...]
            features = [
                node.get("capacity", 1.0),
                node.get("current_load", 0.0),
                node.get("cost", 1.0),
                *node.get("position", {}).values()
            ]
            # Padding/truncate a 10 features
            features = features[:10] + [0.0] * (10 - len(features))
            node_features.append(features)
            node_positions.append(list(node.get("position", {}).values())[:3])
        
        # Construir edge_index y edge_attr
        edge_index = []
        edge_attr = []
        
        for edge_id, edge in edges.items():
            from_node = edge.get("from_node")
            to_node = edge.get("to_node")
            
            if from_node in node_id_to_idx and to_node in node_id_to_idx:
                from_idx = node_id_to_idx[from_node]
                to_idx = node_id_to_idx[to_node]
                
                edge_index.append([from_idx, to_idx])
                
                # Edge attributes: [distance, time, cost, capacity, load]
                edge_attr.append([
                    edge.get("distance", 0.0),
                    edge.get("time", 0.0),
                    edge.get("cost", 0.0),
                    edge.get("capacity", 1.0),
                    edge.get("current_load", 0.0)
                ])
        
        # Convertir a tensores
        node_features_tensor = torch.FloatTensor(node_features)
        edge_index_tensor = torch.LongTensor(edge_index).t().contiguous()
        edge_attr_tensor = torch.FloatTensor(edge_attr)
        node_positions_tensor = torch.FloatTensor(node_positions)
        
        return Data(
            x=node_features_tensor,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            pos=node_positions_tensor
        )
    
    def predict_node_scores(
        self,
        graph_data: Data
    ) -> torch.Tensor:
        """
        Predecir scores de nodos usando GNN.
        
        Args:
            graph_data: Datos del grafo
            
        Returns:
            Scores de nodos [num_nodes, 1]
        """
        self.model.eval()
        
        graph_data = graph_data.to(self.device)
        
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    scores = self.model(graph_data.x, graph_data.edge_index)
            else:
                scores = self.model(graph_data.x, graph_data.edge_index)
        
        return scores
    
    def find_optimal_path_gnn(
        self,
        graph_data: Data,
        start_node_idx: int,
        end_node_idx: int
    ) -> List[int]:
        """
        Encontrar camino óptimo usando scores de GNN.
        
        Args:
            graph_data: Datos del grafo
            start_node_idx: Índice del nodo inicial
            end_node_idx: Índice del nodo final
            
        Returns:
            Lista de índices de nodos del camino
        """
        # Predecir scores
        node_scores = self.predict_node_scores(graph_data)
        node_scores = node_scores.squeeze().cpu().numpy()
        
        # Usar A* con scores de GNN como heurística
        import heapq
        
        # Estructura para A*
        distances = {start_node_idx: 0.0}
        previous = {start_node_idx: None}
        queue = [(0.0, start_node_idx)]
        
        # Construir grafo de adyacencia
        adj_list = {}
        edge_weights = {}
        
        for i in range(graph_data.edge_index.size(1)):
            from_idx = graph_data.edge_index[0, i].item()
            to_idx = graph_data.edge_index[1, i].item()
            
            if from_idx not in adj_list:
                adj_list[from_idx] = []
            
            weight = graph_data.edge_attr[i, 0].item()  # Usar distancia
            adj_list[from_idx].append(to_idx)
            edge_weights[(from_idx, to_idx)] = weight
        
        while queue:
            current_dist, current = heapq.heappop(queue)
            
            if current == end_node_idx:
                break
            
            if current not in adj_list:
                continue
            
            for neighbor in adj_list[current]:
                edge_weight = edge_weights.get((current, neighbor), float('inf'))
                new_dist = current_dist + edge_weight
                
                # Heurística: combinar distancia con score de GNN
                heuristic = node_scores[neighbor] if neighbor < len(node_scores) else 0.0
                priority = new_dist - heuristic  # Menor score = mejor
                
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(queue, (priority, neighbor))
        
        # Reconstruir camino
        path = []
        current = end_node_idx
        while current is not None:
            path.insert(0, current)
            current = previous.get(current)
        
        if path[0] != start_node_idx:
            raise ValueError("No se encontró camino")
        
        return path
    
    def train(
        self,
        train_graphs: List[Data],
        train_targets: List[torch.Tensor],
        val_graphs: Optional[List[Data]] = None,
        val_targets: Optional[List[torch.Tensor]] = None,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Entrenar modelo GNN.
        
        Args:
            train_graphs: Lista de grafos de entrenamiento
            train_targets: Lista de targets
            val_graphs: Lista de grafos de validación
            val_targets: Lista de targets de validación
            epochs: Número de épocas
            batch_size: Tamaño de batch
            
        Returns:
            Historial de entrenamiento
        """
        history = []
        
        for epoch in range(epochs):
            # Entrenamiento
            self.model.train()
            train_loss = 0.0
            
            # Crear batches
            for i in range(0, len(train_graphs), batch_size):
                batch_graphs = train_graphs[i:i + batch_size]
                batch_targets = train_targets[i:i + batch_size]
                
                # Crear batch de PyTorch Geometric
                batch = Batch.from_data_list([g.to(self.device) for g in batch_graphs])
                targets = torch.cat(batch_targets).to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch.x, batch.edge_index, batch.batch)
                        # Pooling global
                        outputs = global_mean_pool(outputs, batch.batch)
                        loss = self.criterion(outputs, targets)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch.x, batch.edge_index, batch.batch)
                    outputs = global_mean_pool(outputs, batch.batch)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_graphs) // batch_size + 1
            
            # Validación
            val_loss = None
            if val_graphs and val_targets:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for i in range(0, len(val_graphs), batch_size):
                        batch_graphs = val_graphs[i:i + batch_size]
                        batch_targets = val_targets[i:i + batch_size]
                        
                        batch = Batch.from_data_list([g.to(self.device) for g in batch_graphs])
                        targets = torch.cat(batch_targets).to(self.device)
                        
                        if self.use_mixed_precision:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(batch.x, batch.edge_index, batch.batch)
                                outputs = global_mean_pool(outputs, batch.batch)
                                loss = self.criterion(outputs, targets)
                        else:
                            outputs = self.model(batch.x, batch.edge_index, batch.batch)
                            outputs = global_mean_pool(outputs, batch.batch)
                            loss = self.criterion(outputs, targets)
                        
                        val_loss += loss.item()
                
                val_loss /= len(val_graphs) // batch_size + 1
            
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Época {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}" +
                    (f" - Val Loss: {val_loss:.4f}" if val_loss else "")
                )
        
        return {"history": history}
    
    def save_model(self, path: str):
        """Guardar modelo."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_type": self.model_type
        }, path)
        logger.info(f"Modelo GNN guardado en: {path}")
    
    def load_model(self, path: str):
        """Cargar modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Modelo GNN cargado desde: {path}")




