"""
Domain Layer
============

Entidades de dominio y value objects.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class RouteStatus(Enum):
    """Estado de una ruta."""
    PENDING = "pending"
    COMPUTING = "computing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class RouteMetrics:
    """Métricas de una ruta."""
    distance: float
    time: float
    cost: float
    efficiency: float = 0.0
    reliability: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "distance": self.distance,
            "time": self.time,
            "cost": self.cost,
            "efficiency": self.efficiency,
            "reliability": self.reliability,
            "confidence": self.confidence,
            **self.metadata
        }


@dataclass
class Route:
    """Entidad de ruta."""
    id: str
    start_node: str
    end_node: str
    path: List[str]
    metrics: RouteMetrics
    strategy: str
    status: RouteStatus = RouteStatus.COMPLETED
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "start_node": self.start_node,
            "end_node": self.end_node,
            "path": self.path,
            "metrics": self.metrics.to_dict(),
            "strategy": self.strategy,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class Node:
    """Nodo del grafo."""
    id: str
    position: Optional[tuple] = None
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "position": self.position,
            "features": self.features,
            "metadata": self.metadata
        }


@dataclass
class Edge:
    """Arista del grafo."""
    source: str
    target: str
    weight: float = 1.0
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "features": self.features,
            "metadata": self.metadata
        }


@dataclass
class Graph:
    """Grafo de routing."""
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: Node):
        """Agregar nodo."""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: Edge):
        """Agregar arista."""
        self.edges.append(edge)
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Obtener vecinos de un nodo."""
        neighbors = []
        for edge in self.edges:
            if edge.source == node_id:
                neighbors.append(edge.target)
            elif edge.target == node_id:
                neighbors.append(edge.source)
        return neighbors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata
        }

