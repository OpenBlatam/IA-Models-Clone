"""
Routing Strategies
===================

Estrategias modulares de enrutamiento separadas del código principal.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .routing_utils import (
    PathGenerator,
    RouteMetricsCalculator,
    NodeFeatureExtractor,
    GraphBuilder
)


class BaseRoutingStrategy:
    """Clase base para estrategias de enrutamiento."""
    
    def __init__(self):
        self.path_generator = PathGenerator()
        self.metrics_calculator = RouteMetricsCalculator()
        self.graph_builder = GraphBuilder()
    
    def find_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str,
        nodes: Dict[str, Any],
        edges: Dict[str, Any]
    ) -> Tuple[Optional[List[str]], float, float, float, float]:
        """
        Encontrar ruta.
        
        Returns:
            (path, distance, time, cost, confidence)
        """
        raise NotImplementedError


class ShortestPathStrategy(BaseRoutingStrategy):
    """Estrategia de ruta más corta."""
    
    def find_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str,
        nodes: Dict[str, Any],
        edges: Dict[str, Any]
    ) -> Tuple[Optional[List[str]], float, float, float, float]:
        path, distance, time, cost = self.path_generator.dijkstra(graph, start, end, weight="distance")
        return path, distance, time, cost, 1.0


class FastestPathStrategy(BaseRoutingStrategy):
    """Estrategia de ruta más rápida."""
    
    def find_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str,
        nodes: Dict[str, Any],
        edges: Dict[str, Any]
    ) -> Tuple[Optional[List[str]], float, float, float, float]:
        path, distance, time, cost = self.path_generator.dijkstra(graph, start, end, weight="time")
        return path, distance, time, cost, 1.0


class LeastCostStrategy(BaseRoutingStrategy):
    """Estrategia de menor costo."""
    
    def find_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str,
        nodes: Dict[str, Any],
        edges: Dict[str, Any]
    ) -> Tuple[Optional[List[str]], float, float, float, float]:
        path, distance, time, cost = self.path_generator.dijkstra(graph, start, end, weight="cost")
        return path, distance, time, cost, 1.0


class LoadBalancedStrategy(BaseRoutingStrategy):
    """Estrategia balanceada por carga."""
    
    def find_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str,
        nodes: Dict[str, Any],
        edges: Dict[str, Any]
    ) -> Tuple[Optional[List[str]], float, float, float, float]:
        # Ajustar grafo por carga
        adjusted_graph = {}
        for from_node, neighbors in graph.items():
            adjusted_graph[from_node] = {}
            for to_node, edge_data in neighbors.items():
                load_factor = edge_data.get("load", 0.0) / max(edge_data.get("capacity", 1.0), 0.001)
                adjusted_cost = edge_data.get("cost", 1.0) * (1.0 + load_factor)
                adjusted_graph[from_node][to_node] = {
                    **edge_data,
                    "cost": adjusted_cost
                }
        
        path, distance, time, cost = self.path_generator.dijkstra(adjusted_graph, start, end, weight="cost")
        return path, distance, time, cost, 0.9


class AdaptiveStrategy(BaseRoutingStrategy):
    """Estrategia adaptativa que combina múltiples factores."""
    
    def find_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str,
        nodes: Dict[str, Any],
        edges: Dict[str, Any]
    ) -> Tuple[Optional[List[str]], float, float, float, float]:
        # Combinar distancia, tiempo, costo y carga
        adjusted_graph = {}
        for from_node, neighbors in graph.items():
            adjusted_graph[from_node] = {}
            for to_node, edge_data in neighbors.items():
                load_factor = edge_data.get("load", 0.0) / max(edge_data.get("capacity", 1.0), 0.001)
                # Peso combinado
                combined_weight = (
                    edge_data.get("distance", 0.0) * 0.3 +
                    edge_data.get("time", 0.0) * 0.3 +
                    edge_data.get("cost", 0.0) * 0.2 +
                    load_factor * 100.0 * 0.2
                )
                adjusted_graph[from_node][to_node] = {
                    **edge_data,
                    "combined": combined_weight
                }
        
        path, distance, time, cost = self.path_generator.dijkstra(adjusted_graph, start, end, weight="combined")
        return path, distance, time, cost, 0.85


class MLBasedStrategy(BaseRoutingStrategy):
    """Estrategia base para métodos basados en ML."""
    
    def __init__(self, model=None, device=None):
        super().__init__()
        self.model = model
        self.device = device
        self.feature_extractor = NodeFeatureExtractor()
    
    def find_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str,
        nodes: Dict[str, Any],
        edges: Dict[str, Any]
    ) -> Tuple[Optional[List[str]], float, float, float, float]:
        """Implementación base que debe ser sobrescrita."""
        # Fallback a adaptativa
        adaptive = AdaptiveStrategy()
        return adaptive.find_route(graph, start, end, nodes, edges)


def create_strategy(strategy_type: str, **kwargs) -> BaseRoutingStrategy:
    """
    Factory function para crear estrategias.
    
    Args:
        strategy_type: Tipo de estrategia
        **kwargs: Parámetros específicos de la estrategia
    
    Returns:
        Estrategia inicializada
    """
    strategy_type = strategy_type.lower()
    
    if strategy_type == "shortest_path":
        return ShortestPathStrategy()
    elif strategy_type == "fastest_path":
        return FastestPathStrategy()
    elif strategy_type == "least_cost":
        return LeastCostStrategy()
    elif strategy_type == "load_balanced":
        return LoadBalancedStrategy()
    elif strategy_type == "adaptive":
        return AdaptiveStrategy()
    elif strategy_type in ["ml_based", "deep_learning", "transformer", "gnn"]:
        return MLBasedStrategy(**kwargs)
    else:
        logger.warning(f"Unknown strategy type: {strategy_type}, using adaptive")
        return AdaptiveStrategy()




