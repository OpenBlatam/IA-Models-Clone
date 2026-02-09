"""
Routing Utilities
=================

Utilidades modulares para routing: feature extraction, path generation, etc.
"""

import logging
import heapq
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class NodeFeatureExtractor:
    """Extractor de características para nodos."""
    
    def __init__(self):
        self.cache: Dict[str, np.ndarray] = {}
    
    def extract(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        edges: Dict[str, Any],
        all_nodes: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extraer características de un nodo.
        
        Args:
            node_id: ID del nodo
            node_data: Datos del nodo
            edges: Diccionario de todas las aristas
            all_nodes: Diccionario de todos los nodos
        
        Returns:
            Array de características [10]
        """
        if node_id in self.cache:
            return self.cache[node_id]
        
        position = node_data.get('position', {})
        features = np.array([
            position.get('x', 0.0),
            position.get('y', 0.0),
            position.get('z', 0.0),
            node_data.get('capacity', 1.0),
            node_data.get('current_load', 0.0),
            node_data.get('cost', 1.0),
            node_data.get('current_load', 0.0) / max(node_data.get('capacity', 1.0), 0.001),  # Load ratio
            len([e for e in edges.values() if e.get('from_node') == node_id]),  # Out degree
            len([e for e in edges.values() if e.get('to_node') == node_id]),    # In degree
            np.mean([e.get('cost', 1.0) for e in edges.values() if e.get('from_node') == node_id] + [1.0])  # Avg edge cost
        ], dtype=np.float32)
        
        self.cache[node_id] = features
        return features
    
    def clear_cache(self):
        """Limpiar cache de características."""
        self.cache.clear()


class PathGenerator:
    """Generador de rutas candidatas."""
    
    @staticmethod
    def dijkstra(
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str,
        weight: str = "distance"
    ) -> Tuple[Optional[List[str]], float, float, float]:
        """
        Algoritmo de Dijkstra.
        
        Returns:
            (path, total_distance, total_time, total_cost) or (None, 0, 0, 0) if no path
        """
        distances = {node: float('inf') for node in graph}
        distances[start] = 0.0
        previous = {node: None for node in graph}
        queue = [(0.0, start)]
        
        while queue:
            current_dist, current = heapq.heappop(queue)
            
            if current == end:
                break
            
            if current_dist > distances[current]:
                continue
            
            if current not in graph:
                continue
            
            for neighbor, edge_data in graph[current].items():
                edge_weight = edge_data.get(weight, float('inf'))
                new_dist = current_dist + edge_weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(queue, (new_dist, neighbor))
        
        # Reconstruir ruta
        path = []
        current = end
        while current is not None:
            path.insert(0, current)
            current = previous[current]
        
        if not path or path[0] != start:
            return None, 0.0, 0.0, 0.0
        
        # Calcular totales
        total_distance = 0.0
        total_time = 0.0
        total_cost = 0.0
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            if from_node in graph and to_node in graph[from_node]:
                edge_data = graph[from_node][to_node]
                total_distance += edge_data.get("distance", 0.0)
                total_time += edge_data.get("time", 0.0)
                total_cost += edge_data.get("cost", 0.0)
        
        return path, total_distance, total_time, total_cost
    
    @staticmethod
    def random_walk(
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str,
        max_steps: int = 100
    ) -> Optional[List[str]]:
        """Generate path using random walk with goal bias."""
        path = [start]
        current = start
        visited = {start}
        
        for _ in range(max_steps):
            if current == end:
                return path
            
            if current not in graph:
                break
            
            neighbors = list(graph[current].keys())
            if not neighbors:
                break
            
            # Bias towards end node
            if end in neighbors:
                current = end
                path.append(current)
                return path
            
            # Weight neighbors
            weights = []
            for neighbor in neighbors:
                if neighbor in visited and neighbor != end:
                    weights.append(0.1)
                else:
                    weights.append(1.0)
            
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            current = np.random.choice(neighbors, p=weights)
            path.append(current)
            visited.add(current)
        
        return None
    
    @staticmethod
    def generate_candidates(
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str,
        num_candidates: int = 10
    ) -> List[List[str]]:
        """Generate candidate paths using various heuristics."""
        candidates = []
        
        # Shortest path
        try:
            path, _, _, _ = PathGenerator.dijkstra(graph, start, end, weight="distance")
            if path:
                candidates.append(path)
        except:
            pass
        
        # Fastest path
        try:
            path, _, _, _ = PathGenerator.dijkstra(graph, start, end, weight="time")
            if path and path not in candidates:
                candidates.append(path)
        except:
            pass
        
        # Least cost path
        try:
            path, _, _, _ = PathGenerator.dijkstra(graph, start, end, weight="cost")
            if path and path not in candidates:
                candidates.append(path)
        except:
            pass
        
        # Random walk variations
        for _ in range(num_candidates - len(candidates)):
            try:
                path = PathGenerator.random_walk(graph, start, end, max_steps=100)
                if path and path not in candidates:
                    candidates.append(path)
            except:
                pass
        
        return candidates[:num_candidates]


class RouteMetricsCalculator:
    """Calculadora de métricas de rutas."""
    
    @staticmethod
    def calculate(
        graph: Dict[str, Dict[str, Dict[str, float]]],
        path: List[str]
    ) -> Tuple[float, float, float]:
        """Calculate distance, time, and cost for a path."""
        total_distance = 0.0
        total_time = 0.0
        total_cost = 0.0
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            if from_node in graph and to_node in graph[from_node]:
                edge_data = graph[from_node][to_node]
                total_distance += edge_data.get("distance", 0.0)
                total_time += edge_data.get("time", 0.0)
                total_cost += edge_data.get("cost", 0.0)
        
        return total_distance, total_time, total_cost
    
    @staticmethod
    def calculate_smoothness(path: List[str], positions: Dict[str, Dict[str, float]]) -> float:
        """Calculate path smoothness based on angles."""
        if len(path) < 3:
            return 1.0
        
        angles = []
        for i in range(1, len(path) - 1):
            p1 = positions.get(path[i-1], {})
            p2 = positions.get(path[i], {})
            p3 = positions.get(path[i+1], {})
            
            v1 = np.array([
                p2.get('x', 0) - p1.get('x', 0),
                p2.get('y', 0) - p1.get('y', 0),
                p2.get('z', 0) - p1.get('z', 0)
            ])
            v2 = np.array([
                p3.get('x', 0) - p2.get('x', 0),
                p3.get('y', 0) - p2.get('y', 0),
                p3.get('z', 0) - p2.get('z', 0)
            ])
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        if not angles:
            return 1.0
        
        # Lower variance in angles = smoother path
        angle_variance = np.var(angles)
        smoothness = 1.0 / (1.0 + angle_variance)
        return smoothness


class GraphBuilder:
    """Constructor de grafos desde nodos y aristas."""
    
    @staticmethod
    def build(
        nodes: Dict[str, Any],
        edges: Dict[str, Any]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Construir grafo de nodos y aristas."""
        graph = {}
        
        for node_id in nodes:
            graph[node_id] = {}
        
        for edge in edges.values():
            from_node = edge.get('from_node') if isinstance(edge, dict) else edge.from_node
            to_node = edge.get('to_node') if isinstance(edge, dict) else edge.to_node
            
            if from_node not in graph:
                graph[from_node] = {}
            
            edge_data = {
                "distance": edge.get('distance', 0.0) if isinstance(edge, dict) else edge.distance,
                "time": edge.get('time', 0.0) if isinstance(edge, dict) else edge.time,
                "cost": edge.get('cost', 0.0) if isinstance(edge, dict) else edge.cost,
                "capacity": edge.get('capacity', 1.0) if isinstance(edge, dict) else edge.capacity,
                "load": edge.get('current_load', 0.0) if isinstance(edge, dict) else edge.current_load
            }
            
            graph[from_node][to_node] = edge_data
        
        return graph




