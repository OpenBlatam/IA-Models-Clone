"""
Graph Analysis Service - Análisis de grafos y redes
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class GraphAnalysisService:
    """Servicio para análisis de grafos y redes"""
    
    def __init__(self):
        self.graphs: Dict[str, Dict[str, Any]] = {}
        self.analyses: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_graph(
        self,
        graph_name: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        graph_type: str = "directed"  # "directed", "undirected", "weighted"
    ) -> Dict[str, Any]:
        """Crear grafo"""
        
        graph_id = f"graph_{graph_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        graph = {
            "graph_id": graph_id,
            "name": graph_name,
            "type": graph_type,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "created_at": datetime.now().isoformat()
        }
        
        self.graphs[graph_id] = graph
        
        return graph
    
    def analyze_graph(
        self,
        graph_id: str
    ) -> Dict[str, Any]:
        """Analizar grafo"""
        
        graph = self.graphs.get(graph_id)
        
        if not graph:
            raise ValueError(f"Grafo {graph_id} no encontrado")
        
        nodes = graph["nodes"]
        edges = graph["edges"]
        
        # Calcular métricas básicas
        degree_centrality = self._calculate_degree_centrality(nodes, edges)
        betweenness = self._calculate_betweenness(nodes, edges)
        communities = self._detect_communities(nodes, edges)
        
        analysis = {
            "analysis_id": f"analysis_{graph_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "graph_id": graph_id,
            "metrics": {
                "density": self._calculate_density(graph),
                "average_degree": self._calculate_average_degree(nodes, edges),
                "clustering_coefficient": self._calculate_clustering(nodes, edges)
            },
            "centrality": {
                "degree": degree_centrality,
                "betweenness": betweenness
            },
            "communities": communities,
            "analyzed_at": datetime.now().isoformat()
        }
        
        if graph_id not in self.analyses:
            self.analyses[graph_id] = []
        
        self.analyses[graph_id].append(analysis)
        
        return analysis
    
    def _calculate_degree_centrality(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calcular centralidad de grado"""
        degrees = defaultdict(int)
        
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source:
                degrees[source] += 1
            if target:
                degrees[target] += 1
        
        max_degree = max(degrees.values()) if degrees else 1
        
        return {
            node.get("id", ""): degrees.get(node.get("id", ""), 0) / max_degree
            for node in nodes
        }
    
    def _calculate_betweenness(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calcular betweenness centrality (simplificado)"""
        # En producción, usar algoritmo real
        return {
            node.get("id", ""): 0.5
            for node in nodes
        }
    
    def _detect_communities(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detectar comunidades (simplificado)"""
        # En producción, usar algoritmo como Louvain
        return [
            {
                "community_id": "community_1",
                "nodes": [node.get("id") for node in nodes[:len(nodes)//2]],
                "size": len(nodes) // 2
            },
            {
                "community_id": "community_2",
                "nodes": [node.get("id") for node in nodes[len(nodes)//2:]],
                "size": len(nodes) - len(nodes) // 2
            }
        ]
    
    def _calculate_density(self, graph: Dict[str, Any]) -> float:
        """Calcular densidad del grafo"""
        n = graph["node_count"]
        m = graph["edge_count"]
        
        if n < 2:
            return 0.0
        
        max_edges = n * (n - 1) if graph["type"] == "directed" else n * (n - 1) / 2
        
        return m / max_edges if max_edges > 0 else 0.0
    
    def _calculate_average_degree(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> float:
        """Calcular grado promedio"""
        if not nodes:
            return 0.0
        
        degrees = defaultdict(int)
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source:
                degrees[source] += 1
            if target:
                degrees[target] += 1
        
        total_degree = sum(degrees.values())
        return total_degree / len(nodes) if nodes else 0.0
    
    def _calculate_clustering(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> float:
        """Calcular coeficiente de clustering (simplificado)"""
        # En producción, calcular clustering real
        return 0.3




