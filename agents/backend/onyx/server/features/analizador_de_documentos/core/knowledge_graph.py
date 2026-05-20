"""
Sistema de Knowledge Graph
============================

Sistema para construcción y consulta de knowledge graphs.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeNode:
    """Nodo en knowledge graph"""
    node_id: str
    label: str
    properties: Dict[str, Any]
    node_type: str


@dataclass
class KnowledgeEdge:
    """Arista en knowledge graph"""
    source_id: str
    target_id: str
    relation: str
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class KnowledgeGraph:
    """
    Knowledge Graph
    
    Proporciona:
    - Construcción de knowledge graphs
    - Consultas de grafos
    - Búsqueda de caminos
    - Análisis de relaciones
    - Visualización de grafos
    """
    
    def __init__(self):
        """Inicializar knowledge graph"""
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        logger.info("KnowledgeGraph inicializado")
    
    def add_node(
        self,
        node_id: str,
        label: str,
        node_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> KnowledgeNode:
        """Agregar nodo al grafo"""
        node = KnowledgeNode(
            node_id=node_id,
            label=label,
            properties=properties or {},
            node_type=node_type
        )
        
        self.nodes[node_id] = node
        logger.debug(f"Nodo agregado: {node_id}")
        
        return node
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> KnowledgeEdge:
        """Agregar arista al grafo"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Nodos deben existir antes de agregar arista")
        
        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            properties=properties or {}
        )
        
        self.edges.append(edge)
        self.adjacency[source_id].append(target_id)
        
        logger.debug(f"Arista agregada: {source_id} -{relation}-> {target_id}")
        
        return edge
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """
        Encontrar camino entre nodos
        
        Args:
            source_id: ID del nodo origen
            target_id: ID del nodo destino
            max_depth: Profundidad máxima
        
        Returns:
            Lista de IDs de nodos en el camino
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        if source_id == target_id:
            return [source_id]
        
        # BFS para encontrar camino
        queue = [(source_id, [source_id])]
        visited = {source_id}
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            for neighbor in self.adjacency.get(current, []):
                if neighbor == target_id:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_related_nodes(
        self,
        node_id: str,
        relation: Optional[str] = None,
        max_depth: int = 1
    ) -> List[str]:
        """
        Obtener nodos relacionados
        
        Args:
            node_id: ID del nodo
            relation: Tipo de relación (opcional)
            max_depth: Profundidad máxima
        
        Returns:
            Lista de IDs de nodos relacionados
        """
        if node_id not in self.nodes:
            return []
        
        related = set()
        to_visit = [(node_id, 0)]
        visited = {node_id}
        
        while to_visit:
            current, depth = to_visit.pop(0)
            
            if depth >= max_depth:
                continue
            
            for edge in self.edges:
                if edge.source_id == current:
                    if relation is None or edge.relation == relation:
                        if edge.target_id not in visited:
                            related.add(edge.target_id)
                            visited.add(edge.target_id)
                            to_visit.append((edge.target_id, depth + 1))
                
                if edge.target_id == current:
                    if relation is None or edge.relation == relation:
                        if edge.source_id not in visited:
                            related.add(edge.source_id)
                            visited.add(edge.source_id)
                            to_visit.append((edge.source_id, depth + 1))
        
        return list(related)
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del grafo"""
        node_types = {}
        relation_types = {}
        
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        
        for edge in self.edges:
            relation_types[edge.relation] = relation_types.get(edge.relation, 0) + 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": node_types,
            "relation_types": relation_types
        }


# Instancia global
_knowledge_graph: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Obtener instancia global del grafo"""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph()
    return _knowledge_graph














