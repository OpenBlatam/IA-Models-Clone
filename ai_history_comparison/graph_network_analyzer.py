"""
Advanced Graph and Network Analysis System
Sistema avanzado de análisis de grafos y redes
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import math
import warnings
warnings.filterwarnings('ignore')

# Graph analysis imports
import networkx as nx
from networkx.algorithms import centrality, community, clustering, shortest_paths
from networkx.algorithms.components import connected_components, strongly_connected_components
from networkx.algorithms.link_analysis import pagerank, hits
from networkx.algorithms.shortest_paths import all_pairs_shortest_path_length
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphType(Enum):
    """Tipos de grafos"""
    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    WEIGHTED = "weighted"
    MULTIGRAPH = "multigraph"
    BIPARTITE = "bipartite"
    TREE = "tree"
    DAG = "dag"  # Directed Acyclic Graph

class AnalysisType(Enum):
    """Tipos de análisis"""
    CENTRALITY = "centrality"
    COMMUNITY = "community"
    CLUSTERING = "clustering"
    PATH_ANALYSIS = "path_analysis"
    CONNECTIVITY = "connectivity"
    STRUCTURAL = "structural"
    DYNAMIC = "dynamic"

class CentralityType(Enum):
    """Tipos de centralidad"""
    DEGREE = "degree"
    BETWEENNESS = "betweenness"
    CLOSENESS = "closeness"
    EIGENVECTOR = "eigenvector"
    PAGERANK = "pagerank"
    HITS = "hits"
    LOAD = "load"
    HARMONIC = "harmonic"

@dataclass
class GraphNode:
    """Nodo del grafo"""
    id: str
    label: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    position: Optional[Tuple[float, float]] = None
    community: Optional[int] = None
    centrality_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class GraphEdge:
    """Arista del grafo"""
    source: str
    target: str
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    edge_type: str = "default"

@dataclass
class GraphAnalysis:
    """Análisis de grafo"""
    id: str
    graph_type: GraphType
    analysis_type: AnalysisType
    nodes_count: int
    edges_count: int
    density: float
    clustering_coefficient: float
    average_path_length: float
    diameter: int
    components_count: int
    largest_component_size: int
    centrality_analysis: Dict[str, Any]
    community_analysis: Dict[str, Any]
    structural_properties: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class NetworkInsight:
    """Insight de red"""
    id: str
    insight_type: str
    description: str
    significance: float
    confidence: float
    related_nodes: List[str]
    related_edges: List[Tuple[str, str]]
    implications: List[str]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedGraphNetworkAnalyzer:
    """
    Analizador avanzado de grafos y redes
    """
    
    def __init__(
        self,
        enable_visualization: bool = True,
        enable_community_detection: bool = True,
        enable_centrality_analysis: bool = True,
        enable_dynamic_analysis: bool = True
    ):
        self.enable_visualization = enable_visualization
        self.enable_community_detection = enable_community_detection
        self.enable_centrality_analysis = enable_centrality_analysis
        self.enable_dynamic_analysis = enable_dynamic_analysis
        
        # Almacenamiento
        self.graphs: Dict[str, nx.Graph] = {}
        self.graph_analyses: Dict[str, GraphAnalysis] = {}
        self.network_insights: Dict[str, NetworkInsight] = {}
        
        # Configuración
        self.config = {
            "default_layout": "spring",
            "visualization_dpi": 300,
            "max_nodes_visualization": 1000,
            "community_resolution": 1.0,
            "centrality_normalization": True,
            "path_length_threshold": 10
        }
    
    async def create_graph(
        self,
        graph_id: str,
        graph_type: GraphType = GraphType.UNDIRECTED,
        nodes: Optional[List[GraphNode]] = None,
        edges: Optional[List[GraphEdge]] = None
    ) -> nx.Graph:
        """
        Crear grafo
        
        Args:
            graph_id: ID del grafo
            graph_type: Tipo de grafo
            nodes: Lista de nodos
            edges: Lista de aristas
            
        Returns:
            Grafo de NetworkX
        """
        try:
            logger.info(f"Creating {graph_type.value} graph: {graph_id}")
            
            # Crear grafo según el tipo
            if graph_type == GraphType.DIRECTED:
                G = nx.DiGraph()
            elif graph_type == GraphType.MULTIGRAPH:
                G = nx.MultiGraph()
            elif graph_type == GraphType.BIPARTITE:
                G = nx.Graph()
            else:
                G = nx.Graph()
            
            # Agregar nodos
            if nodes:
                for node in nodes:
                    G.add_node(
                        node.id,
                        label=node.label,
                        **node.attributes
                    )
            
            # Agregar aristas
            if edges:
                for edge in edges:
                    G.add_edge(
                        edge.source,
                        edge.target,
                        weight=edge.weight,
                        edge_type=edge.edge_type,
                        **edge.attributes
                    )
            
            # Almacenar grafo
            self.graphs[graph_id] = G
            
            logger.info(f"Graph {graph_id} created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Error creating graph: {e}")
            raise
    
    async def analyze_graph(
        self,
        graph_id: str,
        analysis_type: AnalysisType = AnalysisType.STRUCTURAL,
        include_centrality: bool = True,
        include_community: bool = True
    ) -> GraphAnalysis:
        """
        Analizar grafo
        
        Args:
            graph_id: ID del grafo
            analysis_type: Tipo de análisis
            include_centrality: Si incluir análisis de centralidad
            include_community: Si incluir detección de comunidades
            
        Returns:
            Análisis del grafo
        """
        try:
            if graph_id not in self.graphs:
                raise ValueError(f"Graph {graph_id} not found")
            
            G = self.graphs[graph_id]
            logger.info(f"Analyzing graph {graph_id} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            # Propiedades básicas
            nodes_count = G.number_of_nodes()
            edges_count = G.number_of_edges()
            density = nx.density(G)
            
            # Coeficiente de clustering
            clustering_coefficient = nx.average_clustering(G)
            
            # Longitud promedio de caminos
            if nx.is_connected(G):
                average_path_length = nx.average_shortest_path_length(G)
                diameter = nx.diameter(G)
            else:
                # Para grafos no conectados, calcular para el componente más grande
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                average_path_length = nx.average_shortest_path_length(subgraph)
                diameter = nx.diameter(subgraph)
            
            # Componentes conectados
            if G.is_directed():
                components = list(strongly_connected_components(G))
            else:
                components = list(connected_components(G))
            
            components_count = len(components)
            largest_component_size = len(max(components, key=len)) if components else 0
            
            # Análisis de centralidad
            centrality_analysis = {}
            if include_centrality and self.enable_centrality_analysis:
                centrality_analysis = await self._analyze_centrality(G)
            
            # Análisis de comunidades
            community_analysis = {}
            if include_community and self.enable_community_detection:
                community_analysis = await self._analyze_communities(G)
            
            # Propiedades estructurales
            structural_properties = await self._analyze_structural_properties(G)
            
            # Crear análisis
            analysis = GraphAnalysis(
                id=f"analysis_{graph_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                graph_type=self._determine_graph_type(G),
                analysis_type=analysis_type,
                nodes_count=nodes_count,
                edges_count=edges_count,
                density=density,
                clustering_coefficient=clustering_coefficient,
                average_path_length=average_path_length,
                diameter=diameter,
                components_count=components_count,
                largest_component_size=largest_component_size,
                centrality_analysis=centrality_analysis,
                community_analysis=community_analysis,
                structural_properties=structural_properties
            )
            
            # Almacenar análisis
            self.graph_analyses[analysis.id] = analysis
            
            # Generar insights
            await self._generate_network_insights(analysis, G)
            
            logger.info(f"Graph analysis completed: {analysis.id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing graph: {e}")
            raise
    
    async def _analyze_centrality(self, G: nx.Graph) -> Dict[str, Any]:
        """Analizar centralidad del grafo"""
        try:
            centrality_results = {}
            
            # Centralidad de grado
            degree_centrality = nx.degree_centrality(G)
            centrality_results["degree"] = {
                "scores": degree_centrality,
                "top_nodes": sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
            # Centralidad de intermediación
            betweenness_centrality = nx.betweenness_centrality(G)
            centrality_results["betweenness"] = {
                "scores": betweenness_centrality,
                "top_nodes": sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
            # Centralidad de cercanía
            if nx.is_connected(G):
                closeness_centrality = nx.closeness_centrality(G)
                centrality_results["closeness"] = {
                    "scores": closeness_centrality,
                    "top_nodes": sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                }
            
            # Centralidad de vector propio
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
                centrality_results["eigenvector"] = {
                    "scores": eigenvector_centrality,
                    "top_nodes": sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                }
            except:
                centrality_results["eigenvector"] = {"error": "Could not compute eigenvector centrality"}
            
            # PageRank
            try:
                pagerank_scores = nx.pagerank(G)
                centrality_results["pagerank"] = {
                    "scores": pagerank_scores,
                    "top_nodes": sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                }
            except:
                centrality_results["pagerank"] = {"error": "Could not compute PageRank"}
            
            # HITS (solo para grafos dirigidos)
            if G.is_directed():
                try:
                    hits_scores = nx.hits(G)
                    centrality_results["hits"] = {
                        "hubs": hits_scores[0],
                        "authorities": hits_scores[1],
                        "top_hubs": sorted(hits_scores[0].items(), key=lambda x: x[1], reverse=True)[:10],
                        "top_authorities": sorted(hits_scores[1].items(), key=lambda x: x[1], reverse=True)[:10]
                    }
                except:
                    centrality_results["hits"] = {"error": "Could not compute HITS"}
            
            return centrality_results
            
        except Exception as e:
            logger.error(f"Error analyzing centrality: {e}")
            return {}
    
    async def _analyze_communities(self, G: nx.Graph) -> Dict[str, Any]:
        """Analizar comunidades del grafo"""
        try:
            community_results = {}
            
            # Detección de comunidades con algoritmo de Louvain
            try:
                import networkx.algorithms.community as nx_comm
                louvain_communities = nx_comm.louvain_communities(G, resolution=self.config["community_resolution"])
                community_results["louvain"] = {
                    "communities": [list(community) for community in louvain_communities],
                    "modularity": nx_comm.modularity(G, louvain_communities),
                    "number_of_communities": len(louvain_communities),
                    "community_sizes": [len(community) for community in louvain_communities]
                }
            except Exception as e:
                community_results["louvain"] = {"error": f"Could not compute Louvain communities: {e}"}
            
            # Detección de comunidades con algoritmo de Girvan-Newman
            try:
                if G.number_of_nodes() <= 100:  # Solo para grafos pequeños
                    girvan_newman_communities = list(nx_comm.girvan_newman(G))
                    if girvan_newman_communities:
                        best_communities = girvan_newman_communities[-1]
                        community_results["girvan_newman"] = {
                            "communities": [list(community) for community in best_communities],
                            "modularity": nx_comm.modularity(G, best_communities),
                            "number_of_communities": len(best_communities),
                            "community_sizes": [len(community) for community in best_communities]
                        }
            except Exception as e:
                community_results["girvan_newman"] = {"error": f"Could not compute Girvan-Newman communities: {e}"}
            
            # Detección de comunidades con algoritmo de etiquetado
            try:
                label_propagation_communities = list(nx_comm.label_propagation_communities(G))
                community_results["label_propagation"] = {
                    "communities": [list(community) for community in label_propagation_communities],
                    "modularity": nx_comm.modularity(G, label_propagation_communities),
                    "number_of_communities": len(label_propagation_communities),
                    "community_sizes": [len(community) for community in label_propagation_communities]
                }
            except Exception as e:
                community_results["label_propagation"] = {"error": f"Could not compute label propagation communities: {e}"}
            
            return community_results
            
        except Exception as e:
            logger.error(f"Error analyzing communities: {e}")
            return {}
    
    async def _analyze_structural_properties(self, G: nx.Graph) -> Dict[str, Any]:
        """Analizar propiedades estructurales del grafo"""
        try:
            structural_properties = {}
            
            # Propiedades básicas
            structural_properties["basic"] = {
                "is_connected": nx.is_connected(G) if not G.is_directed() else nx.is_strongly_connected(G),
                "is_weakly_connected": nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G),
                "is_directed": G.is_directed(),
                "is_multigraph": G.is_multigraph(),
                "is_bipartite": nx.is_bipartite(G),
                "is_tree": nx.is_tree(G),
                "is_forest": nx.is_forest(G),
                "is_dag": nx.is_directed_acyclic_graph(G) if G.is_directed() else False
            }
            
            # Distribución de grados
            degree_sequence = [d for n, d in G.degree()]
            structural_properties["degree_distribution"] = {
                "mean_degree": np.mean(degree_sequence),
                "std_degree": np.std(degree_sequence),
                "min_degree": np.min(degree_sequence),
                "max_degree": np.max(degree_sequence),
                "degree_histogram": np.histogram(degree_sequence, bins=20)[0].tolist()
            }
            
            # Coeficientes de clustering
            clustering_coefficients = nx.clustering(G)
            structural_properties["clustering"] = {
                "average_clustering": nx.average_clustering(G),
                "global_clustering": nx.transitivity(G),
                "clustering_distribution": {
                    "mean": np.mean(list(clustering_coefficients.values())),
                    "std": np.std(list(clustering_coefficients.values())),
                    "min": np.min(list(clustering_coefficients.values())),
                    "max": np.max(list(clustering_coefficients.values()))
                }
            }
            
            # Análisis de caminos
            if nx.is_connected(G):
                structural_properties["paths"] = {
                    "average_shortest_path_length": nx.average_shortest_path_length(G),
                    "diameter": nx.diameter(G),
                    "radius": nx.radius(G),
                    "center": list(nx.center(G)),
                    "periphery": list(nx.periphery(G))
                }
            else:
                # Para grafos no conectados
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                structural_properties["paths"] = {
                    "average_shortest_path_length": nx.average_shortest_path_length(subgraph),
                    "diameter": nx.diameter(subgraph),
                    "radius": nx.radius(subgraph),
                    "center": list(nx.center(subgraph)),
                    "periphery": list(nx.periphery(subgraph))
                }
            
            # Análisis de conectividad
            if G.is_directed():
                structural_properties["connectivity"] = {
                    "strongly_connected_components": len(list(strongly_connected_components(G))),
                    "weakly_connected_components": len(list(connected_components(G.to_undirected()))),
                    "node_connectivity": nx.node_connectivity(G) if nx.is_strongly_connected(G) else 0,
                    "edge_connectivity": nx.edge_connectivity(G) if nx.is_strongly_connected(G) else 0
                }
            else:
                structural_properties["connectivity"] = {
                    "connected_components": len(list(connected_components(G))),
                    "node_connectivity": nx.node_connectivity(G) if nx.is_connected(G) else 0,
                    "edge_connectivity": nx.edge_connectivity(G) if nx.is_connected(G) else 0
                }
            
            return structural_properties
            
        except Exception as e:
            logger.error(f"Error analyzing structural properties: {e}")
            return {}
    
    def _determine_graph_type(self, G: nx.Graph) -> GraphType:
        """Determinar tipo de grafo"""
        if G.is_directed():
            if nx.is_directed_acyclic_graph(G):
                return GraphType.DAG
            else:
                return GraphType.DIRECTED
        elif G.is_multigraph():
            return GraphType.MULTIGRAPH
        elif nx.is_bipartite(G):
            return GraphType.BIPARTITE
        elif nx.is_tree(G):
            return GraphType.TREE
        else:
            return GraphType.UNDIRECTED
    
    async def _generate_network_insights(self, analysis: GraphAnalysis, G: nx.Graph):
        """Generar insights de red"""
        try:
            insights = []
            
            # Insight 1: Densidad de la red
            if analysis.density > 0.5:
                insight = NetworkInsight(
                    id=f"high_density_{analysis.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    insight_type="density",
                    description=f"Red de alta densidad ({analysis.density:.3f}) - conexiones muy densas",
                    significance=analysis.density,
                    confidence=0.9,
                    related_nodes=[],
                    related_edges=[],
                    implications=[
                        "La red tiene muchas conexiones entre nodos",
                        "La información puede fluir rápidamente",
                        "La red es robusta a fallos de nodos individuales"
                    ],
                    recommendations=[
                        "Considerar la redundancia en la red",
                        "Monitorear la propagación de información",
                        "Evaluar la eficiencia de la red"
                    ]
                )
                insights.append(insight)
            
            # Insight 2: Componentes conectados
            if analysis.components_count > 1:
                insight = NetworkInsight(
                    id=f"multiple_components_{analysis.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    insight_type="connectivity",
                    description=f"Red fragmentada en {analysis.components_count} componentes",
                    significance=1.0 - (analysis.largest_component_size / analysis.nodes_count),
                    confidence=0.8,
                    related_nodes=[],
                    related_edges=[],
                    implications=[
                        "La red no está completamente conectada",
                        "Algunos nodos pueden estar aislados",
                        "La comunicación puede estar limitada"
                    ],
                    recommendations=[
                        "Identificar nodos puente para conectar componentes",
                        "Evaluar la importancia de cada componente",
                        "Considerar estrategias de conexión"
                    ]
                )
                insights.append(insight)
            
            # Insight 3: Coeficiente de clustering
            if analysis.clustering_coefficient > 0.3:
                insight = NetworkInsight(
                    id=f"high_clustering_{analysis.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    insight_type="clustering",
                    description=f"Alto coeficiente de clustering ({analysis.clustering_coefficient:.3f})",
                    significance=analysis.clustering_coefficient,
                    confidence=0.8,
                    related_nodes=[],
                    related_edges=[],
                    implications=[
                        "Los nodos tienden a formar grupos densos",
                        "La red tiene estructura de mundo pequeño",
                        "Las comunidades están bien definidas"
                    ],
                    recommendations=[
                        "Identificar comunidades naturales",
                        "Analizar la estructura de grupos",
                        "Considerar la propagación dentro de comunidades"
                    ]
                )
                insights.append(insight)
            
            # Insight 4: Nodos centrales
            if analysis.centrality_analysis and "degree" in analysis.centrality_analysis:
                top_nodes = analysis.centrality_analysis["degree"]["top_nodes"][:5]
                if top_nodes:
                    insight = NetworkInsight(
                        id=f"central_nodes_{analysis.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        insight_type="centrality",
                        description=f"Nodos centrales identificados: {', '.join([node[0] for node in top_nodes])}",
                        significance=0.7,
                        confidence=0.8,
                        related_nodes=[node[0] for node in top_nodes],
                        related_edges=[],
                        implications=[
                            "Estos nodos son cruciales para la conectividad",
                            "Pueden ser puntos de fallo críticos",
                            "Son importantes para la propagación de información"
                        ],
                        recommendations=[
                            "Monitorear la salud de estos nodos",
                            "Considerar redundancia para nodos críticos",
                            "Analizar su papel en la red"
                        ]
                    )
                    insights.append(insight)
            
            # Almacenar insights
            for insight in insights:
                self.network_insights[insight.id] = insight
                
        except Exception as e:
            logger.error(f"Error generating network insights: {e}")
    
    async def visualize_graph(
        self,
        graph_id: str,
        layout: str = "spring",
        highlight_communities: bool = True,
        highlight_centrality: bool = True,
        output_format: str = "html"
    ) -> str:
        """
        Visualizar grafo
        
        Args:
            graph_id: ID del grafo
            layout: Tipo de layout
            highlight_communities: Si resaltar comunidades
            highlight_centrality: Si resaltar centralidad
            output_format: Formato de salida (html, png, svg)
            
        Returns:
            Ruta del archivo de visualización
        """
        try:
            if graph_id not in self.graphs:
                raise ValueError(f"Graph {graph_id} not found")
            
            G = self.graphs[graph_id]
            
            if G.number_of_nodes() > self.config["max_nodes_visualization"]:
                logger.warning(f"Graph too large for visualization ({G.number_of_nodes()} nodes)")
                return ""
            
            logger.info(f"Visualizing graph {graph_id} with {G.number_of_nodes()} nodes")
            
            if output_format == "html":
                return await self._create_interactive_visualization(
                    G, graph_id, layout, highlight_communities, highlight_centrality
                )
            else:
                return await self._create_static_visualization(
                    G, graph_id, layout, highlight_communities, highlight_centrality, output_format
                )
            
        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")
            return ""
    
    async def _create_interactive_visualization(
        self,
        G: nx.Graph,
        graph_id: str,
        layout: str,
        highlight_communities: bool,
        highlight_centrality: bool
    ) -> str:
        """Crear visualización interactiva con Plotly"""
        try:
            # Calcular posiciones
            if layout == "spring":
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == "circular":
                pos = nx.circular_layout(G)
            elif layout == "random":
                pos = nx.random_layout(G)
            else:
                pos = nx.spring_layout(G)
            
            # Preparar datos de nodos
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            node_sizes = []
            
            # Obtener análisis de centralidad si está disponible
            centrality_scores = {}
            if highlight_centrality:
                for analysis in self.graph_analyses.values():
                    if "degree" in analysis.centrality_analysis:
                        centrality_scores = analysis.centrality_analysis["degree"]["scores"]
                        break
            
            # Obtener comunidades si están disponibles
            community_colors = {}
            if highlight_communities:
                for analysis in self.graph_analyses.values():
                    if "louvain" in analysis.community_analysis:
                        communities = analysis.community_analysis["louvain"]["communities"]
                        for i, community in enumerate(communities):
                            for node in community:
                                community_colors[node] = i
                        break
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"Node: {node}")
                
                # Color por comunidad
                if node in community_colors:
                    node_colors.append(community_colors[node])
                else:
                    node_colors.append(0)
                
                # Tamaño por centralidad
                if node in centrality_scores:
                    node_sizes.append(centrality_scores[node] * 50 + 10)
                else:
                    node_sizes.append(10)
            
            # Preparar datos de aristas
            edge_x = []
            edge_y = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Crear figura
            fig = go.Figure()
            
            # Agregar aristas
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                name='Edges'
            ))
            
            # Agregar nodos
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale='Viridis',
                    line=dict(width=2, color='black')
                ),
                name='Nodes'
            ))
            
            # Configurar layout
            fig.update_layout(
                title=f'Graph Visualization: {graph_id}',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Interactive graph visualization",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='#999', size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            # Guardar archivo
            output_path = f"exports/graph_visualization_{graph_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            fig.write_html(output_path)
            logger.info(f"Interactive visualization saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating interactive visualization: {e}")
            return ""
    
    async def _create_static_visualization(
        self,
        G: nx.Graph,
        graph_id: str,
        layout: str,
        highlight_communities: bool,
        highlight_centrality: bool,
        output_format: str
    ) -> str:
        """Crear visualización estática con Matplotlib"""
        try:
            # Configurar matplotlib
            plt.figure(figsize=(12, 8))
            
            # Calcular posiciones
            if layout == "spring":
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == "circular":
                pos = nx.circular_layout(G)
            elif layout == "random":
                pos = nx.random_layout(G)
            else:
                pos = nx.spring_layout(G)
            
            # Dibujar aristas
            nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
            
            # Dibujar nodos
            if highlight_centrality:
                # Obtener centralidad
                centrality_scores = nx.degree_centrality(G)
                node_sizes = [centrality_scores[node] * 1000 + 100 for node in G.nodes()]
            else:
                node_sizes = 300
            
            if highlight_communities:
                # Obtener comunidades
                try:
                    import networkx.algorithms.community as nx_comm
                    communities = list(nx_comm.louvain_communities(G))
                    node_colors = []
                    for node in G.nodes():
                        for i, community in enumerate(communities):
                            if node in community:
                                node_colors.append(i)
                                break
                        else:
                            node_colors.append(len(communities))
                except:
                    node_colors = 'lightblue'
            else:
                node_colors = 'lightblue'
            
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
            
            # Dibujar etiquetas
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            plt.title(f'Graph Visualization: {graph_id}')
            plt.axis('off')
            
            # Guardar archivo
            output_path = f"exports/graph_visualization_{graph_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            plt.savefig(output_path, format=output_format, dpi=self.config["visualization_dpi"], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Static visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating static visualization: {e}")
            return ""
    
    async def compare_graphs(
        self,
        graph_ids: List[str]
    ) -> Dict[str, Any]:
        """Comparar múltiples grafos"""
        try:
            if len(graph_ids) < 2:
                raise ValueError("Se necesitan al menos 2 grafos para comparar")
            
            comparison = {
                "graph_ids": graph_ids,
                "graphs_found": 0,
                "comparison_results": {}
            }
            
            # Obtener análisis de grafos
            analyses = []
            for graph_id in graph_ids:
                if graph_id in self.graphs:
                    # Buscar análisis existente
                    analysis = None
                    for analysis_id, analysis_data in self.graph_analyses.items():
                        if graph_id in analysis_id:
                            analysis = analysis_data
                            break
                    
                    if analysis:
                        analyses.append(analysis)
                        comparison["graphs_found"] += 1
            
            if len(analyses) < 2:
                raise ValueError("No hay suficientes análisis para comparar")
            
            # Comparar propiedades básicas
            basic_comparison = {
                "nodes_count": [analysis.nodes_count for analysis in analyses],
                "edges_count": [analysis.edges_count for analysis in analyses],
                "density": [analysis.density for analysis in analyses],
                "clustering_coefficient": [analysis.clustering_coefficient for analysis in analyses],
                "average_path_length": [analysis.average_path_length for analysis in analyses],
                "diameter": [analysis.diameter for analysis in analyses],
                "components_count": [analysis.components_count for analysis in analyses]
            }
            
            # Comparar centralidad
            centrality_comparison = {}
            for analysis in analyses:
                if analysis.centrality_analysis:
                    for centrality_type, centrality_data in analysis.centrality_analysis.items():
                        if centrality_type not in centrality_comparison:
                            centrality_comparison[centrality_type] = []
                        
                        if "top_nodes" in centrality_data:
                            centrality_comparison[centrality_type].append({
                                "analysis_id": analysis.id,
                                "top_nodes": centrality_data["top_nodes"][:5]
                            })
            
            # Comparar comunidades
            community_comparison = {}
            for analysis in analyses:
                if analysis.community_analysis:
                    for community_type, community_data in analysis.community_analysis.items():
                        if community_type not in community_comparison:
                            community_comparison[community_type] = []
                        
                        if "number_of_communities" in community_data:
                            community_comparison[community_type].append({
                                "analysis_id": analysis.id,
                                "number_of_communities": community_data["number_of_communities"],
                                "modularity": community_data.get("modularity", 0)
                            })
            
            comparison["comparison_results"] = {
                "basic_properties": basic_comparison,
                "centrality_comparison": centrality_comparison,
                "community_comparison": community_comparison,
                "total_analyses": len(analyses)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing graphs: {e}")
            raise
    
    async def get_graph_network_summary(self) -> Dict[str, Any]:
        """Obtener resumen de análisis de grafos"""
        try:
            return {
                "total_graphs": len(self.graphs),
                "total_analyses": len(self.graph_analyses),
                "total_insights": len(self.network_insights),
                "graph_types": {
                    graph_type.value: len([g for g in self.graphs.values() if self._determine_graph_type(g) == graph_type])
                    for graph_type in GraphType
                },
                "analysis_types": {
                    analysis_type.value: len([a for a in self.graph_analyses.values() if a.analysis_type == analysis_type])
                    for analysis_type in AnalysisType
                },
                "average_nodes": np.mean([analysis.nodes_count for analysis in self.graph_analyses.values()]) if self.graph_analyses else 0,
                "average_edges": np.mean([analysis.edges_count for analysis in self.graph_analyses.values()]) if self.graph_analyses else 0,
                "average_density": np.mean([analysis.density for analysis in self.graph_analyses.values()]) if self.graph_analyses else 0,
                "average_clustering": np.mean([analysis.clustering_coefficient for analysis in self.graph_analyses.values()]) if self.graph_analyses else 0,
                "last_analysis": max([analysis.created_at for analysis in self.graph_analyses.values()]).isoformat() if self.graph_analyses else None
            }
        except Exception as e:
            logger.error(f"Error getting graph network summary: {e}")
            return {}
    
    async def export_graph_network_data(self, filepath: str = None) -> str:
        """Exportar datos de análisis de grafos"""
        try:
            if filepath is None:
                filepath = f"exports/graph_network_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Preparar datos de grafos
            graphs_data = {}
            for graph_id, G in self.graphs.items():
                graphs_data[graph_id] = {
                    "nodes": list(G.nodes(data=True)),
                    "edges": list(G.edges(data=True)),
                    "graph_type": self._determine_graph_type(G).value,
                    "is_directed": G.is_directed(),
                    "is_multigraph": G.is_multigraph()
                }
            
            export_data = {
                "graphs": graphs_data,
                "graph_analyses": {
                    analysis_id: {
                        "graph_type": analysis.graph_type.value,
                        "analysis_type": analysis.analysis_type.value,
                        "nodes_count": analysis.nodes_count,
                        "edges_count": analysis.edges_count,
                        "density": analysis.density,
                        "clustering_coefficient": analysis.clustering_coefficient,
                        "average_path_length": analysis.average_path_length,
                        "diameter": analysis.diameter,
                        "components_count": analysis.components_count,
                        "largest_component_size": analysis.largest_component_size,
                        "centrality_analysis": analysis.centrality_analysis,
                        "community_analysis": analysis.community_analysis,
                        "structural_properties": analysis.structural_properties,
                        "created_at": analysis.created_at.isoformat()
                    }
                    for analysis_id, analysis in self.graph_analyses.items()
                },
                "network_insights": {
                    insight_id: {
                        "insight_type": insight.insight_type,
                        "description": insight.description,
                        "significance": insight.significance,
                        "confidence": insight.confidence,
                        "related_nodes": insight.related_nodes,
                        "related_edges": insight.related_edges,
                        "implications": insight.implications,
                        "recommendations": insight.recommendations,
                        "created_at": insight.created_at.isoformat()
                    }
                    for insight_id, insight in self.network_insights.items()
                },
                "summary": await self.get_graph_network_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Graph network data exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting graph network data: {e}")
            raise
























