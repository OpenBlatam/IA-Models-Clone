"""
Knowledge Graphs Engine for Export IA
Advanced knowledge graph construction, reasoning, and querying with RDF, OWL, and neural approaches
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
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
from rdflib.plugins.sparql import prepareQuery
import spacy
from spacy import displacy
import transformers
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from neo4j import GraphDatabase
import py2neo
from py2neo import Graph as NeoGraph, Node, Relationship
import owlready2
from owlready2 import *
import rdflib
from rdflib import Graph as RDFGraph
import jsonld
from jsonld import compact, expand, flatten
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
import dgl
import dgl.nn as dglnn
from dgl.nn import GraphConv as DGLGraphConv
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import to_networkx, from_networkx

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeGraphConfig:
    """Configuration for knowledge graphs"""
    # Graph types
    graph_type: str = "rdf"  # rdf, neo4j, networkx, custom
    
    # RDF parameters
    rdf_format: str = "turtle"  # turtle, n3, xml, json-ld
    rdf_namespaces: Dict[str, str] = None
    rdf_inference: bool = True
    
    # Neo4j parameters
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    
    # NetworkX parameters
    networkx_directed: bool = True
    networkx_multigraph: bool = False
    
    # Entity extraction
    entity_extraction_method: str = "spacy"  # spacy, transformers, custom
    entity_types: List[str] = None  # PERSON, ORG, GPE, etc.
    entity_confidence_threshold: float = 0.5
    
    # Relation extraction
    relation_extraction_method: str = "spacy"  # spacy, transformers, custom
    relation_types: List[str] = None  # works_for, located_in, etc.
    relation_confidence_threshold: float = 0.5
    
    # Knowledge graph construction
    kg_construction_method: str = "automatic"  # automatic, manual, hybrid
    kg_validation: bool = True
    kg_deduplication: bool = True
    kg_merging: bool = True
    
    # Embedding parameters
    embedding_method: str = "transformer"  # transformer, word2vec, glove, custom
    embedding_model: str = "bert-base-uncased"
    embedding_dimension: int = 768
    embedding_normalization: bool = True
    
    # Reasoning parameters
    reasoning_method: str = "rule_based"  # rule_based, neural, hybrid
    reasoning_rules: List[str] = None
    reasoning_depth: int = 3
    
    # Query parameters
    query_language: str = "sparql"  # sparql, cypher, custom
    query_optimization: bool = True
    query_caching: bool = True
    query_timeout: int = 30
    
    # Visualization parameters
    visualization_method: str = "networkx"  # networkx, d3, plotly, custom
    visualization_layout: str = "spring"  # spring, circular, hierarchical, force
    visualization_node_size: str = "degree"  # degree, centrality, custom
    visualization_edge_width: str = "weight"  # weight, frequency, custom
    
    # Performance parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    num_workers: int = 4
    enable_caching: bool = True

class EntityExtractor:
    """Extract entities from text"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        
        if config.entity_extraction_method == "spacy":
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.nlp = None
        elif config.entity_extraction_method == "transformers":
            self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
            self.model = AutoModel.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        else:
            raise ValueError(f"Unsupported entity extraction method: {config.entity_extraction_method}")
            
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        
        if self.config.entity_extraction_method == "spacy" and self.nlp:
            return self._extract_entities_spacy(text)
        elif self.config.entity_extraction_method == "transformers":
            return self._extract_entities_transformers(text)
        else:
            return []
            
    def _extract_entities_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy"""
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in (self.config.entity_types or ["PERSON", "ORG", "GPE"]):
                entity = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0  # spaCy doesn't provide confidence scores
                }
                entities.append(entity)
                
        return entities
        
    def _extract_entities_transformers(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using transformers"""
        
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
        # Convert predictions to entities
        entities = []
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        current_entity = None
        for i, (token, pred) in enumerate(zip(tokens, predictions[0])):
            if token.startswith('##'):
                continue
                
            pred_label = self.model.config.id2label[pred.item()]
            
            if pred_label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token,
                    'label': pred_label[2:],
                    'start': i,
                    'end': i,
                    'confidence': 0.8  # Placeholder
                }
            elif pred_label.startswith('I-') and current_entity:
                current_entity['text'] += ' ' + token
                current_entity['end'] = i
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                    
        if current_entity:
            entities.append(current_entity)
            
        return entities

class RelationExtractor:
    """Extract relations from text"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        
        if config.relation_extraction_method == "spacy":
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.nlp = None
        elif config.relation_extraction_method == "transformers":
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            self.model = AutoModel.from_pretrained("microsoft/DialoGPT-medium")
        else:
            raise ValueError(f"Unsupported relation extraction method: {config.relation_extraction_method}")
            
    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations from text"""
        
        if self.config.relation_extraction_method == "spacy" and self.nlp:
            return self._extract_relations_spacy(text, entities)
        elif self.config.relation_extraction_method == "transformers":
            return self._extract_relations_transformers(text, entities)
        else:
            return []
            
    def _extract_relations_spacy(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations using spaCy"""
        
        doc = self.nlp(text)
        relations = []
        
        # Simple relation extraction based on dependency parsing
        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "pobj"]:
                # Find related entities
                head_entity = self._find_entity_at_position(entities, token.head.i)
                dep_entity = self._find_entity_at_position(entities, token.i)
                
                if head_entity and dep_entity and head_entity != dep_entity:
                    relation = {
                        'subject': head_entity,
                        'predicate': token.dep_,
                        'object': dep_entity,
                        'confidence': 0.7
                    }
                    relations.append(relation)
                    
        return relations
        
    def _extract_relations_transformers(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations using transformers"""
        
        # Simplified relation extraction
        relations = []
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Check if entities are close in text
                if abs(entity1['start'] - entity2['start']) < 100:
                    # Simple relation based on entity types
                    relation_type = self._infer_relation_type(entity1, entity2)
                    
                    if relation_type:
                        relation = {
                            'subject': entity1,
                            'predicate': relation_type,
                            'object': entity2,
                            'confidence': 0.6
                        }
                        relations.append(relation)
                        
        return relations
        
    def _find_entity_at_position(self, entities: List[Dict[str, Any]], position: int) -> Optional[Dict[str, Any]]:
        """Find entity at specific position"""
        
        for entity in entities:
            if entity['start'] <= position <= entity['end']:
                return entity
        return None
        
    def _infer_relation_type(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> Optional[str]:
        """Infer relation type between entities"""
        
        # Simple rule-based relation inference
        if entity1['label'] == 'PERSON' and entity2['label'] == 'ORG':
            return 'works_for'
        elif entity1['label'] == 'PERSON' and entity2['label'] == 'GPE':
            return 'located_in'
        elif entity1['label'] == 'ORG' and entity2['label'] == 'GPE':
            return 'located_in'
        else:
            return 'related_to'

class KnowledgeGraphBuilder:
    """Build knowledge graphs from extracted entities and relations"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        
        if config.graph_type == "rdf":
            self.graph = RDFGraph()
            self._setup_rdf_namespaces()
        elif config.graph_type == "neo4j":
            self._setup_neo4j()
        elif config.graph_type == "networkx":
            self.graph = nx.DiGraph() if config.networkx_directed else nx.Graph()
        else:
            raise ValueError(f"Unsupported graph type: {config.graph_type}")
            
    def _setup_rdf_namespaces(self):
        """Setup RDF namespaces"""
        
        self.namespaces = {}
        
        if self.config.rdf_namespaces:
            for prefix, uri in self.config.rdf_namespaces.items():
                self.namespaces[prefix] = Namespace(uri)
        else:
            # Default namespaces
            self.namespaces['ex'] = Namespace("http://example.org/")
            self.namespaces['foaf'] = Namespace("http://xmlns.com/foaf/0.1/")
            self.namespaces['schema'] = Namespace("http://schema.org/")
            
    def _setup_neo4j(self):
        """Setup Neo4j connection"""
        
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            self.neo_graph = NeoGraph(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
        except:
            self.driver = None
            self.neo_graph = None
            
    def build_graph(self, entities: List[Dict[str, Any]], 
                   relations: List[Dict[str, Any]]) -> Any:
        """Build knowledge graph from entities and relations"""
        
        if self.config.graph_type == "rdf":
            return self._build_rdf_graph(entities, relations)
        elif self.config.graph_type == "neo4j":
            return self._build_neo4j_graph(entities, relations)
        elif self.config.graph_type == "networkx":
            return self._build_networkx_graph(entities, relations)
        else:
            raise ValueError(f"Unsupported graph type: {self.config.graph_type}")
            
    def _build_rdf_graph(self, entities: List[Dict[str, Any]], 
                        relations: List[Dict[str, Any]]) -> RDFGraph:
        """Build RDF graph"""
        
        # Add entities as subjects
        entity_uris = {}
        for entity in entities:
            uri = self.namespaces['ex'][f"entity_{hash(entity['text'])}"]
            entity_uris[entity['text']] = uri
            
            # Add entity type
            self.graph.add((uri, RDF.type, self.namespaces['ex'][entity['label']]))
            
            # Add entity label
            self.graph.add((uri, RDFS.label, Literal(entity['text'])))
            
        # Add relations as predicates
        for relation in relations:
            subject_uri = entity_uris[relation['subject']['text']]
            object_uri = entity_uris[relation['object']['text']]
            predicate_uri = self.namespaces['ex'][relation['predicate']]
            
            self.graph.add((subject_uri, predicate_uri, object_uri))
            
        return self.graph
        
    def _build_neo4j_graph(self, entities: List[Dict[str, Any]], 
                          relations: List[Dict[str, Any]]) -> bool:
        """Build Neo4j graph"""
        
        if not self.neo_graph:
            return False
            
        try:
            # Clear existing data
            self.neo_graph.delete_all()
            
            # Add entities as nodes
            entity_nodes = {}
            for entity in entities:
                node = Node(entity['label'], name=entity['text'])
                self.neo_graph.create(node)
                entity_nodes[entity['text']] = node
                
            # Add relations as edges
            for relation in relations:
                subject_node = entity_nodes[relation['subject']['text']]
                object_node = entity_nodes[relation['object']['text']]
                
                rel = Relationship(subject_node, relation['predicate'], object_node)
                self.neo_graph.create(rel)
                
            return True
            
        except Exception as e:
            logger.error(f"Error building Neo4j graph: {e}")
            return False
            
    def _build_networkx_graph(self, entities: List[Dict[str, Any]], 
                             relations: List[Dict[str, Any]]) -> nx.Graph:
        """Build NetworkX graph"""
        
        # Add entities as nodes
        for entity in entities:
            self.graph.add_node(
                entity['text'],
                label=entity['label'],
                confidence=entity.get('confidence', 1.0)
            )
            
        # Add relations as edges
        for relation in relations:
            self.graph.add_edge(
                relation['subject']['text'],
                relation['object']['text'],
                relation=relation['predicate'],
                confidence=relation.get('confidence', 1.0)
            )
            
        return self.graph

class KnowledgeGraphEmbedder:
    """Generate embeddings for knowledge graph entities and relations"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        
        if config.embedding_method == "transformer":
            self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
            self.model = AutoModel.from_pretrained(config.embedding_model)
        else:
            raise ValueError(f"Unsupported embedding method: {config.embedding_method}")
            
    def generate_embeddings(self, entities: List[Dict[str, Any]], 
                           relations: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Generate embeddings for entities and relations"""
        
        embeddings = {}
        
        # Generate entity embeddings
        for entity in entities:
            embedding = self._generate_entity_embedding(entity)
            embeddings[entity['text']] = embedding
            
        # Generate relation embeddings
        for relation in relations:
            embedding = self._generate_relation_embedding(relation)
            embeddings[f"{relation['subject']['text']}_{relation['predicate']}_{relation['object']['text']}"] = embedding
            
        return embeddings
        
    def _generate_entity_embedding(self, entity: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for entity"""
        
        # Use entity text and label
        text = f"{entity['text']} {entity['label']}"
        
        # Tokenize and encode
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
        # Normalize if required
        if self.config.embedding_normalization:
            embedding = embedding / np.linalg.norm(embedding)
            
        return embedding
        
    def _generate_relation_embedding(self, relation: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for relation"""
        
        # Use relation text
        text = f"{relation['subject']['text']} {relation['predicate']} {relation['object']['text']}"
        
        # Tokenize and encode
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
        # Normalize if required
        if self.config.embedding_normalization:
            embedding = embedding / np.linalg.norm(embedding)
            
        return embedding

class KnowledgeGraphReasoner:
    """Perform reasoning on knowledge graphs"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        
    def reason(self, graph: Any, query: str = None) -> Dict[str, Any]:
        """Perform reasoning on knowledge graph"""
        
        if self.config.reasoning_method == "rule_based":
            return self._rule_based_reasoning(graph, query)
        elif self.config.reasoning_method == "neural":
            return self._neural_reasoning(graph, query)
        elif self.config.reasoning_method == "hybrid":
            return self._hybrid_reasoning(graph, query)
        else:
            raise ValueError(f"Unsupported reasoning method: {self.config.reasoning_method}")
            
    def _rule_based_reasoning(self, graph: Any, query: str = None) -> Dict[str, Any]:
        """Perform rule-based reasoning"""
        
        results = {
            'inferred_facts': [],
            'consistency_check': True,
            'completeness_check': True
        }
        
        if self.config.graph_type == "rdf":
            # RDF reasoning
            inferred_facts = self._rdf_reasoning(graph)
            results['inferred_facts'] = inferred_facts
            
        elif self.config.graph_type == "networkx":
            # NetworkX reasoning
            inferred_facts = self._networkx_reasoning(graph)
            results['inferred_facts'] = inferred_facts
            
        return results
        
    def _neural_reasoning(self, graph: Any, query: str = None) -> Dict[str, Any]:
        """Perform neural reasoning"""
        
        # Placeholder for neural reasoning
        results = {
            'inferred_facts': [],
            'confidence_scores': [],
            'reasoning_paths': []
        }
        
        return results
        
    def _hybrid_reasoning(self, graph: Any, query: str = None) -> Dict[str, Any]:
        """Perform hybrid reasoning"""
        
        # Combine rule-based and neural reasoning
        rule_results = self._rule_based_reasoning(graph, query)
        neural_results = self._neural_reasoning(graph, query)
        
        results = {
            'rule_based_results': rule_results,
            'neural_results': neural_results,
            'combined_results': {
                'inferred_facts': rule_results['inferred_facts'] + neural_results['inferred_facts'],
                'confidence_scores': neural_results['confidence_scores']
            }
        }
        
        return results
        
    def _rdf_reasoning(self, graph: RDFGraph) -> List[Dict[str, Any]]:
        """Perform RDF reasoning"""
        
        inferred_facts = []
        
        # Simple transitivity reasoning
        for s, p, o in graph:
            if p == RDFS.subClassOf:
                # Find subclasses
                for s2, p2, o2 in graph:
                    if s2 == o and p2 == RDFS.subClassOf:
                        inferred_facts.append({
                            'subject': str(s2),
                            'predicate': str(RDFS.subClassOf),
                            'object': str(s)
                        })
                        
        return inferred_facts
        
    def _networkx_reasoning(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Perform NetworkX reasoning"""
        
        inferred_facts = []
        
        # Simple transitivity reasoning
        for node1 in graph.nodes():
            for node2 in graph.neighbors(node1):
                for node3 in graph.neighbors(node2):
                    if node1 != node3 and not graph.has_edge(node1, node3):
                        inferred_facts.append({
                            'subject': node1,
                            'predicate': 'transitive_relation',
                            'object': node3
                        })
                        
        return inferred_facts

class KnowledgeGraphQuerier:
    """Query knowledge graphs"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        
    def query(self, graph: Any, query: str) -> List[Dict[str, Any]]:
        """Query knowledge graph"""
        
        if self.config.query_language == "sparql":
            return self._sparql_query(graph, query)
        elif self.config.query_language == "cypher":
            return self._cypher_query(graph, query)
        elif self.config.query_language == "custom":
            return self._custom_query(graph, query)
        else:
            raise ValueError(f"Unsupported query language: {self.config.query_language}")
            
    def _sparql_query(self, graph: RDFGraph, query: str) -> List[Dict[str, Any]]:
        """Execute SPARQL query"""
        
        try:
            results = graph.query(query)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"SPARQL query error: {e}")
            return []
            
    def _cypher_query(self, graph: Any, query: str) -> List[Dict[str, Any]]:
        """Execute Cypher query"""
        
        try:
            if hasattr(graph, 'run'):
                results = graph.run(query).data()
                return results
            else:
                return []
        except Exception as e:
            logger.error(f"Cypher query error: {e}")
            return []
            
    def _custom_query(self, graph: Any, query: str) -> List[Dict[str, Any]]:
        """Execute custom query"""
        
        # Simple pattern matching
        results = []
        
        if self.config.graph_type == "networkx":
            # Parse query and match patterns
            # This is a simplified implementation
            pass
            
        return results

class KnowledgeGraphVisualizer:
    """Visualize knowledge graphs"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        
    def visualize(self, graph: Any, save_path: str = None) -> None:
        """Visualize knowledge graph"""
        
        if self.config.visualization_method == "networkx":
            self._visualize_networkx(graph, save_path)
        elif self.config.visualization_method == "d3":
            self._visualize_d3(graph, save_path)
        elif self.config.visualization_method == "plotly":
            self._visualize_plotly(graph, save_path)
        else:
            raise ValueError(f"Unsupported visualization method: {self.config.visualization_method}")
            
    def _visualize_networkx(self, graph: nx.Graph, save_path: str = None) -> None:
        """Visualize using NetworkX"""
        
        plt.figure(figsize=(12, 8))
        
        # Choose layout
        if self.config.visualization_layout == "spring":
            pos = nx.spring_layout(graph)
        elif self.config.visualization_layout == "circular":
            pos = nx.circular_layout(graph)
        elif self.config.visualization_layout == "hierarchical":
            pos = nx.spring_layout(graph)
        else:
            pos = nx.spring_layout(graph)
            
        # Draw nodes
        node_sizes = self._calculate_node_sizes(graph)
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color='lightblue')
        
        # Draw edges
        edge_widths = self._calculate_edge_widths(graph)
        nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=8)
        
        plt.title("Knowledge Graph")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
        
    def _visualize_d3(self, graph: Any, save_path: str = None) -> None:
        """Visualize using D3.js"""
        
        # Convert graph to D3 format
        d3_data = self._convert_to_d3_format(graph)
        
        # Generate HTML with D3.js
        html_content = self._generate_d3_html(d3_data)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html_content)
        else:
            print("D3 visualization HTML generated")
            
    def _visualize_plotly(self, graph: nx.Graph, save_path: str = None) -> None:
        """Visualize using Plotly"""
        
        # Convert to Plotly format
        edge_trace, node_trace = self._convert_to_plotly_format(graph)
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Knowledge Graph',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Knowledge Graph Visualization",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="black", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
            
    def _calculate_node_sizes(self, graph: nx.Graph) -> List[int]:
        """Calculate node sizes for visualization"""
        
        if self.config.visualization_node_size == "degree":
            degrees = dict(graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            return [300 + (degrees[node] / max_degree) * 700 for node in graph.nodes()]
        else:
            return [500] * len(graph.nodes())
            
    def _calculate_edge_widths(self, graph: nx.Graph) -> List[float]:
        """Calculate edge widths for visualization"""
        
        if self.config.visualization_edge_width == "weight":
            weights = [graph[u][v].get('weight', 1.0) for u, v in graph.edges()]
            max_weight = max(weights) if weights else 1.0
            return [1.0 + (w / max_weight) * 3.0 for w in weights]
        else:
            return [1.0] * len(graph.edges())

class KnowledgeGraphEngine:
    """Main Knowledge Graph Engine"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self.entity_extractor = EntityExtractor(config)
        self.relation_extractor = RelationExtractor(config)
        self.graph_builder = KnowledgeGraphBuilder(config)
        self.embedder = KnowledgeGraphEmbedder(config)
        self.reasoner = KnowledgeGraphReasoner(config)
        self.querier = KnowledgeGraphQuerier(config)
        self.visualizer = KnowledgeGraphVisualizer(config)
        
        # Results storage
        self.results = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        
    def build_knowledge_graph(self, texts: List[str]) -> Any:
        """Build knowledge graph from texts"""
        
        all_entities = []
        all_relations = []
        
        # Extract entities and relations from each text
        for text in texts:
            entities = self.entity_extractor.extract_entities(text)
            relations = self.relation_extractor.extract_relations(text, entities)
            
            all_entities.extend(entities)
            all_relations.extend(relations)
            
        # Build knowledge graph
        graph = self.graph_builder.build_graph(all_entities, all_relations)
        
        # Generate embeddings
        embeddings = self.embedder.generate_embeddings(all_entities, all_relations)
        
        # Store results
        self.results['knowledge_graph'] = {
            'graph': graph,
            'entities': all_entities,
            'relations': all_relations,
            'embeddings': embeddings
        }
        
        return graph
        
    def query_knowledge_graph(self, graph: Any, query: str) -> List[Dict[str, Any]]:
        """Query knowledge graph"""
        
        results = self.querier.query(graph, query)
        
        # Store results
        self.results['queries'].append({
            'query': query,
            'results': results
        })
        
        return results
        
    def reason_knowledge_graph(self, graph: Any, query: str = None) -> Dict[str, Any]:
        """Perform reasoning on knowledge graph"""
        
        results = self.reasoner.reason(graph, query)
        
        # Store results
        self.results['reasoning'].append(results)
        
        return results
        
    def visualize_knowledge_graph(self, graph: Any, save_path: str = None) -> None:
        """Visualize knowledge graph"""
        
        self.visualizer.visualize(graph, save_path)
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        metrics = {
            'graph_type': self.config.graph_type,
            'entity_extraction_method': self.config.entity_extraction_method,
            'relation_extraction_method': self.config.relation_extraction_method,
            'reasoning_method': self.config.reasoning_method,
            'total_graphs_built': len(self.results.get('knowledge_graph', [])),
            'total_queries_executed': len(self.results.get('queries', [])),
            'total_reasoning_sessions': len(self.results.get('reasoning', []))
        }
        
        return metrics
        
    def save_knowledge_graph(self, graph: Any, filepath: str):
        """Save knowledge graph"""
        
        if self.config.graph_type == "rdf":
            graph.serialize(filepath, format=self.config.rdf_format)
        elif self.config.graph_type == "networkx":
            nx.write_gpickle(graph, filepath)
        else:
            logger.warning(f"Saving not supported for graph type: {self.config.graph_type}")
            
    def load_knowledge_graph(self, filepath: str) -> Any:
        """Load knowledge graph"""
        
        if self.config.graph_type == "rdf":
            graph = RDFGraph()
            graph.parse(filepath, format=self.config.rdf_format)
            return graph
        elif self.config.graph_type == "networkx":
            return nx.read_gpickle(filepath)
        else:
            logger.warning(f"Loading not supported for graph type: {self.config.graph_type}")
            return None

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test knowledge graph engine
    print("Testing Knowledge Graph Engine...")
    
    # Create config
    config = KnowledgeGraphConfig(
        graph_type="networkx",
        entity_extraction_method="spacy",
        relation_extraction_method="spacy",
        reasoning_method="rule_based",
        query_language="custom",
        visualization_method="networkx"
    )
    
    # Create engine
    kg_engine = KnowledgeGraphEngine(config)
    
    # Test texts
    texts = [
        "John works for Microsoft in Seattle.",
        "Microsoft is located in Redmond, Washington.",
        "Seattle is a city in Washington state.",
        "John is a software engineer at Microsoft."
    ]
    
    # Build knowledge graph
    print("Building knowledge graph...")
    graph = kg_engine.build_knowledge_graph(texts)
    print(f"Knowledge graph built: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    
    # Test reasoning
    print("Testing reasoning...")
    reasoning_results = kg_engine.reason_knowledge_graph(graph)
    print(f"Reasoning completed: {len(reasoning_results['inferred_facts'])} inferred facts")
    
    # Test visualization
    print("Testing visualization...")
    kg_engine.visualize_knowledge_graph(graph, "knowledge_graph.png")
    print("Visualization saved to knowledge_graph.png")
    
    # Get performance metrics
    metrics = kg_engine.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    print("\nKnowledge graph engine initialized successfully!")
























