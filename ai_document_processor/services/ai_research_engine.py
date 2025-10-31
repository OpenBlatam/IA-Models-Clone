"""
Motor de Investigación AI
========================

Motor para investigación automática, análisis de tendencias y descubrimiento de conocimiento.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from pathlib import Path
import hashlib
import aiohttp
import re

logger = logging.getLogger(__name__)

class ResearchType(str, Enum):
    """Tipos de investigación"""
    TREND_ANALYSIS = "trend_analysis"
    KNOWLEDGE_DISCOVERY = "knowledge_discovery"
    PATTERN_RECOGNITION = "pattern_recognition"
    CORRELATION_ANALYSIS = "correlation_analysis"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    COMPARATIVE_STUDY = "comparative_study"

class ResearchStatus(str, Enum):
    """Estados de investigación"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ResearchQuery:
    """Consulta de investigación"""
    id: str
    query_type: ResearchType
    keywords: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ResearchResult:
    """Resultado de investigación"""
    query_id: str
    research_type: ResearchType
    findings: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    trends: List[Dict[str, Any]] = field(default_factory=list)
    correlations: List[Dict[str, Any]] = field(default_factory=list)
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    data_sources: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class KnowledgeGraph:
    """Grafo de conocimiento"""
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AIResearchEngine:
    """Motor de investigación AI"""
    
    def __init__(self):
        self.research_queries: Dict[str, ResearchQuery] = {}
        self.research_results: Dict[str, ResearchResult] = {}
        self.knowledge_graphs: Dict[str, KnowledgeGraph] = {}
        self.running_research: Dict[str, asyncio.Task] = {}
        self.data_cache: Dict[str, Any] = {}
        
        # Configuración de APIs externas
        self.external_apis = {
            "arxiv": "http://export.arxiv.org/api/query",
            "pubmed": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            "google_scholar": "https://scholar.google.com/scholar",
            "news_api": "https://newsapi.org/v2/",
            "twitter_api": "https://api.twitter.com/2/"
        }
        
    async def initialize(self):
        """Inicializa el motor de investigación"""
        logger.info("Inicializando motor de investigación AI...")
        
        # Cargar conocimiento base
        await self._load_knowledge_base()
        
        # Inicializar APIs externas
        await self._initialize_external_apis()
        
        logger.info("Motor de investigación AI inicializado")
    
    async def _load_knowledge_base(self):
        """Carga base de conocimiento"""
        try:
            # Crear directorio de conocimiento
            knowledge_dir = Path("data/knowledge")
            knowledge_dir.mkdir(parents=True, exist_ok=True)
            
            # Cargar grafos de conocimiento existentes
            for graph_file in knowledge_dir.glob("*.json"):
                try:
                    with open(graph_file, 'r', encoding='utf-8') as f:
                        graph_data = json.load(f)
                    
                    graph_id = graph_file.stem
                    self.knowledge_graphs[graph_id] = KnowledgeGraph(
                        nodes=graph_data.get("nodes", {}),
                        edges=graph_data.get("edges", []),
                        metadata=graph_data.get("metadata", {})
                    )
                    
                except Exception as e:
                    logger.warning(f"Error cargando grafo de conocimiento {graph_file}: {e}")
            
            logger.info(f"Cargados {len(self.knowledge_graphs)} grafos de conocimiento")
            
        except Exception as e:
            logger.error(f"Error cargando base de conocimiento: {e}")
    
    async def _initialize_external_apis(self):
        """Inicializa APIs externas"""
        try:
            # Verificar disponibilidad de APIs
            self.api_available = {
                "arxiv": True,
                "pubmed": True,
                "google_scholar": False,  # Requiere scraping
                "news_api": False,  # Requiere API key
                "twitter_api": False  # Requiere API key
            }
            
            logger.info("APIs externas inicializadas")
            
        except Exception as e:
            logger.error(f"Error inicializando APIs externas: {e}")
    
    async def start_research(
        self,
        research_type: ResearchType,
        keywords: List[str],
        parameters: Dict[str, Any] = None,
        filters: Dict[str, Any] = None
    ) -> str:
        """Inicia una investigación"""
        try:
            query_id = f"{research_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            query = ResearchQuery(
                id=query_id,
                query_type=research_type,
                keywords=keywords,
                parameters=parameters or {},
                filters=filters or {}
            )
            
            self.research_queries[query_id] = query
            
            # Ejecutar investigación
            task = asyncio.create_task(self._execute_research(query))
            self.running_research[query_id] = task
            
            logger.info(f"Investigación iniciada: {query_id}")
            return query_id
            
        except Exception as e:
            logger.error(f"Error iniciando investigación: {e}")
            raise
    
    async def _execute_research(self, query: ResearchQuery):
        """Ejecuta una investigación específica"""
        try:
            result = ResearchResult(
                query_id=query.id,
                research_type=query.query_type
            )
            
            # Ejecutar investigación según tipo
            if query.query_type == ResearchType.TREND_ANALYSIS:
                await self._analyze_trends(query, result)
            elif query.query_type == ResearchType.KNOWLEDGE_DISCOVERY:
                await self._discover_knowledge(query, result)
            elif query.query_type == ResearchType.PATTERN_RECOGNITION:
                await self._recognize_patterns(query, result)
            elif query.query_type == ResearchType.CORRELATION_ANALYSIS:
                await self._analyze_correlations(query, result)
            elif query.query_type == ResearchType.PREDICTIVE_ANALYSIS:
                await self._predictive_analysis(query, result)
            elif query.query_type == ResearchType.COMPARATIVE_STUDY:
                await self._comparative_study(query, result)
            
            # Guardar resultado
            self.research_results[query.id] = result
            
            # Actualizar grafo de conocimiento
            await self._update_knowledge_graph(query, result)
            
            logger.info(f"Investigación completada: {query.id}")
            
        except Exception as e:
            logger.error(f"Error ejecutando investigación {query.id}: {e}")
            if query.id in self.research_results:
                self.research_results[query.id].confidence_score = 0.0
    
    async def _analyze_trends(self, query: ResearchQuery, result: ResearchResult):
        """Analiza tendencias"""
        try:
            # Buscar información en múltiples fuentes
            sources_data = []
            
            # ArXiv (artículos académicos)
            if self.api_available["arxiv"]:
                arxiv_data = await self._search_arxiv(query.keywords)
                sources_data.extend(arxiv_data)
                result.data_sources.append("arxiv")
            
            # PubMed (artículos médicos)
            if self.api_available["pubmed"]:
                pubmed_data = await self._search_pubmed(query.keywords)
                sources_data.extend(pubmed_data)
                result.data_sources.append("pubmed")
            
            # Análisis de tendencias temporales
            trends = await self._extract_temporal_trends(sources_data, query.parameters)
            result.trends = trends
            
            # Generar insights
            insights = await self._generate_trend_insights(trends, query.keywords)
            result.insights = insights
            
            # Calcular confianza
            result.confidence_score = min(0.9, len(sources_data) / 100.0)
            
            logger.info(f"Análisis de tendencias completado: {len(trends)} tendencias encontradas")
            
        except Exception as e:
            logger.error(f"Error analizando tendencias: {e}")
            raise
    
    async def _discover_knowledge(self, query: ResearchQuery, result: ResearchResult):
        """Descubre conocimiento"""
        try:
            # Buscar conexiones en grafos de conocimiento existentes
            knowledge_connections = []
            
            for graph_id, graph in self.knowledge_graphs.items():
                connections = await self._find_knowledge_connections(graph, query.keywords)
                knowledge_connections.extend(connections)
            
            # Buscar nuevos patrones
            new_patterns = await self._discover_new_patterns(query.keywords, query.parameters)
            
            # Generar hallazgos
            findings = []
            for connection in knowledge_connections:
                findings.append({
                    "type": "knowledge_connection",
                    "description": connection.get("description", ""),
                    "confidence": connection.get("confidence", 0.0),
                    "source": connection.get("source", "unknown")
                })
            
            for pattern in new_patterns:
                findings.append({
                    "type": "new_pattern",
                    "description": pattern.get("description", ""),
                    "confidence": pattern.get("confidence", 0.0),
                    "evidence": pattern.get("evidence", [])
                })
            
            result.findings = findings
            
            # Generar insights
            insights = await self._generate_knowledge_insights(findings, query.keywords)
            result.insights = insights
            
            # Calcular confianza
            result.confidence_score = min(0.95, len(findings) / 50.0)
            
            logger.info(f"Descubrimiento de conocimiento completado: {len(findings)} hallazgos")
            
        except Exception as e:
            logger.error(f"Error descubriendo conocimiento: {e}")
            raise
    
    async def _recognize_patterns(self, query: ResearchQuery, result: ResearchResult):
        """Reconoce patrones"""
        try:
            # Recopilar datos de múltiples fuentes
            data_sources = await self._collect_pattern_data(query.keywords)
            
            # Aplicar algoritmos de reconocimiento de patrones
            patterns = []
            
            # Patrones temporales
            temporal_patterns = await self._find_temporal_patterns(data_sources)
            patterns.extend(temporal_patterns)
            
            # Patrones de frecuencia
            frequency_patterns = await self._find_frequency_patterns(data_sources)
            patterns.extend(frequency_patterns)
            
            # Patrones de co-ocurrencia
            cooccurrence_patterns = await self._find_cooccurrence_patterns(data_sources)
            patterns.extend(cooccurrence_patterns)
            
            # Patrones semánticos
            semantic_patterns = await self._find_semantic_patterns(data_sources)
            patterns.extend(semantic_patterns)
            
            result.findings = patterns
            
            # Generar insights
            insights = await self._generate_pattern_insights(patterns, query.keywords)
            result.insights = insights
            
            # Calcular confianza
            result.confidence_score = min(0.9, len(patterns) / 30.0)
            
            logger.info(f"Reconocimiento de patrones completado: {len(patterns)} patrones encontrados")
            
        except Exception as e:
            logger.error(f"Error reconociendo patrones: {e}")
            raise
    
    async def _analyze_correlations(self, query: ResearchQuery, result: ResearchResult):
        """Analiza correlaciones"""
        try:
            # Recopilar datos para análisis de correlación
            correlation_data = await self._collect_correlation_data(query.keywords)
            
            # Calcular correlaciones
            correlations = []
            
            # Correlaciones entre variables
            variable_correlations = await self._calculate_variable_correlations(correlation_data)
            correlations.extend(variable_correlations)
            
            # Correlaciones temporales
            temporal_correlations = await self._calculate_temporal_correlations(correlation_data)
            correlations.extend(temporal_correlations)
            
            # Correlaciones causales
            causal_correlations = await self._identify_causal_correlations(correlation_data)
            correlations.extend(causal_correlations)
            
            result.correlations = correlations
            
            # Generar insights
            insights = await self._generate_correlation_insights(correlations, query.keywords)
            result.insights = insights
            
            # Calcular confianza
            result.confidence_score = min(0.85, len(correlations) / 20.0)
            
            logger.info(f"Análisis de correlaciones completado: {len(correlations)} correlaciones encontradas")
            
        except Exception as e:
            logger.error(f"Error analizando correlaciones: {e}")
            raise
    
    async def _predictive_analysis(self, query: ResearchQuery, result: ResearchResult):
        """Análisis predictivo"""
        try:
            # Recopilar datos históricos
            historical_data = await self._collect_historical_data(query.keywords)
            
            # Aplicar modelos predictivos
            predictions = []
            
            # Predicciones de tendencias
            trend_predictions = await self._predict_trends(historical_data, query.parameters)
            predictions.extend(trend_predictions)
            
            # Predicciones de eventos
            event_predictions = await self._predict_events(historical_data, query.parameters)
            predictions.extend(event_predictions)
            
            # Predicciones de impacto
            impact_predictions = await self._predict_impact(historical_data, query.parameters)
            predictions.extend(impact_predictions)
            
            result.predictions = predictions
            
            # Generar insights
            insights = await self._generate_predictive_insights(predictions, query.keywords)
            result.insights = insights
            
            # Calcular confianza
            result.confidence_score = min(0.8, len(predictions) / 15.0)
            
            logger.info(f"Análisis predictivo completado: {len(predictions)} predicciones generadas")
            
        except Exception as e:
            logger.error(f"Error en análisis predictivo: {e}")
            raise
    
    async def _comparative_study(self, query: ResearchQuery, result: ResearchResult):
        """Estudio comparativo"""
        try:
            # Recopilar datos para comparación
            comparison_data = await self._collect_comparison_data(query.keywords)
            
            # Realizar comparaciones
            comparisons = []
            
            # Comparación temporal
            temporal_comparisons = await self._compare_temporal_data(comparison_data)
            comparisons.extend(temporal_comparisons)
            
            # Comparación entre fuentes
            source_comparisons = await self._compare_sources(comparison_data)
            comparisons.extend(source_comparisons)
            
            # Comparación de metodologías
            methodology_comparisons = await self._compare_methodologies(comparison_data)
            comparisons.extend(methodology_comparisons)
            
            result.findings = comparisons
            
            # Generar insights
            insights = await self._generate_comparative_insights(comparisons, query.keywords)
            result.insights = insights
            
            # Calcular confianza
            result.confidence_score = min(0.9, len(comparisons) / 25.0)
            
            logger.info(f"Estudio comparativo completado: {len(comparisons)} comparaciones realizadas")
            
        except Exception as e:
            logger.error(f"Error en estudio comparativo: {e}")
            raise
    
    # Métodos de búsqueda en APIs externas
    async def _search_arxiv(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Busca en ArXiv"""
        try:
            if not self.api_available["arxiv"]:
                return []
            
            query = " OR ".join(keywords)
            url = f"{self.external_apis['arxiv']}?search_query={query}&max_results=50"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Parsear XML (simplificado)
                        papers = self._parse_arxiv_xml(content)
                        return papers
            
            return []
            
        except Exception as e:
            logger.error(f"Error buscando en ArXiv: {e}")
            return []
    
    async def _search_pubmed(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Busca en PubMed"""
        try:
            if not self.api_available["pubmed"]:
                return []
            
            # Implementación simplificada
            query = " ".join(keywords)
            # En implementación real, usar E-utilities de PubMed
            
            # Simular resultados
            papers = []
            for i in range(10):
                papers.append({
                    "title": f"Research paper about {query} - {i}",
                    "abstract": f"This paper discusses {query} and its implications...",
                    "authors": [f"Author {i+1}", f"Author {i+2}"],
                    "year": 2023 - i,
                    "source": "pubmed"
                })
            
            return papers
            
        except Exception as e:
            logger.error(f"Error buscando en PubMed: {e}")
            return []
    
    def _parse_arxiv_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parsea XML de ArXiv"""
        try:
            # Implementación simplificada
            papers = []
            
            # En implementación real, usar xml.etree.ElementTree
            # Por ahora, simular resultados
            for i in range(20):
                papers.append({
                    "title": f"ArXiv paper about AI research - {i}",
                    "abstract": f"This paper presents research on artificial intelligence and machine learning...",
                    "authors": [f"Researcher {i+1}"],
                    "year": 2023 - (i % 5),
                    "source": "arxiv"
                })
            
            return papers
            
        except Exception as e:
            logger.error(f"Error parseando XML de ArXiv: {e}")
            return []
    
    # Métodos de análisis
    async def _extract_temporal_trends(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrae tendencias temporales"""
        try:
            trends = []
            
            # Agrupar por año
            yearly_data = {}
            for item in data:
                year = item.get("year", 2023)
                if year not in yearly_data:
                    yearly_data[year] = []
                yearly_data[year].append(item)
            
            # Analizar tendencias
            for year in sorted(yearly_data.keys()):
                count = len(yearly_data[year])
                trends.append({
                    "year": year,
                    "count": count,
                    "trend_type": "publication_count",
                    "description": f"Publications in {year}: {count}"
                })
            
            return trends
            
        except Exception as e:
            logger.error(f"Error extrayendo tendencias temporales: {e}")
            return []
    
    async def _find_knowledge_connections(self, graph: KnowledgeGraph, keywords: List[str]) -> List[Dict[str, Any]]:
        """Encuentra conexiones de conocimiento"""
        try:
            connections = []
            
            # Buscar nodos relacionados con keywords
            for keyword in keywords:
                for node_id, node_data in graph.nodes.items():
                    if keyword.lower() in node_data.get("name", "").lower():
                        connections.append({
                            "type": "node_connection",
                            "node_id": node_id,
                            "keyword": keyword,
                            "description": f"Found connection to {node_data.get('name', '')}",
                            "confidence": 0.8
                        })
            
            return connections
            
        except Exception as e:
            logger.error(f"Error encontrando conexiones de conocimiento: {e}")
            return []
    
    async def _discover_new_patterns(self, keywords: List[str], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Descubre nuevos patrones"""
        try:
            patterns = []
            
            # Simular descubrimiento de patrones
            for i, keyword in enumerate(keywords):
                patterns.append({
                    "type": "semantic_pattern",
                    "description": f"Pattern discovered for {keyword}",
                    "confidence": 0.7 + (i * 0.05),
                    "evidence": [f"Evidence {j}" for j in range(3)]
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error descubriendo patrones: {e}")
            return []
    
    async def _collect_pattern_data(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Recopila datos para análisis de patrones"""
        try:
            data = []
            
            # Simular recopilación de datos
            for keyword in keywords:
                for i in range(10):
                    data.append({
                        "keyword": keyword,
                        "timestamp": datetime.now() - timedelta(days=i),
                        "frequency": np.random.randint(1, 100),
                        "context": f"Context for {keyword} - {i}"
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"Error recopilando datos de patrones: {e}")
            return []
    
    async def _find_temporal_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encuentra patrones temporales"""
        try:
            patterns = []
            
            # Agrupar por keyword
            keyword_data = {}
            for item in data:
                keyword = item["keyword"]
                if keyword not in keyword_data:
                    keyword_data[keyword] = []
                keyword_data[keyword].append(item)
            
            # Analizar patrones temporales
            for keyword, items in keyword_data.items():
                frequencies = [item["frequency"] for item in items]
                if len(frequencies) > 1:
                    trend = "increasing" if frequencies[-1] > frequencies[0] else "decreasing"
                    patterns.append({
                        "type": "temporal_pattern",
                        "keyword": keyword,
                        "trend": trend,
                        "description": f"Temporal pattern for {keyword}: {trend}",
                        "confidence": 0.75
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error encontrando patrones temporales: {e}")
            return []
    
    async def _find_frequency_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encuentra patrones de frecuencia"""
        try:
            patterns = []
            
            # Analizar frecuencias
            frequencies = [item["frequency"] for item in data]
            if frequencies:
                avg_freq = np.mean(frequencies)
                high_freq_items = [item for item in data if item["frequency"] > avg_freq * 1.5]
                
                for item in high_freq_items:
                    patterns.append({
                        "type": "frequency_pattern",
                        "keyword": item["keyword"],
                        "frequency": item["frequency"],
                        "description": f"High frequency pattern for {item['keyword']}",
                        "confidence": 0.8
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error encontrando patrones de frecuencia: {e}")
            return []
    
    async def _find_cooccurrence_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encuentra patrones de co-ocurrencia"""
        try:
            patterns = []
            
            # Simular análisis de co-ocurrencia
            keywords = list(set(item["keyword"] for item in data))
            for i, keyword1 in enumerate(keywords):
                for keyword2 in keywords[i+1:]:
                    patterns.append({
                        "type": "cooccurrence_pattern",
                        "keywords": [keyword1, keyword2],
                        "description": f"Co-occurrence pattern between {keyword1} and {keyword2}",
                        "confidence": 0.7
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error encontrando patrones de co-ocurrencia: {e}")
            return []
    
    async def _find_semantic_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encuentra patrones semánticos"""
        try:
            patterns = []
            
            # Simular análisis semántico
            for item in data:
                patterns.append({
                    "type": "semantic_pattern",
                    "keyword": item["keyword"],
                    "description": f"Semantic pattern for {item['keyword']}",
                    "confidence": 0.6
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error encontrando patrones semánticos: {e}")
            return []
    
    # Métodos de generación de insights
    async def _generate_trend_insights(self, trends: List[Dict[str, Any]], keywords: List[str]) -> List[str]:
        """Genera insights de tendencias"""
        try:
            insights = []
            
            if trends:
                latest_trend = max(trends, key=lambda x: x.get("year", 0))
                insights.append(f"Latest trend shows {latest_trend['count']} publications in {latest_trend['year']}")
                
                total_publications = sum(trend["count"] for trend in trends)
                insights.append(f"Total publications found: {total_publications}")
                
                insights.append(f"Research areas covered: {', '.join(keywords)}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generando insights de tendencias: {e}")
            return []
    
    async def _generate_knowledge_insights(self, findings: List[Dict[str, Any]], keywords: List[str]) -> List[str]:
        """Genera insights de conocimiento"""
        try:
            insights = []
            
            if findings:
                insights.append(f"Found {len(findings)} knowledge connections")
                
                high_confidence = [f for f in findings if f.get("confidence", 0) > 0.8]
                if high_confidence:
                    insights.append(f"{len(high_confidence)} high-confidence findings identified")
                
                insights.append(f"Knowledge domains: {', '.join(keywords)}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generando insights de conocimiento: {e}")
            return []
    
    async def _generate_pattern_insights(self, patterns: List[Dict[str, Any]], keywords: List[str]) -> List[str]:
        """Genera insights de patrones"""
        try:
            insights = []
            
            if patterns:
                pattern_types = {}
                for pattern in patterns:
                    pattern_type = pattern.get("type", "unknown")
                    pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
                
                insights.append(f"Discovered {len(patterns)} patterns across {len(pattern_types)} types")
                
                for pattern_type, count in pattern_types.items():
                    insights.append(f"{count} {pattern_type} patterns found")
                
                insights.append(f"Pattern analysis for: {', '.join(keywords)}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generando insights de patrones: {e}")
            return []
    
    async def _generate_correlation_insights(self, correlations: List[Dict[str, Any]], keywords: List[str]) -> List[str]:
        """Genera insights de correlaciones"""
        try:
            insights = []
            
            if correlations:
                insights.append(f"Identified {len(correlations)} correlations")
                
                strong_correlations = [c for c in correlations if c.get("strength", 0) > 0.7]
                if strong_correlations:
                    insights.append(f"{len(strong_correlations)} strong correlations found")
                
                insights.append(f"Correlation analysis for: {', '.join(keywords)}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generando insights de correlaciones: {e}")
            return []
    
    async def _generate_predictive_insights(self, predictions: List[Dict[str, Any]], keywords: List[str]) -> List[str]:
        """Genera insights predictivos"""
        try:
            insights = []
            
            if predictions:
                insights.append(f"Generated {len(predictions)} predictions")
                
                high_confidence_preds = [p for p in predictions if p.get("confidence", 0) > 0.8]
                if high_confidence_preds:
                    insights.append(f"{len(high_confidence_preds)} high-confidence predictions")
                
                insights.append(f"Predictive analysis for: {', '.join(keywords)}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generando insights predictivos: {e}")
            return []
    
    async def _generate_comparative_insights(self, comparisons: List[Dict[str, Any]], keywords: List[str]) -> List[str]:
        """Genera insights comparativos"""
        try:
            insights = []
            
            if comparisons:
                insights.append(f"Conducted {len(comparisons)} comparisons")
                
                insights.append(f"Comparative analysis for: {', '.join(keywords)}")
                
                if len(comparisons) > 1:
                    insights.append("Multiple comparison perspectives available")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generando insights comparativos: {e}")
            return []
    
    # Métodos auxiliares
    async def _collect_correlation_data(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Recopila datos para análisis de correlación"""
        return await self._collect_pattern_data(keywords)
    
    async def _collect_historical_data(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Recopila datos históricos"""
        return await self._collect_pattern_data(keywords)
    
    async def _collect_comparison_data(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Recopila datos para comparación"""
        return await self._collect_pattern_data(keywords)
    
    async def _calculate_variable_correlations(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calcula correlaciones entre variables"""
        try:
            correlations = []
            
            # Simular cálculo de correlaciones
            for i in range(5):
                correlations.append({
                    "variable1": f"var_{i}",
                    "variable2": f"var_{i+1}",
                    "correlation": np.random.uniform(-1, 1),
                    "strength": abs(np.random.uniform(-1, 1)),
                    "description": f"Correlation between var_{i} and var_{i+1}"
                })
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculando correlaciones de variables: {e}")
            return []
    
    async def _calculate_temporal_correlations(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calcula correlaciones temporales"""
        try:
            correlations = []
            
            # Simular correlaciones temporales
            for i in range(3):
                correlations.append({
                    "type": "temporal",
                    "lag": i + 1,
                    "correlation": np.random.uniform(-1, 1),
                    "description": f"Temporal correlation with lag {i+1}"
                })
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculando correlaciones temporales: {e}")
            return []
    
    async def _identify_causal_correlations(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identifica correlaciones causales"""
        try:
            correlations = []
            
            # Simular correlaciones causales
            for i in range(2):
                correlations.append({
                    "type": "causal",
                    "cause": f"factor_{i}",
                    "effect": f"outcome_{i}",
                    "strength": np.random.uniform(0.5, 1.0),
                    "description": f"Causal relationship between factor_{i} and outcome_{i}"
                })
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error identificando correlaciones causales: {e}")
            return []
    
    async def _predict_trends(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predice tendencias"""
        try:
            predictions = []
            
            # Simular predicciones de tendencias
            for i in range(3):
                predictions.append({
                    "type": "trend",
                    "timeframe": f"{i+1} year(s)",
                    "prediction": f"Trend prediction {i+1}",
                    "confidence": np.random.uniform(0.6, 0.9),
                    "description": f"Predicted trend for {i+1} year(s) ahead"
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error prediciendo tendencias: {e}")
            return []
    
    async def _predict_events(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predice eventos"""
        try:
            predictions = []
            
            # Simular predicciones de eventos
            for i in range(2):
                predictions.append({
                    "type": "event",
                    "event_type": f"event_type_{i}",
                    "probability": np.random.uniform(0.3, 0.8),
                    "timeframe": f"{i+1} month(s)",
                    "description": f"Predicted event of type {i}"
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error prediciendo eventos: {e}")
            return []
    
    async def _predict_impact(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predice impacto"""
        try:
            predictions = []
            
            # Simular predicciones de impacto
            for i in range(2):
                predictions.append({
                    "type": "impact",
                    "impact_level": ["low", "medium", "high"][i % 3],
                    "confidence": np.random.uniform(0.5, 0.9),
                    "description": f"Predicted impact level {i}"
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error prediciendo impacto: {e}")
            return []
    
    async def _compare_temporal_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compara datos temporales"""
        try:
            comparisons = []
            
            # Simular comparaciones temporales
            for i in range(3):
                comparisons.append({
                    "type": "temporal_comparison",
                    "period1": f"period_{i}",
                    "period2": f"period_{i+1}",
                    "difference": np.random.uniform(-50, 50),
                    "description": f"Comparison between period_{i} and period_{i+1}"
                })
            
            return comparisons
            
        except Exception as e:
            logger.error(f"Error comparando datos temporales: {e}")
            return []
    
    async def _compare_sources(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compara fuentes"""
        try:
            comparisons = []
            
            # Simular comparaciones de fuentes
            sources = ["arxiv", "pubmed", "google_scholar"]
            for i, source1 in enumerate(sources):
                for source2 in sources[i+1:]:
                    comparisons.append({
                        "type": "source_comparison",
                        "source1": source1,
                        "source2": source2,
                        "similarity": np.random.uniform(0.3, 0.9),
                        "description": f"Comparison between {source1} and {source2}"
                    })
            
            return comparisons
            
        except Exception as e:
            logger.error(f"Error comparando fuentes: {e}")
            return []
    
    async def _compare_methodologies(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compara metodologías"""
        try:
            comparisons = []
            
            # Simular comparaciones de metodologías
            methodologies = ["quantitative", "qualitative", "mixed"]
            for i, method1 in enumerate(methodologies):
                for method2 in methodologies[i+1:]:
                    comparisons.append({
                        "type": "methodology_comparison",
                        "method1": method1,
                        "method2": method2,
                        "effectiveness": np.random.uniform(0.4, 0.95),
                        "description": f"Comparison between {method1} and {method2} methodologies"
                    })
            
            return comparisons
            
        except Exception as e:
            logger.error(f"Error comparando metodologías: {e}")
            return []
    
    async def _update_knowledge_graph(self, query: ResearchQuery, result: ResearchResult):
        """Actualiza grafo de conocimiento"""
        try:
            graph_id = f"research_{query.query_type.value}"
            
            if graph_id not in self.knowledge_graphs:
                self.knowledge_graphs[graph_id] = KnowledgeGraph()
            
            graph = self.knowledge_graphs[graph_id]
            
            # Agregar nodos para keywords
            for keyword in query.keywords:
                node_id = hashlib.md5(keyword.encode()).hexdigest()[:8]
                graph.nodes[node_id] = {
                    "name": keyword,
                    "type": "keyword",
                    "research_count": graph.nodes.get(node_id, {}).get("research_count", 0) + 1,
                    "last_updated": datetime.now().isoformat()
                }
            
            # Agregar nodos para insights
            for i, insight in enumerate(result.insights):
                node_id = f"insight_{i}_{datetime.now().strftime('%Y%m%d')}"
                graph.nodes[node_id] = {
                    "name": insight[:50] + "..." if len(insight) > 50 else insight,
                    "type": "insight",
                    "full_text": insight,
                    "confidence": result.confidence_score,
                    "created_at": datetime.now().isoformat()
                }
            
            # Guardar grafo actualizado
            await self._save_knowledge_graph(graph_id, graph)
            
        except Exception as e:
            logger.error(f"Error actualizando grafo de conocimiento: {e}")
    
    async def _save_knowledge_graph(self, graph_id: str, graph: KnowledgeGraph):
        """Guarda grafo de conocimiento"""
        try:
            knowledge_dir = Path("data/knowledge")
            knowledge_dir.mkdir(parents=True, exist_ok=True)
            
            graph_data = {
                "nodes": graph.nodes,
                "edges": graph.edges,
                "metadata": {
                    **graph.metadata,
                    "last_updated": datetime.now().isoformat(),
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edges)
                }
            }
            
            graph_file = knowledge_dir / f"{graph_id}.json"
            with open(graph_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"Error guardando grafo de conocimiento: {e}")
    
    async def get_research_status(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de investigación"""
        try:
            if query_id not in self.research_queries:
                return None
            
            query = self.research_queries[query_id]
            result = self.research_results.get(query_id)
            
            return {
                "query_id": query.id,
                "research_type": query.query_type.value,
                "keywords": query.keywords,
                "status": "completed" if result else "running",
                "created_at": query.created_at.isoformat(),
                "result": {
                    "findings_count": len(result.findings) if result else 0,
                    "insights_count": len(result.insights) if result else 0,
                    "confidence_score": result.confidence_score if result else 0.0,
                    "data_sources": result.data_sources if result else []
                } if result else None
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado de investigación: {e}")
            return None
    
    async def get_research_results(self, query_id: str) -> Optional[ResearchResult]:
        """Obtiene resultados de investigación"""
        try:
            return self.research_results.get(query_id)
        except Exception as e:
            logger.error(f"Error obteniendo resultados de investigación: {e}")
            return None
    
    async def get_knowledge_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """Obtiene grafo de conocimiento"""
        try:
            return self.knowledge_graphs.get(graph_id)
        except Exception as e:
            logger.error(f"Error obteniendo grafo de conocimiento: {e}")
            return None
    
    async def search_knowledge_graph(self, graph_id: str, query: str) -> List[Dict[str, Any]]:
        """Busca en grafo de conocimiento"""
        try:
            if graph_id not in self.knowledge_graphs:
                return []
            
            graph = self.knowledge_graphs[graph_id]
            results = []
            
            query_lower = query.lower()
            
            for node_id, node_data in graph.nodes.items():
                if (query_lower in node_data.get("name", "").lower() or
                    query_lower in node_data.get("full_text", "").lower()):
                    results.append({
                        "node_id": node_id,
                        "name": node_data.get("name", ""),
                        "type": node_data.get("type", ""),
                        "relevance": 0.8  # Simulado
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error buscando en grafo de conocimiento: {e}")
            return []


