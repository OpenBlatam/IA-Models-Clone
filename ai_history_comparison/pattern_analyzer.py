"""
Text Pattern Analysis System for AI History Comparison
Sistema de análisis de patrones de texto para análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import Counter, defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Tipos de patrones"""
    LINGUISTIC = "linguistic"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    QUALITY = "quality"
    STYLE = "style"

class PatternCategory(Enum):
    """Categorías de patrones"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    ANOMALY = "anomaly"
    TREND = "trend"

@dataclass
class TextPattern:
    """Patrón de texto identificado"""
    id: str
    type: PatternType
    category: PatternCategory
    name: str
    description: str
    pattern: str
    frequency: int
    confidence: float
    impact_score: float
    examples: List[str]
    documents: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PatternCluster:
    """Cluster de patrones similares"""
    id: str
    name: str
    patterns: List[TextPattern]
    centroid: np.ndarray
    size: int
    coherence: float
    representative_pattern: TextPattern
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PatternEvolution:
    """Evolución de patrones en el tiempo"""
    pattern_id: str
    time_periods: List[str]
    frequencies: List[int]
    trends: Dict[str, float]
    peak_period: str
    decline_period: Optional[str]
    stability_score: float

class TextPatternAnalyzer:
    """
    Analizador de patrones de texto
    """
    
    def __init__(
        self,
        min_pattern_frequency: int = 3,
        similarity_threshold: float = 0.7,
        enable_clustering: bool = True,
        enable_evolution_analysis: bool = True
    ):
        self.min_pattern_frequency = min_pattern_frequency
        self.similarity_threshold = similarity_threshold
        self.enable_clustering = enable_clustering
        self.enable_evolution_analysis = enable_evolution_analysis
        
        # Almacenamiento de patrones
        self.patterns: Dict[str, TextPattern] = {}
        self.pattern_clusters: Dict[str, PatternCluster] = {}
        self.pattern_evolutions: Dict[str, PatternEvolution] = {}
        
        # Configuración de análisis
        self.config = {
            "linguistic_patterns": {
                "min_length": 3,
                "max_length": 50,
                "include_punctuation": False
            },
            "structural_patterns": {
                "sentence_patterns": True,
                "paragraph_patterns": True,
                "list_patterns": True
            },
            "semantic_patterns": {
                "topic_modeling": True,
                "keyword_clustering": True,
                "concept_extraction": True
            },
            "quality_patterns": {
                "quality_indicators": True,
                "error_patterns": True,
                "improvement_patterns": True
            }
        }
        
        # Patrones predefinidos
        self._initialize_predefined_patterns()
    
    def _initialize_predefined_patterns(self):
        """Inicializar patrones predefinidos"""
        # Patrones de calidad
        self.quality_patterns = {
            "excellent_indicators": [
                r"\b(excelente|outstanding|exceptional|superior|premium)\b",
                r"\b(high.?quality|top.?notch|first.?class)\b",
                r"\b(perfect|flawless|impeccable)\b"
            ],
            "poor_indicators": [
                r"\b(poor|bad|terrible|awful|horrible)\b",
                r"\b(low.?quality|substandard|inferior)\b",
                r"\b(defective|flawed|problematic)\b"
            ],
            "improvement_indicators": [
                r"\b(improve|enhance|optimize|upgrade|refine)\b",
                r"\b(better|best|optimal|ideal)\b",
                r"\b(progress|advance|develop|evolve)\b"
            ]
        }
        
        # Patrones estructurales
        self.structural_patterns = {
            "list_patterns": [
                r"^\s*[\d\w]+[\.\)]\s+",  # Lista numerada
                r"^\s*[-*•]\s+",  # Lista con viñetas
                r"^\s*[a-zA-Z][\.\)]\s+"  # Lista alfabética
            ],
            "heading_patterns": [
                r"^#{1,6}\s+",  # Markdown headers
                r"^[A-Z][A-Z\s]+$",  # Títulos en mayúsculas
                r"^\d+\.\s+[A-Z]"  # Títulos numerados
            ],
            "question_patterns": [
                r"\?$",  # Preguntas
                r"^(what|how|why|when|where|who|which)\b",
                r"^(can|could|would|should|will|shall)\b"
            ]
        }
        
        # Patrones lingüísticos
        self.linguistic_patterns = {
            "formal_language": [
                r"\b(therefore|however|furthermore|moreover|consequently)\b",
                r"\b(in addition|on the other hand|as a result)\b",
                r"\b(it is important to note|it should be noted)\b"
            ],
            "informal_language": [
                r"\b(yeah|yep|nope|gonna|wanna|gotta)\b",
                r"\b(awesome|cool|amazing|fantastic)\b",
                r"\b(btw|fyi|lol|omg)\b"
            ],
            "technical_language": [
                r"\b(algorithm|implementation|optimization|configuration)\b",
                r"\b(API|database|framework|architecture)\b",
                r"\b(performance|scalability|efficiency|reliability)\b"
            ]
        }
    
    async def analyze_documents(
        self,
        documents: List[Dict[str, Any]],
        pattern_types: List[PatternType] = None
    ) -> Dict[str, Any]:
        """
        Analizar patrones en documentos
        
        Args:
            documents: Lista de documentos a analizar
            pattern_types: Tipos de patrones a buscar
            
        Returns:
            Resultados del análisis de patrones
        """
        if pattern_types is None:
            pattern_types = [
                PatternType.LINGUISTIC,
                PatternType.STRUCTURAL,
                PatternType.SEMANTIC,
                PatternType.QUALITY
            ]
        
        try:
            logger.info(f"Analyzing patterns in {len(documents)} documents")
            
            results = {}
            
            # Preparar datos
            texts = [doc.get("content", "") for doc in documents]
            doc_ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]
            
            # Analizar cada tipo de patrón
            for pattern_type in pattern_types:
                if pattern_type == PatternType.LINGUISTIC:
                    results["linguistic"] = await self._analyze_linguistic_patterns(texts, doc_ids)
                elif pattern_type == PatternType.STRUCTURAL:
                    results["structural"] = await self._analyze_structural_patterns(texts, doc_ids)
                elif pattern_type == PatternType.SEMANTIC:
                    results["semantic"] = await self._analyze_semantic_patterns(texts, doc_ids)
                elif pattern_type == PatternType.QUALITY:
                    results["quality"] = await self._analyze_quality_patterns(texts, doc_ids)
                elif pattern_type == PatternType.STYLE:
                    results["style"] = await self._analyze_style_patterns(texts, doc_ids)
            
            # Clustering de patrones si está habilitado
            if self.enable_clustering:
                results["clusters"] = await self._cluster_patterns()
            
            # Análisis de evolución si está habilitado
            if self.enable_evolution_analysis:
                results["evolution"] = await self._analyze_pattern_evolution(documents)
            
            logger.info("Pattern analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return {"error": str(e)}
    
    async def _analyze_linguistic_patterns(self, texts: List[str], doc_ids: List[str]) -> Dict[str, Any]:
        """Analizar patrones lingüísticos"""
        patterns_found = []
        
        for pattern_name, pattern_list in self.linguistic_patterns.items():
            for pattern in pattern_list:
                matches = []
                for i, text in enumerate(texts):
                    found_matches = re.findall(pattern, text, re.IGNORECASE)
                    if found_matches:
                        matches.extend([(doc_ids[i], match) for match in found_matches])
                
                if len(matches) >= self.min_pattern_frequency:
                    pattern_id = f"linguistic_{pattern_name}_{len(patterns_found)}"
                    
                    # Determinar categoría
                    if "formal" in pattern_name:
                        category = PatternCategory.POSITIVE
                    elif "informal" in pattern_name:
                        category = PatternCategory.NEGATIVE
                    else:
                        category = PatternCategory.NEUTRAL
                    
                    text_pattern = TextPattern(
                        id=pattern_id,
                        type=PatternType.LINGUISTIC,
                        category=category,
                        name=f"Linguistic Pattern: {pattern_name}",
                        description=f"Pattern for {pattern_name} language usage",
                        pattern=pattern,
                        frequency=len(matches),
                        confidence=min(len(matches) / 10, 1.0),
                        impact_score=0.5,
                        examples=[match[1] for match in matches[:5]],
                        documents=[match[0] for match in matches]
                    )
                    
                    patterns_found.append(text_pattern)
                    self.patterns[pattern_id] = text_pattern
        
        return {
            "patterns": [self._pattern_to_dict(p) for p in patterns_found],
            "total_patterns": len(patterns_found),
            "pattern_types": list(set(p.category.value for p in patterns_found))
        }
    
    async def _analyze_structural_patterns(self, texts: List[str], doc_ids: List[str]) -> Dict[str, Any]:
        """Analizar patrones estructurales"""
        patterns_found = []
        
        for pattern_name, pattern_list in self.structural_patterns.items():
            for pattern in pattern_list:
                matches = []
                for i, text in enumerate(texts):
                    lines = text.split('\n')
                    for line in lines:
                        if re.search(pattern, line):
                            matches.append((doc_ids[i], line.strip()))
                
                if len(matches) >= self.min_pattern_frequency:
                    pattern_id = f"structural_{pattern_name}_{len(patterns_found)}"
                    
                    # Determinar categoría
                    if "heading" in pattern_name or "list" in pattern_name:
                        category = PatternCategory.POSITIVE
                    else:
                        category = PatternCategory.NEUTRAL
                    
                    text_pattern = TextPattern(
                        id=pattern_id,
                        type=PatternType.STRUCTURAL,
                        category=category,
                        name=f"Structural Pattern: {pattern_name}",
                        description=f"Pattern for {pattern_name} structure",
                        pattern=pattern,
                        frequency=len(matches),
                        confidence=min(len(matches) / 10, 1.0),
                        impact_score=0.6,
                        examples=[match[1] for match in matches[:5]],
                        documents=[match[0] for match in matches]
                    )
                    
                    patterns_found.append(text_pattern)
                    self.patterns[pattern_id] = text_pattern
        
        return {
            "patterns": [self._pattern_to_dict(p) for p in patterns_found],
            "total_patterns": len(patterns_found),
            "pattern_types": list(set(p.category.value for p in patterns_found))
        }
    
    async def _analyze_quality_patterns(self, texts: List[str], doc_ids: List[str]) -> Dict[str, Any]:
        """Analizar patrones de calidad"""
        patterns_found = []
        
        for pattern_name, pattern_list in self.quality_patterns.items():
            for pattern in pattern_list:
                matches = []
                for i, text in enumerate(texts):
                    found_matches = re.findall(pattern, text, re.IGNORECASE)
                    if found_matches:
                        matches.extend([(doc_ids[i], match) for match in found_matches])
                
                if len(matches) >= self.min_pattern_frequency:
                    pattern_id = f"quality_{pattern_name}_{len(patterns_found)}"
                    
                    # Determinar categoría
                    if "excellent" in pattern_name or "improvement" in pattern_name:
                        category = PatternCategory.POSITIVE
                    elif "poor" in pattern_name:
                        category = PatternCategory.NEGATIVE
                    else:
                        category = PatternCategory.NEUTRAL
                    
                    text_pattern = TextPattern(
                        id=pattern_id,
                        type=PatternType.QUALITY,
                        category=category,
                        name=f"Quality Pattern: {pattern_name}",
                        description=f"Pattern for {pattern_name} quality indicators",
                        pattern=pattern,
                        frequency=len(matches),
                        confidence=min(len(matches) / 10, 1.0),
                        impact_score=0.8,
                        examples=[match[1] for match in matches[:5]],
                        documents=[match[0] for match in matches]
                    )
                    
                    patterns_found.append(text_pattern)
                    self.patterns[pattern_id] = text_pattern
        
        return {
            "patterns": [self._pattern_to_dict(p) for p in patterns_found],
            "total_patterns": len(patterns_found),
            "pattern_types": list(set(p.category.value for p in patterns_found))
        }
    
    async def _analyze_semantic_patterns(self, texts: List[str], doc_ids: List[str]) -> Dict[str, Any]:
        """Analizar patrones semánticos"""
        try:
            # Usar TF-IDF para encontrar patrones semánticos
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(2, 3),
                min_df=2
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Encontrar frases más importantes
            patterns_found = []
            for i, doc_id in enumerate(doc_ids):
                doc_scores = tfidf_matrix[i].toarray()[0]
                top_indices = np.argsort(doc_scores)[-10:]  # Top 10 frases
                
                for idx in top_indices:
                    if doc_scores[idx] > 0.1:  # Umbral mínimo
                        phrase = feature_names[idx]
                        pattern_id = f"semantic_{phrase}_{len(patterns_found)}"
                        
                        text_pattern = TextPattern(
                            id=pattern_id,
                            type=PatternType.SEMANTIC,
                            category=PatternCategory.NEUTRAL,
                            name=f"Semantic Pattern: {phrase}",
                            description=f"Important semantic phrase: {phrase}",
                            pattern=phrase,
                            frequency=1,
                            confidence=float(doc_scores[idx]),
                            impact_score=float(doc_scores[idx]),
                            examples=[phrase],
                            documents=[doc_id]
                        )
                        
                        patterns_found.append(text_pattern)
                        self.patterns[pattern_id] = text_pattern
            
            return {
                "patterns": [self._pattern_to_dict(p) for p in patterns_found],
                "total_patterns": len(patterns_found),
                "pattern_types": ["semantic"]
            }
            
        except Exception as e:
            logger.error(f"Error in semantic pattern analysis: {e}")
            return {"error": str(e)}
    
    async def _analyze_style_patterns(self, texts: List[str], doc_ids: List[str]) -> Dict[str, Any]:
        """Analizar patrones de estilo"""
        patterns_found = []
        
        for i, text in enumerate(texts):
            # Análisis de longitud de oraciones
            sentences = text.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            
            if avg_sentence_length > 20:
                pattern_id = f"style_long_sentences_{i}"
                text_pattern = TextPattern(
                    id=pattern_id,
                    type=PatternType.STYLE,
                    category=PatternCategory.NEUTRAL,
                    name="Long Sentences Pattern",
                    description="Document contains long sentences",
                    pattern=f"avg_sentence_length > 20",
                    frequency=1,
                    confidence=0.7,
                    impact_score=0.3,
                    examples=[f"Average sentence length: {avg_sentence_length:.1f}"],
                    documents=[doc_ids[i]]
                )
                patterns_found.append(text_pattern)
                self.patterns[pattern_id] = text_pattern
            
            # Análisis de uso de párrafos
            paragraphs = text.split('\n\n')
            if len(paragraphs) > 5:
                pattern_id = f"style_structured_{i}"
                text_pattern = TextPattern(
                    id=pattern_id,
                    type=PatternType.STYLE,
                    category=PatternCategory.POSITIVE,
                    name="Structured Document Pattern",
                    description="Document has good paragraph structure",
                    pattern="paragraphs > 5",
                    frequency=1,
                    confidence=0.8,
                    impact_score=0.6,
                    examples=[f"Number of paragraphs: {len(paragraphs)}"],
                    documents=[doc_ids[i]]
                )
                patterns_found.append(text_pattern)
                self.patterns[pattern_id] = text_pattern
        
        return {
            "patterns": [self._pattern_to_dict(p) for p in patterns_found],
            "total_patterns": len(patterns_found),
            "pattern_types": list(set(p.category.value for p in patterns_found))
        }
    
    async def _cluster_patterns(self) -> Dict[str, Any]:
        """Agrupar patrones similares"""
        if len(self.patterns) < 3:
            return {"clusters": [], "total_clusters": 0}
        
        try:
            # Preparar datos para clustering
            pattern_texts = [pattern.pattern for pattern in self.patterns.values()]
            pattern_ids = list(self.patterns.keys())
            
            # Vectorizar patrones
            vectorizer = TfidfVectorizer()
            pattern_vectors = vectorizer.fit_transform(pattern_texts)
            
            # Clustering con DBSCAN
            clustering = DBSCAN(eps=0.3, min_samples=2)
            cluster_labels = clustering.fit_predict(pattern_vectors)
            
            # Crear clusters
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label == -1:  # Ruido
                    continue
                
                cluster_id = f"cluster_{label}"
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                
                pattern_id = pattern_ids[i]
                clusters[cluster_id].append(self.patterns[pattern_id])
            
            # Crear objetos PatternCluster
            pattern_clusters = []
            for cluster_id, cluster_patterns in clusters.items():
                if len(cluster_patterns) > 1:
                    # Calcular centroide
                    cluster_vectors = [pattern_vectors[i] for i, pattern in enumerate(self.patterns.values()) 
                                     if pattern in cluster_patterns]
                    centroid = np.mean(cluster_vectors, axis=0)
                    
                    # Encontrar patrón representativo
                    representative = max(cluster_patterns, key=lambda p: p.frequency)
                    
                    pattern_cluster = PatternCluster(
                        id=cluster_id,
                        name=f"Pattern Cluster {cluster_id}",
                        patterns=cluster_patterns,
                        centroid=centroid,
                        size=len(cluster_patterns),
                        coherence=0.7,  # Placeholder
                        representative_pattern=representative
                    )
                    
                    pattern_clusters.append(pattern_cluster)
                    self.pattern_clusters[cluster_id] = pattern_cluster
            
            return {
                "clusters": [self._cluster_to_dict(c) for c in pattern_clusters],
                "total_clusters": len(pattern_clusters),
                "clustering_method": "DBSCAN"
            }
            
        except Exception as e:
            logger.error(f"Error in pattern clustering: {e}")
            return {"error": str(e)}
    
    async def _analyze_pattern_evolution(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analizar evolución de patrones en el tiempo"""
        if not documents or len(documents) < 5:
            return {"evolutions": [], "total_evolutions": 0}
        
        try:
            # Agrupar documentos por período de tiempo
            time_periods = defaultdict(list)
            for doc in documents:
                timestamp = doc.get("timestamp", datetime.now().isoformat())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                
                # Agrupar por semana
                week_key = timestamp.strftime("%Y-W%U")
                time_periods[week_key].append(doc)
            
            # Analizar evolución de cada patrón
            evolutions = []
            for pattern_id, pattern in self.patterns.items():
                frequencies = []
                periods = sorted(time_periods.keys())
                
                for period in periods:
                    period_docs = time_periods[period]
                    period_texts = [doc.get("content", "") for doc in period_docs]
                    
                    # Contar ocurrencias del patrón en este período
                    frequency = 0
                    for text in period_texts:
                        matches = re.findall(pattern.pattern, text, re.IGNORECASE)
                        frequency += len(matches)
                    
                    frequencies.append(frequency)
                
                if sum(frequencies) > 0:
                    # Calcular tendencias
                    trends = self._calculate_trends(frequencies)
                    
                    # Encontrar período pico y declive
                    peak_period = periods[np.argmax(frequencies)]
                    decline_period = None
                    if len(frequencies) > 1 and frequencies[-1] < frequencies[0]:
                        decline_period = periods[-1]
                    
                    # Calcular estabilidad
                    stability_score = 1 - (np.std(frequencies) / (np.mean(frequencies) + 1))
                    
                    evolution = PatternEvolution(
                        pattern_id=pattern_id,
                        time_periods=periods,
                        frequencies=frequencies,
                        trends=trends,
                        peak_period=peak_period,
                        decline_period=decline_period,
                        stability_score=stability_score
                    )
                    
                    evolutions.append(evolution)
                    self.pattern_evolutions[pattern_id] = evolution
            
            return {
                "evolutions": [self._evolution_to_dict(e) for e in evolutions],
                "total_evolutions": len(evolutions),
                "time_periods": list(time_periods.keys())
            }
            
        except Exception as e:
            logger.error(f"Error in pattern evolution analysis: {e}")
            return {"error": str(e)}
    
    def _calculate_trends(self, frequencies: List[int]) -> Dict[str, float]:
        """Calcular tendencias en frecuencias"""
        if len(frequencies) < 2:
            return {"trend": 0.0, "volatility": 0.0}
        
        # Tendencia lineal
        x = np.arange(len(frequencies))
        y = np.array(frequencies)
        slope = np.polyfit(x, y, 1)[0]
        
        # Volatilidad
        volatility = np.std(frequencies) / (np.mean(frequencies) + 1)
        
        return {
            "trend": float(slope),
            "volatility": float(volatility),
            "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        }
    
    def _pattern_to_dict(self, pattern: TextPattern) -> Dict[str, Any]:
        """Convertir patrón a diccionario"""
        return {
            "id": pattern.id,
            "type": pattern.type.value,
            "category": pattern.category.value,
            "name": pattern.name,
            "description": pattern.description,
            "pattern": pattern.pattern,
            "frequency": pattern.frequency,
            "confidence": pattern.confidence,
            "impact_score": pattern.impact_score,
            "examples": pattern.examples,
            "documents": pattern.documents,
            "created_at": pattern.created_at.isoformat()
        }
    
    def _cluster_to_dict(self, cluster: PatternCluster) -> Dict[str, Any]:
        """Convertir cluster a diccionario"""
        return {
            "id": cluster.id,
            "name": cluster.name,
            "size": cluster.size,
            "coherence": cluster.coherence,
            "representative_pattern": self._pattern_to_dict(cluster.representative_pattern),
            "patterns": [self._pattern_to_dict(p) for p in cluster.patterns],
            "created_at": cluster.created_at.isoformat()
        }
    
    def _evolution_to_dict(self, evolution: PatternEvolution) -> Dict[str, Any]:
        """Convertir evolución a diccionario"""
        return {
            "pattern_id": evolution.pattern_id,
            "time_periods": evolution.time_periods,
            "frequencies": evolution.frequencies,
            "trends": evolution.trends,
            "peak_period": evolution.peak_period,
            "decline_period": evolution.decline_period,
            "stability_score": evolution.stability_score
        }
    
    async def get_pattern_summary(self) -> Dict[str, Any]:
        """Obtener resumen de patrones"""
        if not self.patterns:
            return {"message": "No patterns found"}
        
        # Estadísticas por tipo
        type_counts = Counter([pattern.type.value for pattern in self.patterns.values()])
        category_counts = Counter([pattern.category.value for pattern in self.patterns.values()])
        
        # Patrones más frecuentes
        top_patterns = sorted(self.patterns.values(), key=lambda p: p.frequency, reverse=True)[:10]
        
        # Patrones de mayor impacto
        high_impact_patterns = [p for p in self.patterns.values() if p.impact_score > 0.7]
        
        return {
            "total_patterns": len(self.patterns),
            "type_distribution": dict(type_counts),
            "category_distribution": dict(category_counts),
            "top_patterns": [self._pattern_to_dict(p) for p in top_patterns],
            "high_impact_patterns": [self._pattern_to_dict(p) for p in high_impact_patterns],
            "total_clusters": len(self.pattern_clusters),
            "total_evolutions": len(self.pattern_evolutions)
        }
    
    async def find_similar_patterns(self, pattern_id: str, limit: int = 5) -> List[TextPattern]:
        """Encontrar patrones similares"""
        if pattern_id not in self.patterns:
            return []
        
        target_pattern = self.patterns[pattern_id]
        similarities = []
        
        for other_id, other_pattern in self.patterns.items():
            if other_id == pattern_id:
                continue
            
            # Calcular similitud basada en tipo y categoría
            type_similarity = 1.0 if target_pattern.type == other_pattern.type else 0.0
            category_similarity = 1.0 if target_pattern.category == other_pattern.category else 0.0
            
            # Similitud de patrón (simplificada)
            pattern_similarity = 0.0
            if target_pattern.pattern and other_pattern.pattern:
                # Usar similitud de Jaccard en palabras
                target_words = set(target_pattern.pattern.split())
                other_words = set(other_pattern.pattern.split())
                if target_words or other_words:
                    pattern_similarity = len(target_words.intersection(other_words)) / len(target_words.union(other_words))
            
            # Similitud combinada
            combined_similarity = (type_similarity * 0.3 + category_similarity * 0.3 + pattern_similarity * 0.4)
            
            if combined_similarity > self.similarity_threshold:
                similarities.append((other_pattern, combined_similarity))
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [pattern for pattern, _ in similarities[:limit]]
    
    async def export_patterns(self, filepath: str = None) -> str:
        """Exportar patrones a archivo"""
        if filepath is None:
            filepath = f"exports/patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Crear directorio si no existe
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        export_data = {
            "patterns": {k: self._pattern_to_dict(v) for k, v in self.patterns.items()},
            "clusters": {k: self._cluster_to_dict(v) for k, v in self.pattern_clusters.items()},
            "evolutions": {k: self._evolution_to_dict(v) for k, v in self.pattern_evolutions.items()},
            "summary": await self.get_pattern_summary(),
            "exported_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Patterns exported to {filepath}")
        return filepath



























