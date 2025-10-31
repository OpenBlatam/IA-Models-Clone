"""
Advanced Analytics Engine for AI History Comparison System
Motor de análisis avanzado para el sistema de análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pickle
import joblib

# Machine Learning imports
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

# NLP imports
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import spacy

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Tipos de análisis disponibles"""
    CLUSTERING = "clustering"
    SEGMENTATION = "segmentation"
    ANOMALY_DETECTION = "anomaly_detection"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TOPIC_MODELING = "topic_modeling"
    TREND_ANALYSIS = "trend_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    PREDICTIVE_ANALYSIS = "predictive_analysis"

class ClusteringMethod(Enum):
    """Métodos de clustering disponibles"""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    AGGLOMERATIVE = "agglomerative"
    SPECTRAL = "spectral"
    GAUSSIAN_MIXTURE = "gaussian_mixture"

class AnomalyMethod(Enum):
    """Métodos de detección de anomalías"""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    STATISTICAL = "statistical"
    ZSCORE = "zscore"

@dataclass
class DocumentCluster:
    """Cluster de documentos"""
    id: str
    name: str
    documents: List[str]
    centroid: np.ndarray
    size: int
    quality_avg: float
    characteristics: Dict[str, Any]
    representative_documents: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DocumentSegment:
    """Segmento de documentos"""
    id: str
    name: str
    criteria: Dict[str, Any]
    documents: List[str]
    size: int
    metrics: Dict[str, float]
    insights: List[str]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AnomalyResult:
    """Resultado de detección de anomalías"""
    document_id: str
    anomaly_score: float
    anomaly_type: str
    severity: str
    explanation: str
    features: Dict[str, float]
    recommendations: List[str]
    detected_at: datetime = field(default_factory=datetime.now)

@dataclass
class SentimentAnalysis:
    """Análisis de sentimientos"""
    document_id: str
    overall_sentiment: str
    sentiment_score: float
    confidence: float
    emotional_tone: str
    subjectivity: float
    polarity_breakdown: Dict[str, float]
    key_phrases: List[str]
    analyzed_at: datetime = field(default_factory=datetime.now)

@dataclass
class TopicModel:
    """Modelo de tópicos"""
    id: str
    name: str
    keywords: List[str]
    documents: List[str]
    coherence_score: float
    perplexity_score: float
    word_distribution: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedAnalyticsEngine:
    """
    Motor de análisis avanzado para el sistema de análisis de historial de IA
    """
    
    def __init__(
        self,
        cache_directory: str = "cache/analytics/",
        models_directory: str = "models/analytics/",
        enable_nlp: bool = True,
        language: str = "spanish"
    ):
        self.cache_directory = Path(cache_directory)
        self.models_directory = Path(models_directory)
        self.enable_nlp = enable_nlp
        self.language = language
        
        # Crear directorios
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
        # Inicializar componentes
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=self._get_stopwords(),
            ngram_range=(1, 2)
        )
        
        # Inicializar NLP si está habilitado
        if self.enable_nlp:
            self._initialize_nlp()
        
        # Almacenamiento de resultados
        self.clusters: Dict[str, DocumentCluster] = {}
        self.segments: Dict[str, DocumentSegment] = {}
        self.anomalies: List[AnomalyResult] = []
        self.sentiment_analyses: Dict[str, SentimentAnalysis] = {}
        self.topic_models: Dict[str, TopicModel] = {}
        
        # Configuración de análisis
        self.analysis_config = {
            "clustering": {
                "max_clusters": 10,
                "min_cluster_size": 3,
                "random_state": 42
            },
            "anomaly_detection": {
                "contamination": 0.1,
                "random_state": 42
            },
            "sentiment_analysis": {
                "confidence_threshold": 0.7
            }
        }
    
    def _initialize_nlp(self):
        """Inicializar componentes de NLP"""
        try:
            # Descargar recursos de NLTK
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Inicializar analizador de sentimientos
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Inicializar lemmatizador
            self.lemmatizer = WordNetLemmatizer()
            
            # Cargar modelo de spaCy si está disponible
            try:
                if self.language == "spanish":
                    self.nlp = spacy.load("es_core_news_sm")
                else:
                    self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using basic NLP")
                self.nlp = None
            
            logger.info("NLP components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP: {e}")
            self.enable_nlp = False
    
    def _get_stopwords(self) -> List[str]:
        """Obtener lista de stopwords según el idioma"""
        try:
            if self.language == "spanish":
                return list(stopwords.words('spanish'))
            else:
                return list(stopwords.words('english'))
        except:
            return []
    
    async def analyze_documents(
        self,
        documents: List[Dict[str, Any]],
        analysis_types: List[AnalysisType],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Realizar análisis avanzado de documentos
        
        Args:
            documents: Lista de documentos a analizar
            analysis_types: Tipos de análisis a realizar
            **kwargs: Parámetros adicionales para cada tipo de análisis
            
        Returns:
            Resultados del análisis
        """
        results = {}
        
        # Preparar datos
        df = self._prepare_dataframe(documents)
        
        # Realizar análisis según los tipos solicitados
        for analysis_type in analysis_types:
            try:
                if analysis_type == AnalysisType.CLUSTERING:
                    results["clustering"] = await self._perform_clustering(df, **kwargs)
                elif analysis_type == AnalysisType.SEGMENTATION:
                    results["segmentation"] = await self._perform_segmentation(df, **kwargs)
                elif analysis_type == AnalysisType.ANOMALY_DETECTION:
                    results["anomaly_detection"] = await self._perform_anomaly_detection(df, **kwargs)
                elif analysis_type == AnalysisType.SENTIMENT_ANALYSIS:
                    results["sentiment_analysis"] = await self._perform_sentiment_analysis(df, **kwargs)
                elif analysis_type == AnalysisType.TOPIC_MODELING:
                    results["topic_modeling"] = await self._perform_topic_modeling(df, **kwargs)
                elif analysis_type == AnalysisType.TREND_ANALYSIS:
                    results["trend_analysis"] = await self._perform_trend_analysis(df, **kwargs)
                elif analysis_type == AnalysisType.CORRELATION_ANALYSIS:
                    results["correlation_analysis"] = await self._perform_correlation_analysis(df, **kwargs)
                elif analysis_type == AnalysisType.PREDICTIVE_ANALYSIS:
                    results["predictive_analysis"] = await self._perform_predictive_analysis(df, **kwargs)
                    
            except Exception as e:
                logger.error(f"Error in {analysis_type.value} analysis: {e}")
                results[analysis_type.value] = {"error": str(e)}
        
        return results
    
    def _prepare_dataframe(self, documents: List[Dict[str, Any]]) -> pd.DataFrame:
        """Preparar DataFrame para análisis"""
        data = []
        
        for doc in documents:
            row = {
                "id": doc.get("id", ""),
                "content": doc.get("content", ""),
                "query": doc.get("query", ""),
                "quality_score": doc.get("quality_score", 0.0),
                "readability_score": doc.get("readability_score", 0.0),
                "originality_score": doc.get("originality_score", 0.0),
                "word_count": doc.get("word_count", 0),
                "timestamp": doc.get("timestamp", datetime.now().isoformat()),
                "metadata": doc.get("metadata", {})
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Convertir timestamp a datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Extraer características adicionales
        df["content_length"] = df["content"].str.len()
        df["sentence_count"] = df["content"].apply(lambda x: len(sent_tokenize(x)) if x else 0)
        df["avg_sentence_length"] = df["content_length"] / (df["sentence_count"] + 1)
        
        return df
    
    async def _perform_clustering(
        self,
        df: pd.DataFrame,
        method: ClusteringMethod = ClusteringMethod.KMEANS,
        n_clusters: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Realizar clustering de documentos"""
        try:
            # Preparar características para clustering
            features = self._extract_clustering_features(df)
            
            # Determinar número óptimo de clusters si no se especifica
            if n_clusters is None:
                n_clusters = self._find_optimal_clusters(features)
            
            # Aplicar clustering
            if method == ClusteringMethod.KMEANS:
                clusterer = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.analysis_config["clustering"]["random_state"]
                )
            elif method == ClusteringMethod.DBSCAN:
                clusterer = DBSCAN(
                    eps=kwargs.get("eps", 0.5),
                    min_samples=self.analysis_config["clustering"]["min_cluster_size"]
                )
            elif method == ClusteringMethod.AGGLOMERATIVE:
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=kwargs.get("linkage", "ward")
                )
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            cluster_labels = clusterer.fit_predict(features)
            
            # Crear clusters
            clusters = {}
            for i in range(n_clusters):
                cluster_docs = df[cluster_labels == i]
                if len(cluster_docs) > 0:
                    cluster_id = f"cluster_{i}"
                    cluster = DocumentCluster(
                        id=cluster_id,
                        name=f"Cluster {i+1}",
                        documents=cluster_docs["id"].tolist(),
                        centroid=np.mean(features[cluster_labels == i], axis=0),
                        size=len(cluster_docs),
                        quality_avg=cluster_docs["quality_score"].mean(),
                        characteristics=self._analyze_cluster_characteristics(cluster_docs),
                        representative_documents=self._find_representative_documents(cluster_docs, features[cluster_labels == i])
                    )
                    clusters[cluster_id] = cluster
                    self.clusters[cluster_id] = cluster
            
            # Calcular métricas de calidad
            silhouette_avg = silhouette_score(features, cluster_labels) if len(set(cluster_labels)) > 1 else 0
            calinski_harabasz = calinski_harabasz_score(features, cluster_labels) if len(set(cluster_labels)) > 1 else 0
            
            return {
                "method": method.value,
                "n_clusters": n_clusters,
                "clusters": {k: self._cluster_to_dict(v) for k, v in clusters.items()},
                "metrics": {
                    "silhouette_score": silhouette_avg,
                    "calinski_harabasz_score": calinski_harabasz,
                    "total_documents": len(df)
                },
                "cluster_distribution": pd.Series(cluster_labels).value_counts().to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error in clustering analysis: {e}")
            return {"error": str(e)}
    
    def _extract_clustering_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extraer características para clustering"""
        # Características numéricas
        numeric_features = df[["quality_score", "readability_score", "originality_score", 
                              "word_count", "content_length", "sentence_count", "avg_sentence_length"]].values
        
        # Características de texto usando TF-IDF
        text_features = self.tfidf_vectorizer.fit_transform(df["content"].fillna("")).toarray()
        
        # Combinar características
        features = np.hstack([numeric_features, text_features])
        
        # Normalizar
        features = self.scaler.fit_transform(features)
        
        return features
    
    def _find_optimal_clusters(self, features: np.ndarray, max_k: int = 10) -> int:
        """Encontrar número óptimo de clusters usando método del codo"""
        if len(features) < 4:
            return 2
        
        inertias = []
        K_range = range(2, min(max_k + 1, len(features) // 2))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
        
        # Método del codo simplificado
        if len(inertias) > 1:
            # Encontrar el punto de mayor curvatura
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            optimal_k = K_range[np.argmax(second_diffs) + 1] if len(second_diffs) > 0 else K_range[0]
        else:
            optimal_k = 2
        
        return optimal_k
    
    def _analyze_cluster_characteristics(self, cluster_docs: pd.DataFrame) -> Dict[str, Any]:
        """Analizar características de un cluster"""
        return {
            "avg_quality": cluster_docs["quality_score"].mean(),
            "avg_readability": cluster_docs["readability_score"].mean(),
            "avg_originality": cluster_docs["originality_score"].mean(),
            "avg_word_count": cluster_docs["word_count"].mean(),
            "quality_std": cluster_docs["quality_score"].std(),
            "common_queries": cluster_docs["query"].value_counts().head(3).to_dict(),
            "time_span": {
                "start": cluster_docs["timestamp"].min().isoformat(),
                "end": cluster_docs["timestamp"].max().isoformat()
            }
        }
    
    def _find_representative_documents(self, cluster_docs: pd.DataFrame, cluster_features: np.ndarray) -> List[str]:
        """Encontrar documentos representativos del cluster"""
        if len(cluster_docs) == 0:
            return []
        
        # Calcular distancia al centroide
        centroid = np.mean(cluster_features, axis=0)
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        
        # Seleccionar los documentos más cercanos al centroide
        n_representatives = min(3, len(cluster_docs))
        representative_indices = np.argsort(distances)[:n_representatives]
        
        return cluster_docs.iloc[representative_indices]["id"].tolist()
    
    async def _perform_segmentation(
        self,
        df: pd.DataFrame,
        criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Realizar segmentación de documentos"""
        try:
            segments = {}
            
            # Segmentación por calidad
            quality_segments = self._segment_by_quality(df)
            segments.update(quality_segments)
            
            # Segmentación por tiempo
            time_segments = self._segment_by_time(df)
            segments.update(time_segments)
            
            # Segmentación por tipo de query
            query_segments = self._segment_by_query_type(df)
            segments.update(query_segments)
            
            # Segmentación personalizada
            if criteria:
                custom_segments = self._segment_by_criteria(df, criteria)
                segments.update(custom_segments)
            
            # Almacenar segmentos
            for segment_id, segment in segments.items():
                self.segments[segment_id] = segment
            
            return {
                "segments": {k: self._segment_to_dict(v) for k, v in segments.items()},
                "total_segments": len(segments),
                "total_documents": len(df)
            }
            
        except Exception as e:
            logger.error(f"Error in segmentation analysis: {e}")
            return {"error": str(e)}
    
    def _segment_by_quality(self, df: pd.DataFrame) -> Dict[str, DocumentSegment]:
        """Segmentar documentos por calidad"""
        segments = {}
        
        # Definir umbrales de calidad
        excellent_threshold = 0.8
        good_threshold = 0.6
        average_threshold = 0.4
        
        # Segmento de excelente calidad
        excellent_docs = df[df["quality_score"] >= excellent_threshold]
        if len(excellent_docs) > 0:
            segments["excellent_quality"] = DocumentSegment(
                id="excellent_quality",
                name="Excelente Calidad",
                criteria={"quality_score": f">= {excellent_threshold}"},
                documents=excellent_docs["id"].tolist(),
                size=len(excellent_docs),
                metrics=self._calculate_segment_metrics(excellent_docs),
                insights=self._generate_quality_insights(excellent_docs, "excellent"),
                recommendations=self._generate_quality_recommendations(excellent_docs, "excellent")
            )
        
        # Segmento de buena calidad
        good_docs = df[(df["quality_score"] >= good_threshold) & (df["quality_score"] < excellent_threshold)]
        if len(good_docs) > 0:
            segments["good_quality"] = DocumentSegment(
                id="good_quality",
                name="Buena Calidad",
                criteria={"quality_score": f"{good_threshold} - {excellent_threshold}"},
                documents=good_docs["id"].tolist(),
                size=len(good_docs),
                metrics=self._calculate_segment_metrics(good_docs),
                insights=self._generate_quality_insights(good_docs, "good"),
                recommendations=self._generate_quality_recommendations(good_docs, "good")
            )
        
        # Segmento de calidad promedio
        average_docs = df[(df["quality_score"] >= average_threshold) & (df["quality_score"] < good_threshold)]
        if len(average_docs) > 0:
            segments["average_quality"] = DocumentSegment(
                id="average_quality",
                name="Calidad Promedio",
                criteria={"quality_score": f"{average_threshold} - {good_threshold}"},
                documents=average_docs["id"].tolist(),
                size=len(average_docs),
                metrics=self._calculate_segment_metrics(average_docs),
                insights=self._generate_quality_insights(average_docs, "average"),
                recommendations=self._generate_quality_recommendations(average_docs, "average")
            )
        
        # Segmento de baja calidad
        poor_docs = df[df["quality_score"] < average_threshold]
        if len(poor_docs) > 0:
            segments["poor_quality"] = DocumentSegment(
                id="poor_quality",
                name="Baja Calidad",
                criteria={"quality_score": f"< {average_threshold}"},
                documents=poor_docs["id"].tolist(),
                size=len(poor_docs),
                metrics=self._calculate_segment_metrics(poor_docs),
                insights=self._generate_quality_insights(poor_docs, "poor"),
                recommendations=self._generate_quality_recommendations(poor_docs, "poor")
            )
        
        return segments
    
    def _segment_by_time(self, df: pd.DataFrame) -> Dict[str, DocumentSegment]:
        """Segmentar documentos por tiempo"""
        segments = {}
        
        # Ordenar por timestamp
        df_sorted = df.sort_values("timestamp")
        
        # Dividir en períodos
        total_days = (df_sorted["timestamp"].max() - df_sorted["timestamp"].min()).days
        
        if total_days > 0:
            # Segmento reciente (últimos 7 días)
            recent_cutoff = df_sorted["timestamp"].max() - timedelta(days=7)
            recent_docs = df_sorted[df_sorted["timestamp"] >= recent_cutoff]
            
            if len(recent_docs) > 0:
                segments["recent"] = DocumentSegment(
                    id="recent",
                    name="Documentos Recientes",
                    criteria={"time_period": "last_7_days"},
                    documents=recent_docs["id"].tolist(),
                    size=len(recent_docs),
                    metrics=self._calculate_segment_metrics(recent_docs),
                    insights=self._generate_time_insights(recent_docs, "recent"),
                    recommendations=self._generate_time_recommendations(recent_docs, "recent")
                )
            
            # Segmento histórico (más de 30 días)
            if total_days > 30:
                historical_cutoff = df_sorted["timestamp"].max() - timedelta(days=30)
                historical_docs = df_sorted[df_sorted["timestamp"] < historical_cutoff]
                
                if len(historical_docs) > 0:
                    segments["historical"] = DocumentSegment(
                        id="historical",
                        name="Documentos Históricos",
                        criteria={"time_period": "older_than_30_days"},
                        documents=historical_docs["id"].tolist(),
                        size=len(historical_docs),
                        metrics=self._calculate_segment_metrics(historical_docs),
                        insights=self._generate_time_insights(historical_docs, "historical"),
                        recommendations=self._generate_time_recommendations(historical_docs, "historical")
                    )
        
        return segments
    
    def _segment_by_query_type(self, df: pd.DataFrame) -> Dict[str, DocumentSegment]:
        """Segmentar documentos por tipo de query"""
        segments = {}
        
        # Analizar patrones en las queries
        query_patterns = self._analyze_query_patterns(df)
        
        for pattern_name, pattern_docs in query_patterns.items():
            if len(pattern_docs) > 0:
                segments[f"query_{pattern_name}"] = DocumentSegment(
                    id=f"query_{pattern_name}",
                    name=f"Queries {pattern_name.title()}",
                    criteria={"query_pattern": pattern_name},
                    documents=pattern_docs["id"].tolist(),
                    size=len(pattern_docs),
                    metrics=self._calculate_segment_metrics(pattern_docs),
                    insights=self._generate_query_insights(pattern_docs, pattern_name),
                    recommendations=self._generate_query_recommendations(pattern_docs, pattern_name)
                )
        
        return segments
    
    def _segment_by_criteria(self, df: pd.DataFrame, criteria: Dict[str, Any]) -> Dict[str, DocumentSegment]:
        """Segmentar documentos por criterios personalizados"""
        segments = {}
        
        # Implementar segmentación personalizada basada en criterios
        # Por ejemplo: por metadata, por longitud, por características específicas
        
        return segments
    
    def _calculate_segment_metrics(self, segment_docs: pd.DataFrame) -> Dict[str, float]:
        """Calcular métricas para un segmento"""
        return {
            "avg_quality": segment_docs["quality_score"].mean(),
            "avg_readability": segment_docs["readability_score"].mean(),
            "avg_originality": segment_docs["originality_score"].mean(),
            "avg_word_count": segment_docs["word_count"].mean(),
            "quality_std": segment_docs["quality_score"].std(),
            "quality_min": segment_docs["quality_score"].min(),
            "quality_max": segment_docs["quality_score"].max(),
            "total_documents": len(segment_docs)
        }
    
    def _generate_quality_insights(self, docs: pd.DataFrame, quality_level: str) -> List[str]:
        """Generar insights basados en la calidad"""
        insights = []
        
        avg_quality = docs["quality_score"].mean()
        insights.append(f"Calidad promedio: {avg_quality:.2f}")
        
        if quality_level == "excellent":
            insights.append("Documentos con características excepcionales")
            insights.append("Patrones de alta calidad identificados")
        elif quality_level == "good":
            insights.append("Documentos con buena calidad general")
            insights.append("Oportunidades de mejora identificadas")
        elif quality_level == "average":
            insights.append("Documentos con calidad estándar")
            insights.append("Necesidad de optimización")
        else:  # poor
            insights.append("Documentos que requieren revisión urgente")
            insights.append("Problemas de calidad identificados")
        
        return insights
    
    def _generate_quality_recommendations(self, docs: pd.DataFrame, quality_level: str) -> List[str]:
        """Generar recomendaciones basadas en la calidad"""
        recommendations = []
        
        if quality_level == "excellent":
            recommendations.append("Replicar patrones de estos documentos")
            recommendations.append("Usar como referencia para otros documentos")
        elif quality_level == "good":
            recommendations.append("Identificar áreas específicas de mejora")
            recommendations.append("Optimizar queries para mejor rendimiento")
        elif quality_level == "average":
            recommendations.append("Revisar y mejorar estructura de contenido")
            recommendations.append("Optimizar longitud y legibilidad")
        else:  # poor
            recommendations.append("Revisión completa del contenido")
            recommendations.append("Reformular queries problemáticas")
            recommendations.append("Implementar controles de calidad")
        
        return recommendations
    
    def _generate_time_insights(self, docs: pd.DataFrame, time_period: str) -> List[str]:
        """Generar insights basados en el tiempo"""
        insights = []
        
        if time_period == "recent":
            insights.append("Tendencias recientes en la generación de documentos")
            insights.append("Cambios en patrones de calidad")
        else:  # historical
            insights.append("Evolución histórica de la calidad")
            insights.append("Patrones de mejora a lo largo del tiempo")
        
        return insights
    
    def _generate_time_recommendations(self, docs: pd.DataFrame, time_period: str) -> List[str]:
        """Generar recomendaciones basadas en el tiempo"""
        recommendations = []
        
        if time_period == "recent":
            recommendations.append("Monitorear tendencias recientes")
            recommendations.append("Ajustar estrategias basadas en cambios recientes")
        else:  # historical
            recommendations.append("Aplicar lecciones aprendidas del historial")
            recommendations.append("Identificar patrones de mejora sostenible")
        
        return recommendations
    
    def _analyze_query_patterns(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analizar patrones en las queries"""
        patterns = {}
        
        # Patrones básicos basados en longitud y palabras clave
        short_queries = df[df["query"].str.len() < 50]
        long_queries = df[df["query"].str.len() >= 100]
        
        if len(short_queries) > 0:
            patterns["cortas"] = short_queries
        
        if len(long_queries) > 0:
            patterns["largas"] = long_queries
        
        # Patrones por palabras clave comunes
        common_keywords = ["escribe", "genera", "crea", "explica", "describe"]
        for keyword in common_keywords:
            keyword_docs = df[df["query"].str.contains(keyword, case=False, na=False)]
            if len(keyword_docs) > 0:
                patterns[f"con_{keyword}"] = keyword_docs
        
        return patterns
    
    def _generate_query_insights(self, docs: pd.DataFrame, pattern_name: str) -> List[str]:
        """Generar insights basados en patrones de query"""
        insights = []
        
        insights.append(f"Patrón de query: {pattern_name}")
        insights.append(f"Documentos con este patrón: {len(docs)}")
        
        avg_quality = docs["quality_score"].mean()
        insights.append(f"Calidad promedio para este patrón: {avg_quality:.2f}")
        
        return insights
    
    def _generate_query_recommendations(self, docs: pd.DataFrame, pattern_name: str) -> List[str]:
        """Generar recomendaciones basadas en patrones de query"""
        recommendations = []
        
        avg_quality = docs["quality_score"].mean()
        
        if avg_quality > 0.7:
            recommendations.append(f"Patrón '{pattern_name}' genera buena calidad")
            recommendations.append("Considerar replicar este patrón")
        elif avg_quality < 0.4:
            recommendations.append(f"Patrón '{pattern_name}' necesita optimización")
            recommendations.append("Revisar y mejorar este tipo de queries")
        else:
            recommendations.append(f"Patrón '{pattern_name}' con calidad promedio")
            recommendations.append("Identificar oportunidades de mejora")
        
        return recommendations
    
    async def _perform_anomaly_detection(
        self,
        df: pd.DataFrame,
        method: AnomalyMethod = AnomalyMethod.ISOLATION_FOREST,
        **kwargs
    ) -> Dict[str, Any]:
        """Realizar detección de anomalías"""
        try:
            # Preparar características
            features = self._extract_clustering_features(df)
            
            # Aplicar método de detección de anomalías
            if method == AnomalyMethod.ISOLATION_FOREST:
                detector = IsolationForest(
                    contamination=kwargs.get("contamination", self.analysis_config["anomaly_detection"]["contamination"]),
                    random_state=self.analysis_config["anomaly_detection"]["random_state"]
                )
                anomaly_scores = detector.fit_predict(features)
                anomaly_scores = detector.score_samples(features)
            elif method == AnomalyMethod.ONE_CLASS_SVM:
                detector = OneClassSVM(
                    nu=kwargs.get("nu", 0.1),
                    kernel=kwargs.get("kernel", "rbf")
                )
                anomaly_scores = detector.fit_predict(features)
                anomaly_scores = detector.score_samples(features)
            elif method == AnomalyMethod.LOCAL_OUTLIER_FACTOR:
                detector = LocalOutlierFactor(
                    n_neighbors=kwargs.get("n_neighbors", 20),
                    contamination=kwargs.get("contamination", 0.1)
                )
                anomaly_scores = detector.fit_predict(features)
                anomaly_scores = detector.negative_outlier_factor_
            else:
                raise ValueError(f"Unsupported anomaly detection method: {method}")
            
            # Identificar anomalías
            anomalies = []
            threshold = np.percentile(anomaly_scores, 10)  # Bottom 10% as anomalies
            
            for i, (idx, row) in enumerate(df.iterrows()):
                if anomaly_scores[i] < threshold:
                    anomaly = AnomalyResult(
                        document_id=row["id"],
                        anomaly_score=float(anomaly_scores[i]),
                        anomaly_type=self._classify_anomaly_type(row, anomaly_scores[i]),
                        severity=self._classify_anomaly_severity(anomaly_scores[i], threshold),
                        explanation=self._explain_anomaly(row, anomaly_scores[i]),
                        features=self._extract_anomaly_features(row),
                        recommendations=self._generate_anomaly_recommendations(row, anomaly_scores[i])
                    )
                    anomalies.append(anomaly)
                    self.anomalies.append(anomaly)
            
            return {
                "method": method.value,
                "anomalies": [self._anomaly_to_dict(a) for a in anomalies],
                "total_anomalies": len(anomalies),
                "anomaly_rate": len(anomalies) / len(df),
                "threshold": float(threshold),
                "statistics": {
                    "min_score": float(np.min(anomaly_scores)),
                    "max_score": float(np.max(anomaly_scores)),
                    "mean_score": float(np.mean(anomaly_scores)),
                    "std_score": float(np.std(anomaly_scores))
                }
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {"error": str(e)}
    
    def _classify_anomaly_type(self, row: pd.Series, score: float) -> str:
        """Clasificar el tipo de anomalía"""
        if row["quality_score"] < 0.3:
            return "low_quality"
        elif row["readability_score"] < 0.3:
            return "low_readability"
        elif row["originality_score"] < 0.3:
            return "low_originality"
        elif row["word_count"] > 2000:
            return "excessive_length"
        elif row["word_count"] < 50:
            return "insufficient_length"
        else:
            return "general_anomaly"
    
    def _classify_anomaly_severity(self, score: float, threshold: float) -> str:
        """Clasificar la severidad de la anomalía"""
        if score < threshold * 0.5:
            return "critical"
        elif score < threshold * 0.8:
            return "high"
        else:
            return "medium"
    
    def _explain_anomaly(self, row: pd.Series, score: float) -> str:
        """Explicar la anomalía detectada"""
        explanations = []
        
        if row["quality_score"] < 0.3:
            explanations.append("Calidad excepcionalmente baja")
        if row["readability_score"] < 0.3:
            explanations.append("Legibilidad muy pobre")
        if row["originality_score"] < 0.3:
            explanations.append("Falta de originalidad")
        if row["word_count"] > 2000:
            explanations.append("Longitud excesiva")
        if row["word_count"] < 50:
            explanations.append("Longitud insuficiente")
        
        if not explanations:
            explanations.append("Patrón inusual en las características del documento")
        
        return "; ".join(explanations)
    
    def _extract_anomaly_features(self, row: pd.Series) -> Dict[str, float]:
        """Extraer características relevantes para la anomalía"""
        return {
            "quality_score": float(row["quality_score"]),
            "readability_score": float(row["readability_score"]),
            "originality_score": float(row["originality_score"]),
            "word_count": int(row["word_count"]),
            "content_length": int(row["content_length"]),
            "sentence_count": int(row["sentence_count"])
        }
    
    def _generate_anomaly_recommendations(self, row: pd.Series, score: float) -> List[str]:
        """Generar recomendaciones para anomalías"""
        recommendations = []
        
        if row["quality_score"] < 0.3:
            recommendations.append("Revisar y mejorar la estructura del contenido")
            recommendations.append("Verificar la coherencia del documento")
        
        if row["readability_score"] < 0.3:
            recommendations.append("Simplificar el lenguaje utilizado")
            recommendations.append("Mejorar la estructura de oraciones")
        
        if row["originality_score"] < 0.3:
            recommendations.append("Añadir perspectivas únicas al contenido")
            recommendations.append("Evitar repetición de información")
        
        if row["word_count"] > 2000:
            recommendations.append("Dividir el contenido en secciones más pequeñas")
            recommendations.append("Eliminar información redundante")
        
        if row["word_count"] < 50:
            recommendations.append("Expandir el contenido con más detalles")
            recommendations.append("Añadir ejemplos y explicaciones")
        
        return recommendations
    
    async def _perform_sentiment_analysis(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """Realizar análisis de sentimientos"""
        if not self.enable_nlp:
            return {"error": "NLP not enabled"}
        
        try:
            sentiment_results = []
            
            for idx, row in df.iterrows():
                sentiment = self._analyze_document_sentiment(row)
                sentiment_results.append(sentiment)
                self.sentiment_analyses[row["id"]] = sentiment
            
            # Análisis agregado
            overall_sentiment = self._calculate_overall_sentiment(sentiment_results)
            
            return {
                "individual_analyses": [self._sentiment_to_dict(s) for s in sentiment_results],
                "overall_sentiment": overall_sentiment,
                "sentiment_distribution": self._calculate_sentiment_distribution(sentiment_results),
                "total_analyzed": len(sentiment_results)
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_document_sentiment(self, row: pd.Series) -> SentimentAnalysis:
        """Analizar sentimiento de un documento individual"""
        content = row["content"]
        
        # Análisis con VADER
        vader_scores = self.sentiment_analyzer.polarity_scores(content)
        
        # Análisis con TextBlob
        blob = TextBlob(content)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Determinar sentimiento general
        if vader_scores["compound"] >= 0.05:
            overall_sentiment = "positive"
        elif vader_scores["compound"] <= -0.05:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        # Determinar tono emocional
        emotional_tone = self._determine_emotional_tone(vader_scores)
        
        # Extraer frases clave
        key_phrases = self._extract_key_phrases(content)
        
        return SentimentAnalysis(
            document_id=row["id"],
            overall_sentiment=overall_sentiment,
            sentiment_score=vader_scores["compound"],
            confidence=abs(vader_scores["compound"]),
            emotional_tone=emotional_tone,
            subjectivity=textblob_subjectivity,
            polarity_breakdown={
                "positive": vader_scores["pos"],
                "negative": vader_scores["neg"],
                "neutral": vader_scores["neu"],
                "compound": vader_scores["compound"]
            },
            key_phrases=key_phrases
        )
    
    def _determine_emotional_tone(self, vader_scores: Dict[str, float]) -> str:
        """Determinar el tono emocional basado en scores de VADER"""
        if vader_scores["pos"] > 0.5:
            return "enthusiastic"
        elif vader_scores["neg"] > 0.5:
            return "critical"
        elif vader_scores["neu"] > 0.8:
            return "neutral"
        elif vader_scores["compound"] > 0.3:
            return "optimistic"
        elif vader_scores["compound"] < -0.3:
            return "pessimistic"
        else:
            return "balanced"
    
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extraer frases clave del contenido"""
        try:
            if self.nlp:
                doc = self.nlp(content)
                # Extraer frases nominales
                phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
                return phrases[:5]  # Top 5 phrases
            else:
                # Fallback simple
                sentences = sent_tokenize(content)
                return sentences[:3]  # Top 3 sentences
        except:
            return []
    
    def _calculate_overall_sentiment(self, sentiment_results: List[SentimentAnalysis]) -> Dict[str, Any]:
        """Calcular sentimiento general de todos los documentos"""
        if not sentiment_results:
            return {}
        
        avg_sentiment_score = np.mean([s.sentiment_score for s in sentiment_results])
        avg_confidence = np.mean([s.confidence for s in sentiment_results])
        avg_subjectivity = np.mean([s.subjectivity for s in sentiment_results])
        
        # Distribución de sentimientos
        sentiment_counts = {}
        for sentiment in sentiment_results:
            sentiment_type = sentiment.overall_sentiment
            sentiment_counts[sentiment_type] = sentiment_counts.get(sentiment_type, 0) + 1
        
        return {
            "average_sentiment_score": float(avg_sentiment_score),
            "average_confidence": float(avg_confidence),
            "average_subjectivity": float(avg_subjectivity),
            "sentiment_distribution": sentiment_counts,
            "dominant_sentiment": max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else "neutral"
        }
    
    def _calculate_sentiment_distribution(self, sentiment_results: List[SentimentAnalysis]) -> Dict[str, Any]:
        """Calcular distribución de sentimientos"""
        distribution = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
        
        emotional_tones = {}
        
        for sentiment in sentiment_results:
            distribution[sentiment.overall_sentiment] += 1
            emotional_tones[sentiment.emotional_tone] = emotional_tones.get(sentiment.emotional_tone, 0) + 1
        
        # Convertir a porcentajes
        total = len(sentiment_results)
        for key in distribution:
            distribution[key] = (distribution[key] / total) * 100
        
        return {
            "sentiment_percentages": distribution,
            "emotional_tone_distribution": emotional_tones
        }
    
    async def _perform_topic_modeling(
        self,
        df: pd.DataFrame,
        n_topics: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Realizar modelado de tópicos"""
        try:
            from sklearn.decomposition import LatentDirichletAllocation
            
            # Preparar datos de texto
            texts = df["content"].fillna("").tolist()
            
            # Crear matriz TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Aplicar LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100
            )
            lda.fit(tfidf_matrix)
            
            # Extraer tópicos
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [topic[i] for i in top_words_idx]
                
                # Asignar documentos al tópico
                doc_topic_probs = lda.transform(tfidf_matrix)
                topic_docs = df[doc_topic_probs[:, topic_idx] > 0.3]["id"].tolist()
                
                topic_model = TopicModel(
                    id=f"topic_{topic_idx}",
                    name=f"Tópico {topic_idx + 1}",
                    keywords=top_words,
                    documents=topic_docs,
                    coherence_score=self._calculate_topic_coherence(top_words, texts),
                    perplexity_score=lda.perplexity(tfidf_matrix),
                    word_distribution=dict(zip(top_words, top_weights))
                )
                
                topics.append(topic_model)
                self.topic_models[topic_model.id] = topic_model
            
            return {
                "topics": [self._topic_to_dict(t) for t in topics],
                "n_topics": n_topics,
                "perplexity": lda.perplexity(tfidf_matrix),
                "total_documents": len(df)
            }
            
        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            return {"error": str(e)}
    
    def _calculate_topic_coherence(self, top_words: List[str], texts: List[str]) -> float:
        """Calcular coherencia de tópico (simplificado)"""
        # Implementación simplificada de coherencia
        # En una implementación real, se usaría una métrica más sofisticada
        return 0.5  # Placeholder
    
    async def _perform_trend_analysis(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """Realizar análisis de tendencias"""
        try:
            # Ordenar por timestamp
            df_sorted = df.sort_values("timestamp")
            
            # Análisis de tendencias temporales
            time_trends = self._analyze_temporal_trends(df_sorted)
            
            # Análisis de tendencias de calidad
            quality_trends = self._analyze_quality_trends(df_sorted)
            
            # Análisis de tendencias de contenido
            content_trends = self._analyze_content_trends(df_sorted)
            
            return {
                "temporal_trends": time_trends,
                "quality_trends": quality_trends,
                "content_trends": content_trends,
                "trend_summary": self._generate_trend_summary(time_trends, quality_trends, content_trends)
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_temporal_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analizar tendencias temporales"""
        # Agrupar por día
        df["date"] = df["timestamp"].dt.date
        daily_stats = df.groupby("date").agg({
            "quality_score": ["mean", "std", "count"],
            "readability_score": "mean",
            "originality_score": "mean",
            "word_count": "mean"
        }).round(3)
        
        # Calcular tendencias
        quality_trend = self._calculate_trend_direction(daily_stats[("quality_score", "mean")])
        volume_trend = self._calculate_trend_direction(daily_stats[("quality_score", "count")])
        
        return {
            "daily_statistics": daily_stats.to_dict(),
            "quality_trend": quality_trend,
            "volume_trend": volume_trend,
            "time_span": {
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat(),
                "days": (df["timestamp"].max() - df["timestamp"].min()).days
            }
        }
    
    def _analyze_quality_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analizar tendencias de calidad"""
        # Calcular medias móviles
        window_size = min(7, len(df) // 4)  # Ventana de 7 días o 25% de los datos
        if window_size > 1:
            df["quality_ma"] = df["quality_score"].rolling(window=window_size).mean()
            df["readability_ma"] = df["readability_score"].rolling(window=window_size).mean()
            df["originality_ma"] = df["originality_score"].rolling(window=window_size).mean()
        
        # Identificar períodos de mejora/deterioro
        improvement_periods = self._identify_improvement_periods(df)
        deterioration_periods = self._identify_deterioration_periods(df)
        
        return {
            "moving_averages": {
                "quality": df["quality_ma"].dropna().tolist() if "quality_ma" in df.columns else [],
                "readability": df["readability_ma"].dropna().tolist() if "readability_ma" in df.columns else [],
                "originality": df["originality_ma"].dropna().tolist() if "originality_ma" in df.columns else []
            },
            "improvement_periods": improvement_periods,
            "deterioration_periods": deterioration_periods,
            "overall_quality_change": self._calculate_quality_change(df)
        }
    
    def _analyze_content_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analizar tendencias de contenido"""
        # Tendencias en longitud
        length_trend = self._calculate_trend_direction(df["word_count"])
        
        # Tendencias en tipos de queries
        query_trends = self._analyze_query_trends(df)
        
        # Tendencias en características de contenido
        content_features_trends = self._analyze_content_features_trends(df)
        
        return {
            "length_trend": length_trend,
            "query_trends": query_trends,
            "content_features_trends": content_features_trends
        }
    
    def _calculate_trend_direction(self, series: pd.Series) -> Dict[str, Any]:
        """Calcular dirección de tendencia"""
        if len(series) < 2:
            return {"direction": "insufficient_data", "strength": 0.0}
        
        # Calcular pendiente usando regresión lineal simple
        x = np.arange(len(series))
        y = series.values
        
        # Remover NaN
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return {"direction": "insufficient_data", "strength": 0.0}
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Calcular pendiente
        slope = np.polyfit(x_clean, y_clean, 1)[0]
        
        # Determinar dirección
        if slope > 0.01:
            direction = "increasing"
        elif slope < -0.01:
            direction = "decreasing"
        else:
            direction = "stable"
        
        # Calcular fuerza de la tendencia
        strength = abs(slope) * 100  # Normalizar
        
        return {
            "direction": direction,
            "strength": float(strength),
            "slope": float(slope)
        }
    
    def _identify_improvement_periods(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identificar períodos de mejora"""
        periods = []
        
        if "quality_ma" not in df.columns:
            return periods
        
        # Buscar períodos donde la calidad mejora consistentemente
        quality_ma = df["quality_ma"].dropna()
        if len(quality_ma) < 3:
            return periods
        
        for i in range(len(quality_ma) - 2):
            if (quality_ma.iloc[i+1] > quality_ma.iloc[i] and 
                quality_ma.iloc[i+2] > quality_ma.iloc[i+1]):
                
                periods.append({
                    "start_date": df.iloc[i]["timestamp"].isoformat(),
                    "end_date": df.iloc[i+2]["timestamp"].isoformat(),
                    "improvement": float(quality_ma.iloc[i+2] - quality_ma.iloc[i]),
                    "documents_count": 3
                })
        
        return periods
    
    def _identify_deterioration_periods(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identificar períodos de deterioro"""
        periods = []
        
        if "quality_ma" not in df.columns:
            return periods
        
        # Buscar períodos donde la calidad empeora consistentemente
        quality_ma = df["quality_ma"].dropna()
        if len(quality_ma) < 3:
            return periods
        
        for i in range(len(quality_ma) - 2):
            if (quality_ma.iloc[i+1] < quality_ma.iloc[i] and 
                quality_ma.iloc[i+2] < quality_ma.iloc[i+1]):
                
                periods.append({
                    "start_date": df.iloc[i]["timestamp"].isoformat(),
                    "end_date": df.iloc[i+2]["timestamp"].isoformat(),
                    "deterioration": float(quality_ma.iloc[i] - quality_ma.iloc[i+2]),
                    "documents_count": 3
                })
        
        return periods
    
    def _calculate_quality_change(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcular cambio general en la calidad"""
        if len(df) < 2:
            return {"change": 0.0, "percentage": 0.0}
        
        first_half = df.iloc[:len(df)//2]["quality_score"].mean()
        second_half = df.iloc[len(df)//2:]["quality_score"].mean()
        
        change = second_half - first_half
        percentage = (change / first_half) * 100 if first_half > 0 else 0
        
        return {
            "change": float(change),
            "percentage": float(percentage),
            "first_half_avg": float(first_half),
            "second_half_avg": float(second_half)
        }
    
    def _analyze_query_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analizar tendencias en queries"""
        # Agrupar por período y analizar cambios en tipos de queries
        df["period"] = df["timestamp"].dt.to_period("M")  # Agrupar por mes
        
        query_trends = {}
        for period in df["period"].unique():
            period_docs = df[df["period"] == period]
            query_trends[str(period)] = {
                "total_queries": len(period_docs),
                "unique_queries": period_docs["query"].nunique(),
                "avg_quality": period_docs["quality_score"].mean(),
                "common_query_patterns": period_docs["query"].value_counts().head(3).to_dict()
            }
        
        return query_trends
    
    def _analyze_content_features_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analizar tendencias en características de contenido"""
        trends = {}
        
        # Tendencias en longitud
        trends["word_count"] = self._calculate_trend_direction(df["word_count"])
        trends["content_length"] = self._calculate_trend_direction(df["content_length"])
        trends["sentence_count"] = self._calculate_trend_direction(df["sentence_count"])
        
        return trends
    
    def _generate_trend_summary(
        self,
        temporal_trends: Dict[str, Any],
        quality_trends: Dict[str, Any],
        content_trends: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generar resumen de tendencias"""
        summary = {
            "overall_assessment": "stable",
            "key_insights": [],
            "recommendations": []
        }
        
        # Evaluar tendencia general de calidad
        quality_change = quality_trends.get("overall_quality_change", {})
        if quality_change.get("percentage", 0) > 5:
            summary["overall_assessment"] = "improving"
            summary["key_insights"].append("Calidad en tendencia ascendente")
        elif quality_change.get("percentage", 0) < -5:
            summary["overall_assessment"] = "declining"
            summary["key_insights"].append("Calidad en tendencia descendente")
        
        # Agregar insights adicionales
        if temporal_trends.get("volume_trend", {}).get("direction") == "increasing":
            summary["key_insights"].append("Volumen de documentos en aumento")
        
        if content_trends.get("word_count", {}).get("direction") == "increasing":
            summary["key_insights"].append("Longitud promedio de documentos aumentando")
        
        # Generar recomendaciones
        if summary["overall_assessment"] == "declining":
            summary["recommendations"].append("Implementar medidas correctivas urgentes")
            summary["recommendations"].append("Revisar procesos de generación de contenido")
        elif summary["overall_assessment"] == "improving":
            summary["recommendations"].append("Mantener prácticas actuales")
            summary["recommendations"].append("Documentar factores de éxito")
        
        return summary
    
    async def _perform_correlation_analysis(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """Realizar análisis de correlaciones"""
        try:
            # Seleccionar variables numéricas
            numeric_cols = ["quality_score", "readability_score", "originality_score", 
                           "word_count", "content_length", "sentence_count", "avg_sentence_length"]
            
            # Calcular matriz de correlación
            correlation_matrix = df[numeric_cols].corr()
            
            # Encontrar correlaciones significativas
            significant_correlations = self._find_significant_correlations(correlation_matrix)
            
            # Análisis de correlaciones con calidad
            quality_correlations = self._analyze_quality_correlations(correlation_matrix)
            
            return {
                "correlation_matrix": correlation_matrix.to_dict(),
                "significant_correlations": significant_correlations,
                "quality_correlations": quality_correlations,
                "insights": self._generate_correlation_insights(significant_correlations, quality_correlations)
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {"error": str(e)}
    
    def _find_significant_correlations(self, correlation_matrix: pd.DataFrame, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Encontrar correlaciones significativas"""
        significant = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    significant.append({
                        "variable1": correlation_matrix.columns[i],
                        "variable2": correlation_matrix.columns[j],
                        "correlation": float(corr_value),
                        "strength": "strong" if abs(corr_value) >= 0.7 else "moderate"
                    })
        
        return significant
    
    def _analyze_quality_correlations(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Analizar correlaciones con la calidad"""
        quality_correlations = {}
        
        for col in correlation_matrix.columns:
            if col != "quality_score":
                corr_value = correlation_matrix.loc["quality_score", col]
                quality_correlations[col] = {
                    "correlation": float(corr_value),
                    "strength": "strong" if abs(corr_value) >= 0.7 else "moderate" if abs(corr_value) >= 0.5 else "weak",
                    "direction": "positive" if corr_value > 0 else "negative"
                }
        
        return quality_correlations
    
    def _generate_correlation_insights(
        self,
        significant_correlations: List[Dict[str, Any]],
        quality_correlations: Dict[str, Any]
    ) -> List[str]:
        """Generar insights basados en correlaciones"""
        insights = []
        
        # Insights sobre correlaciones con calidad
        for var, corr_data in quality_correlations.items():
            if corr_data["strength"] in ["strong", "moderate"]:
                direction = "positivamente" if corr_data["direction"] == "positive" else "negativamente"
                insights.append(f"La calidad se correlaciona {direction} con {var}")
        
        # Insights sobre correlaciones generales
        for corr in significant_correlations:
            if corr["strength"] == "strong":
                insights.append(f"Fuerte correlación entre {corr['variable1']} y {corr['variable2']}")
        
        return insights
    
    async def _perform_predictive_analysis(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """Realizar análisis predictivo"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Preparar datos para predicción
            features = df[["readability_score", "originality_score", "word_count", 
                          "content_length", "sentence_count", "avg_sentence_length"]].values
            target = df["quality_score"].values
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Entrenar modelo
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Hacer predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Importancia de características
            feature_importance = dict(zip(
                ["readability_score", "originality_score", "word_count", 
                 "content_length", "sentence_count", "avg_sentence_length"],
                model.feature_importances_
            ))
            
            return {
                "model_performance": {
                    "mse": float(mse),
                    "r2_score": float(r2),
                    "rmse": float(np.sqrt(mse))
                },
                "feature_importance": feature_importance,
                "predictions": {
                    "actual": y_test.tolist(),
                    "predicted": y_pred.tolist()
                },
                "insights": self._generate_predictive_insights(feature_importance, r2)
            }
            
        except Exception as e:
            logger.error(f"Error in predictive analysis: {e}")
            return {"error": str(e)}
    
    def _generate_predictive_insights(self, feature_importance: Dict[str, float], r2_score: float) -> List[str]:
        """Generar insights del análisis predictivo"""
        insights = []
        
        # Evaluar capacidad predictiva
        if r2_score > 0.7:
            insights.append("Modelo predictivo con buena capacidad de predicción")
        elif r2_score > 0.5:
            insights.append("Modelo predictivo con capacidad moderada")
        else:
            insights.append("Modelo predictivo con capacidad limitada")
        
        # Identificar características más importantes
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        for feature, importance in top_features:
            insights.append(f"{feature} es un factor importante para predecir calidad (importancia: {importance:.3f})")
        
        return insights
    
    # Métodos de utilidad para conversión a diccionario
    def _cluster_to_dict(self, cluster: DocumentCluster) -> Dict[str, Any]:
        """Convertir cluster a diccionario"""
        return {
            "id": cluster.id,
            "name": cluster.name,
            "size": cluster.size,
            "quality_avg": cluster.quality_avg,
            "characteristics": cluster.characteristics,
            "representative_documents": cluster.representative_documents,
            "created_at": cluster.created_at.isoformat()
        }
    
    def _segment_to_dict(self, segment: DocumentSegment) -> Dict[str, Any]:
        """Convertir segmento a diccionario"""
        return {
            "id": segment.id,
            "name": segment.name,
            "criteria": segment.criteria,
            "size": segment.size,
            "metrics": segment.metrics,
            "insights": segment.insights,
            "recommendations": segment.recommendations,
            "created_at": segment.created_at.isoformat()
        }
    
    def _anomaly_to_dict(self, anomaly: AnomalyResult) -> Dict[str, Any]:
        """Convertir anomalía a diccionario"""
        return {
            "document_id": anomaly.document_id,
            "anomaly_score": anomaly.anomaly_score,
            "anomaly_type": anomaly.anomaly_type,
            "severity": anomaly.severity,
            "explanation": anomaly.explanation,
            "features": anomaly.features,
            "recommendations": anomaly.recommendations,
            "detected_at": anomaly.detected_at.isoformat()
        }
    
    def _sentiment_to_dict(self, sentiment: SentimentAnalysis) -> Dict[str, Any]:
        """Convertir análisis de sentimiento a diccionario"""
        return {
            "document_id": sentiment.document_id,
            "overall_sentiment": sentiment.overall_sentiment,
            "sentiment_score": sentiment.sentiment_score,
            "confidence": sentiment.confidence,
            "emotional_tone": sentiment.emotional_tone,
            "subjectivity": sentiment.subjectivity,
            "polarity_breakdown": sentiment.polarity_breakdown,
            "key_phrases": sentiment.key_phrases,
            "analyzed_at": sentiment.analyzed_at.isoformat()
        }
    
    def _topic_to_dict(self, topic: TopicModel) -> Dict[str, Any]:
        """Convertir modelo de tópico a diccionario"""
        return {
            "id": topic.id,
            "name": topic.name,
            "keywords": topic.keywords,
            "documents": topic.documents,
            "coherence_score": topic.coherence_score,
            "perplexity_score": topic.perplexity_score,
            "word_distribution": topic.word_distribution,
            "created_at": topic.created_at.isoformat()
        }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Obtener resumen de todos los análisis realizados"""
        return {
            "clusters": len(self.clusters),
            "segments": len(self.segments),
            "anomalies": len(self.anomalies),
            "sentiment_analyses": len(self.sentiment_analyses),
            "topic_models": len(self.topic_models),
            "last_analysis": datetime.now().isoformat()
        }
    
    async def save_analysis_results(self, filename: str = None) -> str:
        """Guardar resultados del análisis"""
        if filename is None:
            filename = f"advanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.cache_directory / filename
        
        results = {
            "clusters": {k: self._cluster_to_dict(v) for k, v in self.clusters.items()},
            "segments": {k: self._segment_to_dict(v) for k, v in self.segments.items()},
            "anomalies": [self._anomaly_to_dict(a) for a in self.anomalies],
            "sentiment_analyses": {k: self._sentiment_to_dict(v) for k, v in self.sentiment_analyses.items()},
            "topic_models": {k: self._topic_to_dict(v) for k, v in self.topic_models.items()},
            "summary": self.get_analysis_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Analysis results saved to {filepath}")
        return str(filepath)
    
    async def load_analysis_results(self, filepath: str) -> Dict[str, Any]:
        """Cargar resultados del análisis"""
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Analysis results loaded from {filepath}")
        return results



























