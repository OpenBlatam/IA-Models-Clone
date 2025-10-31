"""
Advanced Behavior Pattern Analysis System for AI History Comparison
Sistema avanzado de análisis de patrones de comportamiento para análisis de historial de IA
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
import math
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BehaviorType(Enum):
    """Tipos de comportamiento"""
    CONSISTENT = "consistent"
    VARIABLE = "variable"
    TRENDING = "trending"
    CYCLICAL = "cyclical"
    ANOMALOUS = "anomalous"
    ADAPTIVE = "adaptive"
    PREDICTABLE = "predictable"
    RANDOM = "random"

class PatternComplexity(Enum):
    """Complejidad de patrones"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class InteractionType(Enum):
    """Tipos de interacción"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    NETWORK = "network"
    FEEDBACK = "feedback"

@dataclass
class BehaviorMetric:
    """Métrica de comportamiento"""
    name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

@dataclass
class BehaviorPattern:
    """Patrón de comportamiento"""
    id: str
    pattern_type: BehaviorType
    complexity: PatternComplexity
    frequency: float
    duration: timedelta
    strength: float
    confidence: float
    start_time: datetime
    end_time: datetime
    metrics: List[BehaviorMetric]
    characteristics: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class BehaviorInsight:
    """Insight de comportamiento"""
    id: str
    insight_type: str
    description: str
    significance: float
    confidence: float
    pattern_id: str
    implications: List[str]
    recommendations: List[str]
    related_patterns: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class InteractionPattern:
    """Patrón de interacción"""
    id: str
    interaction_type: InteractionType
    participants: List[str]
    frequency: float
    strength: float
    duration: timedelta
    characteristics: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedBehaviorPatternAnalyzer:
    """
    Analizador avanzado de patrones de comportamiento
    """
    
    def __init__(
        self,
        enable_clustering: bool = True,
        enable_anomaly_detection: bool = True,
        enable_interaction_analysis: bool = True,
        enable_temporal_analysis: bool = True
    ):
        self.enable_clustering = enable_clustering
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_interaction_analysis = enable_interaction_analysis
        self.enable_temporal_analysis = enable_temporal_analysis
        
        # Almacenamiento de datos y análisis
        self.behavior_metrics: Dict[str, List[BehaviorMetric]] = {}
        self.behavior_patterns: Dict[str, BehaviorPattern] = {}
        self.behavior_insights: Dict[str, BehaviorInsight] = {}
        self.interaction_patterns: Dict[str, InteractionPattern] = {}
        
        # Configuración
        self.config = {
            "min_pattern_duration": timedelta(minutes=5),
            "max_pattern_duration": timedelta(days=30),
            "min_metrics_for_pattern": 10,
            "anomaly_threshold": 0.1,
            "clustering_min_samples": 5,
            "pattern_confidence_threshold": 0.6,
            "interaction_threshold": 0.5
        }
        
        # Modelos ML
        self.clustering_models = {}
        self.anomaly_detectors = {}
        self.scaler = StandardScaler()
    
    async def add_behavior_metrics(
        self,
        entity_id: str,
        metrics: List[BehaviorMetric],
        replace_existing: bool = False
    ) -> bool:
        """
        Agregar métricas de comportamiento
        
        Args:
            entity_id: ID de la entidad
            metrics: Lista de métricas
            replace_existing: Si reemplazar métricas existentes
            
        Returns:
            True si se agregaron exitosamente
        """
        try:
            if len(metrics) < self.config["min_metrics_for_pattern"]:
                logger.warning(f"Insufficient metrics for {entity_id}: {len(metrics)}")
                return False
            
            # Validar y ordenar métricas
            validated_metrics = []
            for metric in metrics:
                if isinstance(metric.timestamp, str):
                    metric.timestamp = datetime.fromisoformat(metric.timestamp)
                validated_metrics.append(metric)
            
            # Ordenar por timestamp
            validated_metrics.sort(key=lambda x: x.timestamp)
            
            if replace_existing or entity_id not in self.behavior_metrics:
                self.behavior_metrics[entity_id] = validated_metrics
            else:
                self.behavior_metrics[entity_id].extend(validated_metrics)
                self.behavior_metrics[entity_id].sort(key=lambda x: x.timestamp)
            
            logger.info(f"Added {len(validated_metrics)} behavior metrics for entity {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding behavior metrics for {entity_id}: {e}")
            return False
    
    async def analyze_behavior_patterns(
        self,
        entity_id: str,
        include_anomalies: bool = True,
        include_interactions: bool = True
    ) -> List[BehaviorPattern]:
        """
        Analizar patrones de comportamiento
        
        Args:
            entity_id: ID de la entidad
            include_anomalies: Si incluir detección de anomalías
            include_interactions: Si incluir análisis de interacciones
            
        Returns:
            Lista de patrones identificados
        """
        try:
            if entity_id not in self.behavior_metrics:
                raise ValueError(f"No behavior metrics available for entity {entity_id}")
            
            metrics = self.behavior_metrics[entity_id]
            logger.info(f"Analyzing behavior patterns for entity {entity_id} with {len(metrics)} metrics")
            
            patterns = []
            
            # 1. Análisis de patrones temporales
            temporal_patterns = await self._analyze_temporal_patterns(entity_id, metrics)
            patterns.extend(temporal_patterns)
            
            # 2. Análisis de patrones de frecuencia
            frequency_patterns = await self._analyze_frequency_patterns(entity_id, metrics)
            patterns.extend(frequency_patterns)
            
            # 3. Análisis de patrones de valores
            value_patterns = await self._analyze_value_patterns(entity_id, metrics)
            patterns.extend(value_patterns)
            
            # 4. Análisis de anomalías
            if include_anomalies and self.enable_anomaly_detection:
                anomaly_patterns = await self._analyze_anomaly_patterns(entity_id, metrics)
                patterns.extend(anomaly_patterns)
            
            # 5. Análisis de interacciones
            if include_interactions and self.enable_interaction_analysis:
                interaction_patterns = await self._analyze_interaction_patterns(entity_id, metrics)
                patterns.extend(interaction_patterns)
            
            # 6. Análisis de clustering
            if self.enable_clustering:
                cluster_patterns = await self._analyze_cluster_patterns(entity_id, metrics)
                patterns.extend(cluster_patterns)
            
            # Almacenar patrones
            for pattern in patterns:
                self.behavior_patterns[pattern.id] = pattern
            
            # Generar insights
            await self._generate_behavior_insights(entity_id, patterns)
            
            logger.info(f"Identified {len(patterns)} behavior patterns for entity {entity_id}")
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing behavior patterns for {entity_id}: {e}")
            raise
    
    async def _analyze_temporal_patterns(
        self,
        entity_id: str,
        metrics: List[BehaviorMetric]
    ) -> List[BehaviorPattern]:
        """Analizar patrones temporales"""
        patterns = []
        
        try:
            if len(metrics) < 10:
                return patterns
            
            # Agrupar métricas por nombre
            metric_groups = defaultdict(list)
            for metric in metrics:
                metric_groups[metric.name].append(metric)
            
            for metric_name, metric_list in metric_groups.items():
                if len(metric_list) < 5:
                    continue
                
                # Ordenar por timestamp
                metric_list.sort(key=lambda x: x.timestamp)
                
                # Análisis de tendencia temporal
                timestamps = [m.timestamp for m in metric_list]
                values = [m.value for m in metric_list]
                
                # Convertir timestamps a números
                time_numeric = [(ts - timestamps[0]).total_seconds() / 3600 for ts in timestamps]
                
                # Regresión lineal para detectar tendencias
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
                
                # Determinar tipo de patrón
                if p_value < 0.05:  # Tendencia significativa
                    if slope > 0:
                        pattern_type = BehaviorType.TRENDING
                        strength = abs(slope) * r_value
                    else:
                        pattern_type = BehaviorType.TRENDING
                        strength = abs(slope) * r_value
                else:
                    # Análisis de variabilidad
                    coefficient_of_variation = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                    
                    if coefficient_of_variation < 0.1:
                        pattern_type = BehaviorType.CONSISTENT
                        strength = 1 - coefficient_of_variation
                    elif coefficient_of_variation > 0.5:
                        pattern_type = BehaviorType.VARIABLE
                        strength = coefficient_of_variation
                    else:
                        pattern_type = BehaviorType.PREDICTABLE
                        strength = 0.5
                
                # Análisis de ciclicidad
                if len(values) > 20:
                    cyclic_score = await self._analyze_cyclicity(values, timestamps)
                    if cyclic_score > 0.7:
                        pattern_type = BehaviorType.CYCLICAL
                        strength = cyclic_score
                
                # Calcular duración
                duration = timestamps[-1] - timestamps[0]
                
                # Determinar complejidad
                complexity = self._determine_pattern_complexity(values, timestamps)
                
                # Crear patrón
                pattern = BehaviorPattern(
                    id=f"temporal_{entity_id}_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    pattern_type=pattern_type,
                    complexity=complexity,
                    frequency=len(metric_list) / duration.total_seconds() * 3600,  # Por hora
                    duration=duration,
                    strength=strength,
                    confidence=min(1.0, r_value ** 2 if p_value < 0.05 else 0.5),
                    start_time=timestamps[0],
                    end_time=timestamps[-1],
                    metrics=metric_list,
                    characteristics={
                        "metric_name": metric_name,
                        "slope": slope,
                        "r_squared": r_value ** 2,
                        "p_value": p_value,
                        "coefficient_of_variation": coefficient_of_variation,
                        "cyclic_score": cyclic_score if 'cyclic_score' in locals() else 0
                    },
                    anomalies=[]
                )
                
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
            return []
    
    async def _analyze_frequency_patterns(
        self,
        entity_id: str,
        metrics: List[BehaviorMetric]
    ) -> List[BehaviorPattern]:
        """Analizar patrones de frecuencia"""
        patterns = []
        
        try:
            # Agrupar métricas por nombre
            metric_groups = defaultdict(list)
            for metric in metrics:
                metric_groups[metric.name].append(metric)
            
            for metric_name, metric_list in metric_groups.items():
                if len(metric_list) < 10:
                    continue
                
                # Ordenar por timestamp
                metric_list.sort(key=lambda x: x.timestamp)
                
                # Calcular intervalos entre métricas
                intervals = []
                for i in range(1, len(metric_list)):
                    interval = (metric_list[i].timestamp - metric_list[i-1].timestamp).total_seconds()
                    intervals.append(interval)
                
                if not intervals:
                    continue
                
                # Análisis de frecuencia
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                coefficient_of_variation = std_interval / mean_interval if mean_interval > 0 else 0
                
                # Determinar tipo de patrón de frecuencia
                if coefficient_of_variation < 0.2:
                    pattern_type = BehaviorType.CONSISTENT
                    strength = 1 - coefficient_of_variation
                elif coefficient_of_variation > 0.8:
                    pattern_type = BehaviorType.RANDOM
                    strength = coefficient_of_variation
                else:
                    pattern_type = BehaviorType.PREDICTABLE
                    strength = 0.5
                
                # Análisis de tendencia en frecuencia
                if len(intervals) > 5:
                    time_numeric = list(range(len(intervals)))
                    slope, _, r_value, p_value, _ = stats.linregress(time_numeric, intervals)
                    
                    if p_value < 0.05:
                        if slope > 0:
                            # Frecuencia decreciente
                            pattern_type = BehaviorType.TRENDING
                            strength = abs(slope) * r_value
                        else:
                            # Frecuencia creciente
                            pattern_type = BehaviorType.TRENDING
                            strength = abs(slope) * r_value
                
                # Calcular duración
                duration = metric_list[-1].timestamp - metric_list[0].timestamp
                
                # Determinar complejidad
                complexity = self._determine_pattern_complexity(intervals, [metric_list[0].timestamp] * len(intervals))
                
                # Crear patrón
                pattern = BehaviorPattern(
                    id=f"frequency_{entity_id}_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    pattern_type=pattern_type,
                    complexity=complexity,
                    frequency=len(metric_list) / duration.total_seconds() * 3600,
                    duration=duration,
                    strength=strength,
                    confidence=min(1.0, 1 - coefficient_of_variation),
                    start_time=metric_list[0].timestamp,
                    end_time=metric_list[-1].timestamp,
                    metrics=metric_list,
                    characteristics={
                        "metric_name": metric_name,
                        "mean_interval": mean_interval,
                        "std_interval": std_interval,
                        "coefficient_of_variation": coefficient_of_variation,
                        "frequency_trend_slope": slope if 'slope' in locals() else 0,
                        "frequency_trend_r_squared": r_value ** 2 if 'r_value' in locals() else 0
                    },
                    anomalies=[]
                )
                
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing frequency patterns: {e}")
            return []
    
    async def _analyze_value_patterns(
        self,
        entity_id: str,
        metrics: List[BehaviorMetric]
    ) -> List[BehaviorPattern]:
        """Analizar patrones de valores"""
        patterns = []
        
        try:
            # Agrupar métricas por nombre
            metric_groups = defaultdict(list)
            for metric in metrics:
                metric_groups[metric.name].append(metric)
            
            for metric_name, metric_list in metric_groups.items():
                if len(metric_list) < 10:
                    continue
                
                values = [m.value for m in metric_list]
                
                # Análisis de distribución de valores
                mean_value = np.mean(values)
                std_value = np.std(values)
                coefficient_of_variation = std_value / mean_value if mean_value != 0 else 0
                
                # Análisis de outliers
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                outlier_threshold = 1.5 * iqr
                outliers = [v for v in values if v < q1 - outlier_threshold or v > q3 + outlier_threshold]
                outlier_ratio = len(outliers) / len(values)
                
                # Determinar tipo de patrón
                if outlier_ratio > 0.2:
                    pattern_type = BehaviorType.ANOMALOUS
                    strength = outlier_ratio
                elif coefficient_of_variation < 0.1:
                    pattern_type = BehaviorType.CONSISTENT
                    strength = 1 - coefficient_of_variation
                elif coefficient_of_variation > 0.5:
                    pattern_type = BehaviorType.VARIABLE
                    strength = coefficient_of_variation
                else:
                    pattern_type = BehaviorType.PREDICTABLE
                    strength = 0.5
                
                # Análisis de autocorrelación
                if len(values) > 20:
                    autocorr_score = await self._analyze_autocorrelation(values)
                    if autocorr_score > 0.5:
                        pattern_type = BehaviorType.CYCLICAL
                        strength = autocorr_score
                
                # Calcular duración
                duration = metric_list[-1].timestamp - metric_list[0].timestamp
                
                # Determinar complejidad
                complexity = self._determine_pattern_complexity(values, [m.timestamp for m in metric_list])
                
                # Crear patrón
                pattern = BehaviorPattern(
                    id=f"value_{entity_id}_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    pattern_type=pattern_type,
                    complexity=complexity,
                    frequency=len(metric_list) / duration.total_seconds() * 3600,
                    duration=duration,
                    strength=strength,
                    confidence=min(1.0, 1 - outlier_ratio),
                    start_time=metric_list[0].timestamp,
                    end_time=metric_list[-1].timestamp,
                    metrics=metric_list,
                    characteristics={
                        "metric_name": metric_name,
                        "mean_value": mean_value,
                        "std_value": std_value,
                        "coefficient_of_variation": coefficient_of_variation,
                        "outlier_ratio": outlier_ratio,
                        "autocorr_score": autocorr_score if 'autocorr_score' in locals() else 0,
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr
                    },
                    anomalies=[]
                )
                
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing value patterns: {e}")
            return []
    
    async def _analyze_anomaly_patterns(
        self,
        entity_id: str,
        metrics: List[BehaviorMetric]
    ) -> List[BehaviorPattern]:
        """Analizar patrones de anomalías"""
        patterns = []
        
        try:
            if not self.enable_anomaly_detection:
                return patterns
            
            # Agrupar métricas por nombre
            metric_groups = defaultdict(list)
            for metric in metrics:
                metric_groups[metric.name].append(metric)
            
            for metric_name, metric_list in metric_groups.items():
                if len(metric_list) < 10:
                    continue
                
                # Preparar datos para detección de anomalías
                values = np.array([m.value for m in metric_list]).reshape(-1, 1)
                timestamps = [m.timestamp for m in metric_list]
                
                # Usar Isolation Forest
                iso_forest = IsolationForest(contamination=self.config["anomaly_threshold"], random_state=42)
                anomaly_labels = iso_forest.fit_predict(values)
                
                # Identificar anomalías
                anomalies = []
                for i, label in enumerate(anomaly_labels):
                    if label == -1:  # Anomalía
                        anomalies.append({
                            "timestamp": timestamps[i].isoformat(),
                            "value": values[i][0],
                            "anomaly_score": iso_forest.decision_function(values[i].reshape(1, -1))[0],
                            "method": "isolation_forest"
                        })
                
                if len(anomalies) > 0:
                    # Crear patrón de anomalías
                    pattern = BehaviorPattern(
                        id=f"anomaly_{entity_id}_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        pattern_type=BehaviorType.ANOMALOUS,
                        complexity=PatternComplexity.MODERATE,
                        frequency=len(anomalies) / (timestamps[-1] - timestamps[0]).total_seconds() * 3600,
                        duration=timestamps[-1] - timestamps[0],
                        strength=len(anomalies) / len(metric_list),
                        confidence=0.8,
                        start_time=timestamps[0],
                        end_time=timestamps[-1],
                        metrics=metric_list,
                        characteristics={
                            "metric_name": metric_name,
                            "total_anomalies": len(anomalies),
                            "anomaly_ratio": len(anomalies) / len(metric_list),
                            "anomaly_method": "isolation_forest"
                        },
                        anomalies=anomalies
                    )
                    
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing anomaly patterns: {e}")
            return []
    
    async def _analyze_interaction_patterns(
        self,
        entity_id: str,
        metrics: List[BehaviorMetric]
    ) -> List[BehaviorPattern]:
        """Analizar patrones de interacción"""
        patterns = []
        
        try:
            if not self.enable_interaction_analysis:
                return patterns
            
            # Agrupar métricas por contexto de interacción
            interaction_groups = defaultdict(list)
            for metric in metrics:
                if 'interaction_partner' in metric.context:
                    partner = metric.context['interaction_partner']
                    interaction_groups[partner].append(metric)
            
            for partner, interaction_metrics in interaction_groups.items():
                if len(interaction_metrics) < 5:
                    continue
                
                # Análisis de frecuencia de interacción
                interaction_metrics.sort(key=lambda x: x.timestamp)
                intervals = []
                for i in range(1, len(interaction_metrics)):
                    interval = (interaction_metrics[i].timestamp - interaction_metrics[i-1].timestamp).total_seconds()
                    intervals.append(interval)
                
                mean_interval = np.mean(intervals) if intervals else 0
                interaction_frequency = 1 / mean_interval if mean_interval > 0 else 0
                
                # Análisis de intensidad de interacción
                interaction_values = [m.value for m in interaction_metrics]
                mean_intensity = np.mean(interaction_values)
                
                # Determinar tipo de interacción
                if mean_interval < 3600:  # Menos de 1 hora
                    interaction_type = InteractionType.SEQUENTIAL
                elif mean_interval < 86400:  # Menos de 1 día
                    interaction_type = InteractionType.PARALLEL
                else:
                    interaction_type = InteractionType.HIERARCHICAL
                
                # Crear patrón de interacción
                pattern = BehaviorPattern(
                    id=f"interaction_{entity_id}_{partner}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    pattern_type=BehaviorType.ADAPTIVE,
                    complexity=PatternComplexity.MODERATE,
                    frequency=interaction_frequency,
                    duration=interaction_metrics[-1].timestamp - interaction_metrics[0].timestamp,
                    strength=mean_intensity,
                    confidence=0.7,
                    start_time=interaction_metrics[0].timestamp,
                    end_time=interaction_metrics[-1].timestamp,
                    metrics=interaction_metrics,
                    characteristics={
                        "interaction_partner": partner,
                        "interaction_type": interaction_type.value,
                        "mean_interval": mean_interval,
                        "mean_intensity": mean_intensity,
                        "total_interactions": len(interaction_metrics)
                    },
                    anomalies=[]
                )
                
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing interaction patterns: {e}")
            return []
    
    async def _analyze_cluster_patterns(
        self,
        entity_id: str,
        metrics: List[BehaviorMetric]
    ) -> List[BehaviorPattern]:
        """Analizar patrones de clustering"""
        patterns = []
        
        try:
            if not self.enable_clustering or len(metrics) < 20:
                return patterns
            
            # Preparar datos para clustering
            feature_matrix = []
            for metric in metrics:
                features = [
                    metric.value,
                    metric.timestamp.hour,
                    metric.timestamp.weekday(),
                    metric.confidence
                ]
                # Agregar características del contexto si están disponibles
                if 'context_features' in metric.context:
                    features.extend(metric.context['context_features'])
                feature_matrix.append(features)
            
            feature_matrix = np.array(feature_matrix)
            
            # Normalizar características
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            
            # Aplicar clustering
            n_clusters = min(5, len(metrics) // 4)  # Máximo 5 clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
            
            # Analizar cada cluster
            for cluster_id in range(n_clusters):
                cluster_metrics = [metrics[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_metrics) < self.config["clustering_min_samples"]:
                    continue
                
                # Características del cluster
                cluster_values = [m.value for m in cluster_metrics]
                cluster_timestamps = [m.timestamp for m in cluster_metrics]
                
                # Análisis de coherencia del cluster
                cluster_coherence = 1 - (np.std(cluster_values) / np.mean(cluster_values)) if np.mean(cluster_values) != 0 else 0
                
                # Determinar tipo de patrón
                if cluster_coherence > 0.8:
                    pattern_type = BehaviorType.CONSISTENT
                elif cluster_coherence > 0.5:
                    pattern_type = BehaviorType.PREDICTABLE
                else:
                    pattern_type = BehaviorType.VARIABLE
                
                # Crear patrón de cluster
                pattern = BehaviorPattern(
                    id=f"cluster_{entity_id}_{cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    pattern_type=pattern_type,
                    complexity=PatternComplexity.MODERATE,
                    frequency=len(cluster_metrics) / (cluster_timestamps[-1] - cluster_timestamps[0]).total_seconds() * 3600,
                    duration=cluster_timestamps[-1] - cluster_timestamps[0],
                    strength=cluster_coherence,
                    confidence=0.7,
                    start_time=cluster_timestamps[0],
                    end_time=cluster_timestamps[-1],
                    metrics=cluster_metrics,
                    characteristics={
                        "cluster_id": cluster_id,
                        "cluster_size": len(cluster_metrics),
                        "cluster_coherence": cluster_coherence,
                        "mean_value": np.mean(cluster_values),
                        "std_value": np.std(cluster_values),
                        "silhouette_score": silhouette_score(feature_matrix_scaled, cluster_labels)
                    },
                    anomalies=[]
                )
                
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing cluster patterns: {e}")
            return []
    
    async def _analyze_cyclicity(self, values: List[float], timestamps: List[datetime]) -> float:
        """Analizar ciclicidad en los valores"""
        try:
            if len(values) < 20:
                return 0.0
            
            # Análisis de autocorrelación
            autocorr_scores = []
            for lag in range(1, min(10, len(values) // 2)):
                if lag < len(values):
                    corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorr_scores.append(abs(corr))
            
            if autocorr_scores:
                return max(autocorr_scores)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error analyzing cyclicity: {e}")
            return 0.0
    
    async def _analyze_autocorrelation(self, values: List[float]) -> float:
        """Analizar autocorrelación"""
        try:
            if len(values) < 10:
                return 0.0
            
            # Calcular autocorrelación para diferentes lags
            max_corr = 0.0
            for lag in range(1, min(5, len(values) // 2)):
                if lag < len(values):
                    corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                    if not np.isnan(corr):
                        max_corr = max(max_corr, abs(corr))
            
            return max_corr
            
        except Exception as e:
            logger.error(f"Error analyzing autocorrelation: {e}")
            return 0.0
    
    def _determine_pattern_complexity(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> PatternComplexity:
        """Determinar complejidad del patrón"""
        try:
            if len(values) < 5:
                return PatternComplexity.SIMPLE
            
            # Análisis de variabilidad
            coefficient_of_variation = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            
            # Análisis de tendencia
            if len(values) > 3:
                time_numeric = list(range(len(values)))
                slope, _, r_value, _, _ = stats.linregress(time_numeric, values)
                trend_strength = abs(slope) * r_value
            else:
                trend_strength = 0
            
            # Análisis de ciclicidad
            cyclic_score = 0
            if len(values) > 10:
                for lag in range(1, min(5, len(values) // 2)):
                    if lag < len(values):
                        corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                        if not np.isnan(corr):
                            cyclic_score = max(cyclic_score, abs(corr))
            
            # Calcular score de complejidad
            complexity_score = (coefficient_of_variation * 0.4 + 
                              trend_strength * 0.3 + 
                              cyclic_score * 0.3)
            
            if complexity_score < 0.2:
                return PatternComplexity.SIMPLE
            elif complexity_score < 0.5:
                return PatternComplexity.MODERATE
            elif complexity_score < 0.8:
                return PatternComplexity.COMPLEX
            else:
                return PatternComplexity.VERY_COMPLEX
                
        except Exception as e:
            logger.error(f"Error determining pattern complexity: {e}")
            return PatternComplexity.SIMPLE
    
    async def _generate_behavior_insights(
        self,
        entity_id: str,
        patterns: List[BehaviorPattern]
    ):
        """Generar insights de comportamiento"""
        insights = []
        
        try:
            if not patterns:
                return
            
            # Insight 1: Patrón dominante
            dominant_pattern = max(patterns, key=lambda p: p.strength)
            insight = BehaviorInsight(
                id=f"dominant_pattern_{entity_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                insight_type="dominant_pattern",
                description=f"Patrón dominante: {dominant_pattern.pattern_type.value} con fuerza {dominant_pattern.strength:.2f}",
                significance=dominant_pattern.strength,
                confidence=dominant_pattern.confidence,
                pattern_id=dominant_pattern.id,
                implications=[
                    f"El comportamiento muestra tendencia hacia {dominant_pattern.pattern_type.value}",
                    f"La complejidad del patrón es {dominant_pattern.complexity.value}"
                ],
                recommendations=[
                    f"Monitorear cambios en el patrón {dominant_pattern.pattern_type.value}",
                    "Considerar factores que puedan influir en este patrón"
                ],
                related_patterns=[p.id for p in patterns if p.pattern_type == dominant_pattern.pattern_type]
            )
            insights.append(insight)
            
            # Insight 2: Anomalías detectadas
            anomaly_patterns = [p for p in patterns if p.pattern_type == BehaviorType.ANOMALOUS]
            if anomaly_patterns:
                total_anomalies = sum(len(p.anomalies) for p in anomaly_patterns)
                insight = BehaviorInsight(
                    id=f"anomalies_{entity_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    insight_type="anomalies",
                    description=f"Se detectaron {total_anomalies} anomalías en {len(anomaly_patterns)} patrones",
                    significance=total_anomalies / len(patterns),
                    confidence=0.8,
                    pattern_id=anomaly_patterns[0].id,
                    implications=[
                        "Presencia de comportamientos inusuales",
                        "Posible necesidad de investigación adicional"
                    ],
                    recommendations=[
                        "Investigar las causas de las anomalías",
                        "Considerar si las anomalías son problemáticas"
                    ],
                    related_patterns=[p.id for p in anomaly_patterns]
                )
                insights.append(insight)
            
            # Insight 3: Consistencia general
            consistent_patterns = [p for p in patterns if p.pattern_type == BehaviorType.CONSISTENT]
            consistency_ratio = len(consistent_patterns) / len(patterns)
            
            insight = BehaviorInsight(
                id=f"consistency_{entity_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                insight_type="consistency",
                description=f"Consistencia general: {consistency_ratio:.1%} de patrones son consistentes",
                significance=consistency_ratio,
                confidence=0.7,
                pattern_id=patterns[0].id,
                implications=[
                    f"El comportamiento es {'muy' if consistency_ratio > 0.7 else 'moderadamente' if consistency_ratio > 0.4 else 'poco'} consistente"
                ],
                recommendations=[
                    "Mantener monitoreo de consistencia" if consistency_ratio > 0.7 else "Investigar causas de inconsistencia"
                ],
                related_patterns=[p.id for p in consistent_patterns]
            )
            insights.append(insight)
            
            # Almacenar insights
            for insight in insights:
                self.behavior_insights[insight.id] = insight
                
        except Exception as e:
            logger.error(f"Error generating behavior insights: {e}")
    
    async def compare_behavior_patterns(
        self,
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """Comparar patrones de comportamiento entre entidades"""
        try:
            if len(entity_ids) < 2:
                raise ValueError("Se necesitan al menos 2 entidades para comparar")
            
            # Obtener patrones para cada entidad
            entity_patterns = {}
            for entity_id in entity_ids:
                patterns = [p for p in self.behavior_patterns.values() if entity_id in p.id]
                entity_patterns[entity_id] = patterns
            
            # Calcular similitudes
            similarities = await self._calculate_pattern_similarities(entity_patterns)
            
            # Encontrar diferencias
            differences = await self._find_pattern_differences(entity_patterns)
            
            # Generar insights comparativos
            comparative_insights = await self._generate_comparative_insights(entity_patterns)
            
            return {
                "entity_ids": entity_ids,
                "entity_patterns": {entity_id: len(patterns) for entity_id, patterns in entity_patterns.items()},
                "similarities": similarities,
                "differences": differences,
                "comparative_insights": comparative_insights,
                "comparison_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing behavior patterns: {e}")
            raise
    
    async def _calculate_pattern_similarities(
        self,
        entity_patterns: Dict[str, List[BehaviorPattern]]
    ) -> Dict[str, float]:
        """Calcular similitudes entre patrones"""
        similarities = {}
        
        try:
            entity_ids = list(entity_patterns.keys())
            
            for i, entity1 in enumerate(entity_ids):
                for j, entity2 in enumerate(entity_ids[i+1:], i+1):
                    patterns1 = entity_patterns[entity1]
                    patterns2 = entity_patterns[entity2]
                    
                    if not patterns1 or not patterns2:
                        similarities[f"{entity1}_vs_{entity2}"] = 0.0
                        continue
                    
                    # Calcular similitud basada en tipos de patrón
                    types1 = [p.pattern_type for p in patterns1]
                    types2 = [p.pattern_type for p in patterns2]
                    
                    # Similitud de tipos
                    common_types = set(types1) & set(types2)
                    type_similarity = len(common_types) / max(len(set(types1)), len(set(types2)))
                    
                    # Similitud de complejidad
                    complexities1 = [p.complexity for p in patterns1]
                    complexities2 = [p.complexity for p in patterns2]
                    
                    common_complexities = set(complexities1) & set(complexities2)
                    complexity_similarity = len(common_complexities) / max(len(set(complexities1)), len(set(complexities2)))
                    
                    # Similitud de fuerza promedio
                    avg_strength1 = np.mean([p.strength for p in patterns1])
                    avg_strength2 = np.mean([p.strength for p in patterns2])
                    strength_similarity = 1 - abs(avg_strength1 - avg_strength2)
                    
                    # Similitud combinada
                    combined_similarity = (type_similarity * 0.4 + 
                                         complexity_similarity * 0.3 + 
                                         strength_similarity * 0.3)
                    
                    similarities[f"{entity1}_vs_{entity2}"] = combined_similarity
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating pattern similarities: {e}")
            return {}
    
    async def _find_pattern_differences(
        self,
        entity_patterns: Dict[str, List[BehaviorPattern]]
    ) -> List[Dict[str, Any]]:
        """Encontrar diferencias entre patrones"""
        differences = []
        
        try:
            # Comparar tipos de patrón
            all_pattern_types = set()
            for patterns in entity_patterns.values():
                all_pattern_types.update(p.pattern_type for p in patterns)
            
            for pattern_type in all_pattern_types:
                entities_with_pattern = [
                    entity_id for entity_id, patterns in entity_patterns.items()
                    if any(p.pattern_type == pattern_type for p in patterns)
                ]
                
                if len(entities_with_pattern) < len(entity_patterns):
                    differences.append({
                        "type": "pattern_type",
                        "description": f"Patrón {pattern_type.value} presente solo en {len(entities_with_pattern)} entidades",
                        "entities": entities_with_pattern,
                        "impact": "medium"
                    })
            
            # Comparar complejidades
            all_complexities = set()
            for patterns in entity_patterns.values():
                all_complexities.update(p.complexity for p in patterns)
            
            for complexity in all_complexities:
                entities_with_complexity = [
                    entity_id for entity_id, patterns in entity_patterns.items()
                    if any(p.complexity == complexity for p in patterns)
                ]
                
                if len(entities_with_complexity) < len(entity_patterns):
                    differences.append({
                        "type": "complexity",
                        "description": f"Complejidad {complexity.value} presente solo en {len(entities_with_complexity)} entidades",
                        "entities": entities_with_complexity,
                        "impact": "low"
                    })
            
            return differences
            
        except Exception as e:
            logger.error(f"Error finding pattern differences: {e}")
            return []
    
    async def _generate_comparative_insights(
        self,
        entity_patterns: Dict[str, List[BehaviorPattern]]
    ) -> List[str]:
        """Generar insights comparativos"""
        insights = []
        
        try:
            # Insight sobre diversidad de patrones
            pattern_counts = [len(patterns) for patterns in entity_patterns.values()]
            avg_patterns = np.mean(pattern_counts)
            
            if np.std(pattern_counts) > avg_patterns * 0.5:
                insights.append("Alta variabilidad en el número de patrones entre entidades")
            else:
                insights.append("Número de patrones relativamente consistente entre entidades")
            
            # Insight sobre tipos de patrón
            all_pattern_types = set()
            for patterns in entity_patterns.values():
                all_pattern_types.update(p.pattern_type for p in patterns)
            
            insights.append(f"Se identificaron {len(all_pattern_types)} tipos diferentes de patrones")
            
            # Insight sobre anomalías
            total_anomalies = sum(
                sum(len(p.anomalies) for p in patterns if p.pattern_type == BehaviorType.ANOMALOUS)
                for patterns in entity_patterns.values()
            )
            
            if total_anomalies > 0:
                insights.append(f"Se detectaron {total_anomalies} anomalías en total")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating comparative insights: {e}")
            return []
    
    async def get_behavior_summary(self) -> Dict[str, Any]:
        """Obtener resumen de análisis de comportamiento"""
        if not self.behavior_patterns:
            return {"message": "No behavior patterns available"}
        
        # Estadísticas generales
        total_patterns = len(self.behavior_patterns)
        total_insights = len(self.behavior_insights)
        
        # Distribución de tipos de patrón
        pattern_types = [pattern.pattern_type for pattern in self.behavior_patterns.values()]
        pattern_type_distribution = Counter(pattern_types)
        
        # Distribución de complejidades
        complexities = [pattern.complexity for pattern in self.behavior_patterns.values()]
        complexity_distribution = Counter(complexities)
        
        # Estadísticas de fuerza y confianza
        strengths = [pattern.strength for pattern in self.behavior_patterns.values()]
        confidences = [pattern.confidence for pattern in self.behavior_patterns.values()]
        
        # Anomalías
        total_anomalies = sum(len(pattern.anomalies) for pattern in self.behavior_patterns.values())
        
        return {
            "total_patterns": total_patterns,
            "total_insights": total_insights,
            "pattern_type_distribution": {pattern_type.value: count for pattern_type, count in pattern_type_distribution.items()},
            "complexity_distribution": {complexity.value: count for complexity, count in complexity_distribution.items()},
            "average_strength": np.mean(strengths),
            "average_confidence": np.mean(confidences),
            "total_anomalies": total_anomalies,
            "entities_analyzed": len(self.behavior_metrics),
            "last_analysis": max([pattern.created_at for pattern in self.behavior_patterns.values()]).isoformat()
        }
    
    async def export_behavior_analysis(self, filepath: str = None) -> str:
        """Exportar análisis de comportamiento"""
        try:
            if filepath is None:
                filepath = f"exports/behavior_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Crear directorio si no existe
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Preparar datos para exportación
            export_data = {
                "behavior_patterns": {
                    pattern_id: {
                        "pattern_type": pattern.pattern_type.value,
                        "complexity": pattern.complexity.value,
                        "frequency": pattern.frequency,
                        "duration_seconds": pattern.duration.total_seconds(),
                        "strength": pattern.strength,
                        "confidence": pattern.confidence,
                        "start_time": pattern.start_time.isoformat(),
                        "end_time": pattern.end_time.isoformat(),
                        "metrics_count": len(pattern.metrics),
                        "characteristics": pattern.characteristics,
                        "anomalies": pattern.anomalies,
                        "created_at": pattern.created_at.isoformat()
                    }
                    for pattern_id, pattern in self.behavior_patterns.items()
                },
                "behavior_insights": {
                    insight_id: {
                        "insight_type": insight.insight_type,
                        "description": insight.description,
                        "significance": insight.significance,
                        "confidence": insight.confidence,
                        "pattern_id": insight.pattern_id,
                        "implications": insight.implications,
                        "recommendations": insight.recommendations,
                        "related_patterns": insight.related_patterns,
                        "created_at": insight.created_at.isoformat()
                    }
                    for insight_id, insight in self.behavior_insights.items()
                },
                "summary": await self.get_behavior_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Behavior analysis exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting behavior analysis: {e}")
            raise

























