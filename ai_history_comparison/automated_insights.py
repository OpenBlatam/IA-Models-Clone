"""
Automated Insights Generator for AI History Comparison System
Generador automático de insights para el sistema de análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsightType(Enum):
    """Tipos de insights disponibles"""
    QUALITY_TREND = "quality_trend"
    PERFORMANCE_PATTERN = "performance_pattern"
    CONTENT_ANALYSIS = "content_analysis"
    QUERY_OPTIMIZATION = "query_optimization"
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTIVE_INSIGHT = "predictive_insight"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    RECOMMENDATION_ENGINE = "recommendation_engine"

class InsightPriority(Enum):
    """Prioridades de insights"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class InsightCategory(Enum):
    """Categorías de insights"""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    OPTIMIZATION = "optimization"
    ANOMALY = "anomaly"
    TREND = "trend"
    PREDICTION = "prediction"

@dataclass
class AutomatedInsight:
    """Insight generado automáticamente"""
    id: str
    type: InsightType
    category: InsightCategory
    priority: InsightPriority
    title: str
    description: str
    confidence: float
    impact_score: float
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    actionable_items: List[str]
    related_documents: List[str]
    generated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class InsightRule:
    """Regla para generar insights automáticamente"""
    id: str
    name: str
    description: str
    condition: str
    insight_type: InsightType
    priority: InsightPriority
    category: InsightCategory
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)

class AutomatedInsightsGenerator:
    """
    Generador automático de insights para el sistema de análisis de historial de IA
    """
    
    def __init__(
        self,
        min_confidence: float = 0.7,
        min_impact_score: float = 0.5,
        max_insights_per_run: int = 50
    ):
        self.min_confidence = min_confidence
        self.min_impact_score = min_impact_score
        self.max_insights_per_run = max_insights_per_run
        
        # Almacenamiento de insights
        self.insights: Dict[str, AutomatedInsight] = {}
        self.insight_rules: Dict[str, InsightRule] = {}
        
        # Configuración de análisis
        self.analysis_config = {
            "quality_thresholds": {
                "excellent": 0.8,
                "good": 0.6,
                "average": 0.4,
                "poor": 0.2
            },
            "trend_analysis": {
                "min_data_points": 5,
                "significance_threshold": 0.05
            },
            "anomaly_detection": {
                "z_score_threshold": 2.0,
                "iqr_multiplier": 1.5
            }
        }
        
        # Inicializar reglas por defecto
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Inicializar reglas de insights por defecto"""
        default_rules = [
            InsightRule(
                id="quality_decline",
                name="Declive de Calidad",
                description="Detecta cuando la calidad promedio de documentos está disminuyendo",
                condition="quality_trend_declining",
                insight_type=InsightType.QUALITY_TREND,
                priority=InsightPriority.HIGH,
                category=InsightCategory.QUALITY
            ),
            InsightRule(
                id="quality_improvement",
                name="Mejora de Calidad",
                description="Detecta cuando la calidad promedio de documentos está mejorando",
                condition="quality_trend_improving",
                insight_type=InsightType.QUALITY_TREND,
                priority=InsightPriority.MEDIUM,
                category=InsightCategory.QUALITY
            ),
            InsightRule(
                id="high_performing_queries",
                name="Queries de Alto Rendimiento",
                description="Identifica queries que consistentemente generan documentos de alta calidad",
                condition="high_performing_queries",
                insight_type=InsightType.QUERY_OPTIMIZATION,
                priority=InsightPriority.MEDIUM,
                category=InsightCategory.OPTIMIZATION
            ),
            InsightRule(
                id="low_performing_queries",
                name="Queries de Bajo Rendimiento",
                description="Identifica queries que consistentemente generan documentos de baja calidad",
                condition="low_performing_queries",
                insight_type=InsightType.QUERY_OPTIMIZATION,
                priority=InsightPriority.HIGH,
                category=InsightCategory.OPTIMIZATION
            ),
            InsightRule(
                id="anomaly_detection",
                name="Detección de Anomalías",
                description="Detecta documentos con características inusuales",
                condition="anomaly_detected",
                insight_type=InsightType.ANOMALY_DETECTION,
                priority=InsightPriority.MEDIUM,
                category=InsightCategory.ANOMALY
            ),
            InsightRule(
                id="content_patterns",
                name="Patrones de Contenido",
                description="Identifica patrones en el contenido de documentos exitosos",
                condition="content_patterns",
                insight_type=InsightType.CONTENT_ANALYSIS,
                priority=InsightPriority.LOW,
                category=InsightCategory.PERFORMANCE
            ),
            InsightRule(
                id="volume_trends",
                name="Tendencias de Volumen",
                description="Analiza tendencias en el volumen de documentos generados",
                condition="volume_trends",
                insight_type=InsightType.PERFORMANCE_PATTERN,
                priority=InsightPriority.LOW,
                category=InsightCategory.PERFORMANCE
            ),
            InsightRule(
                id="predictive_quality",
                name="Predicción de Calidad",
                description="Predice la calidad de futuros documentos basado en patrones históricos",
                condition="predictive_quality",
                insight_type=InsightType.PREDICTIVE_INSIGHT,
                priority=InsightPriority.MEDIUM,
                category=InsightCategory.PREDICTION
            )
        ]
        
        for rule in default_rules:
            self.insight_rules[rule.id] = rule
    
    async def generate_insights(
        self,
        documents: List[Dict[str, Any]],
        analysis_results: Optional[Dict[str, Any]] = None
    ) -> List[AutomatedInsight]:
        """
        Generar insights automáticamente basados en documentos y análisis
        
        Args:
            documents: Lista de documentos a analizar
            analysis_results: Resultados de análisis avanzado (opcional)
            
        Returns:
            Lista de insights generados
        """
        try:
            logger.info(f"Generating automated insights for {len(documents)} documents")
            
            # Preparar datos
            df = self._prepare_dataframe(documents)
            
            # Generar insights basados en reglas
            new_insights = []
            
            for rule_id, rule in self.insight_rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    insights = await self._apply_rule(rule, df, analysis_results)
                    new_insights.extend(insights)
                    
                    # Limitar número de insights por ejecución
                    if len(new_insights) >= self.max_insights_per_run:
                        break
                        
                except Exception as e:
                    logger.error(f"Error applying rule {rule_id}: {e}")
                    continue
            
            # Filtrar insights por confianza e impacto
            filtered_insights = [
                insight for insight in new_insights
                if insight.confidence >= self.min_confidence and insight.impact_score >= self.min_impact_score
            ]
            
            # Almacenar insights
            for insight in filtered_insights:
                self.insights[insight.id] = insight
            
            logger.info(f"Generated {len(filtered_insights)} automated insights")
            return filtered_insights
            
        except Exception as e:
            logger.error(f"Error generating automated insights: {e}")
            return []
    
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
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        
        return df
    
    async def _apply_rule(
        self,
        rule: InsightRule,
        df: pd.DataFrame,
        analysis_results: Optional[Dict[str, Any]] = None
    ) -> List[AutomatedInsight]:
        """Aplicar una regla específica para generar insights"""
        insights = []
        
        if rule.condition == "quality_trend_declining":
            insights.extend(await self._detect_quality_decline(df))
        elif rule.condition == "quality_trend_improving":
            insights.extend(await self._detect_quality_improvement(df))
        elif rule.condition == "high_performing_queries":
            insights.extend(await self._identify_high_performing_queries(df))
        elif rule.condition == "low_performing_queries":
            insights.extend(await self._identify_low_performing_queries(df))
        elif rule.condition == "anomaly_detected":
            insights.extend(await self._detect_anomalies(df))
        elif rule.condition == "content_patterns":
            insights.extend(await self._analyze_content_patterns(df))
        elif rule.condition == "volume_trends":
            insights.extend(await self._analyze_volume_trends(df))
        elif rule.condition == "predictive_quality":
            insights.extend(await self._generate_predictive_insights(df))
        
        return insights
    
    async def _detect_quality_decline(self, df: pd.DataFrame) -> List[AutomatedInsight]:
        """Detectar declive en la calidad"""
        insights = []
        
        if len(df) < self.analysis_config["trend_analysis"]["min_data_points"]:
            return insights
        
        # Calcular tendencia de calidad
        quality_trend = self._calculate_trend(df["quality_score"])
        
        if quality_trend["direction"] == "decreasing" and quality_trend["significance"] > self.analysis_config["trend_analysis"]["significance_threshold"]:
            # Calcular métricas de soporte
            recent_quality = df.tail(5)["quality_score"].mean()
            historical_quality = df.head(5)["quality_score"].mean()
            decline_percentage = ((historical_quality - recent_quality) / historical_quality) * 100
            
            # Determinar severidad
            if decline_percentage > 20:
                priority = InsightPriority.CRITICAL
                impact_score = 0.9
            elif decline_percentage > 10:
                priority = InsightPriority.HIGH
                impact_score = 0.7
            else:
                priority = InsightPriority.MEDIUM
                impact_score = 0.5
            
            insight = AutomatedInsight(
                id=f"quality_decline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=InsightType.QUALITY_TREND,
                category=InsightCategory.QUALITY,
                priority=priority,
                title="Declive Detectado en la Calidad de Documentos",
                description=f"Se ha detectado una disminución del {decline_percentage:.1f}% en la calidad promedio de documentos en el período reciente.",
                confidence=min(quality_trend["significance"] * 2, 1.0),
                impact_score=impact_score,
                supporting_data={
                    "recent_quality_avg": float(recent_quality),
                    "historical_quality_avg": float(historical_quality),
                    "decline_percentage": float(decline_percentage),
                    "trend_strength": float(quality_trend["strength"]),
                    "data_points": len(df)
                },
                recommendations=[
                    "Revisar procesos de generación de contenido",
                    "Analizar queries problemáticas recientes",
                    "Implementar controles de calidad adicionales",
                    "Capacitar al equipo en mejores prácticas"
                ],
                actionable_items=[
                    "Auditar documentos de baja calidad recientes",
                    "Identificar patrones en queries problemáticas",
                    "Establecer umbrales de calidad mínimos",
                    "Programar revisión de procesos"
                ],
                related_documents=df.tail(10)["id"].tolist(),
                tags=["calidad", "tendencia", "declive", "urgente"]
            )
            
            insights.append(insight)
        
        return insights
    
    async def _detect_quality_improvement(self, df: pd.DataFrame) -> List[AutomatedInsight]:
        """Detectar mejora en la calidad"""
        insights = []
        
        if len(df) < self.analysis_config["trend_analysis"]["min_data_points"]:
            return insights
        
        # Calcular tendencia de calidad
        quality_trend = self._calculate_trend(df["quality_score"])
        
        if quality_trend["direction"] == "increasing" and quality_trend["significance"] > self.analysis_config["trend_analysis"]["significance_threshold"]:
            # Calcular métricas de soporte
            recent_quality = df.tail(5)["quality_score"].mean()
            historical_quality = df.head(5)["quality_score"].mean()
            improvement_percentage = ((recent_quality - historical_quality) / historical_quality) * 100
            
            insight = AutomatedInsight(
                id=f"quality_improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=InsightType.QUALITY_TREND,
                category=InsightCategory.QUALITY,
                priority=InsightPriority.MEDIUM,
                title="Mejora Detectada en la Calidad de Documentos",
                description=f"Se ha detectado una mejora del {improvement_percentage:.1f}% en la calidad promedio de documentos en el período reciente.",
                confidence=min(quality_trend["significance"] * 2, 1.0),
                impact_score=0.6,
                supporting_data={
                    "recent_quality_avg": float(recent_quality),
                    "historical_quality_avg": float(historical_quality),
                    "improvement_percentage": float(improvement_percentage),
                    "trend_strength": float(quality_trend["strength"]),
                    "data_points": len(df)
                },
                recommendations=[
                    "Documentar factores que contribuyen a la mejora",
                    "Replicar prácticas exitosas en otros contextos",
                    "Mantener los procesos actuales",
                    "Compartir mejores prácticas con el equipo"
                ],
                actionable_items=[
                    "Identificar documentos de alta calidad recientes",
                    "Analizar queries exitosas",
                    "Crear guía de mejores prácticas",
                    "Programar sesión de compartir conocimientos"
                ],
                related_documents=df.tail(10)["id"].tolist(),
                tags=["calidad", "tendencia", "mejora", "éxito"]
            )
            
            insights.append(insight)
        
        return insights
    
    async def _identify_high_performing_queries(self, df: pd.DataFrame) -> List[AutomatedInsight]:
        """Identificar queries de alto rendimiento"""
        insights = []
        
        # Agrupar por query y calcular métricas
        query_stats = df.groupby("query").agg({
            "quality_score": ["mean", "count", "std"],
            "readability_score": "mean",
            "originality_score": "mean"
        }).round(3)
        
        # Flatten column names
        query_stats.columns = ["_".join(col).strip() for col in query_stats.columns]
        
        # Filtrar queries con suficiente datos
        min_documents = 3
        high_performing_queries = query_stats[
            (query_stats["quality_score_count"] >= min_documents) &
            (query_stats["quality_score_mean"] >= self.analysis_config["quality_thresholds"]["good"])
        ]
        
        if len(high_performing_queries) > 0:
            # Seleccionar las mejores queries
            top_queries = high_performing_queries.nlargest(3, "quality_score_mean")
            
            for idx, (query, stats) in enumerate(top_queries.iterrows()):
                insight = AutomatedInsight(
                    id=f"high_performing_query_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    type=InsightType.QUERY_OPTIMIZATION,
                    category=InsightCategory.OPTIMIZATION,
                    priority=InsightPriority.MEDIUM,
                    title=f"Query de Alto Rendimiento Identificada",
                    description=f"La query '{query[:100]}...' genera consistentemente documentos de alta calidad (promedio: {stats['quality_score_mean']:.2f}).",
                    confidence=min(stats["quality_score_count"] / 10, 1.0),
                    impact_score=0.7,
                    supporting_data={
                        "query": query,
                        "avg_quality": float(stats["quality_score_mean"]),
                        "document_count": int(stats["quality_score_count"]),
                        "quality_std": float(stats["quality_score_std"]),
                        "avg_readability": float(stats["readability_score_mean"]),
                        "avg_originality": float(stats["originality_score_mean"])
                    },
                    recommendations=[
                        "Usar esta query como plantilla para otros documentos",
                        "Analizar características que la hacen exitosa",
                        "Replicar el patrón en queries similares",
                        "Documentar mejores prácticas de esta query"
                    ],
                    actionable_items=[
                        "Crear plantilla basada en esta query",
                        "Identificar elementos clave del éxito",
                        "Entrenar al equipo en este patrón",
                        "Monitorear rendimiento futuro"
                    ],
                    related_documents=df[df["query"] == query]["id"].tolist(),
                    tags=["query", "alto_rendimiento", "optimización", "plantilla"]
                )
                
                insights.append(insight)
        
        return insights
    
    async def _identify_low_performing_queries(self, df: pd.DataFrame) -> List[AutomatedInsight]:
        """Identificar queries de bajo rendimiento"""
        insights = []
        
        # Agrupar por query y calcular métricas
        query_stats = df.groupby("query").agg({
            "quality_score": ["mean", "count", "std"],
            "readability_score": "mean",
            "originality_score": "mean"
        }).round(3)
        
        # Flatten column names
        query_stats.columns = ["_".join(col).strip() for col in query_stats.columns]
        
        # Filtrar queries con suficiente datos
        min_documents = 2
        low_performing_queries = query_stats[
            (query_stats["quality_score_count"] >= min_documents) &
            (query_stats["quality_score_mean"] < self.analysis_config["quality_thresholds"]["average"])
        ]
        
        if len(low_performing_queries) > 0:
            # Seleccionar las peores queries
            worst_queries = low_performing_queries.nsmallest(3, "quality_score_mean")
            
            for idx, (query, stats) in enumerate(worst_queries.iterrows()):
                insight = AutomatedInsight(
                    id=f"low_performing_query_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    type=InsightType.QUERY_OPTIMIZATION,
                    category=InsightCategory.OPTIMIZATION,
                    priority=InsightPriority.HIGH,
                    title=f"Query de Bajo Rendimiento Identificada",
                    description=f"La query '{query[:100]}...' genera consistentemente documentos de baja calidad (promedio: {stats['quality_score_mean']:.2f}).",
                    confidence=min(stats["quality_score_count"] / 5, 1.0),
                    impact_score=0.8,
                    supporting_data={
                        "query": query,
                        "avg_quality": float(stats["quality_score_mean"]),
                        "document_count": int(stats["quality_score_count"]),
                        "quality_std": float(stats["quality_score_std"]),
                        "avg_readability": float(stats["readability_score_mean"]),
                        "avg_originality": float(stats["originality_score_mean"])
                    },
                    recommendations=[
                        "Reformular esta query para mejorar resultados",
                        "Analizar por qué genera documentos de baja calidad",
                        "Probar variaciones de la query",
                        "Evitar usar esta query hasta que se optimice"
                    ],
                    actionable_items=[
                        "Revisar y reformular la query",
                        "Identificar problemas específicos",
                        "Probar versiones mejoradas",
                        "Monitorear mejoras en rendimiento"
                    ],
                    related_documents=df[df["query"] == query]["id"].tolist(),
                    tags=["query", "bajo_rendimiento", "problema", "optimización"]
                )
                
                insights.append(insight)
        
        return insights
    
    async def _detect_anomalies(self, df: pd.DataFrame) -> List[AutomatedInsight]:
        """Detectar anomalías en los datos"""
        insights = []
        
        if len(df) < 5:
            return insights
        
        # Detectar anomalías usando Z-score
        quality_scores = df["quality_score"].values
        z_scores = np.abs((quality_scores - np.mean(quality_scores)) / np.std(quality_scores))
        
        anomaly_indices = np.where(z_scores > self.analysis_config["anomaly_detection"]["z_score_threshold"])[0]
        
        if len(anomaly_indices) > 0:
            anomaly_docs = df.iloc[anomaly_indices]
            
            # Agrupar anomalías por tipo
            low_quality_anomalies = anomaly_docs[anomaly_docs["quality_score"] < np.mean(quality_scores)]
            high_quality_anomalies = anomaly_docs[anomaly_docs["quality_score"] > np.mean(quality_scores)]
            
            if len(low_quality_anomalies) > 0:
                insight = AutomatedInsight(
                    id=f"low_quality_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    type=InsightType.ANOMALY_DETECTION,
                    category=InsightCategory.ANOMALY,
                    priority=InsightPriority.HIGH,
                    title="Anomalías de Baja Calidad Detectadas",
                    description=f"Se detectaron {len(low_quality_anomalies)} documentos con calidad excepcionalmente baja que requieren atención.",
                    confidence=0.8,
                    impact_score=0.7,
                    supporting_data={
                        "anomaly_count": len(low_quality_anomalies),
                        "avg_anomaly_quality": float(low_quality_anomalies["quality_score"].mean()),
                        "overall_avg_quality": float(np.mean(quality_scores)),
                        "z_score_threshold": self.analysis_config["anomaly_detection"]["z_score_threshold"]
                    },
                    recommendations=[
                        "Revisar inmediatamente los documentos anómalos",
                        "Identificar causas de la baja calidad",
                        "Implementar controles de calidad preventivos",
                        "Capacitar al equipo en mejores prácticas"
                    ],
                    actionable_items=[
                        "Auditar documentos anómalos",
                        "Identificar patrones problemáticos",
                        "Establecer alertas de calidad",
                        "Programar revisión de procesos"
                    ],
                    related_documents=low_quality_anomalies["id"].tolist(),
                    tags=["anomalía", "baja_calidad", "urgente", "revisión"]
                )
                
                insights.append(insight)
            
            if len(high_quality_anomalies) > 0:
                insight = AutomatedInsight(
                    id=f"high_quality_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    type=InsightType.ANOMALY_DETECTION,
                    category=InsightCategory.ANOMALY,
                    priority=InsightPriority.MEDIUM,
                    title="Anomalías de Alta Calidad Detectadas",
                    description=f"Se detectaron {len(high_quality_anomalies)} documentos con calidad excepcionalmente alta que pueden servir como referencia.",
                    confidence=0.8,
                    impact_score=0.6,
                    supporting_data={
                        "anomaly_count": len(high_quality_anomalies),
                        "avg_anomaly_quality": float(high_quality_anomalies["quality_score"].mean()),
                        "overall_avg_quality": float(np.mean(quality_scores)),
                        "z_score_threshold": self.analysis_config["anomaly_detection"]["z_score_threshold"]
                    },
                    recommendations=[
                        "Analizar factores que contribuyen a la alta calidad",
                        "Usar estos documentos como ejemplos de mejores prácticas",
                        "Replicar patrones exitosos en otros documentos",
                        "Documentar características de éxito"
                    ],
                    actionable_items=[
                        "Analizar documentos excepcionales",
                        "Identificar patrones de éxito",
                        "Crear guía de mejores prácticas",
                        "Compartir conocimientos con el equipo"
                    ],
                    related_documents=high_quality_anomalies["id"].tolist(),
                    tags=["anomalía", "alta_calidad", "éxito", "mejores_prácticas"]
                )
                
                insights.append(insight)
        
        return insights
    
    async def _analyze_content_patterns(self, df: pd.DataFrame) -> List[AutomatedInsight]:
        """Analizar patrones en el contenido"""
        insights = []
        
        if len(df) < 10:
            return insights
        
        # Analizar patrones de longitud
        length_patterns = self._analyze_length_patterns(df)
        if length_patterns:
            insights.extend(length_patterns)
        
        # Analizar patrones de legibilidad
        readability_patterns = self._analyze_readability_patterns(df)
        if readability_patterns:
            insights.extend(readability_patterns)
        
        return insights
    
    def _analyze_length_patterns(self, df: pd.DataFrame) -> List[AutomatedInsight]:
        """Analizar patrones de longitud"""
        insights = []
        
        # Correlación entre longitud y calidad
        correlation = df["word_count"].corr(df["quality_score"])
        
        if abs(correlation) > 0.3:  # Correlación moderada o fuerte
            direction = "positiva" if correlation > 0 else "negativa"
            
            insight = AutomatedInsight(
                id=f"length_quality_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=InsightType.CONTENT_ANALYSIS,
                category=InsightCategory.PERFORMANCE,
                priority=InsightPriority.LOW,
                title=f"Correlación {direction.title()} entre Longitud y Calidad",
                description=f"Se detectó una correlación {direction} moderada (r={correlation:.2f}) entre la longitud de documentos y su calidad.",
                confidence=abs(correlation),
                impact_score=0.5,
                supporting_data={
                    "correlation_coefficient": float(correlation),
                    "avg_word_count": float(df["word_count"].mean()),
                    "avg_quality": float(df["quality_score"].mean()),
                    "sample_size": len(df)
                },
                recommendations=[
                    f"Considerar {'aumentar' if correlation > 0 else 'reducir'} la longitud de documentos para mejorar calidad",
                    "Analizar la longitud óptima para diferentes tipos de contenido",
                    "Establecer guías de longitud basadas en evidencia"
                ],
                actionable_items=[
                    "Analizar documentos de diferentes longitudes",
                    "Identificar longitud óptima por tipo de contenido",
                    "Crear guías de longitud",
                    "Monitorear impacto de cambios en longitud"
                ],
                related_documents=df.nlargest(5, "quality_score")["id"].tolist(),
                tags=["longitud", "calidad", "correlación", "patrón"]
            )
            
            insights.append(insight)
        
        return insights
    
    def _analyze_readability_patterns(self, df: pd.DataFrame) -> List[AutomatedInsight]:
        """Analizar patrones de legibilidad"""
        insights = []
        
        # Correlación entre legibilidad y calidad
        correlation = df["readability_score"].corr(df["quality_score"])
        
        if abs(correlation) > 0.4:  # Correlación fuerte
            direction = "positiva" if correlation > 0 else "negativa"
            
            insight = AutomatedInsight(
                id=f"readability_quality_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=InsightType.CONTENT_ANALYSIS,
                category=InsightCategory.PERFORMANCE,
                priority=InsightPriority.MEDIUM,
                title=f"Fuerte Correlación {direction.title()} entre Legibilidad y Calidad",
                description=f"Se detectó una correlación {direction} fuerte (r={correlation:.2f}) entre la legibilidad y la calidad de documentos.",
                confidence=abs(correlation),
                impact_score=0.6,
                supporting_data={
                    "correlation_coefficient": float(correlation),
                    "avg_readability": float(df["readability_score"].mean()),
                    "avg_quality": float(df["quality_score"].mean()),
                    "sample_size": len(df)
                },
                recommendations=[
                    f"Priorizar {'mejorar' if correlation > 0 else 'revisar'} la legibilidad de documentos",
                    "Implementar herramientas de análisis de legibilidad",
                    "Capacitar al equipo en técnicas de escritura clara"
                ],
                actionable_items=[
                    "Implementar métricas de legibilidad",
                    "Capacitar en escritura clara",
                    "Revisar documentos con baja legibilidad",
                    "Establecer estándares de legibilidad"
                ],
                related_documents=df.nlargest(5, "readability_score")["id"].tolist(),
                tags=["legibilidad", "calidad", "correlación", "escritura"]
            )
            
            insights.append(insight)
        
        return insights
    
    async def _analyze_volume_trends(self, df: pd.DataFrame) -> List[AutomatedInsight]:
        """Analizar tendencias de volumen"""
        insights = []
        
        if len(df) < 10:
            return insights
        
        # Agrupar por día
        df["date"] = df["timestamp"].dt.date
        daily_counts = df.groupby("date").size()
        
        if len(daily_counts) < 5:
            return insights
        
        # Calcular tendencia de volumen
        volume_trend = self._calculate_trend(pd.Series(daily_counts.values))
        
        if volume_trend["direction"] in ["increasing", "decreasing"] and volume_trend["significance"] > 0.1:
            direction = "aumento" if volume_trend["direction"] == "increasing" else "disminución"
            
            insight = AutomatedInsight(
                id=f"volume_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=InsightType.PERFORMANCE_PATTERN,
                category=InsightCategory.PERFORMANCE,
                priority=InsightPriority.LOW,
                title=f"Tendencia de {direction.title()} en Volumen de Documentos",
                description=f"Se detectó una tendencia de {direction} en el volumen diario de documentos generados.",
                confidence=volume_trend["significance"],
                impact_score=0.4,
                supporting_data={
                    "trend_direction": volume_trend["direction"],
                    "trend_strength": float(volume_trend["strength"]),
                    "avg_daily_volume": float(daily_counts.mean()),
                    "recent_avg_volume": float(daily_counts.tail(3).mean()),
                    "historical_avg_volume": float(daily_counts.head(3).mean())
                },
                recommendations=[
                    f"Monitorear el {direction} en volumen para planificar recursos",
                    "Analizar factores que contribuyen al cambio en volumen",
                    "Ajustar capacidad según la tendencia"
                ],
                actionable_items=[
                    "Revisar patrones de uso",
                    "Planificar recursos según tendencia",
                    "Analizar factores externos",
                    "Ajustar métricas de rendimiento"
                ],
                related_documents=[],
                tags=["volumen", "tendencia", "rendimiento", "planificación"]
            )
            
            insights.append(insight)
        
        return insights
    
    async def _generate_predictive_insights(self, df: pd.DataFrame) -> List[AutomatedInsight]:
        """Generar insights predictivos"""
        insights = []
        
        if len(df) < 20:
            return insights
        
        # Análisis predictivo simple basado en tendencias
        recent_quality = df.tail(5)["quality_score"].mean()
        historical_quality = df.head(5)["quality_score"].mean()
        
        # Calcular tendencia
        quality_trend = self._calculate_trend(df["quality_score"])
        
        if quality_trend["direction"] == "increasing" and quality_trend["significance"] > 0.1:
            # Predecir calidad futura
            predicted_quality = recent_quality + (recent_quality - historical_quality) * 0.5
            
            if predicted_quality > self.analysis_config["quality_thresholds"]["good"]:
                insight = AutomatedInsight(
                    id=f"predictive_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    type=InsightType.PREDICTIVE_INSIGHT,
                    category=InsightCategory.PREDICTION,
                    priority=InsightPriority.MEDIUM,
                    title="Predicción Positiva de Calidad",
                    description=f"Basado en tendencias actuales, se predice que la calidad promedio alcanzará {predicted_quality:.2f} en el próximo período.",
                    confidence=min(quality_trend["significance"] * 1.5, 0.8),
                    impact_score=0.6,
                    supporting_data={
                        "predicted_quality": float(predicted_quality),
                        "current_quality": float(recent_quality),
                        "historical_quality": float(historical_quality),
                        "trend_strength": float(quality_trend["strength"]),
                        "confidence_level": "moderate"
                    },
                    recommendations=[
                        "Mantener las prácticas actuales que están funcionando",
                        "Preparar para un período de alta calidad",
                        "Documentar factores de éxito",
                        "Planificar recursos para el aumento esperado"
                    ],
                    actionable_items=[
                        "Monitorear tendencias de calidad",
                        "Documentar mejores prácticas actuales",
                        "Preparar para período de alta calidad",
                        "Comunicar predicción al equipo"
                    ],
                    related_documents=df.tail(10)["id"].tolist(),
                    tags=["predicción", "calidad", "tendencia", "futuro"]
                )
                
                insights.append(insight)
        
        return insights
    
    def _calculate_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Calcular tendencia de una serie temporal"""
        if len(series) < 3:
            return {"direction": "insufficient_data", "strength": 0.0, "significance": 0.0}
        
        # Calcular pendiente usando regresión lineal simple
        x = np.arange(len(series))
        y = series.values
        
        # Remover NaN
        mask = ~np.isnan(y)
        if np.sum(mask) < 3:
            return {"direction": "insufficient_data", "strength": 0.0, "significance": 0.0}
        
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
        
        # Calcular significancia (simplificado)
        significance = min(strength / 10, 1.0)
        
        return {
            "direction": direction,
            "strength": strength,
            "significance": significance,
            "slope": slope
        }
    
    def get_insights_by_priority(self, priority: InsightPriority) -> List[AutomatedInsight]:
        """Obtener insights por prioridad"""
        return [insight for insight in self.insights.values() if insight.priority == priority]
    
    def get_insights_by_category(self, category: InsightCategory) -> List[AutomatedInsight]:
        """Obtener insights por categoría"""
        return [insight for insight in self.insights.values() if insight.category == category]
    
    def get_recent_insights(self, hours: int = 24) -> List[AutomatedInsight]:
        """Obtener insights recientes"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            insight for insight in self.insights.values()
            if insight.generated_at >= cutoff_time
        ]
    
    def get_expired_insights(self) -> List[AutomatedInsight]:
        """Obtener insights expirados"""
        current_time = datetime.now()
        return [
            insight for insight in self.insights.values()
            if insight.expires_at and insight.expires_at <= current_time
        ]
    
    def cleanup_expired_insights(self) -> int:
        """Limpiar insights expirados"""
        expired_insights = self.get_expired_insights()
        for insight in expired_insights:
            if insight.id in self.insights:
                del self.insights[insight.id]
        
        logger.info(f"Cleaned up {len(expired_insights)} expired insights")
        return len(expired_insights)
    
    def add_insight_rule(self, rule: InsightRule):
        """Agregar nueva regla de insight"""
        self.insight_rules[rule.id] = rule
        logger.info(f"Added insight rule: {rule.name}")
    
    def remove_insight_rule(self, rule_id: str):
        """Remover regla de insight"""
        if rule_id in self.insight_rules:
            del self.insight_rules[rule_id]
            logger.info(f"Removed insight rule: {rule_id}")
    
    def enable_insight_rule(self, rule_id: str):
        """Habilitar regla de insight"""
        if rule_id in self.insight_rules:
            self.insight_rules[rule_id].enabled = True
            logger.info(f"Enabled insight rule: {rule_id}")
    
    def disable_insight_rule(self, rule_id: str):
        """Deshabilitar regla de insight"""
        if rule_id in self.insight_rules:
            self.insight_rules[rule_id].enabled = False
            logger.info(f"Disabled insight rule: {rule_id}")
    
    def get_insights_summary(self) -> Dict[str, Any]:
        """Obtener resumen de insights"""
        total_insights = len(self.insights)
        
        # Contar por prioridad
        priority_counts = defaultdict(int)
        for insight in self.insights.values():
            priority_counts[insight.priority.value] += 1
        
        # Contar por categoría
        category_counts = defaultdict(int)
        for insight in self.insights.values():
            category_counts[insight.category.value] += 1
        
        # Contar por tipo
        type_counts = defaultdict(int)
        for insight in self.insights.values():
            type_counts[insight.type.value] += 1
        
        return {
            "total_insights": total_insights,
            "priority_distribution": dict(priority_counts),
            "category_distribution": dict(category_counts),
            "type_distribution": dict(type_counts),
            "active_rules": len([r for r in self.insight_rules.values() if r.enabled]),
            "total_rules": len(self.insight_rules),
            "recent_insights": len(self.get_recent_insights(24)),
            "expired_insights": len(self.get_expired_insights())
        }
    
    async def export_insights(self, format: str = "json") -> str:
        """Exportar insights a archivo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            filename = f"automated_insights_{timestamp}.json"
            filepath = f"exports/{filename}"
            
            # Crear directorio si no existe
            import os
            os.makedirs("exports", exist_ok=True)
            
            insights_data = {
                "insights": {k: self._insight_to_dict(v) for k, v in self.insights.items()},
                "rules": {k: self._rule_to_dict(v) for k, v in self.insight_rules.items()},
                "summary": self.get_insights_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(insights_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Insights exported to {filepath}")
            return filepath
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _insight_to_dict(self, insight: AutomatedInsight) -> Dict[str, Any]:
        """Convertir insight a diccionario"""
        return {
            "id": insight.id,
            "type": insight.type.value,
            "category": insight.category.value,
            "priority": insight.priority.value,
            "title": insight.title,
            "description": insight.description,
            "confidence": insight.confidence,
            "impact_score": insight.impact_score,
            "supporting_data": insight.supporting_data,
            "recommendations": insight.recommendations,
            "actionable_items": insight.actionable_items,
            "related_documents": insight.related_documents,
            "generated_at": insight.generated_at.isoformat(),
            "expires_at": insight.expires_at.isoformat() if insight.expires_at else None,
            "tags": insight.tags
        }
    
    def _rule_to_dict(self, rule: InsightRule) -> Dict[str, Any]:
        """Convertir regla a diccionario"""
        return {
            "id": rule.id,
            "name": rule.name,
            "description": rule.description,
            "condition": rule.condition,
            "insight_type": rule.insight_type.value,
            "priority": rule.priority.value,
            "category": rule.category.value,
            "enabled": rule.enabled,
            "created_at": rule.created_at.isoformat()
        }



























