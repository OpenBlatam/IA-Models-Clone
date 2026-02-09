"""
Analizador de Tendencias
=========================

Sistema para analizar tendencias y estadísticas temporales en documentos.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics

from .document_analyzer import DocumentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class TrendDataPoint:
    """Punto de datos de tendencia"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Análisis de tendencia"""
    metric_name: str
    data_points: List[TrendDataPoint]
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0-1
    average_value: float
    min_value: float
    max_value: float
    variance: float
    period: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class TrendAnalyzer:
    """
    Analizador de tendencias y estadísticas temporales
    
    Analiza cambios en documentos a lo largo del tiempo:
    - Tendencias de sentimiento
    - Evolución de temas
    - Cambios en keywords
    - Estadísticas de uso
    """
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar analizador de tendencias
        
        Args:
            analyzer: Instancia de DocumentAnalyzer
        """
        self.analyzer = analyzer
        logger.info("TrendAnalyzer inicializado")
    
    async def analyze_sentiment_trend(
        self,
        documents: List[Dict[str, Any]],
        period: str = "day"
    ) -> TrendAnalysis:
        """
        Analizar tendencia de sentimiento
        
        Args:
            documents: Lista de documentos con timestamp y content
            period: Período de agrupación (day, week, month)
        
        Returns:
            TrendAnalysis con análisis de tendencia
        """
        # Agrupar por período
        grouped = self._group_by_period(documents, period)
        
        data_points = []
        for period_key, docs in grouped.items():
            # Calcular sentimiento promedio del período
            sentiments = []
            for doc in docs:
                try:
                    sentiment = await self.analyzer.analyze_sentiment(doc["content"])
                    if sentiment:
                        # Calcular score promedio (positivo - negativo)
                        positive = sentiment.get("positive", 0)
                        negative = sentiment.get("negative", 0)
                        neutral = sentiment.get("neutral", 0)
                        score = positive - negative
                        sentiments.append(score)
                except:
                    continue
            
            if sentiments:
                avg_sentiment = statistics.mean(sentiments)
                data_points.append(TrendDataPoint(
                    timestamp=period_key,
                    value=avg_sentiment,
                    metadata={"count": len(docs)}
                ))
        
        return self._calculate_trend("sentiment", data_points, period)
    
    async def analyze_keyword_trend(
        self,
        documents: List[Dict[str, Any]],
        period: str = "day",
        top_k: int = 10
    ) -> Dict[str, TrendAnalysis]:
        """
        Analizar tendencia de keywords
        
        Args:
            documents: Lista de documentos con timestamp y content
            period: Período de agrupación
            top_k: Número de keywords a analizar
        
        Returns:
            Diccionario de TrendAnalysis por keyword
        """
        # Agrupar por período
        grouped = self._group_by_period(documents, period)
        
        # Contar keywords por período
        keyword_counts = defaultdict(lambda: defaultdict(int))
        
        for period_key, docs in grouped.items():
            for doc in docs:
                try:
                    keywords = await self.analyzer.extract_keywords(doc["content"], top_k=top_k)
                    for keyword in keywords:
                        keyword_counts[keyword][period_key] += 1
                except:
                    continue
        
        # Calcular tendencias para cada keyword
        trends = {}
        all_keywords = set()
        for keyword, counts in keyword_counts.items():
            all_keywords.add(keyword)
        
        # Seleccionar top keywords más frecuentes
        total_counts = {kw: sum(counts.values()) for kw, counts in keyword_counts.items()}
        top_keywords = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        for keyword, _ in top_keywords:
            data_points = []
            for period_key in sorted(grouped.keys()):
                count = keyword_counts[keyword].get(period_key, 0)
                data_points.append(TrendDataPoint(
                    timestamp=period_key,
                    value=float(count),
                    metadata={}
                ))
            
            trends[keyword] = self._calculate_trend(f"keyword_{keyword}", data_points, period)
        
        return trends
    
    async def analyze_topic_trend(
        self,
        documents: List[Dict[str, Any]],
        period: str = "day"
    ) -> Dict[str, TrendAnalysis]:
        """
        Analizar tendencia de temas
        
        Args:
            documents: Lista de documentos con timestamp y content
            period: Período de agrupación
        
        Returns:
            Diccionario de TrendAnalysis por tema
        """
        # Agrupar por período
        grouped = self._group_by_period(documents, period)
        
        # Extraer temas por período
        topic_counts = defaultdict(lambda: defaultdict(int))
        
        for period_key, docs in grouped.items():
            for doc in docs:
                try:
                    topics = await self.analyzer.extract_topics(doc["content"])
                    for topic in topics:
                        topic_id = topic.get("topic_id", "unknown")
                        topic_counts[topic_id][period_key] += 1
                except:
                    continue
        
        # Calcular tendencias para cada tema
        trends = {}
        for topic_id, counts in topic_counts.items():
            data_points = []
            for period_key in sorted(grouped.keys()):
                count = counts.get(period_key, 0)
                data_points.append(TrendDataPoint(
                    timestamp=period_key,
                    value=float(count),
                    metadata={}
                ))
            
            trends[f"topic_{topic_id}"] = self._calculate_trend(
                f"topic_{topic_id}",
                data_points,
                period
            )
        
        return trends
    
    def _group_by_period(
        self,
        documents: List[Dict[str, Any]],
        period: str
    ) -> Dict[datetime, List[Dict[str, Any]]]:
        """Agrupar documentos por período"""
        grouped = defaultdict(list)
        
        for doc in documents:
            timestamp = doc.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.now()
            
            # Normalizar a período
            if period == "day":
                period_key = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            elif period == "week":
                # Lunes de la semana
                days_since_monday = timestamp.weekday()
                period_key = (timestamp - timedelta(days=days_since_monday)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
            elif period == "month":
                period_key = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                period_key = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            
            grouped[period_key].append(doc)
        
        return grouped
    
    def _calculate_trend(
        self,
        metric_name: str,
        data_points: List[TrendDataPoint],
        period: str
    ) -> TrendAnalysis:
        """Calcular análisis de tendencia"""
        if not data_points:
            return TrendAnalysis(
                metric_name=metric_name,
                data_points=[],
                trend_direction="stable",
                trend_strength=0.0,
                average_value=0.0,
                min_value=0.0,
                max_value=0.0,
                variance=0.0,
                period=period
            )
        
        values = [dp.value for dp in data_points]
        avg_value = statistics.mean(values)
        min_value = min(values)
        max_value = max(values)
        variance = statistics.variance(values) if len(values) > 1 else 0.0
        
        # Determinar dirección de tendencia
        if len(values) < 2:
            trend_direction = "stable"
            trend_strength = 0.0
        else:
            # Calcular pendiente (regresión simple)
            x = list(range(len(values)))
            n = len(values)
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(x[i] * values[i] for i in range(n))
            sum_x2 = sum(xi * xi for xi in x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
            
            # Normalizar slope para obtener strength
            range_value = max_value - min_value if max_value != min_value else 1
            trend_strength = min(1.0, abs(slope) / (range_value / len(values)))
            
            if slope > 0.1:
                trend_direction = "increasing"
            elif slope < -0.1:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
        
        return TrendAnalysis(
            metric_name=metric_name,
            data_points=data_points,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            average_value=avg_value,
            min_value=min_value,
            max_value=max_value,
            variance=variance,
            period=period
        )
    
    async def generate_trend_report(
        self,
        documents: List[Dict[str, Any]],
        period: str = "day"
    ) -> Dict[str, Any]:
        """
        Generar reporte completo de tendencias
        
        Args:
            documents: Lista de documentos con timestamp y content
            period: Período de agrupación
        
        Returns:
            Diccionario con análisis completos
        """
        report = {
            "period": period,
            "total_documents": len(documents),
            "date_range": {
                "start": min(doc.get("timestamp", datetime.now()) for doc in documents),
                "end": max(doc.get("timestamp", datetime.now()) for doc in documents)
            },
            "sentiment_trend": None,
            "keyword_trends": {},
            "topic_trends": {}
        }
        
        try:
            # Análisis de sentimiento
            report["sentiment_trend"] = await self.analyze_sentiment_trend(documents, period)
            
            # Análisis de keywords
            report["keyword_trends"] = await self.analyze_keyword_trend(documents, period)
            
            # Análisis de temas
            report["topic_trends"] = await self.analyze_topic_trend(documents, period)
        except Exception as e:
            logger.error(f"Error generando reporte de tendencias: {e}")
        
        return report
















