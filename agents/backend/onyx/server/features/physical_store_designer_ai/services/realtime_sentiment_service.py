"""
Realtime Sentiment Service - Análisis de sentimiento en tiempo real
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from ..services.llm_service import LLMService
from ..services.sentiment_analysis_service import SentimentAnalysisService

logger = logging.getLogger(__name__)


class RealtimeSentimentService:
    """Servicio para análisis de sentimiento en tiempo real"""
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        sentiment_service: Optional[SentimentAnalysisService] = None
    ):
        self.llm_service = llm_service or LLMService()
        self.sentiment_service = sentiment_service or SentimentAnalysisService(llm_service)
        self.streams: Dict[str, List[Dict[str, Any]]] = {}
        self.aggregates: Dict[str, Dict[str, Any]] = {}
    
    async def process_sentiment_stream(
        self,
        store_id: str,
        text: str,
        source: str = "social_media",  # "social_media", "review", "feedback", "chat"
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Procesar stream de sentimiento en tiempo real"""
        
        # Analizar sentimiento
        sentiment = await self.sentiment_service.analyze_sentiment(text)
        
        stream_entry = {
            "entry_id": f"stream_{store_id}_{len(self.streams.get(store_id, [])) + 1}",
            "store_id": store_id,
            "text": text,
            "source": source,
            "sentiment": sentiment,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        if store_id not in self.streams:
            self.streams[store_id] = []
        
        self.streams[store_id].append(stream_entry)
        
        # Actualizar agregados
        self._update_aggregates(store_id, sentiment)
        
        return stream_entry
    
    def _update_aggregates(
        self,
        store_id: str,
        sentiment: Dict[str, Any]
    ):
        """Actualizar agregados de sentimiento"""
        
        if store_id not in self.aggregates:
            self.aggregates[store_id] = {
                "total_entries": 0,
                "sentiment_counts": defaultdict(int),
                "average_score": 0.0,
                "score_sum": 0.0,
                "last_updated": datetime.now().isoformat()
            }
        
        aggregate = self.aggregates[store_id]
        aggregate["total_entries"] += 1
        aggregate["sentiment_counts"][sentiment.get("sentiment", "neutral")] += 1
        aggregate["score_sum"] += sentiment.get("score", 0.0)
        aggregate["average_score"] = aggregate["score_sum"] / aggregate["total_entries"]
        aggregate["last_updated"] = datetime.now().isoformat()
    
    def get_realtime_sentiment(
        self,
        store_id: str,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Obtener sentimiento en tiempo real"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        
        store_streams = self.streams.get(store_id, [])
        recent_streams = [
            s for s in store_streams
            if start_time <= datetime.fromisoformat(s["timestamp"]) <= end_time
        ]
        
        if not recent_streams:
            return {
                "store_id": store_id,
                "time_window_minutes": time_window_minutes,
                "message": "No hay datos en el período especificado"
            }
        
        # Calcular métricas
        scores = [s["sentiment"]["score"] for s in recent_streams]
        sentiments = [s["sentiment"]["sentiment"] for s in recent_streams]
        
        sentiment_counts = defaultdict(int)
        for sent in sentiments:
            sentiment_counts[sent] += 1
        
        return {
            "store_id": store_id,
            "time_window_minutes": time_window_minutes,
            "total_entries": len(recent_streams),
            "average_sentiment_score": round(sum(scores) / len(scores), 2),
            "sentiment_distribution": dict(sentiment_counts),
            "overall_sentiment": self._determine_overall_sentiment(scores),
            "trend": self._calculate_trend(store_id, time_window_minutes),
            "last_updated": datetime.now().isoformat()
        }
    
    def _determine_overall_sentiment(self, scores: List[float]) -> str:
        """Determinar sentimiento general"""
        avg = sum(scores) / len(scores) if scores else 0
        
        if avg > 0.2:
            return "positive"
        elif avg < -0.2:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_trend(
        self,
        store_id: str,
        time_window: int
    ) -> str:
        """Calcular tendencia"""
        store_streams = self.streams.get(store_id, [])
        
        if len(store_streams) < 20:
            return "insufficient_data"
        
        # Comparar primera mitad vs segunda mitad
        mid_point = len(store_streams) // 2
        first_half = store_streams[:mid_point]
        second_half = store_streams[mid_point:]
        
        first_avg = sum(s["sentiment"]["score"] for s in first_half) / len(first_half) if first_half else 0
        second_avg = sum(s["sentiment"]["score"] for s in second_half) / len(second_half) if second_half else 0
        
        change = second_avg - first_avg
        
        if change > 0.1:
            return "improving"
        elif change < -0.1:
            return "declining"
        else:
            return "stable"
    
    def detect_sentiment_alert(
        self,
        store_id: str,
        threshold: float = -0.5
    ) -> Optional[Dict[str, Any]]:
        """Detectar alerta de sentimiento negativo"""
        
        aggregate = self.aggregates.get(store_id)
        
        if not aggregate or aggregate["total_entries"] < 5:
            return None
        
        if aggregate["average_score"] < threshold:
            return {
                "store_id": store_id,
                "alert_type": "negative_sentiment",
                "average_score": aggregate["average_score"],
                "threshold": threshold,
                "severity": "high" if aggregate["average_score"] < -0.7 else "medium",
                "recommendation": "Review recent feedback and address concerns",
                "triggered_at": datetime.now().isoformat()
            }
        
        return None




