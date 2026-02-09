"""
Trend Analyzer - Sistema de análisis de tendencias
"""

import logging
from typing import Dict, Any, Optional, List
from collections import Counter, defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Analizador de tendencias"""

    def __init__(self):
        """Inicializar analizador"""
        self.content_history: List[Dict[str, Any]] = []
        self.max_history = 1000

    def record_content(
        self,
        content_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registrar contenido para análisis de tendencias.

        Args:
            content_id: ID del contenido
            content: Contenido
            metadata: Metadatos adicionales
        """
        record = {
            "content_id": content_id,
            "content": content,
            "word_count": len(content.split()),
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        self.content_history.append(record)
        
        # Limitar tamaño
        if len(self.content_history) > self.max_history:
            self.content_history = self.content_history[-self.max_history:]
        
        logger.debug(f"Contenido registrado para análisis de tendencias: {content_id}")

    def analyze_trends(
        self,
        period_days: int = 30,
        metric: str = "word_count"
    ) -> Dict[str, Any]:
        """
        Analizar tendencias.

        Args:
            period_days: Período en días
            metric: Métrica a analizar

        Returns:
            Análisis de tendencias
        """
        if not self.content_history:
            return {"error": "No hay historial de contenido"}
        
        # Filtrar por período
        cutoff_date = datetime.utcnow() - timedelta(days=period_days)
        recent_content = [
            c for c in self.content_history
            if datetime.fromisoformat(c["timestamp"]) >= cutoff_date
        ]
        
        if not recent_content:
            return {"error": "No hay contenido en el período especificado"}
        
        # Agrupar por día
        daily_data = defaultdict(list)
        for record in recent_content:
            date = datetime.fromisoformat(record["timestamp"]).date()
            daily_data[date].append(record.get(metric, 0))
        
        # Calcular promedios diarios
        daily_averages = {
            str(date): sum(values) / len(values)
            for date, values in daily_data.items()
        }
        
        # Calcular tendencia
        dates = sorted(daily_averages.keys())
        if len(dates) < 2:
            trend = "insufficient_data"
            trend_score = 0.0
        else:
            values = [daily_averages[d] for d in dates]
            # Calcular pendiente (tendencia)
            n = len(values)
            x_mean = sum(range(n)) / n
            y_mean = sum(values) / n
            
            numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                trend = "stable"
                trend_score = 0.0
            else:
                slope = numerator / denominator
                
                if slope > 0.1:
                    trend = "increasing"
                    trend_score = min(1.0, slope / max(values) * 10)
                elif slope < -0.1:
                    trend = "decreasing"
                    trend_score = min(1.0, abs(slope) / max(values) * 10)
                else:
                    trend = "stable"
                    trend_score = 0.0
        
        return {
            "period_days": period_days,
            "metric": metric,
            "total_records": len(recent_content),
            "daily_averages": daily_averages,
            "trend": trend,
            "trend_score": trend_score,
            "current_value": daily_averages.get(dates[-1], 0) if dates else 0,
            "average_value": sum(daily_averages.values()) / len(daily_averages) if daily_averages else 0
        }

    def analyze_keyword_trends(
        self,
        period_days: int = 30,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Analizar tendencias de keywords.

        Args:
            period_days: Período en días
            top_n: Número de keywords top

        Returns:
            Análisis de tendencias de keywords
        """
        if not self.content_history:
            return {"error": "No hay historial de contenido"}
        
        # Filtrar por período
        cutoff_date = datetime.utcnow() - timedelta(days=period_days)
        recent_content = [
            c for c in self.content_history
            if datetime.fromisoformat(c["timestamp"]) >= cutoff_date
        ]
        
        if not recent_content:
            return {"error": "No hay contenido en el período especificado"}
        
        # Extraer keywords de todo el contenido
        all_keywords = []
        for record in recent_content:
            content = record.get("content", "")
            words = content.lower().split()
            # Filtrar stop words básicas
            stop_words = {'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'a', 'y', 'o',
                         'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
            keywords = [w for w in words if w not in stop_words and len(w) > 3]
            all_keywords.extend(keywords)
        
        # Contar frecuencias
        keyword_freq = Counter(all_keywords)
        top_keywords = keyword_freq.most_common(top_n)
        
        return {
            "period_days": period_days,
            "total_keywords": len(all_keywords),
            "unique_keywords": len(keyword_freq),
            "top_keywords": [
                {"keyword": word, "count": count, "percentage": (count / len(all_keywords)) * 100}
                for word, count in top_keywords
            ]
        }

    def predict_future_trend(
        self,
        metric: str = "word_count",
        days_ahead: int = 7
    ) -> Dict[str, Any]:
        """
        Predecir tendencia futura.

        Args:
            metric: Métrica a predecir
            days_ahead: Días a predecir

        Returns:
            Predicción
        """
        if not self.content_history:
            return {"error": "No hay historial de contenido"}
        
        # Obtener datos recientes
        recent_data = [
            record.get(metric, 0)
            for record in self.content_history[-30:]  # Últimos 30 registros
        ]
        
        if len(recent_data) < 3:
            return {"error": "Datos insuficientes para predicción"}
        
        # Calcular promedio reciente
        recent_avg = sum(recent_data) / len(recent_data)
        
        # Calcular tendencia simple
        if len(recent_data) >= 2:
            trend = recent_data[-1] - recent_data[0]
            trend_per_day = trend / len(recent_data)
        else:
            trend_per_day = 0
        
        # Predecir valor futuro
        predicted_value = recent_avg + (trend_per_day * days_ahead)
        
        return {
            "metric": metric,
            "days_ahead": days_ahead,
            "current_value": recent_data[-1] if recent_data else 0,
            "recent_average": recent_avg,
            "trend_per_day": trend_per_day,
            "predicted_value": max(0, predicted_value),
            "confidence": min(1.0, len(recent_data) / 30)  # Más datos = más confianza
        }






