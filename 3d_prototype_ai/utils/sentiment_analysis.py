"""
Sentiment Analysis - Sistema de análisis de sentimientos
=========================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class Sentiment(str, Enum):
    """Tipos de sentimiento"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class SentimentAnalysis:
    """Sistema de análisis de sentimientos"""
    
    def __init__(self):
        self.analyzed_texts: List[Dict[str, Any]] = []
    
    def analyze_text(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Analiza sentimiento de un texto"""
        # Análisis simplificado (en producción usaría NLP real)
        text_lower = text.lower()
        
        positive_words = ["excelente", "bueno", "genial", "perfecto", "me encanta", "recomiendo"]
        negative_words = ["malo", "terrible", "horrible", "no funciona", "problema", "error"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = Sentiment.POSITIVE
            score = 0.5 + (positive_count * 0.1)
        elif negative_count > positive_count:
            sentiment = Sentiment.NEGATIVE
            score = 0.5 - (negative_count * 0.1)
        else:
            sentiment = Sentiment.NEUTRAL
            score = 0.5
        
        score = max(0.0, min(1.0, score))
        
        result = {
            "text": text,
            "sentiment": sentiment.value,
            "score": score,
            "confidence": abs(score - 0.5) * 2,  # 0 a 1
            "context": context,
            "analyzed_at": datetime.now().isoformat()
        }
        
        self.analyzed_texts.append(result)
        
        return result
    
    def analyze_reviews(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza sentimientos de múltiples reseñas"""
        sentiments = []
        
        for review in reviews:
            text = review.get("comment", "") or review.get("text", "")
            if text:
                analysis = self.analyze_text(text, "review")
                sentiments.append(analysis)
        
        if not sentiments:
            return {
                "total_reviews": 0,
                "average_sentiment": "neutral",
                "average_score": 0.5
            }
        
        positive_count = sum(1 for s in sentiments if s["sentiment"] == "positive")
        negative_count = sum(1 for s in sentiments if s["sentiment"] == "negative")
        neutral_count = sum(1 for s in sentiments if s["sentiment"] == "neutral")
        
        avg_score = sum(s["score"] for s in sentiments) / len(sentiments)
        
        if avg_score > 0.6:
            avg_sentiment = "positive"
        elif avg_score < 0.4:
            avg_sentiment = "negative"
        else:
            avg_sentiment = "neutral"
        
        return {
            "total_reviews": len(sentiments),
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "average_sentiment": avg_sentiment,
            "average_score": avg_score,
            "sentiment_distribution": {
                "positive": (positive_count / len(sentiments) * 100) if sentiments else 0,
                "negative": (negative_count / len(sentiments) * 100) if sentiments else 0,
                "neutral": (neutral_count / len(sentiments) * 100) if sentiments else 0
            }
        }
    
    def get_sentiment_trends(self, days: int = 30) -> Dict[str, Any]:
        """Obtiene tendencias de sentimiento"""
        cutoff = datetime.now() - timedelta(days=days)
        
        recent_analyses = [
            a for a in self.analyzed_texts
            if datetime.fromisoformat(a["analyzed_at"]) > cutoff
        ]
        
        if not recent_analyses:
            return {"trend": "stable", "data": []}
        
        # Agrupar por día
        daily_sentiments = defaultdict(list)
        for analysis in recent_analyses:
            date = datetime.fromisoformat(analysis["analyzed_at"]).date()
            daily_sentiments[date].append(analysis["score"])
        
        trend_data = [
            {
                "date": str(date),
                "average_score": sum(scores) / len(scores),
                "count": len(scores)
            }
            for date, scores in sorted(daily_sentiments.items())
        ]
        
        # Calcular tendencia
        if len(trend_data) >= 2:
            first_avg = trend_data[0]["average_score"]
            last_avg = trend_data[-1]["average_score"]
            
            if last_avg > first_avg + 0.1:
                trend = "improving"
            elif last_avg < first_avg - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "data": trend_data,
            "period_days": days
        }




