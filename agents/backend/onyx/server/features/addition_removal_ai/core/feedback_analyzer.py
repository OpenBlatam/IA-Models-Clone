"""
Feedback Analyzer - Sistema de análisis de feedback
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class FeedbackItem:
    """Item de feedback"""
    id: str
    content_id: str
    feedback_text: str
    rating: Optional[float] = None
    sentiment: Optional[str] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None


class FeedbackAnalyzer:
    """Analizador de feedback"""

    def __init__(self):
        """Inicializar analizador"""
        self.feedback_items: List[FeedbackItem] = []
        
        # Palabras positivas
        self.positive_words = {
            'excelente', 'bueno', 'genial', 'perfecto', 'útil', 'ayuda',
            'excellent', 'good', 'great', 'perfect', 'useful', 'helpful',
            'me gusta', 'me encanta', 'recomiendo', 'recomendado',
            'i like', 'i love', 'recommend', 'recommended'
        }
        
        # Palabras negativas
        self.negative_words = {
            'malo', 'mal', 'terrible', 'horrible', 'inútil', 'no sirve',
            'bad', 'terrible', 'horrible', 'useless', 'doesn\'t work',
            'no me gusta', 'no funciona', 'problema', 'error',
            'don\'t like', 'doesn\'t work', 'problem', 'error'
        }

    def add_feedback(
        self,
        content_id: str,
        feedback_text: str,
        rating: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Agregar feedback.

        Args:
            content_id: ID del contenido
            feedback_text: Texto del feedback
            rating: Calificación (opcional)
            metadata: Metadatos adicionales

        Returns:
            ID del feedback
        """
        import uuid
        
        feedback_id = str(uuid.uuid4())
        
        # Analizar sentimiento básico
        sentiment = self._analyze_sentiment(feedback_text)
        
        feedback = FeedbackItem(
            id=feedback_id,
            content_id=content_id,
            feedback_text=feedback_text,
            rating=rating,
            sentiment=sentiment,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.feedback_items.append(feedback)
        logger.debug(f"Feedback agregado: {feedback_id}")
        
        return feedback_id

    def analyze_content_feedback(
        self,
        content_id: str
    ) -> Dict[str, Any]:
        """
        Analizar feedback de un contenido.

        Args:
            content_id: ID del contenido

        Returns:
            Análisis de feedback
        """
        content_feedback = [
            f for f in self.feedback_items
            if f.content_id == content_id
        ]
        
        if not content_feedback:
            return {"error": "No hay feedback para este contenido"}
        
        # Estadísticas básicas
        total_feedback = len(content_feedback)
        
        # Sentimientos
        sentiments = [f.sentiment for f in content_feedback if f.sentiment]
        sentiment_counts = Counter(sentiments)
        
        # Ratings
        ratings = [f.rating for f in content_feedback if f.rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        
        # Temas comunes
        common_themes = self._extract_common_themes(content_feedback)
        
        # Palabras clave
        keywords = self._extract_keywords(content_feedback)
        
        return {
            "content_id": content_id,
            "total_feedback": total_feedback,
            "sentiment_distribution": dict(sentiment_counts),
            "average_rating": avg_rating,
            "rating_count": len(ratings),
            "common_themes": common_themes,
            "keywords": keywords,
            "recent_feedback": [
                {
                    "id": f.id,
                    "text": f.feedback_text[:200],  # Primeros 200 caracteres
                    "rating": f.rating,
                    "sentiment": f.sentiment,
                    "timestamp": f.timestamp.isoformat() if f.timestamp else None
                }
                for f in sorted(content_feedback, key=lambda x: x.timestamp or datetime.min, reverse=True)[:10]
            ]
        }

    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de todo el feedback.

        Returns:
            Resumen de feedback
        """
        if not self.feedback_items:
            return {"error": "No hay feedback disponible"}
        
        total = len(self.feedback_items)
        
        # Sentimientos
        sentiments = [f.sentiment for f in self.feedback_items if f.sentiment]
        sentiment_counts = Counter(sentiments)
        
        # Ratings
        ratings = [f.rating for f in self.feedback_items if f.rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        
        # Contenidos más comentados
        content_counts = Counter(f.content_id for f in self.feedback_items)
        top_content = content_counts.most_common(10)
        
        return {
            "total_feedback": total,
            "average_rating": avg_rating,
            "rating_count": len(ratings),
            "sentiment_distribution": dict(sentiment_counts),
            "top_content": [
                {"content_id": content_id, "feedback_count": count}
                for content_id, count in top_content
            ]
        }

    def _analyze_sentiment(self, text: str) -> str:
        """Analizar sentimiento básico"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _extract_common_themes(
        self,
        feedback_items: List[FeedbackItem]
    ) -> List[Dict[str, Any]]:
        """Extraer temas comunes"""
        # Palabras comunes en feedback
        all_words = []
        for item in feedback_items:
            words = item.feedback_text.lower().split()
            # Filtrar stop words
            stop_words = {'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'a', 'y', 'o',
                         'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
            words = [w for w in words if w not in stop_words and len(w) > 3]
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(5)
        
        return [
            {"theme": word, "frequency": count}
            for word, count in common_words
        ]

    def _extract_keywords(
        self,
        feedback_items: List[FeedbackItem]
    ) -> List[str]:
        """Extraer keywords"""
        all_text = " ".join(item.feedback_text for item in feedback_items)
        words = all_text.lower().split()
        
        stop_words = {'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'a', 'y', 'o',
                     'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        keywords = [w for w in words if w not in stop_words and len(w) > 4]
        
        keyword_counts = Counter(keywords)
        return [word for word, _ in keyword_counts.most_common(10)]






