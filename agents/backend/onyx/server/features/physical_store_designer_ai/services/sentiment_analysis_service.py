"""
Sentiment Analysis Service - Análisis de sentimiento
"""

import logging
from typing import Dict, Any, List, Optional
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class SentimentAnalysisService:
    """Servicio para análisis de sentimiento"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
    
    async def analyze_sentiment(
        self,
        text: str
    ) -> Dict[str, Any]:
        """Analizar sentimiento de texto"""
        
        if self.llm_service.client:
            try:
                return await self._analyze_with_llm(text)
            except Exception as e:
                logger.error(f"Error analizando sentimiento con LLM: {e}")
                return self._analyze_basic(text)
        else:
            return self._analyze_basic(text)
    
    async def _analyze_with_llm(self, text: str) -> Dict[str, Any]:
        """Analizar sentimiento usando LLM"""
        prompt = f"""Analiza el sentimiento del siguiente texto y proporciona:
1. Sentimiento general (positive, neutral, negative)
2. Score de sentimiento (-1 a 1, donde -1 es muy negativo y 1 es muy positivo)
3. Emociones detectadas
4. Aspectos clave mencionados

Texto: {text}

Responde en formato JSON con estas claves:
- sentiment: string (positive/neutral/negative)
- score: float (-1 a 1)
- emotions: array de strings
- key_aspects: array de strings
- confidence: float (0 a 1)"""
        
        result = await self.llm_service.generate_structured(
            prompt=prompt,
            system_prompt="Eres un experto en análisis de sentimiento y emociones."
        )
        
        return result if result else self._analyze_basic(text)
    
    def _analyze_basic(self, text: str) -> Dict[str, Any]:
        """Análisis básico de sentimiento"""
        text_lower = text.lower()
        
        positive_words = ["excelente", "bueno", "genial", "perfecto", "me encanta", "recomiendo", "satisfecho"]
        negative_words = ["malo", "terrible", "horrible", "no recomiendo", "decepcionado", "problema", "error"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            score = min(0.7, 0.3 + (positive_count * 0.1))
        elif negative_count > positive_count:
            sentiment = "negative"
            score = max(-0.7, -0.3 - (negative_count * 0.1))
        else:
            sentiment = "neutral"
            score = 0.0
        
        return {
            "sentiment": sentiment,
            "score": score,
            "emotions": self._detect_emotions(text_lower),
            "key_aspects": [],
            "confidence": 0.6
        }
    
    def _detect_emotions(self, text: str) -> List[str]:
        """Detectar emociones básicas"""
        emotions = []
        
        emotion_keywords = {
            "happy": ["feliz", "contento", "alegre", "satisfecho"],
            "excited": ["emocionado", "entusiasmado", "ansioso"],
            "frustrated": ["frustrado", "molesto", "irritado"],
            "disappointed": ["decepcionado", "desilusionado"],
            "confident": ["confiado", "seguro", "convencido"]
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text for keyword in keywords):
                emotions.append(emotion)
        
        return emotions
    
    async def analyze_feedback_sentiment(
        self,
        feedback_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar sentimiento de múltiples feedbacks"""
        
        sentiments = []
        for feedback in feedback_list:
            content = feedback.get("content", "")
            if content:
                sentiment = await self.analyze_sentiment(content)
                sentiments.append({
                    "feedback_id": feedback.get("id"),
                    "sentiment": sentiment
                })
        
        if not sentiments:
            return {"message": "No hay feedbacks para analizar"}
        
        # Calcular promedios
        scores = [s["sentiment"]["score"] for s in sentiments]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        sentiment_counts = {}
        for s in sentiments:
            sent = s["sentiment"]["sentiment"]
            sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
        
        return {
            "total_feedbacks": len(sentiments),
            "average_sentiment_score": round(avg_score, 2),
            "sentiment_distribution": sentiment_counts,
            "overall_sentiment": "positive" if avg_score > 0.2 else "negative" if avg_score < -0.2 else "neutral",
            "detailed_analysis": sentiments
        }




