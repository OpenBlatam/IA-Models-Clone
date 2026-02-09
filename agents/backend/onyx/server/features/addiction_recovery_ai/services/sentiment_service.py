"""
Servicio de Análisis de Sentimientos - Análisis emocional de texto y voz
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class SentimentType(str, Enum):
    """Tipos de sentimiento"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class EmotionType(str, Enum):
    """Tipos de emociones"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    CALM = "calm"
    HOPEFUL = "hopeful"
    FRUSTRATED = "frustrated"
    CONFIDENT = "confident"


class SentimentService:
    """Servicio de análisis de sentimientos y emociones"""
    
    def __init__(self, openai_client=None):
        """
        Inicializa el servicio de análisis de sentimientos
        
        Args:
            openai_client: Cliente de OpenAI (opcional)
        """
        self.openai_client = openai_client
        self.emotion_keywords = self._load_emotion_keywords()
    
    def analyze_sentiment(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Analiza el sentimiento de un texto
        
        Args:
            text: Texto a analizar
            context: Contexto adicional (opcional)
        
        Returns:
            Análisis de sentimiento
        """
        # Análisis básico con keywords
        sentiment_score = self._calculate_sentiment_score(text)
        emotions = self._detect_emotions(text)
        sentiment_type = self._classify_sentiment(sentiment_score)
        
        analysis = {
            "text": text,
            "sentiment_type": sentiment_type,
            "sentiment_score": sentiment_score,
            "emotions": emotions,
            "risk_indicators": self._detect_risk_indicators(text),
            "analyzed_at": datetime.now().isoformat()
        }
        
        # Si hay cliente de OpenAI, mejorar análisis
        if self.openai_client:
            enhanced = self._enhance_with_ai(text, context)
            analysis.update(enhanced)
        
        return analysis
    
    def analyze_journal_entry(
        self,
        user_id: str,
        entry_text: str,
        entry_date: str
    ) -> Dict:
        """
        Analiza una entrada de diario
        
        Args:
            user_id: ID del usuario
            entry_text: Texto de la entrada
            entry_date: Fecha de la entrada
        
        Returns:
            Análisis completo de la entrada
        """
        sentiment_analysis = self.analyze_sentiment(entry_text)
        
        # Detectar temas importantes
        topics = self._extract_topics(entry_text)
        
        # Recomendaciones basadas en sentimiento
        recommendations = self._generate_sentiment_recommendations(sentiment_analysis)
        
        return {
            "user_id": user_id,
            "entry_date": entry_date,
            "sentiment_analysis": sentiment_analysis,
            "topics": topics,
            "recommendations": recommendations,
            "analyzed_at": datetime.now().isoformat()
        }
    
    def track_emotional_trend(
        self,
        user_id: str,
        sentiment_data: List[Dict]
    ) -> Dict:
        """
        Analiza tendencia emocional a lo largo del tiempo
        
        Args:
            user_id: ID del usuario
            sentiment_data: Lista de análisis de sentimientos
        
        Returns:
            Análisis de tendencia emocional
        """
        if not sentiment_data or len(sentiment_data) < 3:
            return {
                "user_id": user_id,
                "trend": "insufficient_data",
                "message": "Se necesitan más datos para analizar tendencias"
            }
        
        # Calcular promedios
        positive_count = sum(1 for s in sentiment_data if s.get("sentiment_type") == SentimentType.POSITIVE)
        negative_count = sum(1 for s in sentiment_data if s.get("sentiment_type") == SentimentType.NEGATIVE)
        
        avg_score = sum(s.get("sentiment_score", 0) for s in sentiment_data) / len(sentiment_data)
        
        # Determinar tendencia
        recent = sentiment_data[-7:] if len(sentiment_data) >= 7 else sentiment_data
        older = sentiment_data[:-7] if len(sentiment_data) >= 7 else []
        
        recent_avg = sum(s.get("sentiment_score", 0) for s in recent) / len(recent) if recent else 0
        older_avg = sum(s.get("sentiment_score", 0) for s in older) / len(older) if older else 0
        
        trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
        
        return {
            "user_id": user_id,
            "total_entries": len(sentiment_data),
            "positive_ratio": positive_count / len(sentiment_data) if sentiment_data else 0,
            "negative_ratio": negative_count / len(sentiment_data) if sentiment_data else 0,
            "average_sentiment_score": round(avg_score, 2),
            "trend": trend,
            "trend_strength": abs(recent_avg - older_avg) if older else 0,
            "most_common_emotions": self._get_most_common_emotions(sentiment_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calcula score de sentimiento (-1 a 1)"""
        text_lower = text.lower()
        
        positive_words = ["bien", "genial", "feliz", "mejor", "éxito", "progreso", "orgulloso", "motivado"]
        negative_words = ["mal", "triste", "difícil", "frustrado", "ansioso", "preocupado", "desanimado"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        score = (positive_count - negative_count) / max(total_words, 1)
        return max(-1.0, min(1.0, score))
    
    def _detect_emotions(self, text: str) -> List[Dict]:
        """Detecta emociones en el texto"""
        text_lower = text.lower()
        detected = []
        
        emotion_patterns = {
            EmotionType.HAPPY: ["feliz", "contento", "alegre", "genial", "bien"],
            EmotionType.SAD: ["triste", "deprimido", "melancólico", "desanimado"],
            EmotionType.ANXIOUS: ["ansioso", "preocupado", "nervioso", "inquieto"],
            EmotionType.CALM: ["tranquilo", "calmado", "sereno", "relajado"],
            EmotionType.HOPEFUL: ["esperanzado", "optimista", "confiado"],
            EmotionType.FRUSTRATED: ["frustrado", "molesto", "irritado"],
            EmotionType.CONFIDENT: ["seguro", "confiado", "determinado"]
        }
        
        for emotion, keywords in emotion_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                detected.append({
                    "emotion": emotion,
                    "confidence": 0.7  # En implementación real, usar ML
                })
        
        return detected[:3]  # Top 3 emociones
    
    def _classify_sentiment(self, score: float) -> str:
        """Clasifica sentimiento basado en score"""
        if score > 0.2:
            return SentimentType.POSITIVE
        elif score < -0.2:
            return SentimentType.NEGATIVE
        else:
            return SentimentType.NEUTRAL
    
    def _detect_risk_indicators(self, text: str) -> List[str]:
        """Detecta indicadores de riesgo en el texto"""
        text_lower = text.lower()
        indicators = []
        
        risk_keywords = {
            "suicidal": ["suicidio", "morir", "no vale la pena", "sin esperanza"],
            "relapse": ["recaída", "consumir", "volver a", "tentación"],
            "isolation": ["solo", "aislado", "nadie", "abandonado"],
            "hopeless": ["sin esperanza", "inútil", "no puedo", "imposible"]
        }
        
        for risk_type, keywords in risk_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                indicators.append(risk_type)
        
        return indicators
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extrae temas principales del texto"""
        topics = []
        text_lower = text.lower()
        
        topic_keywords = {
            "cravings": ["craving", "deseo", "ansia", "necesito"],
            "stress": ["estrés", "presión", "tensión"],
            "support": ["apoyo", "familia", "amigos", "grupo"],
            "progress": ["progreso", "mejor", "avance", "logro"],
            "health": ["salud", "ejercicio", "sueño", "energía"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _generate_sentiment_recommendations(self, analysis: Dict) -> List[str]:
        """Genera recomendaciones basadas en análisis de sentimiento"""
        recommendations = []
        
        sentiment_type = analysis.get("sentiment_type")
        risk_indicators = analysis.get("risk_indicators", [])
        
        if "suicidal" in risk_indicators:
            recommendations.append("⚠️ CRÍTICO: Busca ayuda profesional inmediatamente. Llama a una línea de crisis.")
        
        if sentiment_type == SentimentType.NEGATIVE:
            recommendations.append("Tu estado emocional parece negativo. Considera contactar tu sistema de apoyo.")
            recommendations.append("Practica técnicas de relajación o ejercicio para mejorar tu estado de ánimo.")
        
        if "relapse" in risk_indicators:
            recommendations.append("Se detectaron señales de riesgo de recaída. Revisa tu plan de emergencia.")
        
        if not recommendations:
            recommendations.append("Continúa con tu plan de recuperación. Estás haciendo un buen trabajo.")
        
        return recommendations
    
    def _get_most_common_emotions(self, sentiment_data: List[Dict]) -> List[Dict]:
        """Obtiene emociones más comunes"""
        emotion_count = {}
        
        for data in sentiment_data:
            emotions = data.get("emotions", [])
            for emotion_obj in emotions:
                emotion = emotion_obj.get("emotion")
                emotion_count[emotion] = emotion_count.get(emotion, 0) + 1
        
        sorted_emotions = sorted(emotion_count.items(), key=lambda x: x[1], reverse=True)
        return [{"emotion": e[0], "count": e[1]} for e in sorted_emotions[:5]]
    
    def _enhance_with_ai(self, text: str, context: Optional[Dict]) -> Dict:
        """Mejora análisis usando IA"""
        if not self.openai_client:
            return {}
        
        try:
            prompt = f"""
            Analiza el sentimiento y emociones de este texto de alguien en recuperación de adicciones:
            
            "{text}"
            
            Proporciona:
            1. Sentimiento general (positivo/negativo/neutral)
            2. Emociones detectadas
            3. Nivel de riesgo (bajo/medio/alto/crítico)
            4. Recomendación breve
            
            Responde en formato JSON con: sentiment, emotions, risk_level, recommendation
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un experto en análisis de sentimientos para recuperación de adicciones."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            import json
            ai_response = response.choices[0].message.content
            try:
                ai_data = json.loads(ai_response)
                return {
                    "ai_sentiment": ai_data.get("sentiment", ""),
                    "ai_emotions": ai_data.get("emotions", []),
                    "ai_risk_level": ai_data.get("risk_level", ""),
                    "ai_recommendation": ai_data.get("recommendation", "")
                }
            except json.JSONDecodeError:
                return {"ai_enhanced_analysis": ai_response}
        except Exception:
            return {}
    
    def _load_emotion_keywords(self) -> Dict:
        """Carga keywords de emociones"""
        return {
            "positive": ["bien", "genial", "feliz", "mejor", "éxito"],
            "negative": ["mal", "triste", "difícil", "frustrado"],
            "anxiety": ["ansioso", "preocupado", "nervioso"],
            "hope": ["esperanza", "optimista", "confiado"]
        }

