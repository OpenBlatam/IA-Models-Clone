"""
Conversation Analyzer - Análisis de conversaciones
==================================================

Sistema avanzado de análisis de conversaciones para extraer insights,
patrones y estadísticas.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter, defaultdict

from .chat_session import ChatSession, ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class ConversationInsights:
    """Insights extraídos de una conversación."""
    session_id: str
    total_messages: int
    user_messages: int
    assistant_messages: int
    average_message_length: float
    topics: List[str] = field(default_factory=list)
    sentiment_trend: Dict[str, float] = field(default_factory=dict)
    key_phrases: List[str] = field(default_factory=list)
    conversation_duration: float = 0.0
    message_frequency: Dict[str, int] = field(default_factory=dict)
    response_time_stats: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationAnalyzer:
    """Analizador de conversaciones."""
    
    def __init__(self):
        self.stop_words = {
            "el", "la", "los", "las", "un", "una", "de", "del", "en", "a",
            "que", "es", "se", "no", "te", "le", "da", "su", "por", "son",
            "con", "para", "al", "the", "a", "an", "and", "or", "but", "in",
            "on", "at", "to", "for", "of", "with", "by", "is", "are", "was",
        }
    
    async def analyze(self, session: ChatSession) -> ConversationInsights:
        """
        Analizar una conversación completa.
        
        Args:
            session: Sesión de chat a analizar
        
        Returns:
            Insights de la conversación
        """
        if not session.messages:
            return ConversationInsights(
                session_id=session.session_id,
                total_messages=0,
                user_messages=0,
                assistant_messages=0,
                average_message_length=0.0,
            )
        
        # Estadísticas básicas
        user_messages = [m for m in session.messages if m.role == "user"]
        assistant_messages = [m for m in session.messages if m.role == "assistant"]
        
        total_length = sum(len(m.content) for m in session.messages)
        avg_length = total_length / len(session.messages) if session.messages else 0
        
        # Duración de la conversación
        if session.messages:
            duration = (session.updated_at - session.created_at).total_seconds()
        else:
            duration = 0.0
        
        # Frecuencia de mensajes por hora
        message_frequency = self._calculate_message_frequency(session.messages)
        
        # Frases clave
        key_phrases = await self._extract_key_phrases(session.messages)
        
        # Temas detectados (simplificado)
        topics = await self._detect_topics(session.messages)
        
        # Tendencias de sentimiento (simplificado)
        sentiment_trend = await self._analyze_sentiment_trend(session.messages)
        
        # Estadísticas de tiempo de respuesta
        response_time_stats = self._calculate_response_times(session.messages)
        
        return ConversationInsights(
            session_id=session.session_id,
            total_messages=len(session.messages),
            user_messages=len(user_messages),
            assistant_messages=len(assistant_messages),
            average_message_length=avg_length,
            topics=topics,
            sentiment_trend=sentiment_trend,
            key_phrases=key_phrases,
            conversation_duration=duration,
            message_frequency=message_frequency,
            response_time_stats=response_time_stats,
        )
    
    def _calculate_message_frequency(self, messages: List[ChatMessage]) -> Dict[str, int]:
        """Calcular frecuencia de mensajes por hora del día."""
        frequency = defaultdict(int)
        
        for msg in messages:
            hour = msg.timestamp.hour
            frequency[f"{hour:02d}:00"] += 1
        
        return dict(frequency)
    
    async def _extract_key_phrases(self, messages: List[ChatMessage], top_n: int = 10) -> List[str]:
        """Extraer frases clave de la conversación."""
        # Contar palabras (simplificado)
        word_counts = Counter()
        
        for msg in messages:
            words = msg.content.lower().split()
            # Filtrar stop words
            words = [w for w in words if w not in self.stop_words and len(w) > 3]
            word_counts.update(words)
        
        # Obtener top N
        top_words = [word for word, count in word_counts.most_common(top_n)]
        return top_words
    
    async def _detect_topics(self, messages: List[ChatMessage]) -> List[str]:
        """Detectar temas en la conversación (simplificado)."""
        # En producción, usaría NLP o embeddings
        topics_keywords = {
            "tecnología": ["código", "programa", "software", "aplicación", "tecnología"],
            "negocios": ["negocio", "empresa", "ventas", "marketing", "cliente"],
            "educación": ["aprender", "estudio", "curso", "educación", "enseñanza"],
            "salud": ["salud", "médico", "tratamiento", "enfermedad", "cura"],
        }
        
        detected_topics = []
        content = " ".join([m.content.lower() for m in messages])
        
        for topic, keywords in topics_keywords.items():
            if any(keyword in content for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics[:5]  # Top 5 temas
    
    async def _analyze_sentiment_trend(self, messages: List[ChatMessage]) -> Dict[str, float]:
        """Analizar tendencia de sentimiento."""
        positive_words = ["bueno", "excelente", "perfecto", "genial", "me gusta", "gracias"]
        negative_words = ["malo", "terrible", "horrible", "no me gusta", "problema"]
        
        positive_count = 0
        negative_count = 0
        
        for msg in messages:
            content_lower = msg.content.lower()
            positive_count += sum(1 for word in positive_words if word in content_lower)
            negative_count += sum(1 for word in negative_words if word in content_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return {"positive": 0.5, "negative": 0.5, "neutral": 1.0}
        
        return {
            "positive": positive_count / total,
            "negative": negative_count / total,
            "neutral": 1.0 - (positive_count + negative_count) / total if total > 0 else 1.0,
        }
    
    def _calculate_response_times(self, messages: List[ChatMessage]) -> Dict[str, float]:
        """Calcular estadísticas de tiempo de respuesta."""
        response_times = []
        
        for i in range(1, len(messages)):
            prev_msg = messages[i - 1]
            curr_msg = messages[i]
            
            if prev_msg.role == "user" and curr_msg.role == "assistant":
                time_diff = (curr_msg.timestamp - prev_msg.timestamp).total_seconds()
                if time_diff > 0:
                    response_times.append(time_diff)
        
        if not response_times:
            return {"average": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "average": sum(response_times) / len(response_times),
            "min": min(response_times),
            "max": max(response_times),
            "median": sorted(response_times)[len(response_times) // 2],
        }
    
    async def generate_summary(self, session: ChatSession) -> str:
        """Generar resumen de la conversación."""
        insights = await self.analyze(session)
        
        summary = f"""
Resumen de Conversación - Sesión {session.session_id[:8]}...

📊 Estadísticas:
- Total de mensajes: {insights.total_messages}
- Mensajes del usuario: {insights.user_messages}
- Mensajes del asistente: {insights.assistant_messages}
- Duración: {insights.conversation_duration / 60:.1f} minutos
- Longitud promedio de mensaje: {insights.average_message_length:.0f} caracteres

🎯 Temas detectados: {', '.join(insights.topics) if insights.topics else 'Ninguno'}

💬 Frases clave: {', '.join(insights.key_phrases[:5]) if insights.key_phrases else 'Ninguna'}

⏱️ Tiempo promedio de respuesta: {insights.response_time_stats.get('average', 0):.2f} segundos
"""
        
        return summary.strip()
    
    async def compare_sessions(self, sessions: List[ChatSession]) -> Dict[str, Any]:
        """Comparar múltiples sesiones."""
        insights_list = [await self.analyze(s) for s in sessions]
        
        return {
            "total_sessions": len(sessions),
            "total_messages": sum(i.total_messages for i in insights_list),
            "average_messages_per_session": sum(i.total_messages for i in insights_list) / len(sessions) if sessions else 0,
            "average_duration": sum(i.conversation_duration for i in insights_list) / len(sessions) if sessions else 0,
            "common_topics": self._find_common_topics(insights_list),
            "average_response_time": sum(i.response_time_stats.get("average", 0) for i in insights_list) / len(sessions) if sessions else 0,
        }
    
    def _find_common_topics(self, insights_list: List[ConversationInsights]) -> List[str]:
        """Encontrar temas comunes."""
        topic_counts = Counter()
        for insights in insights_list:
            topic_counts.update(insights.topics)
        
        return [topic for topic, count in topic_counts.most_common(5)]
































