"""
Advanced Analytics - Analytics Avanzado
======================================

Sistema avanzado de analytics con detección de patrones.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ConversationPattern:
    """Patrón detectado en conversaciones."""
    pattern_type: str
    frequency: int
    confidence: float
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserBehavior:
    """Comportamiento del usuario."""
    user_id: str
    total_sessions: int
    average_session_duration: float
    favorite_topics: List[str] = field(default_factory=list)
    preferred_time: Optional[str] = None
    engagement_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedAnalytics:
    """Analytics avanzado con detección de patrones."""
    
    def __init__(self):
        self.patterns: Dict[str, ConversationPattern] = {}
        self.user_behaviors: Dict[str, UserBehavior] = {}
    
    async def detect_patterns(
        self,
        sessions: List[Any],
        min_frequency: int = 3,
    ) -> List[ConversationPattern]:
        """
        Detectar patrones en conversaciones.
        
        Args:
            sessions: Lista de sesiones
            min_frequency: Frecuencia mínima para considerar patrón
        """
        # Patrones comunes detectados
        patterns = []
        
        # Patrón: Preguntas frecuentes
        questions = []
        for session in sessions:
            for msg in session.messages:
                if msg.role == "user" and "?" in msg.content:
                    questions.append(msg.content.lower())
        
        question_counts = Counter(questions)
        for question, count in question_counts.most_common(10):
            if count >= min_frequency:
                patterns.append(ConversationPattern(
                    pattern_type="frequent_question",
                    frequency=count,
                    confidence=min(count / len(sessions), 1.0),
                    examples=[question],
                ))
        
        # Patrón: Horarios de uso
        hour_usage = defaultdict(int)
        for session in sessions:
            for msg in session.messages:
                hour = msg.timestamp.hour
                hour_usage[hour] += 1
        
        if hour_usage:
            peak_hour = max(hour_usage.items(), key=lambda x: x[1])
            patterns.append(ConversationPattern(
                pattern_type="peak_usage_hour",
                frequency=peak_hour[1],
                confidence=0.8,
                examples=[f"Hour {peak_hour[0]}:00"],
                metadata={"peak_hour": peak_hour[0]},
            ))
        
        return patterns
    
    async def analyze_user_behavior(
        self,
        user_id: str,
        sessions: List[Any],
    ) -> UserBehavior:
        """Analizar comportamiento de usuario."""
        if not sessions:
            return UserBehavior(
                user_id=user_id,
                total_sessions=0,
                average_session_duration=0.0,
            )
        
        total_duration = sum(
            (s.updated_at - s.created_at).total_seconds()
            for s in sessions
        )
        avg_duration = total_duration / len(sessions)
        
        # Temas favoritos
        topics = []
        for session in sessions:
            # Análisis simplificado
            for msg in session.messages:
                if msg.role == "user":
                    # Detectar temas (simplificado)
                    content_lower = msg.content.lower()
                    if "python" in content_lower or "programación" in content_lower:
                        topics.append("programming")
                    elif "negocio" in content_lower or "empresa" in content_lower:
                        topics.append("business")
        
        favorite_topics = [topic for topic, count in Counter(topics).most_common(3)]
        
        # Hora preferida
        hours = [msg.timestamp.hour for s in sessions for msg in s.messages]
        preferred_time = f"{max(set(hours), key=hours.count)}:00" if hours else None
        
        # Engagement score (simplificado)
        total_messages = sum(len(s.messages) for s in sessions)
        engagement_score = min(total_messages / (len(sessions) * 10), 1.0)
        
        behavior = UserBehavior(
            user_id=user_id,
            total_sessions=len(sessions),
            average_session_duration=avg_duration,
            favorite_topics=favorite_topics,
            preferred_time=preferred_time,
            engagement_score=engagement_score,
        )
        
        self.user_behaviors[user_id] = behavior
        return behavior
    
    async def generate_insights(self, sessions: List[Any]) -> Dict[str, Any]:
        """Generar insights avanzados."""
        patterns = await self.detect_patterns(sessions)
        
        # Estadísticas avanzadas
        total_messages = sum(len(s.messages) for s in sessions)
        avg_messages_per_session = total_messages / len(sessions) if sessions else 0
        
        # Análisis de sentimiento general
        positive_count = 0
        negative_count = 0
        
        for session in sessions:
            for msg in session.messages:
                content_lower = msg.content.lower()
                if any(word in content_lower for word in ["bueno", "excelente", "gracias", "perfecto"]):
                    positive_count += 1
                elif any(word in content_lower for word in ["malo", "problema", "error", "no funciona"]):
                    negative_count += 1
        
        total_sentiment = positive_count + negative_count
        sentiment_ratio = positive_count / total_sentiment if total_sentiment > 0 else 0.5
        
        return {
            "patterns_detected": len(patterns),
            "patterns": [
                {
                    "type": p.pattern_type,
                    "frequency": p.frequency,
                    "confidence": p.confidence,
                }
                for p in patterns
            ],
            "avg_messages_per_session": avg_messages_per_session,
            "sentiment_ratio": sentiment_ratio,
            "total_sessions": len(sessions),
            "total_messages": total_messages,
        }
    
    def get_user_behavior(self, user_id: str) -> Optional[UserBehavior]:
        """Obtener comportamiento de usuario."""
        return self.user_behaviors.get(user_id)
































