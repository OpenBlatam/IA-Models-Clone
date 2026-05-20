"""
Servicio de Seguimiento de Emociones Avanzado - Sistema completo de emociones
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class AdvancedEmotionTrackingService:
    """Servicio de seguimiento de emociones avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de emociones"""
        pass
    
    def record_emotion(
        self,
        user_id: str,
        emotion_data: Dict
    ) -> Dict:
        """
        Registra una emoción
        
        Args:
            user_id: ID del usuario
            emotion_data: Datos de emoción
        
        Returns:
            Emoción registrada
        """
        emotion = {
            "id": f"emotion_{datetime.now().timestamp()}",
            "user_id": user_id,
            "emotion_data": emotion_data,
            "emotion_type": emotion_data.get("emotion_type", "neutral"),
            "intensity": emotion_data.get("intensity", 5),
            "triggers": emotion_data.get("triggers", []),
            "context": emotion_data.get("context", {}),
            "timestamp": emotion_data.get("timestamp", datetime.now().isoformat()),
            "recorded_at": datetime.now().isoformat()
        }
        
        return emotion
    
    def analyze_emotion_patterns(
        self,
        user_id: str,
        emotions: List[Dict],
        days: int = 30
    ) -> Dict:
        """
        Analiza patrones emocionales
        
        Args:
            user_id: ID del usuario
            emotions: Lista de emociones
            days: Número de días
        
        Returns:
            Análisis de patrones emocionales
        """
        if not emotions:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_emotions": len(emotions),
            "emotion_distribution": self._analyze_emotion_distribution(emotions),
            "intensity_trends": self._analyze_intensity_trends(emotions),
            "emotional_triggers": self._identify_emotional_triggers(emotions),
            "emotional_stability": self._calculate_emotional_stability(emotions),
            "recommendations": self._generate_emotion_recommendations(emotions),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_emotional_state(
        self,
        user_id: str,
        current_context: Dict,
        emotion_history: List[Dict]
    ) -> Dict:
        """
        Predice estado emocional
        
        Args:
            user_id: ID del usuario
            current_context: Contexto actual
            emotion_history: Historial emocional
        
        Returns:
            Predicción de estado emocional
        """
        predicted_emotion = self._predict_emotion(current_context, emotion_history)
        
        return {
            "user_id": user_id,
            "predicted_emotion": predicted_emotion,
            "predicted_intensity": self._predict_intensity(current_context),
            "confidence": 0.78,
            "factors": self._identify_emotion_factors(current_context),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _analyze_emotion_distribution(self, emotions: List[Dict]) -> Dict:
        """Analiza distribución de emociones"""
        distribution = defaultdict(int)
        
        for emotion in emotions:
            emotion_type = emotion.get("emotion_type", "neutral")
            distribution[emotion_type] += 1
        
        return dict(distribution)
    
    def _analyze_intensity_trends(self, emotions: List[Dict]) -> Dict:
        """Analiza tendencias de intensidad"""
        if len(emotions) < 2:
            return {"trend": "insufficient_data"}
        
        intensities = [e.get("intensity", 5) for e in emotions]
        avg_intensity = statistics.mean(intensities)
        
        first_half = intensities[:len(intensities)//2]
        second_half = intensities[len(intensities)//2:]
        
        avg_first = statistics.mean(first_half) if first_half else 0
        avg_second = statistics.mean(second_half) if second_half else 0
        
        if avg_second < avg_first * 0.9:
            trend = "decreasing"
        elif avg_second > avg_first * 1.1:
            trend = "increasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "average_intensity": round(avg_intensity, 2),
            "change": round(avg_second - avg_first, 2)
        }
    
    def _identify_emotional_triggers(self, emotions: List[Dict]) -> List[Dict]:
        """Identifica triggers emocionales"""
        trigger_counts = defaultdict(int)
        
        for emotion in emotions:
            triggers = emotion.get("triggers", [])
            for trigger in triggers:
                trigger_counts[trigger] += 1
        
        sorted_triggers = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"trigger": trigger, "frequency": count}
            for trigger, count in sorted_triggers[:5]
        ]
    
    def _calculate_emotional_stability(self, emotions: List[Dict]) -> float:
        """Calcula estabilidad emocional"""
        if len(emotions) < 2:
            return 0.5
        
        intensities = [e.get("intensity", 5) for e in emotions]
        std_dev = statistics.stdev(intensities) if len(intensities) > 1 else 0
        
        # Menor desviación = mayor estabilidad
        stability = max(0, 1 - (std_dev / 10))
        
        return round(stability, 2)
    
    def _generate_emotion_recommendations(self, emotions: List[Dict]) -> List[str]:
        """Genera recomendaciones emocionales"""
        recommendations = []
        
        negative_emotions = [e for e in emotions if e.get("emotion_type") in ["sad", "anxious", "angry"]]
        if len(negative_emotions) > len(emotions) * 0.4:
            recommendations.append("Considera técnicas de regulación emocional")
        
        return recommendations
    
    def _predict_emotion(self, context: Dict, history: List[Dict]) -> str:
        """Predice emoción"""
        stress_level = context.get("stress_level", 5)
        
        if stress_level >= 7:
            return "anxious"
        elif stress_level <= 3:
            return "calm"
        else:
            return "neutral"
    
    def _predict_intensity(self, context: Dict) -> float:
        """Predice intensidad"""
        stress_level = context.get("stress_level", 5)
        return round(stress_level / 2, 1)
    
    def _identify_emotion_factors(self, context: Dict) -> List[str]:
        """Identifica factores emocionales"""
        factors = []
        
        if context.get("stress_level", 5) >= 7:
            factors.append("Estrés elevado")
        
        return factors

