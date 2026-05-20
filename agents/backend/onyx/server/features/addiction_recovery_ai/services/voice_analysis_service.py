"""
Servicio de Análisis de Voz - Detección de emociones y estado mediante análisis de voz
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class VoiceEmotion(str, Enum):
    """Emociones detectables en la voz"""
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"
    CALM = "calm"
    ANGRY = "angry"
    TIRED = "tired"
    STRESSED = "stressed"
    CONFIDENT = "confident"


class VoiceAnalysisService:
    """Servicio de análisis de voz para detección de emociones"""
    
    def __init__(self, openai_client=None):
        """
        Inicializa el servicio de análisis de voz
        
        Args:
            openai_client: Cliente de OpenAI (opcional, para Whisper/transcripción)
        """
        self.openai_client = openai_client
    
    def analyze_voice_recording(
        self,
        user_id: str,
        audio_data: bytes,
        duration_seconds: float,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Analiza una grabación de voz
        
        Args:
            user_id: ID del usuario
            audio_data: Datos de audio (bytes)
            duration_seconds: Duración en segundos
            metadata: Metadatos adicionales (opcional)
        
        Returns:
            Análisis de voz
        """
        # En implementación real, esto procesaría el audio
        # Por ahora, simulamos análisis básico
        
        analysis = {
            "user_id": user_id,
            "duration_seconds": duration_seconds,
            "transcription": self._transcribe_audio(audio_data) if self.openai_client else None,
            "emotions_detected": self._detect_voice_emotions(audio_data),
            "speech_characteristics": self._analyze_speech_characteristics(audio_data, duration_seconds),
            "risk_indicators": self._detect_voice_risk_indicators(audio_data),
            "analyzed_at": datetime.now().isoformat()
        }
        
        return analysis
    
    def analyze_voice_patterns(
        self,
        user_id: str,
        recordings: List[Dict]
    ) -> Dict:
        """
        Analiza patrones de voz a lo largo del tiempo
        
        Args:
            user_id: ID del usuario
            recordings: Lista de grabaciones con análisis
        
        Returns:
            Análisis de patrones
        """
        if not recordings or len(recordings) < 3:
            return {
                "user_id": user_id,
                "pattern_analysis": "insufficient_data",
                "message": "Se necesitan más grabaciones para análisis de patrones"
            }
        
        # Analizar tendencias emocionales
        emotions_over_time = [r.get("emotions_detected", []) for r in recordings]
        emotion_trends = self._analyze_emotion_trends(emotions_over_time)
        
        # Analizar características de habla
        speech_trends = self._analyze_speech_trends(recordings)
        
        return {
            "user_id": user_id,
            "total_recordings": len(recordings),
            "emotion_trends": emotion_trends,
            "speech_trends": speech_trends,
            "recommendations": self._generate_voice_recommendations(emotion_trends, speech_trends),
            "generated_at": datetime.now().isoformat()
        }
    
    def _transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio a texto usando Whisper"""
        if not self.openai_client:
            return None
        
        try:
            # En implementación real, usaría OpenAI Whisper API
            # Por ahora, retornamos None
            return None
        except Exception:
            return None
    
    def _detect_voice_emotions(self, audio_data: bytes) -> List[Dict]:
        """Detecta emociones en la voz"""
        # En implementación real, esto usaría análisis de audio avanzado
        # Por ahora, simulamos detección básica
        return [
            {
                "emotion": VoiceEmotion.CALM,
                "confidence": 0.75,
                "description": "Voz calmada y estable"
            }
        ]
    
    def _analyze_speech_characteristics(
        self,
        audio_data: bytes,
        duration: float
    ) -> Dict:
        """Analiza características del habla"""
        # En implementación real, esto analizaría:
        # - Velocidad del habla
        # - Tono
        # - Volumen
        # - Pausas
        # - Claridad
        
        return {
            "speech_rate": "normal",  # lento, normal, rápido
            "pitch": "medium",  # bajo, medio, alto
            "volume": "normal",  # bajo, normal, alto
            "clarity": "clear",  # claro, poco claro
            "pauses": "normal"  # muchas, normales, pocas
        }
    
    def _detect_voice_risk_indicators(self, audio_data: bytes) -> List[str]:
        """Detecta indicadores de riesgo en la voz"""
        # En implementación real, esto detectaría:
        # - Voz temblorosa (ansiedad)
        # - Voz muy baja (depresión)
        # - Voz agitada (estrés)
        # - Pausas largas (pensamientos negativos)
        
        return []
    
    def _analyze_emotion_trends(self, emotions_over_time: List[List[Dict]]) -> Dict:
        """Analiza tendencias emocionales"""
        emotion_counts = {}
        
        for emotions in emotions_over_time:
            for emotion_obj in emotions:
                emotion = emotion_obj.get("emotion")
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return {
            "most_common_emotions": sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "trend": "stable"  # improving, declining, stable
        }
    
    def _analyze_speech_trends(self, recordings: List[Dict]) -> Dict:
        """Analiza tendencias en características del habla"""
        return {
            "speech_rate_trend": "stable",
            "clarity_trend": "stable"
        }
    
    def _generate_voice_recommendations(
        self,
        emotion_trends: Dict,
        speech_trends: Dict
    ) -> List[str]:
        """Genera recomendaciones basadas en análisis de voz"""
        recommendations = []
        
        # Si hay muchas emociones negativas detectadas
        if emotion_trends.get("most_common_emotions"):
            top_emotion = emotion_trends["most_common_emotions"][0][0] if emotion_trends["most_common_emotions"] else None
            if top_emotion in [VoiceEmotion.SAD, VoiceEmotion.ANXIOUS, VoiceEmotion.STRESSED]:
                recommendations.append("Se detectaron emociones negativas en tu voz. Considera contactar tu sistema de apoyo.")
        
        return recommendations

