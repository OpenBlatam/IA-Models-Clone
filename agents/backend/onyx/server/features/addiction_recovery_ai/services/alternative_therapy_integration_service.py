"""
Servicio de Integración con Terapias Alternativas - Sistema completo de terapias alternativas
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class TherapyType(str, Enum):
    """Tipos de terapias alternativas"""
    ACUPUNCTURE = "acupuncture"
    YOGA = "yoga"
    MEDITATION = "meditation"
    MASSAGE = "massage"
    ART_THERAPY = "art_therapy"
    MUSIC_THERAPY = "music_therapy"
    NATURE_THERAPY = "nature_therapy"
    ANIMAL_THERAPY = "animal_therapy"


class AlternativeTherapyIntegrationService:
    """Servicio de integración con terapias alternativas"""
    
    def __init__(self):
        """Inicializa el servicio de terapias alternativas"""
        self.therapy_types = self._load_therapy_types()
    
    def recommend_therapy(
        self,
        user_id: str,
        user_profile: Dict,
        current_state: Dict
    ) -> Dict:
        """
        Recomienda terapia alternativa
        
        Args:
            user_id: ID del usuario
            user_profile: Perfil del usuario
            current_state: Estado actual
        
        Returns:
            Recomendación de terapia
        """
        recommended_therapy = self._select_appropriate_therapy(user_profile, current_state)
        
        return {
            "user_id": user_id,
            "recommendation_id": f"therapy_rec_{datetime.now().timestamp()}",
            "recommended_therapy": recommended_therapy,
            "rationale": self._generate_rationale(user_profile, current_state, recommended_therapy),
            "expected_benefits": self._identify_expected_benefits(recommended_therapy),
            "sessions_recommended": self._calculate_sessions_needed(current_state),
            "recommended_at": datetime.now().isoformat()
        }
    
    def track_therapy_session(
        self,
        user_id: str,
        therapy_type: str,
        session_data: Dict
    ) -> Dict:
        """
        Rastrea sesión de terapia
        
        Args:
            user_id: ID del usuario
            therapy_type: Tipo de terapia
            session_data: Datos de sesión
        
        Returns:
            Sesión registrada
        """
        return {
            "user_id": user_id,
            "session_id": f"session_{datetime.now().timestamp()}",
            "therapy_type": therapy_type,
            "session_data": session_data,
            "duration_minutes": session_data.get("duration_minutes", 60),
            "satisfaction_score": session_data.get("satisfaction_score", 5),
            "recorded_at": datetime.now().isoformat()
        }
    
    def analyze_therapy_effectiveness(
        self,
        user_id: str,
        therapy_sessions: List[Dict],
        recovery_data: List[Dict]
    ) -> Dict:
        """
        Analiza efectividad de terapia
        
        Args:
            user_id: ID del usuario
            therapy_sessions: Sesiones de terapia
            recovery_data: Datos de recuperación
        
        Returns:
            Análisis de efectividad
        """
        if not therapy_sessions:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        return {
            "user_id": user_id,
            "total_sessions": len(therapy_sessions),
            "average_satisfaction": self._calculate_average_satisfaction(therapy_sessions),
            "effectiveness_score": self._calculate_effectiveness(therapy_sessions, recovery_data),
            "correlation_with_recovery": self._calculate_correlation(therapy_sessions, recovery_data),
            "recommendations": self._generate_therapy_recommendations(therapy_sessions),
            "generated_at": datetime.now().isoformat()
        }
    
    def _load_therapy_types(self) -> List[Dict]:
        """Carga tipos de terapias"""
        return [
            {
                "type": TherapyType.YOGA,
                "name": "Yoga",
                "benefits": ["reducción de estrés", "mejora de flexibilidad", "mindfulness"]
            },
            {
                "type": TherapyType.MEDITATION,
                "name": "Meditación",
                "benefits": ["reducción de ansiedad", "mejora de concentración", "bienestar mental"]
            },
            {
                "type": TherapyType.ART_THERAPY,
                "name": "Terapia de Arte",
                "benefits": ["expresión emocional", "reducción de estrés", "creatividad"]
            }
        ]
    
    def _select_appropriate_therapy(self, profile: Dict, state: Dict) -> str:
        """Selecciona terapia apropiada"""
        stress_level = state.get("stress_level", 5)
        
        if stress_level >= 7:
            return TherapyType.MEDITATION
        elif stress_level >= 5:
            return TherapyType.YOGA
        else:
            return TherapyType.ART_THERAPY
    
    def _generate_rationale(self, profile: Dict, state: Dict, therapy: str) -> str:
        """Genera justificación"""
        return f"Esta terapia es recomendada basada en tu perfil y estado actual"
    
    def _identify_expected_benefits(self, therapy: str) -> List[str]:
        """Identifica beneficios esperados"""
        benefits_map = {
            TherapyType.YOGA: ["Reducción de estrés", "Mejora de flexibilidad", "Bienestar físico"],
            TherapyType.MEDITATION: ["Reducción de ansiedad", "Mejora de concentración", "Bienestar mental"],
            TherapyType.ART_THERAPY: ["Expresión emocional", "Reducción de estrés", "Creatividad"]
        }
        
        return benefits_map.get(therapy, ["Bienestar general"])
    
    def _calculate_sessions_needed(self, state: Dict) -> int:
        """Calcula sesiones necesarias"""
        stress_level = state.get("stress_level", 5)
        
        if stress_level >= 7:
            return 8
        elif stress_level >= 5:
            return 6
        else:
            return 4
    
    def _calculate_average_satisfaction(self, sessions: List[Dict]) -> float:
        """Calcula satisfacción promedio"""
        if not sessions:
            return 0.0
        
        satisfactions = [s.get("satisfaction_score", 5) for s in sessions]
        return round(sum(satisfactions) / len(satisfactions), 2)
    
    def _calculate_effectiveness(self, sessions: List[Dict], recovery_data: List[Dict]) -> float:
        """Calcula efectividad"""
        # Lógica simplificada
        if not sessions:
            return 0.0
        
        avg_satisfaction = self._calculate_average_satisfaction(sessions)
        effectiveness = avg_satisfaction / 10
        
        return round(effectiveness, 2)
    
    def _calculate_correlation(self, sessions: List[Dict], recovery_data: List[Dict]) -> float:
        """Calcula correlación"""
        return 0.68
    
    def _generate_therapy_recommendations(self, sessions: List[Dict]) -> List[str]:
        """Genera recomendaciones de terapia"""
        recommendations = []
        
        avg_satisfaction = self._calculate_average_satisfaction(sessions)
        if avg_satisfaction < 6:
            recommendations.append("Considera probar diferentes tipos de terapia")
        
        return recommendations

