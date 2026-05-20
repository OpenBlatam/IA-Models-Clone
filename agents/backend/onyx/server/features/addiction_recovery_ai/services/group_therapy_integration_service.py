"""
Servicio de Integración con Terapias Grupales - Sistema completo de terapias grupales
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import statistics


class GroupType(str, Enum):
    """Tipos de grupos"""
    SUPPORT_GROUP = "support_group"
    THERAPY_GROUP = "therapy_group"
    PEER_GROUP = "peer_group"
    FAMILY_GROUP = "family_group"
    SPECIALIZED_GROUP = "specialized_group"


class GroupTherapyIntegrationService:
    """Servicio de integración con terapias grupales"""
    
    def __init__(self):
        """Inicializa el servicio de terapias grupales"""
        pass
    
    def find_suitable_groups(
        self,
        user_id: str,
        user_profile: Dict,
        preferences: Dict
    ) -> Dict:
        """
        Encuentra grupos adecuados
        
        Args:
            user_id: ID del usuario
            user_profile: Perfil del usuario
            preferences: Preferencias
        
        Returns:
            Grupos recomendados
        """
        return {
            "user_id": user_id,
            "recommended_groups": self._recommend_groups(user_profile, preferences),
            "group_matching_score": self._calculate_matching_score(user_profile, preferences),
            "recommendations": self._generate_group_recommendations(user_profile),
            "generated_at": datetime.now().isoformat()
        }
    
    def join_group(
        self,
        user_id: str,
        group_id: str,
        join_data: Dict
    ) -> Dict:
        """
        Une usuario a grupo
        
        Args:
            user_id: ID del usuario
            group_id: ID del grupo
            join_data: Datos de unión
        
        Returns:
            Unión al grupo
        """
        return {
            "user_id": user_id,
            "group_id": group_id,
            "join_id": f"join_{datetime.now().timestamp()}",
            "join_data": join_data,
            "joined_at": datetime.now().isoformat(),
            "status": "active"
        }
    
    def track_group_participation(
        self,
        user_id: str,
        group_id: str,
        sessions: List[Dict]
    ) -> Dict:
        """
        Rastrea participación en grupo
        
        Args:
            user_id: ID del usuario
            group_id: ID del grupo
            sessions: Lista de sesiones
        
        Returns:
            Análisis de participación
        """
        return {
            "user_id": user_id,
            "group_id": group_id,
            "total_sessions": len(sessions),
            "attendance_rate": self._calculate_attendance_rate(sessions),
            "participation_level": self._assess_participation_level(sessions),
            "benefits_received": self._identify_benefits(sessions),
            "recommendations": self._generate_participation_recommendations(sessions),
            "generated_at": datetime.now().isoformat()
        }
    
    def _recommend_groups(self, profile: Dict, preferences: Dict) -> List[Dict]:
        """Recomienda grupos"""
        groups = []
        
        groups.append({
            "group_id": "group_1",
            "name": "Grupo de Apoyo Semanal",
            "type": GroupType.SUPPORT_GROUP,
            "description": "Grupo de apoyo para personas en recuperación",
            "meeting_frequency": "weekly",
            "match_score": 0.85
        })
        
        return groups
    
    def _calculate_matching_score(self, profile: Dict, preferences: Dict) -> float:
        """Calcula puntuación de coincidencia"""
        return 0.85
    
    def _generate_group_recommendations(self, profile: Dict) -> List[str]:
        """Genera recomendaciones de grupos"""
        recommendations = []
        
        recommendations.append("Unirse a grupos de apoyo puede fortalecer significativamente tu recuperación")
        
        return recommendations
    
    def _calculate_attendance_rate(self, sessions: List[Dict]) -> float:
        """Calcula tasa de asistencia"""
        if not sessions:
            return 0.0
        
        attended = sum(1 for s in sessions if s.get("attended", False))
        return round((attended / len(sessions)) * 100, 2)
    
    def _assess_participation_level(self, sessions: List[Dict]) -> str:
        """Evalúa nivel de participación"""
        if not sessions:
            return "none"
        
        participation_scores = [s.get("participation_score", 5) for s in sessions]
        avg_score = statistics.mean(participation_scores) if participation_scores else 0
        
        if avg_score >= 7:
            return "high"
        elif avg_score >= 5:
            return "moderate"
        else:
            return "low"
    
    def _identify_benefits(self, sessions: List[Dict]) -> List[str]:
        """Identifica beneficios"""
        benefits = []
        
        if len(sessions) >= 5:
            benefits.append("Mejora en habilidades de afrontamiento")
            benefits.append("Aumento en apoyo social")
        
        return benefits
    
    def _generate_participation_recommendations(self, sessions: List[Dict]) -> List[str]:
        """Genera recomendaciones de participación"""
        recommendations = []
        
        attendance_rate = self._calculate_attendance_rate(sessions)
        if attendance_rate < 70:
            recommendations.append("Aumenta tu asistencia a sesiones de grupo para mejores resultados")
        
        return recommendations

