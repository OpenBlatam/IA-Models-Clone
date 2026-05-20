"""
Servicio de Logros Avanzado - Sistema completo de logros
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class AchievementType(str, Enum):
    """Tipos de logros"""
    MILESTONE = "milestone"
    STREAK = "streak"
    CONSISTENCY = "consistency"
    IMPROVEMENT = "improvement"
    COMMUNITY = "community"
    PERSONAL_GROWTH = "personal_growth"


class AdvancedAchievementsService:
    """Servicio de logros avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de logros"""
        pass
    
    def award_achievement(
        self,
        user_id: str,
        achievement_data: Dict
    ) -> Dict:
        """
        Otorga logro
        
        Args:
            user_id: ID del usuario
            achievement_data: Datos del logro
        
        Returns:
            Logro otorgado
        """
        achievement = {
            "id": f"achievement_{datetime.now().timestamp()}",
            "user_id": user_id,
            "achievement_data": achievement_data,
            "achievement_type": achievement_data.get("achievement_type", AchievementType.MILESTONE),
            "title": achievement_data.get("title", ""),
            "description": achievement_data.get("description", ""),
            "rarity": achievement_data.get("rarity", "common"),
            "points": achievement_data.get("points", 100),
            "awarded_at": datetime.now().isoformat(),
            "status": "awarded"
        }
        
        return achievement
    
    def check_achievement_eligibility(
        self,
        user_id: str,
        achievement_criteria: Dict,
        user_data: Dict
    ) -> Dict:
        """
        Verifica elegibilidad para logro
        
        Args:
            user_id: ID del usuario
            achievement_criteria: Criterios del logro
            user_data: Datos del usuario
        
        Returns:
            Verificación de elegibilidad
        """
        is_eligible = self._check_eligibility(achievement_criteria, user_data)
        
        return {
            "user_id": user_id,
            "is_eligible": is_eligible,
            "progress": self._calculate_progress(achievement_criteria, user_data),
            "remaining_requirements": self._identify_remaining(achievement_criteria, user_data) if not is_eligible else [],
            "checked_at": datetime.now().isoformat()
        }
    
    def analyze_achievement_impact(
        self,
        user_id: str,
        achievements: List[Dict],
        recovery_data: List[Dict]
    ) -> Dict:
        """
        Analiza impacto de logros
        
        Args:
            user_id: ID del usuario
            achievements: Lista de logros
            recovery_data: Datos de recuperación
        
        Returns:
            Análisis de impacto
        """
        return {
            "user_id": user_id,
            "total_achievements": len(achievements),
            "total_points": sum(a.get("points", 0) for a in achievements),
            "achievement_categories": self._categorize_achievements(achievements),
            "motivation_impact": self._calculate_motivation_impact(achievements, recovery_data),
            "recommendations": self._generate_achievement_recommendations(achievements),
            "generated_at": datetime.now().isoformat()
        }
    
    def _check_eligibility(self, criteria: Dict, user_data: Dict) -> bool:
        """Verifica elegibilidad"""
        # Lógica simplificada
        days_sober = user_data.get("days_sober", 0)
        required_days = criteria.get("required_days", 0)
        
        return days_sober >= required_days
    
    def _calculate_progress(self, criteria: Dict, user_data: Dict) -> float:
        """Calcula progreso"""
        days_sober = user_data.get("days_sober", 0)
        required_days = criteria.get("required_days", 100)
        
        if required_days > 0:
            return round(min(1.0, days_sober / required_days), 2)
        
        return 0.0
    
    def _identify_remaining(self, criteria: Dict, user_data: Dict) -> List[str]:
        """Identifica requisitos restantes"""
        remaining = []
        
        days_sober = user_data.get("days_sober", 0)
        required_days = criteria.get("required_days", 100)
        
        if days_sober < required_days:
            remaining.append(f"Faltan {required_days - days_sober} días de sobriedad")
        
        return remaining
    
    def _categorize_achievements(self, achievements: List[Dict]) -> Dict:
        """Categoriza logros"""
        categories = {}
        
        for achievement in achievements:
            achievement_type = achievement.get("achievement_type", "unknown")
            categories[achievement_type] = categories.get(achievement_type, 0) + 1
        
        return categories
    
    def _calculate_motivation_impact(self, achievements: List[Dict], recovery_data: List[Dict]) -> float:
        """Calcula impacto en motivación"""
        # Lógica simplificada
        if not achievements:
            return 0.0
        
        total_points = sum(a.get("points", 0) for a in achievements)
        if total_points > 1000:
            return 0.8
        elif total_points > 500:
            return 0.6
        else:
            return 0.4
    
    def _generate_achievement_recommendations(self, achievements: List[Dict]) -> List[str]:
        """Genera recomendaciones de logros"""
        recommendations = []
        
        if len(achievements) < 5:
            recommendations.append("Continúa trabajando en tus objetivos para desbloquear más logros")
        
        return recommendations

