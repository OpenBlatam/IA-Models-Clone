"""
Servicio de motivación - Proporciona mensajes motivacionales y celebra logros
"""

from typing import Dict, List, Optional
from datetime import datetime
from utils.helpers import get_milestone_message, format_time_sober


class MotivationService:
    """Servicio de motivación y celebración de logros"""
    
    def __init__(self):
        """Inicializa el servicio de motivación"""
        self.motivational_messages = self._load_messages()
        self.achievements = self._load_achievements()
    
    def get_motivational_messages(
        self,
        user_id: str,
        user_data: Dict
    ) -> Dict:
        """
        Obtiene mensajes motivacionales personalizados
        
        Args:
            user_id: ID del usuario
            user_data: Datos del usuario (días sobrio, etc.)
        
        Returns:
            Mensajes motivacionales
        """
        days_sober = user_data.get("days_sober", 0)
        
        messages = {
            "user_id": user_id,
            "daily_message": self._get_daily_message(days_sober),
            "milestone_message": get_milestone_message(days_sober),
            "time_sober": format_time_sober(days_sober),
            "encouragement": self._get_encouragement(days_sober),
            "tips": self._get_tips(days_sober),
            "generated_at": datetime.now().isoformat()
        }
        
        return messages
    
    def celebrate_milestone(
        self,
        user_id: str,
        milestone_days: int
    ) -> Dict:
        """
        Celebra un hito alcanzado
        
        Args:
            user_id: ID del usuario
            milestone_days: Días del hito alcanzado
        
        Returns:
            Celebración del hito
        """
        milestone_info = self._get_milestone_info(milestone_days)
        
        celebration = {
            "user_id": user_id,
            "milestone_days": milestone_days,
            "title": milestone_info["title"],
            "message": milestone_info["message"],
            "achievement_unlocked": True,
            "reward_suggestion": milestone_info["reward"],
            "next_milestone": self._get_next_milestone(milestone_days),
            "celebrated_at": datetime.now().isoformat()
        }
        
        return celebration
    
    def get_achievements(self, user_id: str) -> Dict:
        """
        Obtiene todos los logros del usuario
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Lista de logros
        """
        return {
            "user_id": user_id,
            "achievements": self.achievements,
            "total_achievements": len(self.achievements),
            "status": "success"
        }
    
    def _get_daily_message(self, days_sober: int) -> str:
        """Obtiene mensaje diario basado en días de sobriedad"""
        if days_sober == 0:
            return "Hoy es el primer día de tu nueva vida. ¡Tú puedes hacerlo!"
        elif days_sober < 7:
            return f"¡Llevas {days_sober} días! Cada día es una victoria. Sigue adelante."
        elif days_sober < 30:
            return f"¡Increíble! Has completado {days_sober} días. Estás construyendo un hábito poderoso."
        elif days_sober < 90:
            return f"¡{days_sober} días de sobriedad! Estás transformando tu vida día a día."
        else:
            return f"¡{format_time_sober(days_sober)} de sobriedad! Eres una inspiración."
    
    def _get_encouragement(self, days_sober: int) -> str:
        """Obtiene mensaje de aliento"""
        if days_sober < 7:
            return "Los primeros días son los más difíciles. Estás haciendo un trabajo increíble."
        elif days_sober < 30:
            return "Has superado la fase inicial. Tu determinación es admirable."
        elif days_sober < 90:
            return "Estás en un punto crucial. Sigue adelante, estás haciendo cambios reales."
        else:
            return "Has demostrado una fuerza increíble. La recuperación es un estilo de vida y lo estás logrando."
    
    def _get_tips(self, days_sober: int) -> List[str]:
        """Obtiene tips según días de sobriedad"""
        if days_sober < 7:
            return [
                "Mantén tu rutina ocupada",
                "Evita triggers conocidos",
                "Contacta tu apoyo diariamente",
                "Practica técnicas de relajación"
            ]
        elif days_sober < 30:
            return [
                "Continúa evitando situaciones de riesgo",
                "Mantén contacto regular con apoyo",
                "Celebra pequeños logros",
                "Desarrolla nuevos hobbies"
            ]
        else:
            return [
                "Mantén tu plan de recuperación activo",
                "Ayuda a otros en su recuperación",
                "Continúa aprendiendo sobre adicción",
                "Cuida tu salud física y mental"
            ]
    
    def _get_milestone_info(self, days: int) -> Dict:
        """Obtiene información del hito"""
        milestones = {
            1: {
                "title": "Primer Día",
                "message": "¡Felicitaciones por tu primer día! Este es el comienzo de algo grande.",
                "reward": "Reconocimiento personal y auto-reflexión"
            },
            7: {
                "title": "Primera Semana",
                "message": "¡Una semana completa! Has demostrado que puedes hacerlo.",
                "reward": "Actividad especial que disfrutes"
            },
            30: {
                "title": "Primer Mes",
                "message": "¡Un mes de sobriedad! Has alcanzado un hito significativo.",
                "reward": "Celebración significativa con seres queridos"
            },
            90: {
                "title": "Tres Meses",
                "message": "¡Tres meses! Has transformado tu vida de manera increíble.",
                "reward": "Recompensa importante que hayas estado esperando"
            },
            180: {
                "title": "Seis Meses",
                "message": "¡Seis meses de sobriedad! Eres una inspiración.",
                "reward": "Celebración grande y reconocimiento de tu logro"
            },
            365: {
                "title": "Un Año",
                "message": "¡UN AÑO COMPLETO! Has transformado tu vida. ¡Eres increíble!",
                "reward": "Celebración mayor y reflexión sobre tu viaje"
            }
        }
        
        return milestones.get(days, {
            "title": f"{days} Días",
            "message": f"¡Felicitaciones por {days} días de sobriedad!",
            "reward": "Reconocimiento y celebración"
        })
    
    def _get_next_milestone(self, current_days: int) -> Dict:
        """Obtiene próximo hito"""
        milestones = [1, 7, 30, 90, 180, 365]
        
        for milestone in milestones:
            if current_days < milestone:
                return {
                    "days": milestone,
                    "title": f"{milestone} días",
                    "days_remaining": milestone - current_days
                }
        
        return {
            "days": None,
            "title": "Has alcanzado todos los hitos principales",
            "days_remaining": None
        }
    
    def _load_messages(self) -> List[str]:
        """Carga mensajes motivacionales"""
        return [
            "Cada día es una nueva oportunidad",
            "Eres más fuerte de lo que crees",
            "La recuperación es posible",
            "Cada paso cuenta, no importa cuán pequeño",
            "Estás escribiendo una nueva historia"
        ]
    
    def _load_achievements(self) -> List[Dict]:
        """Carga logros disponibles"""
        return [
            {
                "id": "first_day",
                "title": "Primer Día",
                "description": "Completar el primer día de sobriedad",
                "days_required": 1
            },
            {
                "id": "first_week",
                "title": "Primera Semana",
                "description": "Completar 7 días consecutivos",
                "days_required": 7
            },
            {
                "id": "first_month",
                "title": "Primer Mes",
                "description": "Alcanzar 30 días de sobriedad",
                "days_required": 30
            },
            {
                "id": "three_months",
                "title": "Tres Meses",
                "description": "Completar 90 días de recuperación",
                "days_required": 90
            },
            {
                "id": "six_months",
                "title": "Seis Meses",
                "description": "Alcanzar 180 días de sobriedad",
                "days_required": 180
            },
            {
                "id": "one_year",
                "title": "Un Año",
                "description": "Completar 365 días de sobriedad",
                "days_required": 365
            }
        ]

