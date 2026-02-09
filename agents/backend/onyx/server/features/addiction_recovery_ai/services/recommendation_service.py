"""
Servicio de Recomendaciones Personalizadas - Sistema inteligente de recomendaciones
"""

from typing import Dict, List, Optional
from datetime import datetime


class RecommendationService:
    """Servicio de recomendaciones personalizadas"""
    
    def __init__(self):
        """Inicializa el servicio de recomendaciones"""
        self.recommendation_engine = self._initialize_engine()
    
    def get_personalized_recommendations(
        self,
        user_id: str,
        context: Dict
    ) -> Dict:
        """
        Obtiene recomendaciones personalizadas
        
        Args:
            user_id: ID del usuario
            context: Contexto actual del usuario
        
        Returns:
            Recomendaciones personalizadas
        """
        recommendations = []
        
        # Analizar contexto
        days_sober = context.get("days_sober", 0)
        current_mood = context.get("mood", "neutral")
        stress_level = context.get("stress_level", 5)
        cravings_level = context.get("cravings_level", 3)
        
        # Recomendaciones basadas en días sobrio
        if days_sober < 7:
            recommendations.extend(self._get_early_stage_recommendations())
        elif days_sober < 30:
            recommendations.extend(self._get_early_recovery_recommendations())
        elif days_sober < 90:
            recommendations.extend(self._get_mid_recovery_recommendations())
        else:
            recommendations.extend(self._get_long_term_recovery_recommendations())
        
        # Recomendaciones basadas en estado de ánimo
        if current_mood in ["sad", "anxious", "stressed"]:
            recommendations.extend(self._get_mood_based_recommendations(current_mood))
        
        # Recomendaciones basadas en nivel de estrés
        if stress_level >= 7:
            recommendations.extend(self._get_stress_reduction_recommendations())
        
        # Recomendaciones basadas en cravings
        if cravings_level >= 6:
            recommendations.extend(self._get_craving_management_recommendations())
        
        return {
            "user_id": user_id,
            "recommendations": recommendations[:10],  # Top 10
            "total": len(recommendations),
            "generated_at": datetime.now().isoformat(),
            "context_used": list(context.keys())
        }
    
    def get_resource_recommendations(
        self,
        user_id: str,
        resource_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtiene recomendaciones de recursos
        
        Args:
            user_id: ID del usuario
            resource_type: Tipo de recurso (opcional)
        
        Returns:
            Lista de recursos recomendados
        """
        resources = [
            {
                "id": "resource_1",
                "type": "article",
                "title": "Técnicas de Manejo de Cravings",
                "description": "Guía completa sobre cómo manejar los deseos intensos",
                "url": "/resources/cravings-management",
                "relevance_score": 0.9
            },
            {
                "id": "resource_2",
                "type": "video",
                "title": "Meditación para la Recuperación",
                "description": "Sesión guiada de meditación de 10 minutos",
                "url": "/resources/meditation-recovery",
                "relevance_score": 0.85
            },
            {
                "id": "resource_3",
                "type": "exercise",
                "title": "Ejercicios de Respiración",
                "description": "Técnicas de respiración para reducir ansiedad",
                "url": "/resources/breathing-exercises",
                "relevance_score": 0.8
            }
        ]
        
        if resource_type:
            resources = [r for r in resources if r.get("type") == resource_type]
        
        return resources
    
    def get_activity_recommendations(
        self,
        user_id: str,
        time_of_day: str,
        available_time: int
    ) -> List[Dict]:
        """
        Obtiene recomendaciones de actividades
        
        Args:
            user_id: ID del usuario
            time_of_day: Momento del día (morning, afternoon, evening, night)
            available_time: Tiempo disponible en minutos
        
        Returns:
            Lista de actividades recomendadas
        """
        activities = []
        
        if time_of_day == "morning":
            activities = [
                {
                    "id": "activity_1",
                    "name": "Meditación Matutina",
                    "duration": 10,
                    "type": "mindfulness",
                    "benefits": ["reduce_anxiety", "improve_focus"]
                },
                {
                    "id": "activity_2",
                    "name": "Ejercicio Ligero",
                    "duration": 20,
                    "type": "exercise",
                    "benefits": ["boost_energy", "improve_mood"]
                }
            ]
        elif time_of_day == "evening":
            activities = [
                {
                    "id": "activity_3",
                    "name": "Journaling",
                    "duration": 15,
                    "type": "reflection",
                    "benefits": ["process_emotions", "track_progress"]
                },
                {
                    "id": "activity_4",
                    "name": "Lectura Relajante",
                    "duration": 30,
                    "type": "relaxation",
                    "benefits": ["reduce_stress", "improve_sleep"]
                }
            ]
        
        # Filtrar por tiempo disponible
        activities = [a for a in activities if a.get("duration", 0) <= available_time]
        
        return activities
    
    def _initialize_engine(self) -> Dict:
        """Inicializa motor de recomendaciones"""
        return {
            "version": "1.0",
            "algorithms": ["collaborative", "content_based", "contextual"]
        }
    
    def _get_early_stage_recommendations(self) -> List[Dict]:
        """Recomendaciones para etapa temprana"""
        return [
            {
                "type": "support",
                "title": "Conecta con tu Sistema de Apoyo",
                "description": "Establece contacto diario con personas de apoyo",
                "priority": "high"
            },
            {
                "type": "education",
                "title": "Aprende sobre Síntomas de Abstinencia",
                "description": "Entender qué esperar puede reducir la ansiedad",
                "priority": "medium"
            }
        ]
    
    def _get_early_recovery_recommendations(self) -> List[Dict]:
        """Recomendaciones para recuperación temprana"""
        return [
            {
                "type": "routine",
                "title": "Establece una Rutina Diaria",
                "description": "Las rutinas ayudan a mantener el enfoque",
                "priority": "high"
            },
            {
                "type": "exercise",
                "title": "Incorpora Ejercicio Regular",
                "description": "El ejercicio mejora el estado de ánimo y reduce cravings",
                "priority": "medium"
            }
        ]
    
    def _get_mid_recovery_recommendations(self) -> List[Dict]:
        """Recomendaciones para recuperación media"""
        return [
            {
                "type": "growth",
                "title": "Explora Nuevos Intereses",
                "description": "Descubre actividades que te apasionen",
                "priority": "medium"
            },
            {
                "type": "community",
                "title": "Participa en Grupos de Apoyo",
                "description": "Conectar con otros en recuperación fortalece tu viaje",
                "priority": "medium"
            }
        ]
    
    def _get_long_term_recovery_recommendations(self) -> List[Dict]:
        """Recomendaciones para recuperación a largo plazo"""
        return [
            {
                "type": "mentorship",
                "title": "Considera Convertirte en Mentor",
                "description": "Ayudar a otros puede fortalecer tu propia recuperación",
                "priority": "low"
            },
            {
                "type": "maintenance",
                "title": "Mantén tus Estrategias de Prevención",
                "description": "Continúa practicando las técnicas que te han funcionado",
                "priority": "medium"
            }
        ]
    
    def _get_mood_based_recommendations(self, mood: str) -> List[Dict]:
        """Recomendaciones basadas en estado de ánimo"""
        if mood == "sad":
            return [
                {
                    "type": "support",
                    "title": "Contacta a tu Sistema de Apoyo",
                    "description": "No estás solo, hay personas que quieren ayudarte",
                    "priority": "high"
                }
            ]
        elif mood == "anxious":
            return [
                {
                    "type": "breathing",
                    "title": "Practica Respiración Profunda",
                    "description": "Técnicas de respiración pueden reducir la ansiedad",
                    "priority": "high"
                }
            ]
        return []
    
    def _get_stress_reduction_recommendations(self) -> List[Dict]:
        """Recomendaciones para reducir estrés"""
        return [
            {
                "type": "mindfulness",
                "title": "Sesión de Mindfulness",
                "description": "La meditación puede reducir significativamente el estrés",
                "priority": "high"
            },
            {
                "type": "exercise",
                "title": "Ejercicio Físico",
                "description": "El ejercicio libera endorfinas que reducen el estrés",
                "priority": "medium"
            }
        ]
    
    def _get_craving_management_recommendations(self) -> List[Dict]:
        """Recomendaciones para manejo de cravings"""
        return [
            {
                "type": "distraction",
                "title": "Técnica de Distracción",
                "description": "Involúcrate en una actividad que disfrutes",
                "priority": "high"
            },
            {
                "type": "delay",
                "title": "Técnica de Retraso",
                "description": "Espera 15 minutos antes de tomar cualquier decisión",
                "priority": "high"
            }
        ]

