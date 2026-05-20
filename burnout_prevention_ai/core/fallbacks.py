"""
Fallback Data for Responses
===========================
Centralized fallback data structures for JSON parsing.

Provides default fallback values when AI responses cannot be parsed
or when API calls fail, ensuring the application always returns
valid responses to users.
"""

from typing import Dict, Any


def get_assessment_fallback(content: str = "") -> Dict[str, Any]:
    """
    Get fallback data for burnout assessment.
    
    Provides sensible defaults when API response cannot be parsed.
    These defaults ensure users still receive helpful information.
    
    Args:
        content: Optional content string (unused but kept for API compatibility)
        
    Returns:
        Dictionary with fallback assessment data
    """
    return {
        "burnout_risk_level": "medium",
        "burnout_score": 50.0,
        "risk_factors": [
            "No se pudo analizar completamente. Por favor, intenta nuevamente."
        ],
        "recommendations": [
            "Mantén un registro de tus síntomas y niveles de estrés",
            "Considera hablar con un profesional de salud mental",
            "Practica técnicas de relajación y mindfulness",
            "Asegúrate de tener tiempo de descanso adecuado",
            "Establece límites claros entre trabajo y vida personal"
        ],
        "immediate_actions": [
            "Tómate un descanso de 10-15 minutos",
            "Respira profundamente y practica relajación",
            "Hidrátate y come algo nutritivo"
        ],
        "long_term_strategies": [
            "Establece una rutina de sueño consistente",
            "Incorpora ejercicio regular en tu rutina",
            "Desarrolla un sistema de apoyo social"
        ]
    }


def get_wellness_fallback(content: str = "") -> Dict[str, Any]:
    """
    Get fallback data for wellness check.
    
    Args:
        content: Optional content string to extract partial analysis from
        
    Returns:
        Dictionary with fallback wellness data
    """
    return {
        "wellness_score": 50.0,
        "mood_analysis": content[:200] if content else "Análisis no disponible. Por favor, intenta nuevamente.",
        "support_recommendations": [
            "Habla con amigos o familiares sobre cómo te sientes",
            "Considera unirse a un grupo de apoyo",
            "Busca ayuda profesional si es necesario"
        ],
        "self_care_suggestions": [
            "Practica meditación o mindfulness diariamente",
            "Mantén una rutina de ejercicio regular",
            "Asegúrate de dormir lo suficiente",
            "Dedica tiempo a actividades que disfrutas"
        ]
    }


def get_coping_fallback() -> Dict[str, Any]:
    """
    Get fallback data for coping strategies.
    
    Returns:
        Dictionary with fallback coping strategy data
    """
    return {
        "strategies": [
            {
                "name": "Respiración profunda",
                "description": "Técnica de respiración para reducir el estrés inmediato",
                "effectiveness": "alta",
                "time_required": "5-10 minutos",
                "difficulty": "fácil"
            },
            {
                "name": "Ejercicio físico",
                "description": "Actividad física para liberar endorfinas y reducir estrés",
                "effectiveness": "alta",
                "time_required": "30 minutos",
                "difficulty": "media"
            }
        ],
        "implementation_plan": [
            "Identifica el tipo de estresor que enfrentas",
            "Elige una estrategia que se adapte a tu situación",
            "Practica la estrategia regularmente",
            "Evalúa su efectividad después de una semana",
            "Ajusta según sea necesario"
        ],
        "resources": [
            "Aplicaciones de meditación (Headspace, Calm)",
            "Libros sobre manejo del estrés",
            "Videos de YouTube sobre técnicas de relajación"
        ]
    }


def get_progress_fallback(content: str = "") -> Dict[str, Any]:
    """
    Get fallback data for progress tracking.
    
    Args:
        content: Optional content string to extract partial insights from
        
    Returns:
        Dictionary with fallback progress data
    """
    return {
        "progress_score": 50.0,
        "trend": "stable",
        "milestones_achieved": [
            "Sistema de seguimiento establecido"
        ],
        "next_steps": [
            "Continúa registrando tus evaluaciones regularmente",
            "Revisa tus metas y ajusta según sea necesario",
            "Celebra los pequeños logros en el camino"
        ],
        "insights": content[:300] if content else "Análisis no disponible. Por favor, intenta nuevamente."
    }


def get_trend_fallback() -> Dict[str, Any]:
    """
    Get fallback data for trend analysis.
    
    Returns:
        Dictionary with fallback trend analysis data
    """
    return {
        "overall_trend": "stable",
        "key_metrics": {
            "burnout_score_avg": 50.0,
            "stress_level_trend": "stable"
        },
        "patterns": [
            "Se necesitan más datos para identificar patrones claros"
        ],
        "predictions": {
            "next_week": "Continúa monitoreando tus niveles de estrés",
            "next_month": "Mantén un registro consistente para mejores predicciones"
        },
        "recommendations": [
            "Registra evaluaciones regularmente para mejor análisis",
            "Revisa tus tendencias semanalmente",
            "Ajusta tus estrategias basándote en los datos recopilados"
        ]
    }


def get_resource_fallback() -> Dict[str, Any]:
    """
    Get fallback data for resources.
    
    Returns:
        Dictionary with fallback resource data
    """
    return {
        "resources": [
            {
                "title": "Guía de Prevención de Burnout",
                "type": "article",
                "description": "Información general sobre burnout y prevención",
                "url": "",
                "duration": "10 min lectura"
            }
        ],
        "learning_path": [
            "Comprende qué es el burnout",
            "Identifica los signos tempranos",
            "Aprende estrategias de prevención",
            "Implementa cambios en tu rutina"
        ],
        "key_concepts": [
            "Burnout es un estado de agotamiento físico y mental",
            "La prevención temprana es clave",
            "El autocuidado es fundamental",
            "El apoyo social ayuda significativamente",
            "Los cambios pequeños pueden tener gran impacto"
        ],
        "action_items": [
            "Investiga más sobre burnout",
            "Habla con un profesional si es necesario",
            "Únete a comunidades de apoyo"
        ]
    }


def get_plan_fallback() -> Dict[str, Any]:
    """
    Get fallback data for personalized plan.
    
    Returns:
        Dictionary with fallback personalized plan data
    """
    return {
        "plan_name": "Plan Personalizado de Prevención",
        "duration_weeks": 8,
        "weekly_goals": [
            {
                "week": 1,
                "goal": "Establecer rutina de seguimiento",
                "actions": ["Registra tus niveles de estrés diariamente", "Identifica tus principales estresores"],
                "focus_area": "Autoconocimiento"
            }
        ],
        "daily_actions": [
            "Practica 10 minutos de meditación o respiración profunda",
            "Tómate descansos regulares durante el trabajo",
            "Hidrátate adecuadamente",
            "Realiza al menos 20 minutos de actividad física",
            "Mantén un horario de sueño consistente"
        ],
        "milestones": [
            "Semana 2: Establecer rutina de autocuidado",
            "Semana 4: Implementar estrategias de manejo de estrés",
            "Semana 6: Evaluar progreso y ajustar",
            "Semana 8: Consolidar hábitos saludables"
        ],
        "resources": [
            "Aplicaciones de meditación",
            "Libros sobre manejo del estrés",
            "Recursos de salud mental"
        ]
    }

