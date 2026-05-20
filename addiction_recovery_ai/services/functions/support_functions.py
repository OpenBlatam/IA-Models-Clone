"""
Pure functions for support and motivation logic
Functional programming approach - no classes
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import random


def generate_motivational_message(
    days_sober: int,
    milestone_days: Optional[int] = None
) -> str:
    """Generate personalized motivational message"""
    messages = [
        f"¡Increíble! Has estado sobrio por {days_sober} días. Cada día es una victoria.",
        f"Llevas {days_sober} días de progreso. Sigue adelante, estás haciendo un gran trabajo.",
        f"Has llegado a {days_sober} días. Tu determinación es inspiradora.",
        f"{days_sober} días de libertad. Cada día te hace más fuerte.",
        f"Estás en el día {days_sober} de tu recuperación. ¡Sigue así!"
    ]
    
    if milestone_days and days_sober >= milestone_days:
        milestone_messages = [
            f"¡FELICITACIONES! Has alcanzado {milestone_days} días. Este es un hito increíble.",
            f"¡LO LOGASTE! {milestone_days} días de sobriedad. Eres un ejemplo de fuerza.",
            f"¡CELEBRA! Has llegado a {milestone_days} días. Este logro es tuyo."
        ]
        messages.extend(milestone_messages)
    
    return random.choice(messages)


def calculate_milestone_progress(
    days_sober: int,
    milestones: List[int] = [7, 30, 60, 90, 180, 365]
) -> Dict[str, Any]:
    """Calculate progress towards next milestone"""
    milestones.sort()
    
    current_milestone = None
    next_milestone = None
    progress_to_next = 0.0
    
    for milestone in milestones:
        if days_sober >= milestone:
            current_milestone = milestone
        else:
            next_milestone = milestone
            if current_milestone:
                progress_to_next = (days_sober - current_milestone) / (milestone - current_milestone) * 100
            else:
                progress_to_next = (days_sober / milestone) * 100
            break
    
    if not next_milestone:
        # Past all milestones
        next_milestone = milestones[-1] + 365
        progress_to_next = ((days_sober - milestones[-1]) / 365) * 100
    
    return {
        "current_milestone": current_milestone,
        "next_milestone": next_milestone,
        "progress_to_next": min(100.0, max(0.0, progress_to_next)),
        "days_until_next": next_milestone - days_sober if next_milestone else None
    }


def generate_coaching_guidance(
    topic: str,
    situation: str
) -> str:
    """Generate coaching guidance based on topic"""
    guidance_map = {
        "cravings": "Los cravings son temporales. La mayoría pasan en 15-20 minutos. Usa técnicas de distracción y respiración profunda.",
        "relapse": "Si has tenido una recaída, no es el fin del mundo. Aprende de la experiencia y continúa. La recuperación es un proceso.",
        "motivation": "Recuerda tus razones para estar sobrio. Escribe una lista y léela cuando sientas que pierdes motivación.",
        "stress": "El estrés es un trigger común. Desarrolla técnicas de manejo de estrés como ejercicio, meditación o hablar con alguien.",
        "social": "Las situaciones sociales pueden ser desafiantes. Ten un plan antes de ir, lleva apoyo si es posible, y ten una salida preparada."
    }
    
    topic_lower = topic.lower()
    for key, guidance in guidance_map.items():
        if key in topic_lower:
            return guidance
    
    return "Continúa trabajando en tu recuperación. Cada día es una oportunidad de crecimiento."


def generate_action_items(
    topic: str,
    priority: str = "medium"
) -> List[Dict[str, str]]:
    """Generate action items based on topic"""
    base_actions = [
        {
            "action": "Contactar sistema de apoyo",
            "priority": "high",
            "description": "Llama o contacta a alguien de tu red de apoyo"
        },
        {
            "action": "Practicar técnica de relajación",
            "priority": "medium",
            "description": "Usa respiración profunda o meditación"
        },
        {
            "action": "Revisar plan de recuperación",
            "priority": "medium",
            "description": "Revisa y ajusta tu plan si es necesario"
        }
    ]
    
    if topic.lower() in ["cravings", "relapse"]:
        base_actions.insert(0, {
            "action": "Usar técnicas de distracción",
            "priority": "high",
            "description": "Distráete con una actividad que disfrutes"
        })
    
    return base_actions


def create_coaching_session_data(
    user_id: str,
    topic: str,
    situation: str,
    guidance: str,
    action_items: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Create coaching session data structure"""
    return {
        "user_id": user_id,
        "session_id": f"coaching_{datetime.now().timestamp()}",
        "topic": topic,
        "guidance": guidance,
        "action_items": action_items,
        "created_at": datetime.now().isoformat()
    }

