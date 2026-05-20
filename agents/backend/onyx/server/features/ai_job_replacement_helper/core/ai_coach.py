"""
AI Coach Service - Coach de IA avanzado
=========================================

Sistema de coaching inteligente con IA que adapta su estilo según el usuario.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CoachPersonality(str, Enum):
    """Personalidades del coach"""
    ENCOURAGING = "encouraging"
    DIRECT = "direct"
    ANALYTICAL = "analytical"
    SUPPORTIVE = "supportive"
    MOTIVATIONAL = "motivational"


@dataclass
class CoachMessage:
    """Mensaje del coach"""
    id: str
    message: str
    personality: CoachPersonality
    suggestions: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class UserProfile:
    """Perfil del usuario para el coach"""
    user_id: str
    personality_preference: CoachPersonality = CoachPersonality.SUPPORTIVE
    communication_style: str = "balanced"
    goals: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)


class AICoachService:
    """Servicio de coach de IA"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.user_profiles: Dict[str, UserProfile] = {}
        logger.info("AICoachService initialized")
    
    def get_daily_coaching(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CoachMessage:
        """Obtener coaching diario"""
        profile = self._get_or_create_profile(user_id)
        
        # Generar mensaje según personalidad
        message = self._generate_coaching_message(profile, context)
        
        return message
    
    def get_motivational_message(
        self,
        user_id: str,
        current_situation: str
    ) -> CoachMessage:
        """Obtener mensaje motivacional"""
        profile = self._get_or_create_profile(user_id)
        
        messages = {
            CoachPersonality.ENCOURAGING: f"¡Sigue adelante! Estás haciendo un gran progreso. {current_situation} es solo un paso más en tu viaje.",
            CoachPersonality.DIRECT: f"Enfócate en lo que puedes controlar. {current_situation} requiere acción, no solo pensamiento.",
            CoachPersonality.ANALYTICAL: f"Analicemos {current_situation}. ¿Qué datos tenemos? ¿Qué opciones son viables?",
            CoachPersonality.SUPPORTIVE: f"Entiendo que {current_situation} puede ser desafiante. Estoy aquí para apoyarte en cada paso.",
            CoachPersonality.MOTIVATIONAL: f"¡Tú puedes hacerlo! {current_situation} es una oportunidad para crecer y demostrar tu valía.",
        }
        
        message_text = messages.get(profile.personality_preference, messages[CoachPersonality.SUPPORTIVE])
        
        return CoachMessage(
            id=f"coach_msg_{user_id}_{int(datetime.now().timestamp())}",
            message=message_text,
            personality=profile.personality_preference,
            suggestions=[
                "Revisa tu progreso de hoy",
                "Identifica una acción pequeña que puedas tomar ahora",
                "Celebra tus logros, por pequeños que sean",
            ]
        )
    
    def analyze_progress_and_advise(
        self,
        user_id: str,
        progress_data: Dict[str, Any]
    ) -> CoachMessage:
        """Analizar progreso y dar consejos"""
        profile = self._get_or_create_profile(user_id)
        
        # Analizar datos
        steps_completed = progress_data.get("steps_completed", 0)
        applications_sent = progress_data.get("applications_sent", 0)
        skills_learned = progress_data.get("skills_learned", 0)
        
        # Generar análisis
        analysis = []
        if steps_completed < 3:
            analysis.append("Has completado pocos pasos. Te recomiendo acelerar el ritmo.")
        if applications_sent == 0:
            analysis.append("Aún no has enviado aplicaciones. Es hora de empezar a aplicar.")
        if skills_learned < 2:
            analysis.append("Considera aprender más habilidades para aumentar tus oportunidades.")
        
        message = "Basándome en tu progreso, aquí están mis observaciones:\n\n" + "\n".join(analysis)
        
        return CoachMessage(
            id=f"coach_analysis_{user_id}_{int(datetime.now().timestamp())}",
            message=message,
            personality=profile.personality_preference,
            action_items=[
                "Completa al menos 1 paso esta semana",
                "Envía 3 aplicaciones esta semana",
                "Aprende 1 nueva habilidad",
            ]
        )
    
    def _generate_coaching_message(
        self,
        profile: UserProfile,
        context: Optional[Dict[str, Any]]
    ) -> CoachMessage:
        """Generar mensaje de coaching"""
        # En producción, esto usaría un modelo de IA real
        default_message = "Hola, soy tu coach de IA. Estoy aquí para ayudarte en tu transición profesional."
        
        if context:
            if context.get("low_motivation"):
                default_message = "Veo que tu motivación está baja. Es normal. Vamos a trabajar juntos para recuperarla."
            elif context.get("recent_rejection"):
                default_message = "Las rechazos son parte del proceso. Cada 'no' te acerca más a un 'sí'."
        
        return CoachMessage(
            id=f"daily_coach_{profile.user_id}_{int(datetime.now().timestamp())}",
            message=default_message,
            personality=profile.personality_preference,
            suggestions=[
                "Revisa tus objetivos de hoy",
                "Identifica un pequeño paso que puedas dar",
            ]
        )
    
    def _get_or_create_profile(self, user_id: str) -> UserProfile:
        """Obtener o crear perfil"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        return self.user_profiles[user_id]




