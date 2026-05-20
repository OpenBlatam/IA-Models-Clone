"""
AI Personality Service - Personalidades de IA
==============================================

Sistema para personalizar la personalidad de los asistentes de IA.
"""

import logging
from typing import Dict, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class AIPersonality(str, Enum):
    """Personalidades de IA"""
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    MOTIVATIONAL = "motivational"
    ANALYTICAL = "analytical"
    SUPPORTIVE = "supportive"
    DIRECT = "direct"


@dataclass
class PersonalityProfile:
    """Perfil de personalidad"""
    personality: AIPersonality
    tone: str
    greeting_style: str
    response_style: str
    emoji_usage: bool = True
    formality_level: str = "balanced"  # formal, casual, balanced


class AIPersonalityService:
    """Servicio de personalidades de IA"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.personalities = self._initialize_personalities()
        logger.info("AIPersonalityService initialized")
    
    def _initialize_personalities(self) -> Dict[AIPersonality, PersonalityProfile]:
        """Inicializar personalidades"""
        return {
            AIPersonality.FRIENDLY: PersonalityProfile(
                personality=AIPersonality.FRIENDLY,
                tone="warm and approachable",
                greeting_style="Hey! 👋",
                response_style="conversational and friendly",
                emoji_usage=True,
                formality_level="casual",
            ),
            AIPersonality.PROFESSIONAL: PersonalityProfile(
                personality=AIPersonality.PROFESSIONAL,
                tone="formal and business-like",
                greeting_style="Good day",
                response_style="structured and informative",
                emoji_usage=False,
                formality_level="formal",
            ),
            AIPersonality.MOTIVATIONAL: PersonalityProfile(
                personality=AIPersonality.MOTIVATIONAL,
                tone="energetic and inspiring",
                greeting_style="Let's do this! 💪",
                response_style="encouraging and action-oriented",
                emoji_usage=True,
                formality_level="casual",
            ),
            AIPersonality.ANALYTICAL: PersonalityProfile(
                personality=AIPersonality.ANALYTICAL,
                tone="data-driven and logical",
                greeting_style="Let's analyze",
                response_style="fact-based and detailed",
                emoji_usage=False,
                formality_level="formal",
            ),
            AIPersonality.SUPPORTIVE: PersonalityProfile(
                personality=AIPersonality.SUPPORTIVE,
                tone="empathetic and understanding",
                greeting_style="Hi there, how are you?",
                response_style="caring and supportive",
                emoji_usage=True,
                formality_level="balanced",
            ),
            AIPersonality.DIRECT: PersonalityProfile(
                personality=AIPersonality.DIRECT,
                tone="straightforward and no-nonsense",
                greeting_style="Hello",
                response_style="concise and to the point",
                emoji_usage=False,
                formality_level="balanced",
            ),
        }
    
    def get_personality_profile(self, personality: AIPersonality) -> PersonalityProfile:
        """Obtener perfil de personalidad"""
        return self.personalities.get(personality, self.personalities[AIPersonality.SUPPORTIVE])
    
    def generate_message(
        self,
        personality: AIPersonality,
        base_message: str
    ) -> str:
        """Generar mensaje con personalidad"""
        profile = self.get_personality_profile(personality)
        
        # Ajustar mensaje según personalidad
        if personality == AIPersonality.MOTIVATIONAL:
            return f"💪 {base_message} ¡Tú puedes hacerlo!"
        elif personality == AIPersonality.PROFESSIONAL:
            return base_message  # Sin modificaciones
        elif personality == AIPersonality.FRIENDLY:
            return f"👋 {base_message} ¡Estoy aquí para ayudarte!"
        elif personality == AIPersonality.ANALYTICAL:
            return f"📊 Análisis: {base_message}"
        elif personality == AIPersonality.SUPPORTIVE:
            return f"💙 {base_message} Estoy aquí para apoyarte."
        else:
            return base_message




