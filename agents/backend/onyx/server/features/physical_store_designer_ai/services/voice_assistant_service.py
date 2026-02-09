"""
Voice Assistant Service - Integración con asistentes de voz
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class VoicePlatform(str, Enum):
    """Plataformas de voz"""
    ALEXA = "alexa"
    GOOGLE_ASSISTANT = "google_assistant"
    SIRI = "siri"
    CUSTOM = "custom"


class VoiceAssistantService:
    """Servicio para asistentes de voz"""
    
    def __init__(self):
        self.skills: Dict[str, Dict[str, Any]] = {}
        self.interactions: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_voice_skill(
        self,
        store_id: str,
        skill_name: str,
        platform: VoicePlatform,
        intents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Crear skill de voz"""
        
        skill_id = f"skill_{store_id}_{platform.value}_{len(self.skills.get(store_id, [])) + 1}"
        
        skill = {
            "skill_id": skill_id,
            "store_id": store_id,
            "name": skill_name,
            "platform": platform.value,
            "intents": intents,
            "is_active": True,
            "created_at": datetime.now().isoformat(),
            "invocation_name": skill_name.lower().replace(" ", "")
        }
        
        if store_id not in self.skills:
            self.skills[store_id] = []
        
        self.skills[store_id].append(skill)
        
        return skill
    
    def process_voice_command(
        self,
        skill_id: str,
        command: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Procesar comando de voz"""
        
        skill = self._find_skill(skill_id)
        
        if not skill:
            raise ValueError(f"Skill {skill_id} no encontrado")
        
        # Procesar comando (en producción, usar NLP)
        intent = self._detect_intent(command, skill["intents"])
        
        response = self._generate_response(intent, command)
        
        interaction = {
            "interaction_id": f"int_{skill_id}_{len(self.interactions.get(skill_id, [])) + 1}",
            "skill_id": skill_id,
            "user_id": user_id,
            "command": command,
            "intent": intent,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        if skill_id not in self.interactions:
            self.interactions[skill_id] = []
        
        self.interactions[skill_id].append(interaction)
        
        return interaction
    
    def _find_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Encontrar skill"""
        for store_skills in self.skills.values():
            for skill in store_skills:
                if skill["skill_id"] == skill_id:
                    return skill
        return None
    
    def _detect_intent(
        self,
        command: str,
        intents: List[Dict[str, Any]]
    ) -> str:
        """Detectar intent del comando"""
        command_lower = command.lower()
        
        # Búsqueda simple de palabras clave
        for intent in intents:
            keywords = intent.get("keywords", [])
            if any(keyword in command_lower for keyword in keywords):
                return intent.get("name", "unknown")
        
        return "unknown"
    
    def _generate_response(
        self,
        intent: str,
        command: str
    ) -> str:
        """Generar respuesta"""
        responses = {
            "store_hours": "Nuestra tienda está abierta de 9 AM a 8 PM de lunes a sábado.",
            "location": "Estamos ubicados en [dirección].",
            "products": "Tenemos una amplia variedad de productos disponibles.",
            "unknown": "Lo siento, no entendí tu pregunta. ¿Puedes repetirla?"
        }
        
        return responses.get(intent, responses["unknown"])
    
    def get_voice_analytics(
        self,
        skill_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Obtener analytics de voz"""
        
        interactions = self.interactions.get(skill_id, [])
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        recent_interactions = [
            i for i in interactions
            if start_date <= datetime.fromisoformat(i["timestamp"]) <= end_date
        ]
        
        intent_counts = {}
        for interaction in recent_interactions:
            intent = interaction["intent"]
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            "skill_id": skill_id,
            "period_days": days,
            "total_interactions": len(recent_interactions),
            "intent_distribution": intent_counts,
            "most_common_intent": max(intent_counts.items(), key=lambda x: x[1])[0] if intent_counts else None
        }

