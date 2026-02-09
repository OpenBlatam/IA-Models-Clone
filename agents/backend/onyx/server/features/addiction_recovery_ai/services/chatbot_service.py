"""
Servicio de Chatbot de IA - Asistente conversacional 24/7
"""

from typing import Dict, List, Optional
from datetime import datetime
import json


class ChatbotService:
    """Servicio de chatbot conversacional con IA"""
    
    def __init__(self, openai_client=None):
        """
        Inicializa el servicio de chatbot
        
        Args:
            openai_client: Cliente de OpenAI (opcional)
        """
        self.openai_client = openai_client
        self.conversation_contexts = {}
        self.quick_responses = self._load_quick_responses()
    
    def process_message(
        self,
        user_id: str,
        message: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Procesa un mensaje del usuario y genera respuesta
        
        Args:
            user_id: ID del usuario
            message: Mensaje del usuario
            context: Contexto adicional (progreso, estado, etc.)
        
        Returns:
            Respuesta del chatbot
        """
        # Verificar respuestas rápidas primero
        quick_response = self._check_quick_responses(message)
        if quick_response:
            return {
                "user_id": user_id,
                "message": message,
                "response": quick_response,
                "timestamp": datetime.now().isoformat(),
                "type": "quick_response"
            }
        
        # Si hay cliente de OpenAI, usar IA
        if self.openai_client:
            ai_response = self._generate_ai_response(user_id, message, context)
            return {
                "user_id": user_id,
                "message": message,
                "response": ai_response,
                "timestamp": datetime.now().isoformat(),
                "type": "ai_response"
            }
        
        # Respuesta por defecto
        return {
            "user_id": user_id,
            "message": message,
            "response": self._generate_default_response(message, context),
            "timestamp": datetime.now().isoformat(),
            "type": "default_response"
        }
    
    def start_conversation(self, user_id: str, user_data: Optional[Dict] = None) -> Dict:
        """
        Inicia una nueva conversación
        
        Args:
            user_id: ID del usuario
            user_data: Datos del usuario (opcional)
        
        Returns:
            Mensaje de bienvenida
        """
        welcome_messages = [
            "¡Hola! Soy tu asistente de recuperación. Estoy aquí para ayudarte 24/7. ¿En qué puedo ayudarte hoy?",
            "Hola, estoy aquí para apoyarte en tu viaje de recuperación. ¿Cómo te sientes hoy?",
            "¡Bienvenido! Estoy aquí para escucharte y ayudarte. ¿Qué te gustaría compartir?"
        ]
        
        import random
        welcome = random.choice(welcome_messages)
        
        # Guardar contexto de conversación
        self.conversation_contexts[user_id] = {
            "started_at": datetime.now().isoformat(),
            "messages": [],
            "user_data": user_data or {}
        }
        
        return {
            "user_id": user_id,
            "response": welcome,
            "timestamp": datetime.now().isoformat(),
            "type": "welcome"
        }
    
    def get_conversation_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """
        Obtiene historial de conversación
        
        Args:
            user_id: ID del usuario
            limit: Límite de mensajes
        
        Returns:
            Historial de conversación
        """
        if user_id not in self.conversation_contexts:
            return []
        
        messages = self.conversation_contexts[user_id].get("messages", [])
        return messages[-limit:] if len(messages) > limit else messages
    
    def _check_quick_responses(self, message: str) -> Optional[str]:
        """Verifica si hay una respuesta rápida para el mensaje"""
        message_lower = message.lower()
        
        for keyword, response in self.quick_responses.items():
            if keyword in message_lower:
                return response
        
        return None
    
    def _generate_ai_response(
        self,
        user_id: str,
        message: str,
        context: Optional[Dict]
    ) -> str:
        """Genera respuesta usando IA"""
        if not self.openai_client:
            return self._generate_default_response(message, context)
        
        try:
            # Construir contexto para la IA
            system_prompt = """Eres un asistente compasivo y experto en recuperación de adicciones. 
            Tu objetivo es proporcionar apoyo emocional, información útil y motivación.
            Sé empático, positivo pero realista, y siempre enfócate en la recuperación."""
            
            user_prompt = message
            if context:
                context_str = f"\n\nContexto del usuario: Días de sobriedad: {context.get('days_sober', 0)}, Estado: {context.get('mood', 'N/A')}"
                user_prompt += context_str
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return self._generate_default_response(message, context)
    
    def _generate_default_response(self, message: str, context: Optional[Dict]) -> str:
        """Genera respuesta por defecto"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["craving", "deseo", "ansiedad", "necesito"]):
            return "Entiendo que estás pasando por un momento difícil. Los cravings son temporales. ¿Has probado alguna técnica de distracción o respiración?"
        
        if any(word in message_lower for word in ["triste", "deprimido", "mal", "malo"]):
            return "Lamento escuchar que te sientes así. Es normal tener días difíciles. ¿Tienes alguien con quien hablar o quieres que te sugiera algunas estrategias?"
        
        if any(word in message_lower for word in ["bien", "genial", "feliz", "mejor"]):
            return "¡Me alegra saber que estás bien! Continúa con tu plan de recuperación. Cada día positivo es una victoria."
        
        return "Entiendo. Estoy aquí para ayudarte. ¿Hay algo específico sobre tu recuperación en lo que pueda asistirte?"
    
    def _load_quick_responses(self) -> Dict[str, str]:
        """Carga respuestas rápidas predefinidas"""
        return {
            "hola": "¡Hola! ¿Cómo puedo ayudarte hoy?",
            "ayuda": "Estoy aquí para ayudarte. Puedo ayudarte con cravings, motivación, estrategias de afrontamiento, o simplemente escucharte.",
            "craving": "Los cravings son temporales, generalmente duran 15-20 minutos. Prueba: respiración profunda, beber agua, distraerte con una actividad, o contactar a tu apoyo.",
            "recaída": "Si has tenido una recaída, no es el fin. Aprende de la experiencia y continúa. La recuperación es un proceso. ¿Quieres que te ayude a crear un plan?",
            "motivación": "Recuerda tus razones para estar sobrio. Eres más fuerte de lo que crees. Cada día es una oportunidad de crecimiento.",
            "gracias": "De nada. Estoy aquí siempre que me necesites. ¡Sigue adelante!"
        }

