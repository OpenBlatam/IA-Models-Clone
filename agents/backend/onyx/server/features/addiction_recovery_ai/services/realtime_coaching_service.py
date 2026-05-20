"""
Servicio de Coaching en Tiempo Real - Sistema de coaching interactivo
"""

from typing import Dict, List, Optional
from datetime import datetime


class RealtimeCoachingService:
    """Servicio de coaching en tiempo real"""
    
    def __init__(self):
        """Inicializa el servicio de coaching"""
        pass
    
    def start_coaching_session(
        self,
        user_id: str,
        session_type: str = "general",
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Inicia sesión de coaching
        
        Args:
            user_id: ID del usuario
            session_type: Tipo de sesión
            context: Contexto adicional
        
        Returns:
            Sesión de coaching iniciada
        """
        session = {
            "id": f"coaching_{datetime.now().timestamp()}",
            "user_id": user_id,
            "session_type": session_type,
            "context": context or {},
            "started_at": datetime.now().isoformat(),
            "status": "active",
            "messages": []
        }
        
        return session
    
    def send_coaching_message(
        self,
        session_id: str,
        user_id: str,
        message: str
    ) -> Dict:
        """
        Envía mensaje en sesión de coaching
        
        Args:
            session_id: ID de la sesión
            user_id: ID del usuario
            message: Mensaje del usuario
        
        Returns:
            Respuesta de coaching
        """
        # Analizar mensaje y generar respuesta
        response = self._generate_coaching_response(message, user_id)
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "user_message": message,
            "coach_response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_coaching_suggestions(
        self,
        user_id: str,
        current_situation: Dict
    ) -> List[Dict]:
        """
        Obtiene sugerencias de coaching
        
        Args:
            user_id: ID del usuario
            current_situation: Situación actual
        
        Returns:
            Lista de sugerencias
        """
        suggestions = []
        
        # Basado en situación actual
        if current_situation.get("cravings_level", 0) >= 6:
            suggestions.append({
                "type": "craving_management",
                "title": "Manejo de Cravings",
                "suggestion": "Practica la técnica de retraso: espera 15 minutos antes de tomar cualquier decisión",
                "priority": "high"
            })
        
        if current_situation.get("stress_level", 0) >= 7:
            suggestions.append({
                "type": "stress_relief",
                "title": "Alivio de Estrés",
                "suggestion": "Intenta una sesión de respiración profunda de 5 minutos",
                "priority": "high"
            })
        
        return suggestions
    
    def end_coaching_session(
        self,
        session_id: str,
        user_id: str,
        feedback: Optional[Dict] = None
    ) -> Dict:
        """
        Termina sesión de coaching
        
        Args:
            session_id: ID de la sesión
            user_id: ID del usuario
            feedback: Feedback del usuario
        
        Returns:
            Sesión terminada
        """
        return {
            "session_id": session_id,
            "user_id": user_id,
            "ended_at": datetime.now().isoformat(),
            "status": "completed",
            "feedback": feedback,
            "summary": "Sesión completada exitosamente"
        }
    
    def _generate_coaching_response(self, message: str, user_id: str) -> str:
        """Genera respuesta de coaching"""
        message_lower = message.lower()
        
        # Respuestas basadas en palabras clave
        if any(word in message_lower for word in ["difícil", "difícil", "lucha"]):
            return "Entiendo que esto es difícil. Recuerda que cada día que pasas sin consumir es un logro. ¿Qué estrategias has usado antes que te han ayudado?"
        
        if any(word in message_lower for word in ["tentación", "deseo", "craving"]):
            return "Los cravings son normales y temporales. Prueba la técnica de retraso: espera 15 minutos y el deseo probablemente disminuirá. ¿Quieres que te guíe a través de una técnica de respiración?"
        
        if any(word in message_lower for word in ["éxito", "logro", "progreso"]):
            return "¡Excelente! Celebra tus logros, por pequeños que parezcan. Cada paso cuenta. ¿Qué te ha funcionado mejor hasta ahora?"
        
        # Respuesta por defecto
        return "Gracias por compartir. Estoy aquí para apoyarte. ¿Hay algo específico en lo que te gustaría trabajar hoy?"

