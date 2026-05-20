"""
Sistema de coaching personalizado
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class CoachingSession:
    """Sesión de coaching"""
    id: str
    user_id: str
    session_type: str  # "daily", "weekly", "goal_focused", "problem_solving"
    topics: List[str]
    advice: List[str]
    action_items: List[str]
    motivation_message: str
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_type": self.session_type,
            "topics": self.topics,
            "advice": self.advice,
            "action_items": self.action_items,
            "motivation_message": self.motivation_message,
            "created_at": self.created_at
        }


class PersonalizedCoaching:
    """Sistema de coaching personalizado"""
    
    def __init__(self):
        """Inicializa el coaching"""
        self.sessions: Dict[str, List[CoachingSession]] = {}  # user_id -> [sessions]
    
    def create_coaching_session(self, user_id: str, session_type: str,
                               user_data: Dict) -> CoachingSession:
        """Crea sesión de coaching"""
        topics = []
        advice = []
        action_items = []
        
        # Analizar datos del usuario
        skin_scores = user_data.get("skin_analysis", {}).get("quality_scores", {})
        goals = user_data.get("goals", [])
        habits = user_data.get("habits", {})
        
        # Generar contenido basado en tipo de sesión
        if session_type == "daily":
            topics.append("Rutina diaria")
            advice.append("Mantén tu rutina consistente")
            action_items.append("Completa tu rutina matutina y nocturna")
        
        elif session_type == "goal_focused":
            if goals:
                goal = goals[0]
                topics.append(f"Objetivo: {goal.get('title', 'Mejorar piel')}")
                current = goal.get("current_value", 0)
                target = goal.get("target_value", 100)
                progress = (current / target * 100) if target > 0 else 0
                
                if progress < 50:
                    advice.append("Estás en el camino correcto. Continúa con tu rutina.")
                    action_items.append("Mantén la consistencia en tu rutina")
                else:
                    advice.append("¡Excelente progreso! Estás cerca de tu objetivo.")
                    action_items.append("Continúa con tu rutina actual")
        
        elif session_type == "problem_solving":
            overall_score = skin_scores.get("overall_score", 0)
            
            if overall_score < 60:
                topics.append("Mejora de condición de piel")
                advice.append("Tu piel necesita atención. Considera ajustar tu rutina.")
                action_items.append("Revisa tus productos actuales")
                action_items.append("Considera agregar productos específicos para tus necesidades")
        
        # Mensaje motivacional
        motivation = self._generate_motivation_message(session_type, user_data)
        
        session = CoachingSession(
            id=str(uuid.uuid4()),
            user_id=user_id,
            session_type=session_type,
            topics=topics,
            advice=advice,
            action_items=action_items,
            motivation_message=motivation
        )
        
        if user_id not in self.sessions:
            self.sessions[user_id] = []
        
        self.sessions[user_id].append(session)
        return session
    
    def _generate_motivation_message(self, session_type: str, user_data: Dict) -> str:
        """Genera mensaje motivacional"""
        messages = {
            "daily": "Cada día es una oportunidad para cuidar tu piel. ¡Tú puedes!",
            "weekly": "Una semana más de cuidado. ¡Sigue así!",
            "goal_focused": "Cada paso te acerca a tu objetivo. ¡Continúa!",
            "problem_solving": "Los desafíos son oportunidades de crecimiento. ¡Vamos a resolverlo!"
        }
        
        return messages.get(session_type, "Sigue cuidando tu piel. ¡Estás haciendo un gran trabajo!")
    
    def get_user_sessions(self, user_id: str, session_type: Optional[str] = None) -> List[CoachingSession]:
        """Obtiene sesiones del usuario"""
        user_sessions = self.sessions.get(user_id, [])
        
        if session_type:
            user_sessions = [s for s in user_sessions if s.session_type == session_type]
        
        user_sessions.sort(key=lambda x: x.created_at, reverse=True)
        return user_sessions






