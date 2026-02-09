"""
Real-Time Mentoring Service - Mentoría en tiempo real
======================================================

Sistema de mentoría y coaching en tiempo real con IA.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MentorType(str, Enum):
    """Tipos de mentores"""
    CAREER = "career"
    TECHNICAL = "technical"
    INTERVIEW = "interview"
    SALARY = "salary"
    NETWORKING = "networking"
    LEADERSHIP = "leadership"


class SessionStatus(str, Enum):
    """Estados de sesión"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class MentoringSession:
    """Sesión de mentoría"""
    id: str
    user_id: str
    mentor_type: MentorType
    status: SessionStatus
    messages: List[Dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    duration_minutes: int = 0
    topics_covered: List[str] = field(default_factory=list)


@dataclass
class MentorResponse:
    """Respuesta del mentor"""
    message: str
    suggestions: List[str] = field(default_factory=list)
    resources: List[Dict[str, str]] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    confidence: float = 0.8


class RealTimeMentoringService:
    """Servicio de mentoría en tiempo real"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.sessions: Dict[str, MentoringSession] = {}
        self.active_sessions: Dict[str, str] = {}  # user_id -> session_id
        logger.info("RealTimeMentoringService initialized")
    
    def start_session(
        self,
        user_id: str,
        mentor_type: MentorType,
        initial_question: Optional[str] = None
    ) -> MentoringSession:
        """Iniciar sesión de mentoría"""
        session_id = f"mentoring_{user_id}_{int(datetime.now().timestamp())}"
        
        session = MentoringSession(
            id=session_id,
            user_id=user_id,
            mentor_type=mentor_type,
            status=SessionStatus.ACTIVE,
        )
        
        if initial_question:
            response = self._generate_response(mentor_type, initial_question, [])
            session.messages.append({
                "role": "user",
                "content": initial_question,
                "timestamp": datetime.now().isoformat(),
            })
            session.messages.append({
                "role": "mentor",
                "content": response.message,
                "suggestions": response.suggestions,
                "resources": response.resources,
                "timestamp": datetime.now().isoformat(),
            })
        
        self.sessions[session_id] = session
        self.active_sessions[user_id] = session_id
        
        logger.info(f"Mentoring session started: {session_id}")
        return session
    
    def send_message(
        self,
        session_id: str,
        message: str
    ) -> MentorResponse:
        """Enviar mensaje en sesión activa"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.status != SessionStatus.ACTIVE:
            raise ValueError(f"Session {session_id} is not active")
        
        # Agregar mensaje del usuario
        session.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Generar respuesta del mentor
        response = self._generate_response(
            session.mentor_type,
            message,
            session.messages[:-1]  # Contexto previo
        )
        
        # Agregar respuesta del mentor
        session.messages.append({
            "role": "mentor",
            "content": response.message,
            "suggestions": response.suggestions,
            "resources": response.resources,
            "next_steps": response.next_steps,
            "timestamp": datetime.now().isoformat(),
        })
        
        session.last_activity = datetime.now()
        session.duration_minutes = int(
            (datetime.now() - session.started_at).total_seconds() / 60
        )
        
        # Extraer topics
        if message.lower() in ["carrera", "career", "trabajo", "job"]:
            session.topics_covered.append("Career Planning")
        
        return response
    
    def _generate_response(
        self,
        mentor_type: MentorType,
        message: str,
        context: List[Dict[str, Any]]
    ) -> MentorResponse:
        """Generar respuesta del mentor"""
        # En producción, esto usaría un modelo de IA real (GPT, Claude, etc.)
        # Por ahora, generamos respuestas basadas en templates
        
        message_lower = message.lower()
        
        if mentor_type == MentorType.CAREER:
            if "cambio" in message_lower or "transition" in message_lower:
                return MentorResponse(
                    message="Los cambios de carrera requieren planificación cuidadosa. Primero, identifica tus habilidades transferibles y las brechas que necesitas cerrar.",
                    suggestions=[
                        "Haz un inventario de tus habilidades actuales",
                        "Investiga el mercado objetivo",
                        "Crea un plan de transición de 6-12 meses",
                    ],
                    resources=[
                        {"type": "article", "title": "Career Transition Guide", "url": "https://example.com"},
                    ],
                    next_steps=["Completa un assessment de habilidades", "Define tu objetivo de carrera"],
                )
            else:
                return MentorResponse(
                    message="Entiendo tu pregunta sobre carrera. ¿Podrías ser más específico sobre qué aspecto te gustaría explorar?",
                    suggestions=["Planificación de carrera", "Desarrollo de habilidades", "Networking"],
                )
        
        elif mentor_type == MentorType.TECHNICAL:
            if "aprender" in message_lower or "learn" in message_lower:
                return MentorResponse(
                    message="Para aprender nuevas tecnologías, recomiendo un enfoque práctico: proyectos reales, contribuciones open source, y práctica constante.",
                    suggestions=[
                        "Elige un proyecto que te motive",
                        "Únete a comunidades de desarrolladores",
                        "Practica diariamente, aunque sea 30 minutos",
                    ],
                    resources=[
                        {"type": "course", "title": "Technical Skills Course", "url": "https://example.com"},
                    ],
                )
            else:
                return MentorResponse(
                    message="¿Qué tecnología específica te interesa aprender o mejorar?",
                )
        
        elif mentor_type == MentorType.INTERVIEW:
            return MentorResponse(
                message="Para prepararte para entrevistas, practica respuestas usando el método STAR (Situation, Task, Action, Result) y prepárate para preguntas técnicas específicas del rol.",
                suggestions=[
                    "Practica con simuladores de entrevistas",
                    "Prepara ejemplos concretos de tus logros",
                    "Investiga la empresa y el rol",
                ],
                next_steps=["Inicia una sesión de práctica de entrevista", "Revisa preguntas comunes del rol"],
            )
        
        else:
            return MentorResponse(
                message="Estoy aquí para ayudarte. ¿En qué puedo asistirte específicamente?",
            )
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """Finalizar sesión"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.status = SessionStatus.COMPLETED
        session.duration_minutes = int(
            (datetime.now() - session.started_at).total_seconds() / 60
        )
        
        # Remover de sesiones activas
        if session.user_id in self.active_sessions:
            del self.active_sessions[session.user_id]
        
        return {
            "session_id": session_id,
            "duration_minutes": session.duration_minutes,
            "messages_count": len(session.messages),
            "topics_covered": session.topics_covered,
            "summary": self._generate_session_summary(session),
        }
    
    def _generate_session_summary(self, session: MentoringSession) -> str:
        """Generar resumen de sesión"""
        return f"Sesión de {session.mentor_type.value} mentoría completada. Se cubrieron {len(session.topics_covered)} temas principales en {session.duration_minutes} minutos."
    
    def get_active_session(self, user_id: str) -> Optional[MentoringSession]:
        """Obtener sesión activa del usuario"""
        session_id = self.active_sessions.get(user_id)
        if session_id:
            return self.sessions.get(session_id)
        return None
    
    def get_session_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtener historial de sesiones"""
        user_sessions = [
            s for s in self.sessions.values()
            if s.user_id == user_id and s.status == SessionStatus.COMPLETED
        ]
        
        return [
            {
                "session_id": s.id,
                "mentor_type": s.mentor_type.value,
                "duration_minutes": s.duration_minutes,
                "topics_covered": s.topics_covered,
                "completed_at": s.last_activity.isoformat(),
            }
            for s in sorted(user_sessions, key=lambda x: x.last_activity, reverse=True)
        ]




