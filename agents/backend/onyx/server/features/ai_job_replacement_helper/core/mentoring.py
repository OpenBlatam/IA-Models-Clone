"""
Mentoring Service - Sistema de mentoría y coaching con IA
==========================================================

Sistema de mentoría personalizada con IA que guía al usuario
en su transición profesional.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class MentorType(str, Enum):
    """Tipos de mentores"""
    CAREER_COACH = "career_coach"
    TECH_MENTOR = "tech_mentor"
    NETWORKING_EXPERT = "networking_expert"
    INTERVIEW_COACH = "interview_coach"
    MOTIVATION_COACH = "motivation_coach"


class SessionType(str, Enum):
    """Tipos de sesiones"""
    QUICK_CHAT = "quick_chat"
    DEEP_DIVE = "deep_dive"
    SKILL_REVIEW = "skill_review"
    CAREER_PLANNING = "career_planning"
    INTERVIEW_PREP = "interview_prep"
    MOTIVATIONAL = "motivational"


@dataclass
class MentorMessage:
    """Mensaje del mentor"""
    id: str
    mentor_type: MentorType
    message: str
    suggestions: List[str] = field(default_factory=list)
    resources: List[Dict[str, str]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CoachingSession:
    """Sesión de coaching"""
    id: str
    user_id: str
    session_type: SessionType
    mentor_type: MentorType
    messages: List[MentorMessage] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    notes: Optional[str] = None


class MentoringService:
    """Servicio de mentoría con IA"""
    
    def __init__(self):
        """Inicializar servicio de mentoría"""
        self.sessions: Dict[str, List[CoachingSession]] = {}
        logger.info("MentoringService initialized")
    
    def start_session(
        self,
        user_id: str,
        session_type: SessionType,
        mentor_type: Optional[MentorType] = None
    ) -> CoachingSession:
        """Iniciar una sesión de coaching"""
        # Determinar tipo de mentor si no se especifica
        if not mentor_type:
            mentor_type = self._determine_mentor_type(session_type)
        
        session = CoachingSession(
            id=f"session_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            session_type=session_type,
            mentor_type=mentor_type
        )
        
        if user_id not in self.sessions:
            self.sessions[user_id] = []
        
        self.sessions[user_id].append(session)
        
        # Mensaje inicial del mentor
        initial_message = self._generate_initial_message(session_type, mentor_type)
        session.messages.append(initial_message)
        
        logger.info(f"Coaching session started for user {user_id}")
        return session
    
    def ask_mentor(
        self,
        user_id: str,
        session_id: str,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MentorMessage:
        """Hacer una pregunta al mentor"""
        session = self._get_session(user_id, session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Generar respuesta del mentor basada en la pregunta y contexto
        response = self._generate_mentor_response(
            question=question,
            mentor_type=session.mentor_type,
            session_type=session.session_type,
            context=context or {}
        )
        
        session.messages.append(response)
        return response
    
    def get_career_advice(
        self,
        user_id: str,
        current_situation: str,
        goals: List[str]
    ) -> MentorMessage:
        """Obtener consejo de carrera"""
        mentor = MentorType.CAREER_COACH
        
        advice = self._generate_career_advice(current_situation, goals)
        
        message = MentorMessage(
            id=f"advice_{user_id}_{int(datetime.now().timestamp())}",
            mentor_type=mentor,
            message=advice["message"],
            suggestions=advice["suggestions"],
            resources=advice["resources"]
        )
        
        return message
    
    def get_interview_tips(
        self,
        user_id: str,
        job_title: str,
        company: str
    ) -> MentorMessage:
        """Obtener tips para entrevista"""
        mentor = MentorType.INTERVIEW_COACH
        
        tips = self._generate_interview_tips(job_title, company)
        
        message = MentorMessage(
            id=f"tips_{user_id}_{int(datetime.now().timestamp())}",
            mentor_type=mentor,
            message=tips["message"],
            suggestions=tips["suggestions"],
            resources=tips["resources"]
        )
        
        return message
    
    def get_motivational_message(
        self,
        user_id: str,
        current_mood: Optional[str] = None
    ) -> MentorMessage:
        """Obtener mensaje motivacional personalizado"""
        mentor = MentorType.MOTIVATION_COACH
        
        message_text = self._generate_motivational_message(current_mood)
        
        message = MentorMessage(
            id=f"motivation_{user_id}_{int(datetime.now().timestamp())}",
            mentor_type=mentor,
            message=message_text,
            suggestions=[
                "Recuerda tus logros hasta ahora",
                "Visualiza tu objetivo final",
                "Toma un descanso si lo necesitas",
            ]
        )
        
        return message
    
    def _determine_mentor_type(self, session_type: SessionType) -> MentorType:
        """Determinar tipo de mentor según tipo de sesión"""
        mapping = {
            SessionType.CAREER_PLANNING: MentorType.CAREER_COACH,
            SessionType.SKILL_REVIEW: MentorType.TECH_MENTOR,
            SessionType.INTERVIEW_PREP: MentorType.INTERVIEW_COACH,
            SessionType.MOTIVATIONAL: MentorType.MOTIVATION_COACH,
        }
        return mapping.get(session_type, MentorType.CAREER_COACH)
    
    def _generate_initial_message(
        self,
        session_type: SessionType,
        mentor_type: MentorType
    ) -> MentorMessage:
        """Generar mensaje inicial del mentor"""
        greetings = {
            MentorType.CAREER_COACH: "Hola, soy tu coach de carrera. Estoy aquí para ayudarte a navegar esta transición profesional.",
            MentorType.TECH_MENTOR: "¡Hola! Soy tu mentor técnico. Vamos a trabajar juntos en desarrollar tus habilidades.",
            MentorType.INTERVIEW_COACH: "Hola, soy tu coach de entrevistas. Te ayudaré a prepararte para tener éxito.",
            MentorType.MOTIVATION_COACH: "¡Hola! Soy tu coach de motivación. Juntos vamos a mantenerte enfocado y positivo.",
        }
        
        return MentorMessage(
            id=f"initial_{int(datetime.now().timestamp())}",
            mentor_type=mentor_type,
            message=greetings.get(mentor_type, "Hola, estoy aquí para ayudarte."),
            suggestions=[
                "Cuéntame sobre tu situación actual",
                "¿Qué objetivos tienes?",
                "¿En qué puedo ayudarte hoy?",
            ]
        )
    
    def _generate_mentor_response(
        self,
        question: str,
        mentor_type: MentorType,
        session_type: SessionType,
        context: Dict[str, Any]
    ) -> MentorMessage:
        """Generar respuesta del mentor (simulado - en producción usaría IA real)"""
        # En producción, esto usaría un modelo de lenguaje como GPT-4
        # Por ahora, generamos respuestas basadas en templates
        
        responses = {
            MentorType.CAREER_COACH: {
                "default": "Entiendo tu situación. Te recomiendo enfocarte en identificar tus habilidades transferibles y explorar nuevas oportunidades que se alineen con tus intereses.",
                "skills": "Las habilidades más demandadas ahora son Python, Machine Learning, y Cloud Computing. ¿Cuál te interesa más?",
            },
            MentorType.TECH_MENTOR: {
                "default": "Para mejorar tus habilidades técnicas, te sugiero practicar regularmente, construir proyectos personales, y contribuir a código abierto.",
            },
            MentorType.INTERVIEW_COACH: {
                "default": "Para prepararte para entrevistas, practica respuestas a preguntas comunes, investiga la empresa, y prepárate con ejemplos de tu experiencia.",
            },
        }
        
        mentor_responses = responses.get(mentor_type, {})
        message_text = mentor_responses.get("default", "Entiendo. Déjame ayudarte con eso.")
        
        return MentorMessage(
            id=f"response_{int(datetime.now().timestamp())}",
            mentor_type=mentor_type,
            message=message_text,
            suggestions=[
                "Continúa con el siguiente paso de tu roadmap",
                "Practica las habilidades recomendadas",
                "Mantén un registro de tu progreso",
            ]
        )
    
    def _generate_career_advice(
        self,
        current_situation: str,
        goals: List[str]
    ) -> Dict[str, Any]:
        """Generar consejo de carrera"""
        return {
            "message": f"Basándome en tu situación actual y tus objetivos, te recomiendo crear un plan estructurado que combine desarrollo de habilidades con búsqueda activa de oportunidades.",
            "suggestions": [
                "Identifica 3-5 habilidades clave para tus objetivos",
                "Crea un timeline realista (3-6 meses)",
                "Establece métricas de progreso semanales",
                "Conecta con profesionales en tu área objetivo",
            ],
            "resources": [
                {"type": "article", "title": "Cómo hacer transición de carrera", "url": "#"},
                {"type": "template", "title": "Plantilla de plan de carrera", "url": "#"},
            ]
        }
    
    def _generate_interview_tips(
        self,
        job_title: str,
        company: str
    ) -> Dict[str, Any]:
        """Generar tips para entrevista"""
        return {
            "message": f"Para la entrevista de {job_title} en {company}, prepárate investigando la empresa, revisando el job description, y preparando ejemplos de tu experiencia.",
            "suggestions": [
                "Investiga la cultura y valores de la empresa",
                "Prepara ejemplos usando el método STAR",
                "Practica respuestas a preguntas comunes",
                "Prepara preguntas inteligentes para el entrevistador",
            ],
            "resources": [
                {"type": "video", "title": "Cómo responder preguntas de entrevista", "url": "#"},
                {"type": "tool", "title": "Simulador de entrevistas", "url": "#"},
            ]
        }
    
    def _generate_motivational_message(self, current_mood: Optional[str]) -> str:
        """Generar mensaje motivacional"""
        messages = [
            "Recuerda que cada experto fue alguna vez un principiante. Estás en el camino correcto.",
            "El progreso no siempre es lineal. Celebra los pequeños avances.",
            "La transición profesional es un maratón, no un sprint. Mantén el ritmo constante.",
            "Tus habilidades anteriores no desaparecen, se transforman en nuevas oportunidades.",
        ]
        
        return messages[datetime.now().day % len(messages)]
    
    def _get_session(self, user_id: str, session_id: str) -> Optional[CoachingSession]:
        """Obtener sesión por ID"""
        if user_id not in self.sessions:
            return None
        
        for session in self.sessions[user_id]:
            if session.id == session_id:
                return session
        
        return None
    
    def get_user_sessions(self, user_id: str) -> List[CoachingSession]:
        """Obtener todas las sesiones del usuario"""
        return self.sessions.get(user_id, [])




