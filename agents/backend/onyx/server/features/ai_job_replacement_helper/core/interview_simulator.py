"""
Interview Simulator Service - Simulador de entrevistas con IA
==============================================================

Sistema que simula entrevistas de trabajo con IA para practicar.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class InterviewType(str, Enum):
    """Tipos de entrevistas"""
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    CULTURAL_FIT = "cultural_fit"
    CASE_STUDY = "case_study"
    SYSTEM_DESIGN = "system_design"


class QuestionDifficulty(str, Enum):
    """Dificultad de preguntas"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class InterviewQuestion:
    """Pregunta de entrevista"""
    id: str
    question: str
    type: InterviewType
    difficulty: QuestionDifficulty
    category: str
    expected_keywords: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)


@dataclass
class UserAnswer:
    """Respuesta del usuario"""
    question_id: str
    answer: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: Optional[int] = None


@dataclass
class AnswerFeedback:
    """Feedback sobre la respuesta"""
    question_id: str
    score: float  # 0.0 a 1.0
    strengths: List[str]
    improvements: List[str]
    keyword_match: float
    suggestions: List[str]


@dataclass
class InterviewSession:
    """Sesión de entrevista simulada"""
    id: str
    user_id: str
    interview_type: InterviewType
    job_title: Optional[str] = None
    company: Optional[str] = None
    questions: List[InterviewQuestion] = field(default_factory=list)
    answers: List[UserAnswer] = field(default_factory=list)
    feedback: List[AnswerFeedback] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    overall_score: Optional[float] = None


class InterviewSimulatorService:
    """Servicio de simulador de entrevistas"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.sessions: Dict[str, List[InterviewSession]] = {}
        self.question_bank = self._initialize_question_bank()
        logger.info("InterviewSimulatorService initialized")
    
    def start_interview(
        self,
        user_id: str,
        interview_type: InterviewType,
        job_title: Optional[str] = None,
        company: Optional[str] = None,
        num_questions: int = 5
    ) -> InterviewSession:
        """Iniciar una entrevista simulada"""
        session = InterviewSession(
            id=f"interview_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            interview_type=interview_type,
            job_title=job_title,
            company=company
        )
        
        # Seleccionar preguntas del banco
        questions = self._select_questions(interview_type, num_questions)
        session.questions = questions
        
        if user_id not in self.sessions:
            self.sessions[user_id] = []
        
        self.sessions[user_id].append(session)
        
        logger.info(f"Interview session started for user {user_id}")
        return session
    
    def submit_answer(
        self,
        user_id: str,
        session_id: str,
        question_id: str,
        answer: str,
        duration_seconds: Optional[int] = None
    ) -> AnswerFeedback:
        """Enviar respuesta a una pregunta"""
        session = self._get_session(user_id, session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Guardar respuesta
        user_answer = UserAnswer(
            question_id=question_id,
            answer=answer,
            duration_seconds=duration_seconds
        )
        session.answers.append(user_answer)
        
        # Generar feedback
        question = next((q for q in session.questions if q.id == question_id), None)
        if not question:
            raise ValueError(f"Question {question_id} not found")
        
        feedback = self._analyze_answer(answer, question)
        session.feedback.append(feedback)
        
        return feedback
    
    def complete_interview(
        self,
        user_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Completar entrevista y obtener resultados"""
        session = self._get_session(user_id, session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.completed_at = datetime.now()
        
        # Calcular score general
        if session.feedback:
            session.overall_score = sum(f.score for f in session.feedback) / len(session.feedback)
        else:
            session.overall_score = 0.0
        
        # Generar resumen
        summary = self._generate_summary(session)
        
        return {
            "session_id": session_id,
            "overall_score": session.overall_score,
            "total_questions": len(session.questions),
            "answered_questions": len(session.answers),
            "summary": summary,
            "feedback": [
                {
                    "question_id": f.question_id,
                    "score": f.score,
                    "strengths": f.strengths,
                    "improvements": f.improvements,
                }
                for f in session.feedback
            ]
        }
    
    def _initialize_question_bank(self) -> Dict[InterviewType, List[InterviewQuestion]]:
        """Inicializar banco de preguntas"""
        bank = {}
        
        # Preguntas técnicas
        bank[InterviewType.TECHNICAL] = [
            InterviewQuestion(
                id="tech_1",
                question="Explícame qué es la programación orientada a objetos y da un ejemplo.",
                type=InterviewType.TECHNICAL,
                difficulty=QuestionDifficulty.MEDIUM,
                category="Programming Concepts",
                expected_keywords=["clases", "objetos", "herencia", "encapsulación", "polimorfismo"],
                tips=["Menciona los 4 pilares de OOP", "Da un ejemplo práctico"]
            ),
            InterviewQuestion(
                id="tech_2",
                question="¿Cuál es la diferencia entre una lista y un diccionario en Python?",
                type=InterviewType.TECHNICAL,
                difficulty=QuestionDifficulty.EASY,
                category="Python",
                expected_keywords=["lista", "diccionario", "indexación", "clave-valor"],
                tips=["Menciona complejidad temporal", "Da ejemplos de uso"]
            ),
        ]
        
        # Preguntas de comportamiento
        bank[InterviewType.BEHAVIORAL] = [
            InterviewQuestion(
                id="beh_1",
                question="Cuéntame sobre un momento en que tuviste que trabajar bajo presión.",
                type=InterviewType.BEHAVIORAL,
                difficulty=QuestionDifficulty.MEDIUM,
                category="Stress Management",
                expected_keywords=["situación", "acción", "resultado", "aprendizaje"],
                tips=["Usa el método STAR", "Sé específico con números"]
            ),
            InterviewQuestion(
                id="beh_2",
                question="Describe una situación en que tuviste que resolver un conflicto en el equipo.",
                type=InterviewType.BEHAVIORAL,
                difficulty=QuestionDifficulty.MEDIUM,
                category="Teamwork",
                expected_keywords=["conflicto", "resolución", "comunicación", "compromiso"],
                tips=["Muestra habilidades de comunicación", "Enfócate en el resultado positivo"]
            ),
        ]
        
        # Preguntas de fit cultural
        bank[InterviewType.CULTURAL_FIT] = [
            InterviewQuestion(
                id="cult_1",
                question="¿Qué te motiva en el trabajo?",
                type=InterviewType.CULTURAL_FIT,
                difficulty=QuestionDifficulty.EASY,
                category="Motivation",
                expected_keywords=["crecimiento", "aprendizaje", "impacto", "colaboración"],
                tips=["Sé auténtico", "Conecta con los valores de la empresa"]
            ),
        ]
        
        return bank
    
    def _select_questions(
        self,
        interview_type: InterviewType,
        num_questions: int
    ) -> List[InterviewQuestion]:
        """Seleccionar preguntas del banco"""
        questions = self.question_bank.get(interview_type, [])
        return questions[:num_questions]
    
    def _analyze_answer(
        self,
        answer: str,
        question: InterviewQuestion
    ) -> AnswerFeedback:
        """Analizar respuesta del usuario"""
        answer_lower = answer.lower()
        
        # Verificar keywords esperadas
        keyword_matches = sum(
            1 for keyword in question.expected_keywords
            if keyword.lower() in answer_lower
        )
        keyword_match = keyword_matches / len(question.expected_keywords) if question.expected_keywords else 0.5
        
        # Calcular score
        score = keyword_match * 0.7 + 0.3  # Base score + keyword match
        
        # Generar feedback
        strengths = []
        improvements = []
        suggestions = []
        
        if keyword_match >= 0.7:
            strengths.append("Mencionaste los conceptos clave")
        else:
            improvements.append("Falta mencionar algunos conceptos importantes")
            suggestions.append(f"Intenta incluir: {', '.join(question.expected_keywords[:3])}")
        
        if len(answer.split()) >= 50:
            strengths.append("Respuesta con suficiente detalle")
        else:
            improvements.append("La respuesta podría ser más detallada")
            suggestions.append("Expande tu respuesta con ejemplos específicos")
        
        if "ejemplo" in answer_lower or "ejemplo" in answer_lower:
            strengths.append("Incluiste ejemplos")
        else:
            suggestions.append("Agrega ejemplos concretos para ilustrar tu punto")
        
        return AnswerFeedback(
            question_id=question.id,
            score=min(score, 1.0),
            strengths=strengths,
            improvements=improvements,
            keyword_match=keyword_match,
            suggestions=suggestions + question.tips
        )
    
    def _generate_summary(self, session: InterviewSession) -> Dict[str, Any]:
        """Generar resumen de la entrevista"""
        if not session.feedback:
            return {
                "overall_assessment": "Completa más preguntas para obtener un análisis completo",
                "strengths": [],
                "areas_for_improvement": [],
            }
        
        all_strengths = []
        all_improvements = []
        
        for feedback in session.feedback:
            all_strengths.extend(feedback.strengths)
            all_improvements.extend(feedback.improvements)
        
        # Remover duplicados
        all_strengths = list(set(all_strengths))
        all_improvements = list(set(all_improvements))
        
        overall_assessment = "Excelente" if session.overall_score >= 0.8 else \
                            "Bueno" if session.overall_score >= 0.6 else \
                            "Necesita mejorar"
        
        return {
            "overall_assessment": overall_assessment,
            "strengths": all_strengths[:5],
            "areas_for_improvement": all_improvements[:5],
            "recommendation": "Continúa practicando y enfócate en las áreas de mejora identificadas"
        }
    
    def _get_session(self, user_id: str, session_id: str) -> Optional[InterviewSession]:
        """Obtener sesión por ID"""
        if user_id not in self.sessions:
            return None
        
        for session in self.sessions[user_id]:
            if session.id == session_id:
                return session
        
        return None




