"""
Skill Assessment Service - Evaluación de habilidades
====================================================

Sistema para evaluar y certificar habilidades del usuario.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class SkillLevel(str, Enum):
    """Niveles de habilidad"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class AssessmentQuestion:
    """Pregunta de evaluación"""
    id: str
    question: str
    skill: str
    difficulty: str
    options: List[str]
    correct_answer: int
    explanation: str


@dataclass
class SkillAssessment:
    """Evaluación de habilidad"""
    id: str
    user_id: str
    skill: str
    questions: List[AssessmentQuestion]
    answers: Dict[str, int] = field(default_factory=dict)
    score: Optional[float] = None
    level: Optional[SkillLevel] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)


class SkillAssessmentService:
    """Servicio de evaluación de habilidades"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.assessments: Dict[str, List[SkillAssessment]] = {}  # user_id -> [assessments]
        self.question_bank = self._initialize_question_bank()
        logger.info("SkillAssessmentService initialized")
    
    def _initialize_question_bank(self) -> Dict[str, List[AssessmentQuestion]]:
        """Inicializar banco de preguntas"""
        bank = {}
        
        # Preguntas de Python
        bank["Python"] = [
            AssessmentQuestion(
                id="py_1",
                question="¿Qué es una lista en Python?",
                skill="Python",
                difficulty="beginner",
                options=[
                    "Una colección ordenada y mutable",
                    "Una colección desordenada",
                    "Un tipo de dato primitivo",
                    "Una función",
                ],
                correct_answer=0,
                explanation="Una lista es una colección ordenada y mutable de elementos.",
            ),
            AssessmentQuestion(
                id="py_2",
                question="¿Cuál es la diferencia entre '==' y 'is' en Python?",
                skill="Python",
                difficulty="intermediate",
                options=[
                    "'==' compara valores, 'is' compara identidad",
                    "Son iguales",
                    "'is' compara valores, '==' compara identidad",
                    "Ninguna diferencia",
                ],
                correct_answer=0,
                explanation="'==' compara si los valores son iguales, 'is' compara si son el mismo objeto.",
            ),
        ]
        
        return bank
    
    def create_assessment(
        self,
        user_id: str,
        skill: str,
        num_questions: int = 10
    ) -> SkillAssessment:
        """Crear evaluación"""
        questions = self.question_bank.get(skill, [])[:num_questions]
        
        if not questions:
            # Crear preguntas genéricas
            questions = [
                AssessmentQuestion(
                    id=f"q_{i+1}",
                    question=f"Pregunta {i+1} sobre {skill}",
                    skill=skill,
                    difficulty="intermediate",
                    options=["Opción A", "Opción B", "Opción C", "Opción D"],
                    correct_answer=0,
                    explanation="Explicación de la respuesta correcta",
                )
                for i in range(num_questions)
            ]
        
        assessment = SkillAssessment(
            id=f"assessment_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            skill=skill,
            questions=questions,
        )
        
        if user_id not in self.assessments:
            self.assessments[user_id] = []
        
        self.assessments[user_id].append(assessment)
        
        logger.info(f"Assessment created for user {user_id}: {skill}")
        return assessment
    
    def submit_answer(
        self,
        user_id: str,
        assessment_id: str,
        question_id: str,
        answer: int
    ) -> SkillAssessment:
        """Enviar respuesta"""
        assessment = self._get_assessment(user_id, assessment_id)
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found")
        
        assessment.answers[question_id] = answer
        return assessment
    
    def complete_assessment(
        self,
        user_id: str,
        assessment_id: str
    ) -> SkillAssessment:
        """Completar evaluación y calcular score"""
        assessment = self._get_assessment(user_id, assessment_id)
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found")
        
        # Calcular score
        correct = 0
        total = len(assessment.questions)
        
        for question in assessment.questions:
            user_answer = assessment.answers.get(question.id)
            if user_answer == question.correct_answer:
                correct += 1
        
        assessment.score = (correct / total * 100) if total > 0 else 0
        assessment.completed_at = datetime.now()
        
        # Determinar nivel
        if assessment.score >= 90:
            assessment.level = SkillLevel.EXPERT
        elif assessment.score >= 75:
            assessment.level = SkillLevel.ADVANCED
        elif assessment.score >= 60:
            assessment.level = SkillLevel.INTERMEDIATE
        else:
            assessment.level = SkillLevel.BEGINNER
        
        logger.info(f"Assessment completed: score={assessment.score:.1f}%, level={assessment.level.value}")
        return assessment
    
    def get_user_assessments(self, user_id: str) -> List[SkillAssessment]:
        """Obtener evaluaciones del usuario"""
        return self.assessments.get(user_id, [])
    
    def _get_assessment(self, user_id: str, assessment_id: str) -> Optional[SkillAssessment]:
        """Obtener evaluación por ID"""
        assessments = self.assessments.get(user_id, [])
        return next((a for a in assessments if a.id == assessment_id), None)




