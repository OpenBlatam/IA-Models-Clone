"""
Skill Assessments Service - Sistema de evaluaciones de habilidades
===================================================================

Sistema completo de quizzes y assessments para evaluar habilidades.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    """Tipos de preguntas"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    CODING = "coding"
    ESSAY = "essay"
    PRACTICAL = "practical"


@dataclass
class Question:
    """Pregunta de assessment"""
    id: str
    question_text: str
    question_type: QuestionType
    options: List[str] = field(default_factory=list)
    correct_answer: Any = None
    points: int = 1
    difficulty: str = "medium"  # easy, medium, hard
    explanation: Optional[str] = None


@dataclass
class Assessment:
    """Assessment completo"""
    id: str
    title: str
    skill: str
    description: str
    questions: List[Question]
    passing_score: float = 0.7
    time_limit_minutes: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AssessmentResult:
    """Resultado de assessment"""
    assessment_id: str
    user_id: str
    score: float
    total_points: int
    earned_points: int
    passed: bool
    answers: List[Dict[str, Any]]
    completed_at: datetime = field(default_factory=datetime.now)
    time_taken_minutes: Optional[int] = None
    feedback: List[str] = field(default_factory=list)


class SkillAssessmentsService:
    """Servicio de evaluaciones de habilidades"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.assessments: Dict[str, Assessment] = {}
        self.results: Dict[str, List[AssessmentResult]] = {}  # user_id -> results
        logger.info("SkillAssessmentsService initialized")
    
    def create_assessment(
        self,
        title: str,
        skill: str,
        description: str,
        questions: List[Dict[str, Any]],
        passing_score: float = 0.7,
        time_limit_minutes: Optional[int] = None
    ) -> Assessment:
        """Crear nuevo assessment"""
        assessment_id = f"assessment_{int(datetime.now().timestamp())}"
        
        # Convertir preguntas
        question_objects = []
        for q_data in questions:
            question = Question(
                id=f"q_{len(question_objects)}",
                question_text=q_data.get("question_text", ""),
                question_type=QuestionType(q_data.get("question_type", "multiple_choice")),
                options=q_data.get("options", []),
                correct_answer=q_data.get("correct_answer"),
                points=q_data.get("points", 1),
                difficulty=q_data.get("difficulty", "medium"),
                explanation=q_data.get("explanation"),
            )
            question_objects.append(question)
        
        assessment = Assessment(
            id=assessment_id,
            title=title,
            skill=skill,
            description=description,
            questions=question_objects,
            passing_score=passing_score,
            time_limit_minutes=time_limit_minutes,
        )
        
        self.assessments[assessment_id] = assessment
        
        logger.info(f"Assessment created: {assessment_id}")
        return assessment
    
    def take_assessment(
        self,
        assessment_id: str,
        user_id: str,
        answers: List[Dict[str, Any]],
        time_taken_minutes: Optional[int] = None
    ) -> AssessmentResult:
        """Tomar assessment"""
        assessment = self.assessments.get(assessment_id)
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found")
        
        # Calcular score
        total_points = sum(q.points for q in assessment.questions)
        earned_points = 0
        
        answer_details = []
        for i, answer in enumerate(answers):
            question = assessment.questions[i] if i < len(assessment.questions) else None
            if not question:
                continue
            
            is_correct = self._check_answer(question, answer.get("answer"))
            if is_correct:
                earned_points += question.points
            
            answer_details.append({
                "question_id": question.id,
                "user_answer": answer.get("answer"),
                "correct_answer": question.correct_answer,
                "is_correct": is_correct,
                "points_earned": question.points if is_correct else 0,
            })
        
        score = earned_points / total_points if total_points > 0 else 0.0
        passed = score >= assessment.passing_score
        
        # Generar feedback
        feedback = self._generate_feedback(assessment, answer_details, score, passed)
        
        result = AssessmentResult(
            assessment_id=assessment_id,
            user_id=user_id,
            score=score,
            total_points=total_points,
            earned_points=earned_points,
            passed=passed,
            answers=answer_details,
            time_taken_minutes=time_taken_minutes,
            feedback=feedback,
        )
        
        if user_id not in self.results:
            self.results[user_id] = []
        self.results[user_id].append(result)
        
        logger.info(f"Assessment completed: {assessment_id} by {user_id}")
        return result
    
    def _check_answer(self, question: Question, user_answer: Any) -> bool:
        """Verificar respuesta"""
        if question.question_type == QuestionType.MULTIPLE_CHOICE:
            return str(user_answer).lower() == str(question.correct_answer).lower()
        elif question.question_type == QuestionType.TRUE_FALSE:
            return bool(user_answer) == bool(question.correct_answer)
        elif question.question_type == QuestionType.CODING:
            # En producción, esto ejecutaría y compararía código
            return str(user_answer).strip() == str(question.correct_answer).strip()
        else:
            # Para essay y practical, requeriría evaluación manual o IA
            return False
    
    def _generate_feedback(
        self,
        assessment: Assessment,
        answers: List[Dict[str, Any]],
        score: float,
        passed: bool
    ) -> List[str]:
        """Generar feedback"""
        feedback = []
        
        if passed:
            feedback.append(f"¡Felicitaciones! Pasaste el assessment con un {score*100:.0f}%")
        else:
            feedback.append(f"Obtuviste un {score*100:.0f}%. Necesitas {assessment.passing_score*100:.0f}% para pasar")
        
        # Análisis por dificultad
        correct_by_difficulty = {"easy": 0, "medium": 0, "hard": 0}
        total_by_difficulty = {"easy": 0, "medium": 0, "hard": 0}
        
        for i, answer in enumerate(answers):
            if i < len(assessment.questions):
                question = assessment.questions[i]
                total_by_difficulty[question.difficulty] += 1
                if answer["is_correct"]:
                    correct_by_difficulty[question.difficulty] += 1
        
        for difficulty in ["easy", "medium", "hard"]:
            if total_by_difficulty[difficulty] > 0:
                accuracy = correct_by_difficulty[difficulty] / total_by_difficulty[difficulty]
                if accuracy < 0.7:
                    feedback.append(
                        f"Considera practicar más preguntas de dificultad {difficulty}"
                    )
        
        return feedback
    
    def get_user_assessments(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtener assessments del usuario"""
        results = self.results.get(user_id, [])
        
        return [
            {
                "assessment_id": r.assessment_id,
                "score": r.score,
                "passed": r.passed,
                "completed_at": r.completed_at.isoformat(),
            }
            for r in results
        ]
    
    def get_assessment_statistics(self, assessment_id: str) -> Dict[str, Any]:
        """Obtener estadísticas de un assessment"""
        all_results = []
        for user_results in self.results.values():
            all_results.extend([r for r in user_results if r.assessment_id == assessment_id])
        
        if not all_results:
            return {"assessment_id": assessment_id, "total_taken": 0}
        
        scores = [r.score for r in all_results]
        
        return {
            "assessment_id": assessment_id,
            "total_taken": len(all_results),
            "average_score": sum(scores) / len(scores),
            "pass_rate": sum(1 for r in all_results if r.passed) / len(all_results),
            "min_score": min(scores),
            "max_score": max(scores),
        }




