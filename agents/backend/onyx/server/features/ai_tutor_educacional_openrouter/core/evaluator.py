"""
Automatic evaluation system for student responses and quizzes.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of an evaluation."""
    score: float
    max_score: float
    percentage: float
    feedback: str
    correct_answers: List[str]
    incorrect_answers: List[str]
    suggestions: List[str]
    evaluated_at: datetime


class AnswerEvaluator:
    """
    Automatically evaluates student answers and provides feedback.
    """
    
    def __init__(self):
        self.evaluation_history: Dict[str, List[EvaluationResult]] = {}
    
    def evaluate_answer(
        self,
        student_answer: str,
        correct_answer: str,
        question_type: str = "short_answer",
        tolerance: float = 0.8
    ) -> EvaluationResult:
        """
        Evaluate a student's answer against the correct answer.
        
        Args:
            student_answer: Student's answer
            correct_answer: Correct answer
            question_type: Type of question (multiple_choice, short_answer, true_false)
            tolerance: Similarity threshold for partial credit (0.0-1.0)
        
        Returns:
            Evaluation result with score and feedback
        """
        student_answer = student_answer.strip().lower()
        correct_answer = correct_answer.strip().lower()
        
        if question_type == "multiple_choice":
            score = 1.0 if student_answer == correct_answer else 0.0
            feedback = "Correcto" if score == 1.0 else "Incorrecto"
        
        elif question_type == "true_false":
            score = 1.0 if student_answer == correct_answer else 0.0
            feedback = "Correcto" if score == 1.0 else "Incorrecto"
        
        elif question_type == "short_answer":
            similarity = self._calculate_similarity(student_answer, correct_answer)
            score = 1.0 if similarity >= tolerance else similarity
            feedback = self._generate_feedback(similarity, student_answer, correct_answer)
        
        else:
            score = 0.0
            feedback = "Tipo de pregunta no soportado"
        
        percentage = score * 100
        suggestions = self._generate_suggestions(score, student_answer, correct_answer)
        
        result = EvaluationResult(
            score=score,
            max_score=1.0,
            percentage=percentage,
            feedback=feedback,
            correct_answers=[correct_answer] if score >= tolerance else [],
            incorrect_answers=[student_answer] if score < tolerance else [],
            suggestions=suggestions,
            evaluated_at=datetime.now()
        )
        
        return result
    
    def evaluate_quiz(
        self,
        student_answers: Dict[str, str],
        quiz_answers: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a complete quiz.
        
        Args:
            student_answers: Dict mapping question_id to student answer
            quiz_answers: Dict mapping question_id to correct answer and metadata
        
        Returns:
            Complete quiz evaluation
        """
        results = []
        total_score = 0.0
        max_score = len(quiz_answers)
        
        for question_id, student_answer in student_answers.items():
            if question_id not in quiz_answers:
                continue
            
            question_data = quiz_answers[question_id]
            correct_answer = question_data.get("correct_answer", "")
            question_type = question_data.get("type", "short_answer")
            
            result = self.evaluate_answer(
                student_answer=student_answer,
                correct_answer=correct_answer,
                question_type=question_type
            )
            
            results.append({
                "question_id": question_id,
                "score": result.score,
                "feedback": result.feedback,
                "suggestions": result.suggestions
            })
            
            total_score += result.score
        
        percentage = (total_score / max_score * 100) if max_score > 0 else 0.0
        
        overall_feedback = self._generate_overall_feedback(percentage)
        
        return {
            "total_score": total_score,
            "max_score": max_score,
            "percentage": percentage,
            "grade": self._calculate_grade(percentage),
            "overall_feedback": overall_feedback,
            "question_results": results,
            "evaluated_at": datetime.now().isoformat()
        }
    
    def _calculate_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate similarity between two answers."""
        # Simple word-based similarity
        words1 = set(re.findall(r'\w+', answer1))
        words2 = set(re.findall(r'\w+', answer2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Also check for exact substring match
        if answer1 in answer2 or answer2 in answer1:
            jaccard = max(jaccard, 0.9)
        
        return jaccard
    
    def _generate_feedback(self, similarity: float, student_answer: str, correct_answer: str) -> str:
        """Generate feedback based on similarity."""
        if similarity >= 0.9:
            return "Excelente respuesta. Muy cerca de la respuesta correcta."
        elif similarity >= 0.7:
            return "Buena respuesta, pero puedes mejorar algunos detalles."
        elif similarity >= 0.5:
            return "Estás en el camino correcto, pero falta información importante."
        else:
            return f"La respuesta correcta es: {correct_answer}. Revisa el concepto relacionado."
    
    def _generate_suggestions(self, score: float, student_answer: str, correct_answer: str) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if score < 0.7:
            suggestions.append("Revisa el material de estudio relacionado")
            suggestions.append("Practica más ejercicios similares")
        
        if score < 0.5:
            suggestions.append("Considera pedir ayuda al tutor")
            suggestions.append("Estudia los conceptos fundamentales primero")
        
        return suggestions
    
    def _generate_overall_feedback(self, percentage: float) -> str:
        """Generate overall feedback for quiz performance."""
        if percentage >= 90:
            return "¡Excelente trabajo! Dominas muy bien este tema."
        elif percentage >= 80:
            return "Muy buen trabajo. Tienes un buen entendimiento del tema."
        elif percentage >= 70:
            return "Buen trabajo. Hay algunas áreas que puedes mejorar."
        elif percentage >= 60:
            return "Aprobado. Necesitas practicar más para mejorar."
        else:
            return "Necesitas estudiar más este tema. Considera revisar los conceptos básicos."
    
    def _calculate_grade(self, percentage: float) -> str:
        """Calculate letter grade."""
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"
    
    def save_evaluation(self, student_id: str, evaluation: EvaluationResult):
        """Save evaluation to history."""
        if student_id not in self.evaluation_history:
            self.evaluation_history[student_id] = []
        self.evaluation_history[student_id].append(evaluation)
    
    def get_evaluation_history(self, student_id: str) -> List[EvaluationResult]:
        """Get evaluation history for a student."""
        return self.evaluation_history.get(student_id, [])






