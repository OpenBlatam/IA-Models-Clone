"""
Advanced assessment system for comprehensive evaluation.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AssessmentType(Enum):
    """Types of assessments."""
    FORMATIVE = "formative"
    SUMMATIVE = "summative"
    DIAGNOSTIC = "diagnostic"
    PLACEMENT = "placement"
    PERFORMANCE = "performance"


class QuestionType(Enum):
    """Types of questions."""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    MATCHING = "matching"
    FILL_BLANK = "fill_blank"


class AssessmentSystem:
    """
    Comprehensive assessment system for educational evaluation.
    """
    
    def __init__(self):
        self.assessments: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_assessment(
        self,
        assessment_name: str,
        assessment_type: AssessmentType,
        subject: str,
        topic: str,
        questions: List[Dict[str, Any]],
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new assessment.
        
        Args:
            assessment_name: Name of the assessment
            assessment_type: Type of assessment
            subject: Subject area
            topic: Topic covered
            questions: List of questions
            settings: Assessment settings
        
        Returns:
            Assessment information
        """
        assessment_id = f"assessment_{datetime.now().timestamp()}"
        
        assessment = {
            "assessment_id": assessment_id,
            "assessment_name": assessment_name,
            "assessment_type": assessment_type.value,
            "subject": subject,
            "topic": topic,
            "questions": questions,
            "total_points": sum(q.get("points", 1) for q in questions),
            "settings": settings or {
                "time_limit": None,
                "allow_retake": True,
                "show_results": True,
                "randomize_questions": False
            },
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        self.assessments[assessment_id] = assessment
        logger.info(f"Created assessment {assessment_name} ({assessment_id})")
        
        return assessment
    
    def take_assessment(
        self,
        assessment_id: str,
        student_id: str,
        answers: Dict[str, Any],
        time_taken: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Submit assessment answers for grading.
        
        Args:
            assessment_id: Assessment identifier
            student_id: Student identifier
            answers: Student answers
            time_taken: Time taken in seconds
        
        Returns:
            Assessment results
        """
        if assessment_id not in self.assessments:
            return {"error": "Assessment not found"}
        
        assessment = self.assessments[assessment_id]
        
        # Grade assessment
        results = self._grade_assessment(assessment, answers)
        
        # Calculate score
        total_points = assessment["total_points"]
        earned_points = sum(r["points_earned"] for r in results["question_results"])
        percentage = (earned_points / total_points * 100) if total_points > 0 else 0
        
        # Determine grade
        grade = self._calculate_grade(percentage)
        
        result = {
            "assessment_id": assessment_id,
            "student_id": student_id,
            "assessment_name": assessment["assessment_name"],
            "earned_points": earned_points,
            "total_points": total_points,
            "percentage": percentage,
            "grade": grade,
            "question_results": results["question_results"],
            "time_taken": time_taken,
            "submitted_at": datetime.now().isoformat()
        }
        
        # Store result
        if student_id not in self.results:
            self.results[student_id] = []
        self.results[student_id].append(result)
        
        logger.info(f"Student {student_id} completed assessment {assessment_id}: {percentage}%")
        
        return result
    
    def _grade_assessment(
        self,
        assessment: Dict[str, Any],
        answers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Grade assessment answers."""
        question_results = []
        
        for question in assessment["questions"]:
            question_id = question.get("question_id", str(question.get("id", "")))
            student_answer = answers.get(question_id)
            
            # Grade based on question type
            if question.get("type") == QuestionType.MULTIPLE_CHOICE.value:
                is_correct = student_answer == question.get("correct_answer")
            elif question.get("type") == QuestionType.TRUE_FALSE.value:
                is_correct = student_answer == question.get("correct_answer")
            elif question.get("type") == QuestionType.SHORT_ANSWER.value:
                is_correct = self._grade_short_answer(
                    student_answer,
                    question.get("correct_answer")
                )
            elif question.get("type") == QuestionType.ESSAY.value:
                is_correct = None  # Requires manual grading
            else:
                is_correct = student_answer == question.get("correct_answer")
            
            points_earned = question.get("points", 1) if is_correct else 0
            
            question_results.append({
                "question_id": question_id,
                "question": question.get("question"),
                "student_answer": student_answer,
                "correct_answer": question.get("correct_answer"),
                "is_correct": is_correct,
                "points_earned": points_earned,
                "max_points": question.get("points", 1)
            })
        
        return {"question_results": question_results}
    
    def _grade_short_answer(
        self,
        student_answer: str,
        correct_answer: str
    ) -> bool:
        """Grade short answer question (simple keyword matching)."""
        if not student_answer or not correct_answer:
            return False
        
        student_lower = student_answer.lower().strip()
        correct_lower = correct_answer.lower().strip()
        
        # Exact match
        if student_lower == correct_lower:
            return True
        
        # Keyword matching (simple version)
        correct_keywords = set(correct_lower.split())
        student_keywords = set(student_lower.split())
        
        # If 80% of keywords match, consider correct
        if len(correct_keywords) > 0:
            match_ratio = len(student_keywords & correct_keywords) / len(correct_keywords)
            return match_ratio >= 0.8
        
        return False
    
    def _calculate_grade(self, percentage: float) -> str:
        """Calculate letter grade from percentage."""
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
    
    def get_student_progress(
        self,
        student_id: str,
        subject: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get student assessment progress.
        
        Args:
            student_id: Student identifier
            subject: Optional subject filter
        
        Returns:
            Progress summary
        """
        if student_id not in self.results:
            return {"error": "No assessments found for student"}
        
        student_results = self.results[student_id]
        
        if subject:
            student_results = [
                r for r in student_results
                if self.assessments.get(r["assessment_id"], {}).get("subject") == subject
            ]
        
        if not student_results:
            return {"error": "No assessments found"}
        
        total_assessments = len(student_results)
        avg_percentage = sum(r["percentage"] for r in student_results) / total_assessments
        avg_grade = self._calculate_grade(avg_percentage)
        
        # Grade distribution
        grade_distribution = {}
        for result in student_results:
            grade = result["grade"]
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
        
        return {
            "student_id": student_id,
            "total_assessments": total_assessments,
            "average_percentage": avg_percentage,
            "average_grade": avg_grade,
            "grade_distribution": grade_distribution,
            "recent_results": student_results[-5:]
        }
    
    def get_assessment_statistics(
        self,
        assessment_id: str
    ) -> Dict[str, Any]:
        """
        Get statistics for an assessment.
        
        Args:
            assessment_id: Assessment identifier
        
        Returns:
            Assessment statistics
        """
        if assessment_id not in self.assessments:
            return {"error": "Assessment not found"}
        
        # Find all results for this assessment
        all_results = []
        for student_results in self.results.values():
            for result in student_results:
                if result["assessment_id"] == assessment_id:
                    all_results.append(result)
        
        if not all_results:
            return {"error": "No results found for assessment"}
        
        percentages = [r["percentage"] for r in all_results]
        
        return {
            "assessment_id": assessment_id,
            "total_completions": len(all_results),
            "average_score": sum(percentages) / len(percentages),
            "highest_score": max(percentages),
            "lowest_score": min(percentages),
            "pass_rate": len([p for p in percentages if p >= 60]) / len(percentages) * 100
        }




