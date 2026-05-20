"""
Skill Assessments endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.skill_assessments import SkillAssessmentsService

router = APIRouter()
assessments_service = SkillAssessmentsService()


@router.post("/create")
async def create_assessment(
    title: str,
    skill: str,
    description: str,
    questions: List[Dict[str, Any]],
    passing_score: float = 0.7,
    time_limit_minutes: Optional[int] = None
) -> Dict[str, Any]:
    """Crear nuevo assessment"""
    try:
        assessment = assessments_service.create_assessment(
            title, skill, description, questions, passing_score, time_limit_minutes
        )
        return {
            "id": assessment.id,
            "title": assessment.title,
            "skill": assessment.skill,
            "questions_count": len(assessment.questions),
            "passing_score": assessment.passing_score,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/take/{assessment_id}")
async def take_assessment(
    assessment_id: str,
    user_id: str,
    answers: List[Dict[str, Any]],
    time_taken_minutes: Optional[int] = None
) -> Dict[str, Any]:
    """Tomar assessment"""
    try:
        result = assessments_service.take_assessment(
            assessment_id, user_id, answers, time_taken_minutes
        )
        return {
            "assessment_id": result.assessment_id,
            "user_id": result.user_id,
            "score": result.score,
            "total_points": result.total_points,
            "earned_points": result.earned_points,
            "passed": result.passed,
            "feedback": result.feedback,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}")
async def get_user_assessments(user_id: str) -> Dict[str, Any]:
    """Obtener assessments del usuario"""
    try:
        assessments = assessments_service.get_user_assessments(user_id)
        return {
            "user_id": user_id,
            "assessments": assessments,
            "total": len(assessments),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/{assessment_id}")
async def get_assessment_statistics(assessment_id: str) -> Dict[str, Any]:
    """Obtener estadísticas de un assessment"""
    try:
        stats = assessments_service.get_assessment_statistics(assessment_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




