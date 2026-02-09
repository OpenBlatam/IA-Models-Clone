"""
Skill Assessment endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.skill_assessment import SkillAssessmentService

router = APIRouter()
assessment_service = SkillAssessmentService()


@router.post("/create/{user_id}")
async def create_assessment(
    user_id: str,
    skill: str,
    num_questions: int = 10
) -> Dict[str, Any]:
    """Crear evaluación de habilidad"""
    try:
        assessment = assessment_service.create_assessment(user_id, skill, num_questions)
        return {
            "id": assessment.id,
            "skill": assessment.skill,
            "questions_count": len(assessment.questions),
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "options": q.options,
                    "difficulty": q.difficulty,
                }
                for q in assessment.questions
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer/{user_id}/{assessment_id}")
async def submit_answer(
    user_id: str,
    assessment_id: str,
    question_id: str,
    answer: int
) -> Dict[str, Any]:
    """Enviar respuesta"""
    try:
        assessment = assessment_service.submit_answer(user_id, assessment_id, question_id, answer)
        return {
            "assessment_id": assessment.id,
            "answered_questions": len(assessment.answers),
            "total_questions": len(assessment.questions),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complete/{user_id}/{assessment_id}")
async def complete_assessment(user_id: str, assessment_id: str) -> Dict[str, Any]:
    """Completar evaluación"""
    try:
        assessment = assessment_service.complete_assessment(user_id, assessment_id)
        return {
            "id": assessment.id,
            "score": assessment.score,
            "level": assessment.level.value if assessment.level else None,
            "completed_at": assessment.completed_at.isoformat() if assessment.completed_at else None,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




