"""
Interview Simulator endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.interview_simulator import InterviewSimulatorService, InterviewType

router = APIRouter()
interview_service = InterviewSimulatorService()


@router.post("/start/{user_id}")
async def start_interview(
    user_id: str,
    interview_type: str,
    job_title: Optional[str] = None,
    company: Optional[str] = None,
    num_questions: int = 5
) -> Dict[str, Any]:
    """Iniciar entrevista simulada"""
    try:
        interview_type_enum = InterviewType(interview_type)
        session = interview_service.start_interview(
            user_id, interview_type_enum, job_title, company, num_questions
        )
        
        return {
            "session_id": session.id,
            "interview_type": session.interview_type.value,
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "category": q.category,
                    "difficulty": q.difficulty.value,
                    "tips": q.tips,
                }
                for q in session.questions
            ],
            "started_at": session.started_at.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer/{user_id}/{session_id}")
async def submit_answer(
    user_id: str,
    session_id: str,
    question_id: str,
    answer: str,
    duration_seconds: Optional[int] = None
) -> Dict[str, Any]:
    """Enviar respuesta a pregunta"""
    try:
        feedback = interview_service.submit_answer(
            user_id, session_id, question_id, answer, duration_seconds
        )
        return {
            "question_id": question_id,
            "score": feedback.score,
            "strengths": feedback.strengths,
            "improvements": feedback.improvements,
            "keyword_match": feedback.keyword_match,
            "suggestions": feedback.suggestions,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complete/{user_id}/{session_id}")
async def complete_interview(
    user_id: str,
    session_id: str
) -> Dict[str, Any]:
    """Completar entrevista y obtener resultados"""
    try:
        results = interview_service.complete_interview(user_id, session_id)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




