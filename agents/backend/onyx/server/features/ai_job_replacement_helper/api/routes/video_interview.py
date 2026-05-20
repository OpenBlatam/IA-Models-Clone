"""
Video Interview endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.video_interview import VideoInterviewService, InterviewMode

router = APIRouter()
video_service = VideoInterviewService()


@router.post("/start/{user_id}")
async def start_video_interview(
    user_id: str,
    job_title: str,
    company: str,
    mode: str = "practice"
) -> Dict[str, Any]:
    """Iniciar entrevista por video"""
    try:
        interview_mode = InterviewMode(mode)
        session = video_service.start_video_interview(user_id, job_title, company, interview_mode)
        return {
            "session_id": session.id,
            "job_title": session.job_title,
            "company": session.company,
            "mode": session.mode.value,
            "questions": session.questions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/{session_id}")
async def analyze_video_response(
    session_id: str,
    question_id: str,
    video_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Analizar respuesta de video"""
    try:
        analysis = video_service.analyze_video_response(session_id, question_id, video_data)
        return {
            "eye_contact_score": analysis.eye_contact_score,
            "posture_score": analysis.posture_score,
            "energy_level": analysis.energy_level,
            "speech_clarity": analysis.speech_clarity,
            "filler_words_count": analysis.filler_words_count,
            "overall_score": analysis.overall_score,
            "feedback": analysis.feedback,
            "improvements": analysis.improvements,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complete/{session_id}")
async def complete_interview(session_id: str) -> Dict[str, Any]:
    """Completar entrevista y generar reporte final"""
    try:
        result = video_service.complete_interview(session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




