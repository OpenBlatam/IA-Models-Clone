"""
Content Generator endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.content_generator import ContentGeneratorService

router = APIRouter()
content_service = ContentGeneratorService()


@router.post("/cover-letter")
async def generate_cover_letter(
    job_title: str,
    company: str,
    user_skills: List[str],
    user_experience: str
) -> Dict[str, Any]:
    """Generar carta de presentación"""
    try:
        cover_letter = content_service.generate_cover_letter(
            job_title, company, user_skills, user_experience
        )
        return {
            "content": cover_letter,
            "type": "cover_letter",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/linkedin-post")
async def generate_linkedin_post(
    achievement_type: str,
    achievement_details: Dict[str, Any]
) -> Dict[str, Any]:
    """Generar post de LinkedIn"""
    try:
        post = content_service.generate_linkedin_post(achievement_type, achievement_details)
        return {
            "content": post,
            "type": "linkedin_post",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/follow-up-email")
async def generate_follow_up_email(
    company: str,
    job_title: str,
    days_since_application: int
) -> Dict[str, Any]:
    """Generar email de follow-up"""
    try:
        email = content_service.generate_follow_up_email(company, job_title, days_since_application)
        return {
            "content": email,
            "type": "follow_up_email",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/improve-text")
async def improve_text(
    text: str,
    style: str = "professional"
) -> Dict[str, Any]:
    """Mejorar texto con IA"""
    try:
        improved = content_service.improve_text(text, style)
        return {
            "original": text,
            "improved": improved,
            "style": style,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




