"""
Advanced AI Content endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.advanced_ai_content import AdvancedAIContentService

router = APIRouter()
ai_content_service = AdvancedAIContentService()


@router.post("/cover-letter")
async def generate_cover_letter_advanced(
    job_title: str,
    company: str,
    user_profile: Dict[str, Any],
    job_description: str
) -> Dict[str, Any]:
    """Generar carta de presentación avanzada"""
    try:
        content = await ai_content_service.generate_cover_letter_advanced(
            job_title, company, user_profile, job_description
        )
        return {
            "content": content.content,
            "content_type": content.content_type,
            "metadata": content.metadata,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/linkedin-post")
async def generate_linkedin_post_advanced(
    achievement_type: str,
    achievement_data: Dict[str, Any],
    style: str = "professional"
) -> Dict[str, Any]:
    """Generar post de LinkedIn avanzado"""
    try:
        content = await ai_content_service.generate_linkedin_post_advanced(
            achievement_type, achievement_data, style
        )
        return {
            "content": content.content,
            "content_type": content.content_type,
            "metadata": content.metadata,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/improve-text")
async def improve_text_with_ai(
    text: str,
    improvement_type: str,
    target_style: str = "professional"
) -> Dict[str, Any]:
    """Mejorar texto con IA"""
    try:
        content = await ai_content_service.improve_text_with_ai(
            text, improvement_type, target_style
        )
        return {
            "original": content.metadata.get("original"),
            "improved": content.content,
            "improvement_type": improvement_type,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interview-prep")
async def generate_interview_prep(
    job_title: str,
    company: str,
    job_description: str,
    user_skills: List[str]
) -> Dict[str, Any]:
    """Generar preparación para entrevista"""
    try:
        prep = await ai_content_service.generate_interview_prep(
            job_title, company, job_description, user_skills
        )
        return prep
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




