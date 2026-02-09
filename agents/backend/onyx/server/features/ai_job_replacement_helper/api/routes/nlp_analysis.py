"""
NLP Analysis endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.nlp_analysis import NLPAnalysisService

router = APIRouter()
nlp_service = NLPAnalysisService()


@router.post("/analyze-text")
async def analyze_text(
    text: str,
    language: Optional[str] = None
) -> Dict[str, Any]:
    """Analizar texto"""
    try:
        analysis = nlp_service.analyze_text(text, language)
        return {
            "sentiment": analysis.sentiment,
            "sentiment_score": analysis.sentiment_score,
            "entities": analysis.entities,
            "keywords": analysis.keywords,
            "topics": analysis.topics,
            "summary": analysis.summary,
            "readability_score": analysis.readability_score,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-cv")
async def analyze_cv(cv_text: str) -> Dict[str, Any]:
    """Analizar CV con NLP"""
    try:
        analysis = nlp_service.analyze_cv(cv_text)
        return {
            "skills_extracted": analysis.skills_extracted,
            "experience_years": analysis.experience_years,
            "education_level": analysis.education_level,
            "certifications": analysis.certifications,
            "languages": analysis.languages,
            "summary": analysis.summary,
            "keywords_density": analysis.keywords_density,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/match-cv-job")
async def match_cv_to_job(
    cv_text: str,
    job_description: str
) -> Dict[str, Any]:
    """Hacer match de CV con descripción de trabajo"""
    try:
        match = nlp_service.match_cv_to_job(cv_text, job_description)
        return match
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

