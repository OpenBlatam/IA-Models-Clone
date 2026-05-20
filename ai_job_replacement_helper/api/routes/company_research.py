"""
Company Research endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.company_research import CompanyResearchService

router = APIRouter()
research_service = CompanyResearchService()


@router.get("/research/{company_name}")
async def research_company(company_name: str) -> Dict[str, Any]:
    """Investigar empresa"""
    try:
        profile = research_service.research_company(company_name)
        return {
            "company_id": profile.company_id,
            "name": profile.name,
            "industry": profile.industry,
            "size": profile.size,
            "location": profile.location,
            "culture": profile.culture,
            "benefits": profile.benefits,
            "tech_stack": profile.tech_stack,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prepare/{company_id}")
async def prepare_for_interview(
    company_id: str,
    job_title: str
) -> Dict[str, Any]:
    """Preparar para entrevista específica"""
    try:
        prep = research_service.prepare_for_interview(company_id, job_title)
        return {
            "company_id": prep.company_id,
            "job_title": prep.job_title,
            "key_points": prep.key_points,
            "questions_to_ask": prep.questions_to_ask,
            "red_flags": prep.red_flags,
            "talking_points": prep.talking_points,
            "research_summary": prep.research_summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_companies(company_ids: List[str]) -> Dict[str, Any]:
    """Comparar múltiples empresas"""
    try:
        comparison = research_service.compare_companies(company_ids)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




