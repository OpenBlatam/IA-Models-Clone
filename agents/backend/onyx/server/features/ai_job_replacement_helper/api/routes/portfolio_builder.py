"""
Portfolio Builder endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.portfolio_builder import PortfolioBuilderService, ProjectType

router = APIRouter()
portfolio_service = PortfolioBuilderService()


@router.post("/create/{user_id}")
async def create_portfolio(
    user_id: str,
    bio: str,
    skills: List[str],
    contact_info: Dict[str, str]
) -> Dict[str, Any]:
    """Crear portafolio"""
    try:
        portfolio = portfolio_service.create_portfolio(user_id, bio, skills, contact_info)
        return {
            "user_id": portfolio.user_id,
            "bio": portfolio.bio,
            "skills": portfolio.skills,
            "projects_count": len(portfolio.projects),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-project/{user_id}")
async def add_project(
    user_id: str,
    title: str,
    description: str,
    project_type: str,
    technologies: List[str],
    github_url: Optional[str] = None,
    live_url: Optional[str] = None,
    featured: bool = False
) -> Dict[str, Any]:
    """Agregar proyecto al portafolio"""
    try:
        project_type_enum = ProjectType(project_type)
        project = portfolio_service.add_project(
            user_id, title, description, project_type_enum,
            technologies, github_url, live_url, featured
        )
        return {
            "id": project.id,
            "title": project.title,
            "type": project.project_type.value,
            "technologies": project.technologies,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/html/{user_id}")
async def get_portfolio_html(user_id: str) -> Dict[str, Any]:
    """Generar HTML del portafolio"""
    try:
        html = portfolio_service.generate_portfolio_html(user_id)
        return {"html": html}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{user_id}")
async def analyze_portfolio(user_id: str) -> Dict[str, Any]:
    """Analizar portafolio y dar recomendaciones"""
    try:
        analysis = portfolio_service.analyze_portfolio(user_id)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{user_id}")
async def export_portfolio(
    user_id: str,
    format: str = "json"
) -> Dict[str, Any]:
    """Exportar portafolio"""
    try:
        exported = portfolio_service.export_portfolio(user_id, format)
        return exported
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




