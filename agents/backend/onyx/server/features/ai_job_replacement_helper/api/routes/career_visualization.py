"""
Career Visualization endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.career_visualization import CareerVisualizationService, CareerStage

router = APIRouter()
career_service = CareerVisualizationService()


@router.post("/create-path/{user_id}")
async def create_career_path(
    user_id: str,
    current_stage: str,
    target_role: Optional[str] = None,
    timeline_years: int = 5
) -> Dict[str, Any]:
    """Crear trayectoria profesional"""
    try:
        stage = CareerStage(current_stage)
        path = career_service.create_career_path(user_id, stage, target_role, timeline_years)
        return {
            "user_id": path.user_id,
            "current_stage": path.current_stage.value,
            "target_role": path.target_role,
            "timeline_years": path.timeline_years,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-milestone/{user_id}")
async def add_milestone(
    user_id: str,
    title: str,
    company: str,
    role: str,
    stage: str,
    start_date: str,
    end_date: Optional[str] = None,
    achievements: Optional[list] = None
) -> Dict[str, Any]:
    """Agregar hito a la trayectoria"""
    try:
        stage_enum = CareerStage(stage)
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date) if end_date else None
        
        milestone = career_service.add_milestone(
            user_id, title, company, role, stage_enum, start, end, achievements
        )
        return {
            "id": milestone.id,
            "title": milestone.title,
            "company": milestone.company,
            "role": milestone.role,
            "stage": milestone.stage.value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualize/{user_id}")
async def visualize_career_path(
    user_id: str,
    target_role: Optional[str] = None
) -> Dict[str, Any]:
    """Visualizar trayectoria hacia objetivo"""
    try:
        visualization = career_service.visualize_career_path(user_id, target_role)
        return {
            "current_position": visualization.current_position,
            "path_to_target": visualization.path_to_target,
            "estimated_timeline": visualization.estimated_timeline,
            "required_skills": visualization.required_skills,
            "recommended_steps": visualization.recommended_steps,
            "growth_rate": visualization.growth_rate,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




