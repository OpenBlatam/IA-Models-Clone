"""
Advanced Skill Gap Analysis endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.advanced_skill_gap import AdvancedSkillGapService

router = APIRouter()
skill_gap_service = AdvancedSkillGapService()


@router.post("/analyze/{user_id}")
async def analyze_skill_gaps(
    user_id: str,
    target_role: str,
    current_skills: Dict[str, float],
    required_skills: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Analizar brechas de habilidades"""
    try:
        analysis = skill_gap_service.analyze_skill_gaps(
            user_id, target_role, current_skills, required_skills
        )
        return {
            "user_id": analysis.user_id,
            "target_role": analysis.target_role,
            "gaps": [
                {
                    "skill": g.skill,
                    "current_level": g.current_level,
                    "required_level": g.required_level,
                    "gap_size": g.gap_size,
                    "priority": g.priority,
                    "learning_path": g.learning_path,
                    "estimated_time": g.estimated_time,
                    "resources": g.resources,
                }
                for g in analysis.gaps
            ],
            "overall_gap_score": analysis.overall_gap_score,
            "readiness_score": analysis.readiness_score,
            "recommendations": analysis.recommendations,
            "learning_plan": analysis.learning_plan,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track-progress/{user_id}")
async def track_progress(
    user_id: str,
    skill: str,
    new_level: float
) -> Dict[str, Any]:
    """Rastrear progreso en una habilidad"""
    try:
        result = skill_gap_service.track_progress(user_id, skill, new_level)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




