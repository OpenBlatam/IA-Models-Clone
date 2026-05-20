"""
ML Recommendations endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.ml_recommendations import MLRecommendationsService, SalaryOffer

router = APIRouter()
ml_service = MLRecommendationsService()


@router.post("/train/{user_id}")
async def train_user_model(
    user_id: str,
    interactions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Entrenar modelo personalizado para usuario"""
    try:
        result = ml_service.train_user_model(user_id, interactions)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend-jobs/{user_id}")
async def recommend_jobs_ml(
    user_id: str,
    job_pool: List[Dict[str, Any]],
    top_k: int = 10
) -> Dict[str, Any]:
    """Recomendar trabajos usando ML"""
    try:
        recommendations = ml_service.recommend_jobs_ml(user_id, job_pool, top_k)
        return {
            "recommendations": [
                {
                    "item_id": r.item_id,
                    "item_type": r.item_type,
                    "score": r.score,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "features": r.features,
                }
                for r in recommendations
            ],
            "total": len(recommendations),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-profile/{user_id}")
async def update_user_profile(
    user_id: str,
    profile_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Actualizar perfil de usuario"""
    try:
        ml_service.update_user_profile(user_id, profile_data)
        return {"status": "updated", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




