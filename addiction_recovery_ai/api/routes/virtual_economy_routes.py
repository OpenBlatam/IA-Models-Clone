"""
Virtual economy routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.virtual_economy_service import VirtualEconomyService
except ImportError:
    from ...services.virtual_economy_service import VirtualEconomyService

router = APIRouter()

virtual_economy = VirtualEconomyService()


@router.post("/economy/earn-points")
async def earn_points(
    user_id: str = Body(...),
    action_type: str = Body(...),
    amount: int = Body(...),
    description: str = Body(...)
):
    """Otorga puntos al usuario"""
    try:
        transaction = virtual_economy.earn_points(user_id, action_type, amount, description)
        return JSONResponse(content=transaction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error otorgando puntos: {str(e)}")


@router.get("/economy/balance/{user_id}")
async def get_user_balance(user_id: str):
    """Obtiene balance de puntos del usuario"""
    try:
        balance = virtual_economy.get_user_balance(user_id)
        return JSONResponse(content=balance)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo balance: {str(e)}")


@router.get("/economy/rewards")
async def get_rewards_catalog(
    category: Optional[str] = Query(None),
    max_points: Optional[int] = Query(None)
):
    """Obtiene catálogo de recompensas"""
    try:
        rewards = virtual_economy.get_rewards_catalog(category, max_points)
        return JSONResponse(content={
            "rewards": rewards,
            "total": len(rewards),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo recompensas: {str(e)}")



