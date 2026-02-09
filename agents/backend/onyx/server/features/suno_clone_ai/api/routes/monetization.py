"""
API de Monetización

Endpoints para:
- Suscripciones
- Pagos
- Créditos
- Estadísticas de ingresos
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body

from services.monetization import (
    get_monetization_service,
    SubscriptionTier
)
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/monetization",
    tags=["monetization"]
)


@router.post("/subscriptions")
async def create_subscription(
    tier: str = Body(..., description="Nivel de suscripción"),
    duration_days: int = Body(30, description="Duración en días"),
    auto_renew: bool = Body(True, description="Auto-renovación"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Crea una suscripción.
    """
    try:
        try:
            tier_enum = SubscriptionTier(tier)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid tier: {tier}"
            )
        
        user_id = current_user.get("user_id", "unknown")
        service = get_monetization_service()
        subscription = service.create_subscription(
            user_id=user_id,
            tier=tier_enum,
            duration_days=duration_days,
            auto_renew=auto_renew
        )
        
        return {
            "user_id": subscription.user_id,
            "tier": subscription.tier.value,
            "start_date": subscription.start_date.isoformat(),
            "end_date": subscription.end_date.isoformat() if subscription.end_date else None,
            "status": subscription.status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating subscription: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating subscription: {str(e)}"
        )


@router.get("/subscriptions/me")
async def get_my_subscription(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Obtiene la suscripción del usuario actual.
    """
    try:
        user_id = current_user.get("user_id", "unknown")
        service = get_monetization_service()
        subscription = service.get_user_subscription(user_id)
        
        if not subscription:
            return {"subscription": None}
        
        return {
            "subscription": {
                "tier": subscription.tier.value,
                "start_date": subscription.start_date.isoformat(),
                "end_date": subscription.end_date.isoformat() if subscription.end_date else None,
                "status": subscription.status
            }
        }
    except Exception as e:
        logger.error(f"Error getting subscription: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving subscription: {str(e)}"
        )


@router.post("/credits/add")
async def add_credits(
    amount: int = Body(..., description="Cantidad de créditos"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Agrega créditos a un usuario.
    """
    try:
        user_id = current_user.get("user_id", "unknown")
        service = get_monetization_service()
        
        transaction = service.add_credits(
            user_id=user_id,
            amount=amount,
            transaction_type="purchased"
        )
        
        return {
            "transaction_id": transaction.transaction_id,
            "amount": amount,
            "new_balance": service.get_user_credits(user_id)
        }
    except Exception as e:
        logger.error(f"Error adding credits: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding credits: {str(e)}"
        )


@router.get("/credits/balance")
async def get_credits_balance(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Obtiene el balance de créditos del usuario.
    """
    try:
        user_id = current_user.get("user_id", "unknown")
        service = get_monetization_service()
        balance = service.get_user_credits(user_id)
        
        return {"balance": balance}
    except Exception as e:
        logger.error(f"Error getting credits: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving credits: {str(e)}"
        )


@router.get("/revenue/stats")
async def get_revenue_stats(
    start_date: Optional[str] = Query(None, description="Fecha de inicio (ISO)"),
    end_date: Optional[str] = Query(None, description="Fecha de fin (ISO)"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Obtiene estadísticas de ingresos (requiere permisos admin).
    """
    try:
        from datetime import datetime
        
        service = get_monetization_service()
        
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None
        
        stats = service.get_revenue_stats(start, end)
        return stats
    except Exception as e:
        logger.error(f"Error getting revenue stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )

