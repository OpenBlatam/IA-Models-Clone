"""
Subscriptions endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.subscriptions import SubscriptionsService, SubscriptionPlan

router = APIRouter()
subscriptions_service = SubscriptionsService()


@router.post("/create/{user_id}")
async def create_subscription(
    user_id: str,
    plan: str,
    trial_days: int = 7
) -> Dict[str, Any]:
    """Crear suscripción"""
    try:
        plan_enum = SubscriptionPlan(plan)
        subscription = subscriptions_service.create_subscription(user_id, plan_enum, trial_days)
        return {
            "user_id": subscription.user_id,
            "plan": subscription.plan.value,
            "status": subscription.status.value,
            "start_date": subscription.start_date.isoformat(),
            "end_date": subscription.end_date.isoformat() if subscription.end_date else None,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}")
async def get_subscription(user_id: str) -> Dict[str, Any]:
    """Obtener suscripción del usuario"""
    try:
        subscription = subscriptions_service.get_subscription(user_id)
        if not subscription:
            # Crear suscripción free por defecto
            subscription = subscriptions_service.create_subscription(user_id, SubscriptionPlan.FREE)
        
        return {
            "user_id": subscription.user_id,
            "plan": subscription.plan.value,
            "status": subscription.status.value,
            "end_date": subscription.end_date.isoformat() if subscription.end_date else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plans/{plan}")
async def get_plan_features(plan: str) -> Dict[str, Any]:
    """Obtener características del plan"""
    try:
        plan_enum = SubscriptionPlan(plan)
        features = subscriptions_service.get_plan_features(plan_enum)
        return features
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upgrade/{user_id}")
async def upgrade_plan(user_id: str, new_plan: str) -> Dict[str, Any]:
    """Actualizar plan"""
    try:
        plan_enum = SubscriptionPlan(new_plan)
        subscription = subscriptions_service.upgrade_plan(user_id, plan_enum)
        return {
            "user_id": subscription.user_id,
            "plan": subscription.plan.value,
            "status": subscription.status.value,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




