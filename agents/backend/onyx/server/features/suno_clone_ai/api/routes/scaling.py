"""
API de Auto-Scaling

Endpoints para:
- Configurar políticas de escalado
- Evaluar escalado
- Obtener estadísticas
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body

from services.auto_scaler import (
    get_auto_scaler,
    ScalingPolicy,
    ScalingAction
)
from middleware.auth_middleware import require_role

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/scaling",
    tags=["scaling"],
    dependencies=[Depends(require_role("admin"))]  # Requiere rol admin
)


@router.post("/policies")
async def add_scaling_policy(
    name: str = Body(..., description="Nombre de la política"),
    metric: str = Body(..., description="Métrica (cpu, memory, requests, queue_size)"),
    threshold_up: float = Body(..., description="Umbral para scale up"),
    threshold_down: float = Body(..., description="Umbral para scale down"),
    min_replicas: int = Body(1, description="Mínimo de réplicas"),
    max_replicas: int = Body(10, description="Máximo de réplicas"),
    scale_up_cooldown: int = Body(60, description="Cooldown para scale up (segundos)"),
    scale_down_cooldown: int = Body(300, description="Cooldown para scale down (segundos)")
) -> Dict[str, Any]:
    """
    Agrega una política de escalado.
    """
    try:
        policy = ScalingPolicy(
            name=name,
            metric=metric,
            threshold_up=threshold_up,
            threshold_down=threshold_down,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            scale_up_cooldown=scale_up_cooldown,
            scale_down_cooldown=scale_down_cooldown
        )
        
        scaler = get_auto_scaler()
        scaler.add_policy(policy)
        
        return {
            "message": "Scaling policy added successfully",
            "policy": {
                "name": policy.name,
                "metric": policy.metric,
                "threshold_up": policy.threshold_up,
                "threshold_down": policy.threshold_down
            }
        }
    except Exception as e:
        logger.error(f"Error adding scaling policy: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding policy: {str(e)}"
        )


@router.post("/evaluate")
async def evaluate_scaling() -> Dict[str, Any]:
    """
    Evalúa si se necesita escalar.
    """
    try:
        scaler = get_auto_scaler()
        decision = scaler.evaluate_scaling()
        
        if decision:
            return {
                "action_required": True,
                "action": decision.action.value,
                "current_replicas": decision.current_replicas,
                "target_replicas": decision.target_replicas,
                "reason": decision.reason,
                "timestamp": decision.timestamp.isoformat()
            }
        else:
            return {
                "action_required": False,
                "message": "No scaling needed"
            }
    except Exception as e:
        logger.error(f"Error evaluating scaling: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error evaluating scaling: {str(e)}"
        )


@router.post("/apply")
async def apply_scaling(
    action: str = Body(..., description="Acción (scale_up, scale_down)"),
    target_replicas: int = Body(..., description="Número objetivo de réplicas"),
    reason: str = Body(..., description="Razón del escalado")
) -> Dict[str, Any]:
    """
    Aplica una acción de escalado.
    """
    try:
        from services.auto_scaler import ScalingDecision, ScalingAction
        
        try:
            action_enum = ScalingAction(action)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: {action}"
            )
        
        scaler = get_auto_scaler()
        decision = ScalingDecision(
            action=action_enum,
            current_replicas=scaler.current_replicas,
            target_replicas=target_replicas,
            reason=reason
        )
        
        success = scaler.apply_scaling(decision)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to apply scaling"
            )
        
        return {
            "message": "Scaling applied successfully",
            "action": action,
            "replicas": f"{decision.current_replicas} -> {target_replicas}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying scaling: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error applying scaling: {str(e)}"
        )


@router.post("/metrics")
async def record_metric(
    metric_name: str = Body(..., description="Nombre de la métrica"),
    value: float = Body(..., description="Valor de la métrica")
) -> Dict[str, Any]:
    """
    Registra una métrica para evaluación de escalado.
    """
    try:
        scaler = get_auto_scaler()
        scaler.record_metric(metric_name, value)
        
        return {
            "message": "Metric recorded successfully",
            "metric": metric_name,
            "value": value
        }
    except Exception as e:
        logger.error(f"Error recording metric: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recording metric: {str(e)}"
        )


@router.get("/stats")
async def get_scaling_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas del auto-scaler.
    """
    try:
        scaler = get_auto_scaler()
        stats = scaler.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting scaling stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )

