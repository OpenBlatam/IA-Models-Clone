"""
API de A/B Testing

Endpoints para:
- Crear experimentos
- Asignar variantes
- Registrar resultados
- Analizar experimentos
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body

from services.ab_testing import get_ab_testing_service
from middleware.auth_middleware import require_role, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ab-testing",
    tags=["ab-testing"]
)


@router.post("/experiments")
async def create_experiment(
    name: str = Body(..., description="Nombre del experimento"),
    variants: List[str] = Body(["control", "variant_a"], description="Lista de variantes"),
    traffic_split: Optional[Dict[str, float]] = Body(None, description="División de tráfico"),
    description: Optional[str] = Body(None, description="Descripción"),
    current_user: Dict[str, Any] = Depends(require_role("admin"))
) -> Dict[str, Any]:
    """
    Crea un nuevo experimento A/B (requiere rol admin).
    """
    try:
        service = get_ab_testing_service()
        experiment_id = service.create_experiment(
            name=name,
            variants=variants,
            traffic_split=traffic_split,
            description=description
        )
        
        return {
            "experiment_id": experiment_id,
            "message": "Experiment created successfully",
            "name": name
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating experiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating experiment: {str(e)}"
        )


@router.get("/experiments/{experiment_id}/assign")
async def assign_variant(
    experiment_id: str,
    user_id: Optional[str] = Query(None, description="ID del usuario"),
    force_variant: Optional[str] = Query(None, description="Forzar variante (testing)"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Asigna una variante a un usuario.
    """
    try:
        # Usar user_id del token si está disponible
        if not user_id and current_user:
            user_id = current_user.get("user_id") or current_user.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required"
            )
        
        service = get_ab_testing_service()
        variant = service.assign_variant(
            experiment_id=experiment_id,
            user_id=user_id,
            force_variant=force_variant
        )
        
        return {
            "experiment_id": experiment_id,
            "user_id": user_id,
            "variant": variant
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error assigning variant: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error assigning variant: {str(e)}"
        )


@router.post("/experiments/{experiment_id}/results")
async def record_result(
    experiment_id: str,
    user_id: str = Body(..., description="ID del usuario"),
    variant: str = Body(..., description="Variante asignada"),
    metrics: Dict[str, float] = Body(..., description="Métricas del resultado")
) -> Dict[str, Any]:
    """
    Registra resultado de un experimento.
    """
    try:
        service = get_ab_testing_service()
        service.record_result(
            experiment_id=experiment_id,
            user_id=user_id,
            variant=variant,
            metrics=metrics
        )
        
        return {
            "message": "Result recorded successfully",
            "experiment_id": experiment_id,
            "user_id": user_id,
            "variant": variant
        }
    except Exception as e:
        logger.error(f"Error recording result: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recording result: {str(e)}"
        )


@router.get("/experiments/{experiment_id}/analyze")
async def analyze_experiment(
    experiment_id: str,
    metric: str = Query(..., description="Métrica a analizar"),
    confidence_level: float = Query(0.95, ge=0.0, le=1.0, description="Nivel de confianza"),
    current_user: Dict[str, Any] = Depends(require_role("admin"))
) -> Dict[str, Any]:
    """
    Analiza un experimento (requiere rol admin).
    """
    try:
        service = get_ab_testing_service()
        analysis = service.analyze_experiment(
            experiment_id=experiment_id,
            metric=metric,
            confidence_level=confidence_level
        )
        
        return analysis
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error analyzing experiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing experiment: {str(e)}"
        )


@router.get("/experiments/{experiment_id}/stats")
async def get_experiment_stats(
    experiment_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Obtiene estadísticas de un experimento.
    """
    try:
        service = get_ab_testing_service()
        stats = service.get_experiment_stats(experiment_id)
        
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )

