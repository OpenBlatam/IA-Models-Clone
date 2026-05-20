"""
Rutas para Advanced NAS
========================

Endpoints para búsqueda avanzada de arquitecturas.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.advanced_nas import (
    get_advanced_nas,
    AdvancedNAS,
    NASStrategy
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/advanced-nas",
    tags=["Advanced NAS"]
)


class CreateExperimentRequest(BaseModel):
    """Request para crear experimento"""
    search_space: Dict[str, Any] = Field(..., description="Espacio de búsqueda")
    strategy: str = Field("evolutionary", description="Estrategia")


@router.post("/experiments")
async def create_experiment(
    request: CreateExperimentRequest,
    system: AdvancedNAS = Depends(get_advanced_nas)
):
    """Crear experimento de NAS"""
    try:
        strategy = NASStrategy(request.strategy)
        experiment = system.create_experiment(request.search_space, strategy)
        
        return {
            "experiment_id": experiment.experiment_id,
            "strategy": experiment.strategy.value,
            "search_space": experiment.search_space,
            "status": experiment.status
        }
    except Exception as e:
        logger.error(f"Error creando experimento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{experiment_id}/search")
async def search_architecture(
    experiment_id: str,
    max_iterations: int = Field(100, description="Máximo de iteraciones"),
    system: AdvancedNAS = Depends(get_advanced_nas)
):
    """Buscar arquitectura"""
    try:
        architecture = system.search_architecture(experiment_id, max_iterations)
        
        return {
            "arch_id": architecture.arch_id,
            "layers": architecture.layers,
            "parameters": architecture.parameters,
            "flops": architecture.flops,
            "accuracy": architecture.accuracy,
            "latency": architecture.latency,
            "score": architecture.score
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error buscando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/architectures/analyze")
async def analyze_architecture(
    architecture: Dict[str, Any] = Field(..., description="Arquitectura"),
    system: AdvancedNAS = Depends(get_advanced_nas)
):
    """Analizar arquitectura"""
    try:
        from ..core.advanced_nas import Architecture
        
        arch = Architecture(
            arch_id=architecture.get("arch_id", "unknown"),
            layers=architecture.get("layers", []),
            parameters=architecture.get("parameters", 0),
            flops=architecture.get("flops", 0),
            accuracy=architecture.get("accuracy", 0.0),
            latency=architecture.get("latency", 0.0),
            score=architecture.get("score", 0.0)
        )
        
        analysis = system.analyze_architecture(arch)
        
        return analysis
    except Exception as e:
        logger.error(f"Error analizando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


