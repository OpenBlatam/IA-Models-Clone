"""
Neural Architecture Search endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.neural_architecture_search import (
    NeuralArchitectureSearchService,
    NASMethod
)

router = APIRouter()
nas_service = NeuralArchitectureSearchService()


@router.post("/create-space")
async def create_search_space(
    space_id: str,
    num_layers_min: int = 2,
    num_layers_max: int = 10,
    hidden_sizes: List[int] = None
) -> Dict[str, Any]:
    """Crear espacio de búsqueda"""
    try:
        if hidden_sizes is None:
            hidden_sizes = [64, 128, 256, 512]
        
        space = nas_service.create_search_space(
            space_id,
            num_layers={"min": num_layers_min, "max": num_layers_max},
            hidden_sizes=hidden_sizes
        )
        
        return {
            "space_id": space_id,
            "created": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_architectures(
    space_id: str,
    input_size: int,
    output_size: int,
    num_trials: int = 50
) -> Dict[str, Any]:
    """Buscar arquitecturas"""
    try:
        candidates = nas_service.search_architectures(
            space_id, input_size, output_size, num_trials
        )
        
        return {
            "space_id": space_id,
            "candidates": [
                {
                    "architecture_id": c.architecture_id,
                    "config": c.config,
                    "performance": c.performance,
                }
                for c in candidates
            ],
            "total": len(candidates),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best/{space_id}")
async def get_best_architecture(space_id: str) -> Dict[str, Any]:
    """Obtener mejor arquitectura"""
    try:
        best = nas_service.get_best_architecture(space_id)
        if not best:
            return {"error": "No architecture found"}
        
        return {
            "architecture_id": best.architecture_id,
            "config": best.config,
            "performance": best.performance,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




