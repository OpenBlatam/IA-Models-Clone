"""
Meta Learning endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.meta_learning import (
    MetaLearningService,
    MetaLearningConfig,
    MetaLearningMethod
)

router = APIRouter()
meta_learning_service = MetaLearningService()


@router.post("/maml-step")
async def maml_step(
    method: str = "maml",
    inner_lr: float = 0.01,
    num_inner_steps: int = 1
) -> Dict[str, Any]:
    """Ejecutar paso de meta aprendizaje"""
    try:
        method_enum = MetaLearningMethod(method)
        config = MetaLearningConfig(
            method=method_enum,
            inner_lr=inner_lr,
            num_inner_steps=num_inner_steps
        )
        
        return {
            "method": method,
            "inner_lr": inner_lr,
            "num_inner_steps": num_inner_steps,
            "status": "ready",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




