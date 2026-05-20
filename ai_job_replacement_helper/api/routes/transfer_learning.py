"""
Transfer Learning endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.transfer_learning import (
    TransferLearningService,
    TransferLearningConfig,
    TransferStrategy
)

router = APIRouter()
transfer_service = TransferLearningService()


@router.post("/create-model")
async def create_transfer_model(
    model_id: str,
    base_model: str,
    num_labels: int,
    strategy: str = "fine_tuning",
    learning_rate: float = 2e-5
) -> Dict[str, Any]:
    """Crear modelo con transfer learning"""
    try:
        strategy_enum = TransferStrategy(strategy)
        config = TransferLearningConfig(
            base_model=base_model,
            num_labels=num_labels,
            strategy=strategy_enum,
            learning_rate=learning_rate,
        )
        
        model = transfer_service.create_transfer_model(model_id, config)
        
        return {
            "model_id": model_id,
            "base_model": base_model,
            "strategy": strategy,
            "created": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trainable-params/{model_id}")
async def get_trainable_parameters(model_id: str) -> Dict[str, Any]:
    """Obtener parámetros entrenables"""
    try:
        params = transfer_service.get_trainable_parameters(model_id)
        return params
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




