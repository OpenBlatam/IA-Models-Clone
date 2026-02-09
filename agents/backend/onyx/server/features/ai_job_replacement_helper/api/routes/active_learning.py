"""
Active Learning endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.active_learning import (
    ActiveLearningService,
    ActiveLearningConfig,
    SamplingStrategy
)

router = APIRouter()
active_learning_service = ActiveLearningService()


@router.post("/select-samples")
async def select_samples(
    strategy: str = "uncertainty",
    num_samples: int = 100
) -> Dict[str, Any]:
    """Seleccionar muestras para etiquetar"""
    try:
        strategy_enum = SamplingStrategy(strategy)
        config = ActiveLearningConfig(
            strategy=strategy_enum,
            num_samples=num_samples
        )
        
        return {
            "strategy": strategy,
            "num_samples": num_samples,
            "status": "ready",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




