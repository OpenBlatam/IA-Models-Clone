"""
Federated Learning endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.federated_learning import (
    FederatedLearningService,
    AggregationMethod
)

router = APIRouter()
federated_service = FederatedLearningService()


@router.post("/update")
async def receive_client_update(
    round_number: int,
    client_id: str,
    num_samples: int,
    loss: float
) -> Dict[str, Any]:
    """Recibir actualización de cliente"""
    try:
        federated_service.receive_client_update(
            round_number, client_id, {}, num_samples, loss
        )
        
        return {
            "round_number": round_number,
            "client_id": client_id,
            "received": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/round/{round_number}")
async def get_round_updates(round_number: int) -> Dict[str, Any]:
    """Obtener actualizaciones de una ronda"""
    try:
        updates = federated_service.get_round_updates(round_number)
        
        return {
            "round_number": round_number,
            "num_updates": len(updates),
            "clients": [u.client_id for u in updates],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




