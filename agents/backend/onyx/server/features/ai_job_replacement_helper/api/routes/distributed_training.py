"""
Distributed Training endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.distributed_training import (
    DistributedTrainingService,
    DistributedConfig,
    DistributedStrategy
)

router = APIRouter()
distributed_service = DistributedTrainingService()


@router.post("/setup")
async def setup_distributed(
    job_id: str,
    strategy: str = "data_parallel",
    num_nodes: int = 1,
    num_gpus_per_node: int = 1
) -> Dict[str, Any]:
    """Configurar entrenamiento distribuido"""
    try:
        strategy_enum = DistributedStrategy(strategy)
        config = DistributedConfig(
            strategy=strategy_enum,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
        )
        
        success = distributed_service.setup_distributed(job_id, config)
        
        return {
            "job_id": job_id,
            "strategy": strategy,
            "setup": success,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/world-info")
async def get_world_info() -> Dict[str, Any]:
    """Obtener información del mundo distribuido"""
    try:
        info = distributed_service.get_world_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




