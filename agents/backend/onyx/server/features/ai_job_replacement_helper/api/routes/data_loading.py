"""
Data Loading endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data_loading import (
    DataLoadingService,
    DataLoaderConfig,
    DataSplitConfig
)

router = APIRouter()
data_loading_service = DataLoadingService()


@router.post("/optimize-config")
async def optimize_dataloader_config(
    dataset_size: int,
    batch_size: int,
    available_memory_gb: float = None
) -> Dict[str, Any]:
    """Optimizar configuración de DataLoader"""
    try:
        config = data_loading_service.optimize_dataloader_config(
            dataset_size, batch_size, available_memory_gb
        )
        
        return {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "persistent_workers": config.persistent_workers,
            "drop_last": config.drop_last,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




