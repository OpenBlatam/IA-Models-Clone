"""
Data Augmentation endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data_augmentation import (
    DataAugmentationService,
    AugmentationConfig
)

router = APIRouter()
augmentation_service = DataAugmentationService()


@router.post("/augment")
async def augment_data(
    horizontal_flip: bool = False,
    rotation: float = 0.0,
    brightness: float = 0.0,
    use_mixup: bool = False
) -> Dict[str, Any]:
    """Aumentar datos"""
    try:
        config = AugmentationConfig(
            horizontal_flip=horizontal_flip,
            rotation=rotation,
            brightness=brightness,
            use_mixup=use_mixup,
        )
        
        return {
            "config": {
                "horizontal_flip": horizontal_flip,
                "rotation": rotation,
                "brightness": brightness,
                "use_mixup": use_mixup,
            },
            "status": "ready",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




