"""
Data Preprocessing endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data_preprocessing import (
    DataPreprocessingService,
    PreprocessingConfig
)

router = APIRouter()
preprocessing_service = DataPreprocessingService()


@router.post("/preprocess")
async def preprocess_dataset(
    normalize: bool = True,
    normalization_method: str = "standard",
    test_size: float = 0.2,
    validation_size: float = 0.1
) -> Dict[str, Any]:
    """Preprocesar dataset"""
    try:
        config = PreprocessingConfig(
            normalize=normalize,
            normalization_method=normalization_method,
            test_size=test_size,
            validation_size=validation_size,
        )
        
        # In production, you would pass actual data
        # For now, return config info
        return {
            "config": {
                "normalize": normalize,
                "normalization_method": normalization_method,
                "test_size": test_size,
                "validation_size": validation_size,
            },
            "status": "ready",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




