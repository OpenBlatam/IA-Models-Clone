"""
Data Analysis Router
====================

Endpoints for data analysis and validation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import logging

from ..utils import validate_array_shape, handle_model_operation_error
from ..config import settings
from ..constants import BYTES_PER_MB, MAX_MODEL_SIZE_MB

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])


class DataAnalysisRequest(BaseModel):
    """Request schema for data analysis."""
    data: List[List[float]] = Field(..., description="Data to analyze")
    
    class Config:
        schema_extra = {
            "example": {
                "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            }
        }


@router.post("/analyze")
async def analyze_data(request: DataAnalysisRequest):
    """
    Analyze input data.
    
    Returns comprehensive statistics about the data including:
    - Shape and dimensions
    - Statistical measures (mean, std, min, max)
    - Data quality checks
    - Memory usage estimate
    """
    try:
        array = validate_array_shape(request.data, expected_dim=2)
        
        analysis = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "size": int(array.size),
            "dimensions": int(array.ndim),
            "statistics": {
                "mean": float(np.mean(array)),
                "std": float(np.std(array)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "median": float(np.median(array)),
                "percentiles": {
                    "25": float(np.percentile(array, 25)),
                    "50": float(np.percentile(array, 50)),
                    "75": float(np.percentile(array, 75)),
                    "95": float(np.percentile(array, 95)),
                    "99": float(np.percentile(array, 99))
                }
            },
            "quality": {
                "has_nan": bool(np.any(np.isnan(array))),
                "has_inf": bool(np.any(np.isinf(array))),
                "nan_count": int(np.sum(np.isnan(array))),
                "inf_count": int(np.sum(np.isinf(array))),
                "zero_count": int(np.sum(array == 0)),
                "unique_values": int(len(np.unique(array)))
            },
            "memory": {
                "bytes": int(array.nbytes),
                "mb": round(array.nbytes / BYTES_PER_MB, 4)
            }
        }
        
        if array.size > 0:
            analysis["statistics"]["range"] = float(np.max(array) - np.min(array))
            analysis["statistics"]["variance"] = float(np.var(array))
        
        return analysis
    except Exception as e:
        handle_model_operation_error("analyzing data", error=e)
        raise HTTPException(status_code=400, detail=f"Error analyzing data: {str(e)}")


@router.post("/validate")
async def validate_data(request: DataAnalysisRequest):
    """
    Validate data format and quality.
    
    Checks data for:
    - Format validity
    - Missing values (NaN)
    - Infinite values
    - Shape consistency
    - Data type compatibility
    """
    try:
        array = validate_array_shape(request.data, expected_dim=2)
        
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "shape": list(array.shape),
            "dtype": str(array.dtype)
        }
        
        if np.any(np.isnan(array)):
            validation["warnings"].append("Data contains NaN values")
            validation["valid"] = False
        
        if np.any(np.isinf(array)):
            validation["warnings"].append("Data contains infinite values")
            validation["valid"] = False
        
        if array.size == 0:
            validation["errors"].append("Data is empty")
            validation["valid"] = False
        
        if array.size > settings.max_training_samples:
            validation["warnings"].append(
                f"Data size ({array.size}) exceeds recommended maximum ({settings.max_training_samples})"
            )
        
        if array.nbytes / BYTES_PER_MB > MAX_MODEL_SIZE_MB:
            validation["warnings"].append("Data size exceeds 100MB")
        
        if len(set(array.shape)) == 1 and array.shape[0] == 1:
            validation["warnings"].append("Data appears to be a single sample")
        
        return validation
    except ValueError as e:
        return {
            "valid": False,
            "errors": [str(e)],
            "warnings": [],
            "shape": None,
            "dtype": None
        }
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error validating data: {str(e)}")


@router.post("/normalize")
async def normalize_data(
    request: DataAnalysisRequest,
    method: str = "standard",
    axis: int = 0
):
    """
    Normalize data using various methods.
    
    - **data**: Data to normalize
    - **method**: Normalization method (standard, minmax, l2)
    - **axis**: Axis along which to normalize (0 for features, 1 for samples)
    """
    try:
        array = validate_array_shape(request.data, expected_dim=2)
        
        if method == "standard":
            mean = np.mean(array, axis=axis, keepdims=True)
            std = np.std(array, axis=axis, keepdims=True)
            std = np.where(std == 0, 1, std)
            normalized = (array - mean) / std
        elif method == "minmax":
            min_val = np.min(array, axis=axis, keepdims=True)
            max_val = np.max(array, axis=axis, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            normalized = (array - min_val) / range_val
        elif method == "l2":
            norms = np.linalg.norm(array, axis=axis, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normalized = array / norms
        else:
            raise HTTPException(status_code=400, detail=f"Unknown normalization method: {method}")
        
        return {
            "original_shape": list(array.shape),
            "normalized_shape": list(normalized.shape),
            "method": method,
            "axis": axis,
            "normalized_data": normalized.tolist(),
            "statistics": {
                "original": {
                    "mean": float(np.mean(array)),
                    "std": float(np.std(array)),
                    "min": float(np.min(array)),
                    "max": float(np.max(array))
                },
                "normalized": {
                    "mean": float(np.mean(normalized)),
                    "std": float(np.std(normalized)),
                    "min": float(np.min(normalized)),
                    "max": float(np.max(normalized))
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        handle_model_operation_error("normalizing data", error=e)
        raise HTTPException(status_code=400, detail=f"Error normalizing data: {str(e)}")


@router.post("/split")
async def split_data(
    request: DataAnalysisRequest,
    test_size: float = Field(0.2, ge=0.0, le=1.0, description="Proportion of data for test set"),
    random_state: Optional[int] = Field(None, description="Random seed for reproducibility"),
    shuffle: bool = Field(True, description="Whether to shuffle data before splitting")
):
    """
    Split data into train and test sets.
    
    - **data**: Data to split
    - **test_size**: Proportion of data for test set (0.0-1.0)
    - **random_state**: Random seed for reproducibility
    - **shuffle**: Whether to shuffle data before splitting
    """
    try:
        array = validate_array_shape(request.data, expected_dim=2)
        
        n_samples = array.shape[0]
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test
        
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            indices = np.random.permutation(n_samples)
            array = array[indices]
        
        train_data = array[:n_train]
        test_data = array[n_test:] if n_test > 0 else array[n_train:]
        
        return {
            "original_shape": list(array.shape),
            "train": {
                "shape": list(train_data.shape),
                "samples": int(train_data.shape[0]),
                "data": train_data.tolist()
            },
            "test": {
                "shape": list(test_data.shape),
                "samples": int(test_data.shape[0]),
                "data": test_data.tolist()
            },
            "split_ratio": {
                "train": round(n_train / n_samples, 4),
                "test": round(n_test / n_samples, 4) if n_test > 0 else 0.0
            }
        }
    except Exception as e:
        handle_model_operation_error("splitting data", error=e)
        raise HTTPException(status_code=400, detail=f"Error splitting data: {str(e)}")

