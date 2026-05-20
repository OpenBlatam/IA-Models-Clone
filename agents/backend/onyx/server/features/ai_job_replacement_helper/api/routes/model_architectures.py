"""
Model Architectures endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_architectures import (
    ModelArchitecturesService,
    ModelConfig
)

router = APIRouter()
architectures_service = ModelArchitecturesService()


@router.post("/create-mlp")
async def create_mlp(
    model_id: str,
    input_size: int,
    output_size: int,
    hidden_sizes: List[int],
    activation: str = "relu",
    dropout: float = 0.1
) -> Dict[str, Any]:
    """Crear MLP"""
    try:
        config = ModelConfig(
            name=model_id,
            architecture_type="mlp",
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
        )
        
        model = architectures_service.create_mlp(model_id, config)
        
        return {
            "model_id": model_id,
            "architecture": "mlp",
            "created": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-text-classifier")
async def create_text_classifier(
    model_id: str,
    vocab_size: int,
    embedding_dim: int,
    num_classes: int,
    hidden_dim: int = 128,
    num_layers: int = 2
) -> Dict[str, Any]:
    """Crear clasificador de texto"""
    try:
        model = architectures_service.create_text_classifier(
            model_id, vocab_size, embedding_dim, num_classes, hidden_dim, num_layers
        )
        
        return {
            "model_id": model_id,
            "architecture": "text_classifier",
            "created": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize-weights/{model_id}")
async def initialize_weights(
    model_id: str,
    method: str = "xavier_uniform"
) -> Dict[str, Any]:
    """Inicializar pesos del modelo"""
    try:
        success = architectures_service.initialize_weights(model_id, method)
        return {
            "model_id": model_id,
            "method": method,
            "initialized": success,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary/{model_id}")
async def get_model_summary(model_id: str) -> Dict[str, Any]:
    """Obtener resumen del modelo"""
    try:
        summary = architectures_service.get_model_summary(model_id)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




