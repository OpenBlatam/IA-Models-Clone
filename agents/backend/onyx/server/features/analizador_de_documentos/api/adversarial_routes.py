"""
Rutas para Adversarial Training
=================================

Endpoints para adversarial training.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.adversarial_training import (
    get_adversarial_training,
    AdversarialTraining,
    AttackType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/adversarial",
    tags=["Adversarial Training"]
)


class GenerateAdversarialRequest(BaseModel):
    """Request para generar ejemplo adversarial"""
    original_input: Dict[str, Any] = Field(..., description="Entrada original")
    attack_type: str = Field("fgsm", description="Tipo de ataque")
    epsilon: float = Field(0.1, description="Tamaño de perturbación")


class TrainAdversarialRequest(BaseModel):
    """Request para entrenar con adversariales"""
    training_data: List[Dict[str, Any]] = Field(..., description="Datos de entrenamiento")
    adversarial_ratio: float = Field(0.3, description="Proporción adversarial")
    epochs: int = Field(5, description="Número de épocas")


@router.post("/generate")
async def generate_adversarial(
    request: GenerateAdversarialRequest,
    system: AdversarialTraining = Depends(get_adversarial_training)
):
    """Generar ejemplo adversarial"""
    try:
        attack_type = AttackType(request.attack_type)
        example = system.generate_adversarial_example(
            request.original_input,
            attack_type,
            request.epsilon
        )
        
        return {
            "example_id": example.example_id,
            "original": example.original,
            "adversarial": example.adversarial,
            "attack_type": example.attack_type.value,
            "perturbation": example.perturbation,
            "success": example.success
        }
    except Exception as e:
        logger.error(f"Error generando ejemplo adversarial: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def train_with_adversarials(
    model_id: str = Field(..., description="ID del modelo"),
    request: TrainAdversarialRequest = ...,
    system: AdversarialTraining = Depends(get_adversarial_training)
):
    """Entrenar con ejemplos adversariales"""
    try:
        result = system.train_with_adversarials(
            model_id,
            request.training_data,
            request.adversarial_ratio,
            request.epochs
        )
        
        return result
    except Exception as e:
        logger.error(f"Error entrenando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate-robustness")
async def evaluate_robustness(
    model_id: str = Field(..., description="ID del modelo"),
    test_data: List[Dict[str, Any]] = Field(..., description="Datos de prueba"),
    attack_types: List[str] = Field(..., description="Tipos de ataques"),
    system: AdversarialTraining = Depends(get_adversarial_training)
):
    """Evaluar robustez"""
    try:
        attack_types_enum = [AttackType(at) for at in attack_types]
        robustness = system.evaluate_robustness(model_id, test_data, attack_types_enum)
        
        return robustness
    except Exception as e:
        logger.error(f"Error evaluando robustez: {e}")
        raise HTTPException(status_code=500, detail=str(e))



