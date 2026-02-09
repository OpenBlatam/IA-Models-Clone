"""Training Tools Routes"""
from fastapi import APIRouter
from typing import Dict, Any, List
from pydantic import BaseModel

from ..services.advanced_validation_service import AdvancedValidationService
from ..services.advanced_augmentation_service import AdvancedAugmentationService
from ..services.custom_loss_service import CustomLossService
from ..services.advanced_optimizers_service import AdvancedOptimizersService
from ..services.lr_finder_service import LRFinderService
from ..services.model_debugging_service import ModelDebuggingService
from ..core.route_helpers import handle_route_errors, track_route_metrics

router = APIRouter()

validation_service = AdvancedValidationService()
augmentation_service = AdvancedAugmentationService()
loss_service = CustomLossService()
optimizer_service = AdvancedOptimizersService()
lr_service = LRFinderService()
debug_service = ModelDebuggingService()

@router.post("/validation/validate", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("training_tools.validate")
async def validate_model(model_id: str, test_data: List[Any], metrics: List[str]):
    return validation_service.validate_model(model_id, test_data, metrics)

@router.post("/augmentation/pipeline", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("training_tools.augmentation")
async def create_augmentation_pipeline(pipeline_name: str, techniques: List[str]):
    return augmentation_service.create_augmentation_pipeline(pipeline_name, techniques)

@router.post("/loss/custom", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("training_tools.custom_loss")
async def create_custom_loss(loss_name: str, loss_type: str, config: Dict[str, Any]):
    return loss_service.create_custom_loss(loss_name, loss_type, config)

@router.post("/optimizers/create", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("training_tools.optimizer")
async def create_optimizer(optimizer_type: str, lr: float, config: Dict[str, Any]):
    return optimizer_service.create_optimizer(optimizer_type, lr, config)

@router.post("/lr-finder/search", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("training_tools.lr_finder")
async def find_optimal_lr(model_id: str, start_lr: float = 1e-7, end_lr: float = 1.0):
    return lr_service.find_optimal_lr(model_id, start_lr, end_lr)

@router.post("/debug/model", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("training_tools.debug")
async def debug_model(model_id: str, check_gradients: bool = True, check_nan: bool = True):
    return debug_service.debug_model(model_id, check_gradients, check_nan)




