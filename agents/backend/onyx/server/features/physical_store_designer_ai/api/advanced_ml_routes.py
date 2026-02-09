"""Advanced ML Routes"""
from fastapi import APIRouter
from typing import Dict, Any, List
from pydantic import BaseModel

from ..services.transfer_learning_service import TransferLearningService
from ..services.multitask_learning_service import MultiTaskLearningService
from ..services.continual_learning_service import ContinualLearningService
from ..services.nas_service import NASService
from ..services.automl_service import AutoMLService
from ..services.ensembling_service import EnsemblingService
from ..core.route_helpers import handle_route_errors, track_route_metrics

router = APIRouter()

transfer_service = TransferLearningService()
multitask_service = MultiTaskLearningService()
continual_service = ContinualLearningService()
nas_service = NASService()
automl_service = AutoMLService()
ensembling_service = EnsemblingService()

@router.post("/transfer-learning/create", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_ml.transfer_learning")
async def create_transfer_model(base_model_id: str, target_task: str, freeze_backbone: bool = True):
    return transfer_service.create_transfer_model(base_model_id, target_task, freeze_backbone)

@router.post("/multitask/create", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_ml.multitask")
async def create_multi_task_model(model_name: str, tasks: List[str], shared_layers: int = 3):
    return multitask_service.create_multi_task_model(model_name, tasks, shared_layers)

@router.post("/continual-learning/create", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_ml.continual")
async def create_continual_model(model_id: str, method: str = "ewc"):
    return continual_service.create_continual_model(model_id, method)

@router.post("/nas/search", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_ml.nas")
async def create_nas_search(search_space: Dict[str, Any], search_strategy: str = "darts"):
    return nas_service.create_search(search_space, search_strategy)

@router.post("/automl/experiment", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_ml.automl")
async def create_automl_experiment(task_type: str, dataset_id: str, time_budget_hours: float = 1.0):
    return automl_service.create_automl_experiment(task_type, dataset_id, time_budget_hours)

@router.post("/ensembling/create", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_ml.ensembling")
async def create_ensemble(model_ids: List[str], ensemble_method: str = "voting"):
    return ensembling_service.create_ensemble(model_ids, ensemble_method)




