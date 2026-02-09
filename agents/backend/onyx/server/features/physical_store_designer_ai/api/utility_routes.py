"""Utility Routes"""
from fastapi import APIRouter
from typing import Dict, Any, List
from pydantic import BaseModel

from ..services.visualization_service import VisualizationService
from ..services.model_comparison_service import ModelComparisonService
from ..services.batch_processing_service import BatchProcessingService
from ..services.memory_optimization_service import MemoryOptimizationService
from ..services.model_conversion_service import ModelConversionService
from ..services.advanced_metrics_service import AdvancedMetricsService
from ..core.route_helpers import handle_route_errors, track_route_metrics

router = APIRouter()

viz_service = VisualizationService()
comparison_service = ModelComparisonService()
batch_service = BatchProcessingService()
memory_service = MemoryOptimizationService()
conversion_service = ModelConversionService()
metrics_service = AdvancedMetricsService()

@router.post("/visualization/architecture", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("utility.visualize_architecture")
async def visualize_architecture(model_id: str):
    return viz_service.visualize_model_architecture(model_id)

@router.post("/visualization/training-curves", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("utility.visualize_curves")
async def visualize_training_curves(training_id: str):
    return viz_service.visualize_training_curves(training_id)

@router.post("/comparison/models", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("utility.compare_models")
async def compare_models(model_ids: List[str], metrics: List[str]):
    return comparison_service.compare_models(model_ids, metrics)

@router.post("/batch/process", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("utility.process_batch")
async def process_batch(model_id: str, inputs: List[Any], batch_size: int = 32):
    return batch_service.process_batch(model_id, inputs, batch_size)

@router.post("/memory/optimize", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("utility.optimize_memory")
async def optimize_memory(model_id: str, technique: str = "gradient_checkpointing"):
    return memory_service.optimize_memory(model_id, technique)

@router.post("/conversion/convert", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("utility.convert_format")
async def convert_format(model_id: str, target_format: str):
    return conversion_service.convert_format(model_id, target_format)

@router.post("/metrics/track", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("utility.track_metrics")
async def track_metrics(model_id: str, metrics: Dict[str, float], step: int):
    return metrics_service.track_metrics(model_id, metrics, step)

@router.post("/metrics/compute", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("utility.compute_metrics")
async def compute_advanced_metrics(predictions: List[Any], targets: List[Any]):
    return metrics_service.compute_advanced_metrics(predictions, targets)




