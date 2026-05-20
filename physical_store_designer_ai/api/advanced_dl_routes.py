"""
Advanced Deep Learning API Routes - Optimización y técnicas avanzadas
"""

from fastapi import APIRouter
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from ..services.performance_optimization_service import PerformanceOptimizationService, OptimizationStrategy
from ..services.data_pipeline_service import DataPipelineService
from ..services.config_service import ConfigService
from ..services.experiment_logging_service import ExperimentLoggingService, LoggingBackend
from ..services.model_serving_service import ModelServingService, ServingFormat
from ..services.advanced_training_service import AdvancedTrainingService
from ..core.route_helpers import handle_route_errors, track_route_metrics

router = APIRouter()

# Initialize services
perf_service = PerformanceOptimizationService()
data_service = DataPipelineService()
config_service = ConfigService()
logging_service = ExperimentLoggingService()
serving_service = ModelServingService()
training_service = AdvancedTrainingService()


# Performance Optimization
@router.get("/performance/devices", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.detect_devices")
async def detect_devices():
    """Detectar dispositivos disponibles"""
    return perf_service.detect_devices()


@router.post("/performance/data-parallel", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.data_parallel")
async def setup_data_parallel(model_id: str, device_ids: Optional[List[int]] = None):
    """Configurar DataParallel"""
    return perf_service.setup_data_parallel(model_id, device_ids)


@router.post("/performance/distributed", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.distributed")
async def setup_distributed(model_id: str, backend: str = "nccl", world_size: Optional[int] = None):
    """Configurar DistributedDataParallel"""
    return perf_service.setup_distributed_training(model_id, backend, world_size)


@router.post("/performance/mixed-precision", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.mixed_precision")
async def setup_mixed_precision(model_id: str, enabled: bool = True, opt_level: str = "O1"):
    """Configurar mixed precision"""
    return perf_service.setup_mixed_precision(model_id, enabled, opt_level)


@router.post("/performance/gradient-accumulation", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.gradient_accumulation")
async def setup_gradient_accumulation(model_id: str, accumulation_steps: int = 4):
    """Configurar gradient accumulation"""
    return perf_service.setup_gradient_accumulation(model_id, accumulation_steps)


@router.post("/performance/profile", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.profile")
async def profile_model(model_id: str, input_shape: List[int], num_iterations: int = 10):
    """Perfilar modelo"""
    return perf_service.profile_model(model_id, input_shape, num_iterations)


@router.post("/performance/optimize-batch-size", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.optimize_batch_size")
async def optimize_batch_size(model_id: str, start_batch_size: int = 32, max_memory_gb: float = 8.0):
    """Optimizar batch size"""
    return perf_service.optimize_batch_size(model_id, start_batch_size, max_memory_gb)


# Data Pipeline
class CreateDatasetRequest(BaseModel):
    dataset_name: str
    data: List[Dict[str, Any]]


class CreateDataLoaderRequest(BaseModel):
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


@router.post("/data/datasets", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.create_dataset")
async def create_dataset(request: CreateDatasetRequest):
    """Crear dataset"""
    return data_service.create_dataset(request.dataset_name, request.data)


@router.post("/data/datasets/{dataset_id}/loaders", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.create_dataloader")
async def create_dataloader(dataset_id: str, request: CreateDataLoaderRequest):
    """Crear DataLoader"""
    return data_service.create_dataloader(
        dataset_id,
        request.batch_size,
        request.shuffle,
        request.num_workers,
        request.pin_memory,
        request.prefetch_factor
    )


@router.post("/data/datasets/{dataset_id}/split", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.split_dataset")
async def split_dataset(
    dataset_id: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
):
    """Dividir dataset"""
    return data_service.split_dataset(dataset_id, train_ratio, val_ratio, test_ratio)


@router.get("/data/datasets/{dataset_id}/stats", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.get_dataset_stats")
async def get_dataset_stats(dataset_id: str):
    """Obtener estadísticas del dataset"""
    return data_service.get_dataset_stats(dataset_id)


# Config Service
class CreateConfigRequest(BaseModel):
    config_name: str
    config_data: Dict[str, Any]


class LoadYAMLRequest(BaseModel):
    yaml_content: str


@router.post("/config/create", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.create_config")
async def create_config(request: CreateConfigRequest):
    """Crear configuración"""
    return config_service.create_config(request.config_name, request.config_data)


@router.post("/config/load-yaml", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.load_yaml")
async def load_yaml_config(request: LoadYAMLRequest):
    """Cargar configuración desde YAML"""
    return config_service.load_config_from_yaml(request.yaml_content)


@router.post("/config/training", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.training_config")
async def create_training_config(
    model_name: str,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 10,
    optimizer: str = "adam",
    scheduler: Optional[str] = None,
    loss_function: str = "mse"
):
    """Crear configuración de entrenamiento"""
    return config_service.create_training_config(
        model_name, learning_rate, batch_size, epochs,
        optimizer, scheduler, loss_function
    )


@router.post("/config/merge", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.merge_configs")
async def merge_configs(base_config_id: str, override_config_id: str):
    """Fusionar configuraciones"""
    return config_service.merge_configs(base_config_id, override_config_id)


@router.get("/config/{config_id}/export", response_model=str)
@handle_route_errors
@track_route_metrics("advanced_dl.export_config")
async def export_config(config_id: str, format: str = "yaml"):
    """Exportar configuración"""
    return config_service.export_config(config_id, format)


# Experiment Logging
class InitializeLoggingRequest(BaseModel):
    backend: str = LoggingBackend.TENSORBOARD
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


@router.post("/logging/initialize", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.init_logging")
async def initialize_logging(experiment_id: str, request: InitializeLoggingRequest):
    """Inicializar logging"""
    return logging_service.initialize_logging(
        experiment_id,
        request.backend,
        request.project_name,
        request.run_name,
        request.config
    )


@router.post("/logging/metrics", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.log_metrics")
async def log_metrics(experiment_id: str, metrics: Dict[str, float], step: Optional[int] = None):
    """Registrar métricas"""
    return logging_service.log_metrics(experiment_id, metrics, step)


@router.post("/logging/finish", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.finish_logging")
async def finish_logging(experiment_id: str):
    """Finalizar logging"""
    return logging_service.finish_logging(experiment_id)


# Model Serving
@router.post("/serving/export", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.export_model")
async def export_model(
    model_id: str,
    format: str = ServingFormat.TORCHSCRIPT.value,
    input_shape: Optional[List[int]] = None
):
    """Exportar modelo para serving"""
    return serving_service.export_model(model_id, format, input_shape)


@router.post("/serving/quantize", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.quantize")
async def quantize_model(model_id: str, quantization_type: str = "int8"):
    """Cuantizar modelo"""
    return serving_service.quantize_model(model_id, quantization_type)


@router.post("/serving/deploy", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.deploy")
async def create_deployment(
    model_id: str,
    deployment_name: str,
    endpoint_url: Optional[str] = None,
    replicas: int = 1
):
    """Crear deployment"""
    return serving_service.create_deployment(model_id, deployment_name, endpoint_url, replicas)


@router.get("/serving/deployments/{deployment_id}/status", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.get_deployment_status")
async def get_deployment_status(deployment_id: str):
    """Obtener estado del deployment"""
    return serving_service.get_deployment_status(deployment_id)


@router.post("/serving/deployments/{deployment_id}/scale", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.scale_deployment")
async def scale_deployment(deployment_id: str, target_replicas: int):
    """Escalar deployment"""
    return serving_service.scale_deployment(deployment_id, target_replicas)


# Advanced Training
@router.post("/training/scheduler", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.create_scheduler")
async def create_scheduler(
    scheduler_type: str = "cosine",
    initial_lr: float = 0.001,
    **kwargs
):
    """Crear scheduler de learning rate"""
    return training_service.create_learning_rate_scheduler(scheduler_type, initial_lr, **kwargs)


@router.post("/training/early-stopping", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.setup_early_stopping")
async def setup_early_stopping(
    patience: int = 10,
    min_delta: float = 0.0,
    monitor: str = "val_loss",
    mode: str = "min"
):
    """Configurar early stopping"""
    return training_service.setup_early_stopping(patience, min_delta, monitor, mode)


@router.post("/training/gradient-clipping", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.setup_gradient_clipping")
async def setup_gradient_clipping(clip_value: float = 1.0, clip_type: str = "norm"):
    """Configurar gradient clipping"""
    return training_service.setup_gradient_clipping(clip_value, clip_type)


@router.post("/training/checkpointing", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.setup_checkpointing")
async def setup_checkpointing(
    checkpoint_dir: str = "checkpoints",
    save_best: bool = True,
    save_last: bool = True,
    monitor: str = "val_loss"
):
    """Configurar checkpointing"""
    return training_service.setup_checkpointing(checkpoint_dir, save_best, save_last, monitor)


@router.post("/training/loop", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("advanced_dl.create_training_loop")
async def create_training_loop(model_id: str, config: Dict[str, Any]):
    """Crear loop de entrenamiento avanzado"""
    return training_service.create_training_loop(model_id, config)




