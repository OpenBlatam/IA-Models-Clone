"""
ML Ops API Routes - Evaluación, tuning, compresión, interpretabilidad, registry, monitoring
"""

from fastapi import APIRouter
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from ..services.model_evaluation_service import ModelEvaluationService, MetricType
from ..services.hyperparameter_tuning_service import HyperparameterTuningService
from ..services.model_compression_service import ModelCompressionService, CompressionMethod
from ..services.model_interpretability_service import ModelInterpretabilityService, InterpretabilityMethod
from ..services.model_registry_service import ModelRegistryService, ModelStage
from ..services.production_monitoring_service import ProductionMonitoringService, AlertLevel
from ..core.route_helpers import handle_route_errors, track_route_metrics

router = APIRouter()

# Initialize services
eval_service = ModelEvaluationService()
tuning_service = HyperparameterTuningService()
compression_service = ModelCompressionService()
interpretability_service = ModelInterpretabilityService()
registry_service = ModelRegistryService()
monitoring_service = ProductionMonitoringService()


# Model Evaluation
class EvaluateModelRequest(BaseModel):
    predictions: List[Any]
    ground_truth: List[Any]
    metric_type: str = MetricType.REGRESSION.value
    task_name: str = "default"


@router.post("/evaluation/evaluate", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.evaluate")
async def evaluate_model(model_id: str, request: EvaluateModelRequest):
    """Evaluar modelo"""
    return eval_service.evaluate_model(
        model_id,
        request.predictions,
        request.ground_truth,
        request.metric_type,
        request.task_name
    )


@router.post("/evaluation/cross-validate", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.cross_validate")
async def cross_validate(
    model_id: str,
    data: List[Dict[str, Any]],
    n_folds: int = 5,
    metric_type: str = MetricType.REGRESSION.value
):
    """Cross-validation"""
    return eval_service.cross_validate(model_id, data, n_folds, metric_type)


@router.post("/evaluation/compare", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.compare_models")
async def compare_models(model_ids: List[str], evaluation_ids: List[str]):
    """Comparar modelos"""
    return eval_service.compare_models(model_ids, evaluation_ids)


# Hyperparameter Tuning
class CreateStudyRequest(BaseModel):
    study_name: str
    direction: str = "minimize"
    sampler: str = "tpe"


@router.post("/tuning/studies", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.create_study")
async def create_study(request: CreateStudyRequest):
    """Crear estudio de optimización"""
    return tuning_service.create_study(request.study_name, request.direction, request.sampler)


@router.post("/tuning/studies/{study_id}/search-space", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.define_search_space")
async def define_search_space(study_id: str, hyperparameters: Dict[str, Dict[str, Any]]):
    """Definir espacio de búsqueda"""
    return tuning_service.define_search_space(study_id, hyperparameters)


@router.post("/tuning/studies/{study_id}/optimize", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.optimize")
async def run_optimization(study_id: str, n_trials: int = 100):
    """Ejecutar optimización"""
    return tuning_service.run_optimization(study_id, n_trials)


@router.get("/tuning/studies/{study_id}/best-params", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.get_best_params")
async def get_best_params(study_id: str):
    """Obtener mejores parámetros"""
    return tuning_service.get_best_params(study_id)


# Model Compression
@router.post("/compression/prune", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.prune")
async def prune_model(
    model_id: str,
    pruning_method: str = "magnitude",
    pruning_ratio: float = 0.3,
    structured: bool = False
):
    """Aplicar pruning"""
    return compression_service.prune_model(model_id, pruning_method, pruning_ratio, structured)


@router.post("/compression/quantize", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.quantize")
async def apply_quantization(model_id: str, quantization_type: str = "int8"):
    """Aplicar cuantización"""
    return compression_service.apply_quantization(model_id, quantization_type)


@router.post("/compression/distill", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.distill")
async def apply_distillation(
    teacher_model_id: str,
    student_model_id: str,
    temperature: float = 3.0,
    alpha: float = 0.7
):
    """Aplicar knowledge distillation"""
    return compression_service.apply_knowledge_distillation(
        teacher_model_id, student_model_id, temperature, alpha
    )


@router.get("/compression/{compression_id}/stats", response_model=Dict[str, Any])
async def get_compression_stats(compression_id: str):
    """Obtener estadísticas de compresión"""
    try:
        return compression_service.get_compression_stats(compression_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Model Interpretability
@router.post("/interpretability/explain", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.explain")
async def explain_prediction(
    model_id: str,
    input_data: Any,
    method: str = InterpretabilityMethod.GRADIENT.value,
    target_class: Optional[int] = None
):
    """Explicar predicción"""
    return interpretability_service.explain_prediction(model_id, input_data, method, target_class)


@router.post("/interpretability/attention", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.visualize_attention")
async def visualize_attention(model_id: str, input_sequence: List[str], layer: int = -1):
    """Visualizar atención"""
    return interpretability_service.visualize_attention(model_id, input_sequence, layer)


@router.get("/interpretability/importance/{model_id}", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.feature_importance")
async def feature_importance(model_id: str, n_features: int = 10):
    """Calcular importancia de características"""
    return interpretability_service.feature_importance(model_id, n_features)


# Model Registry
class RegisterModelRequest(BaseModel):
    model_name: str
    model_type: str
    description: str
    tags: Optional[List[str]] = None


@router.post("/registry/models", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.register_model")
async def register_model(request: RegisterModelRequest):
    """Registrar modelo"""
    return registry_service.register_model(
        request.model_name,
        request.model_type,
        request.description,
        request.tags
    )


@router.post("/registry/models/{model_id}/versions", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.create_version")
async def create_version(
    model_id: str,
    version_name: str,
    checkpoint_path: str,
    metrics: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Crear versión"""
    return registry_service.create_version(
        model_id, version_name, checkpoint_path, metrics, metadata
    )


@router.post("/registry/versions/{version_id}/promote", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.promote_version")
async def promote_version(model_id: str, version_id: str, target_stage: str):
    """Promover versión"""
    return registry_service.promote_version(model_id, version_id, target_stage)


@router.get("/registry/models/{model_id}/latest", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.get_latest_version")
async def get_latest_version(model_id: str, stage: Optional[str] = None):
    """Obtener última versión"""
    version = registry_service.get_latest_version(model_id, stage)
    if not version:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Versión no encontrada")
    return version


@router.post("/registry/models/{model_id}/compare", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.compare_versions")
async def compare_versions(model_id: str, version_ids: List[str]):
    """Comparar versiones"""
    return registry_service.compare_versions(model_id, version_ids)


@router.get("/registry/search", response_model=List[Dict[str, Any]])
@handle_route_errors
@track_route_metrics("ml_ops.search_models")
async def search_models(
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    stage: Optional[str] = None
):
    """Buscar modelos"""
    return registry_service.search_models(query, tags, stage)


# Production Monitoring
@router.post("/monitoring/register", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.register_monitoring")
async def register_for_monitoring(
    model_id: str,
    deployment_id: str,
    thresholds: Dict[str, float]
):
    """Registrar modelo para monitoreo"""
    return monitoring_service.register_model_for_monitoring(model_id, deployment_id, thresholds)


@router.post("/monitoring/predictions", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.record_prediction")
async def record_prediction(
    model_id: str,
    prediction: Any,
    latency_ms: float
):
    """Registrar predicción"""
    return monitoring_service.record_prediction(model_id, prediction, latency_ms)


@router.get("/monitoring/health/{model_id}", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.check_health")
async def check_model_health(model_id: str, time_window_minutes: int = 60):
    """Verificar salud del modelo"""
    return monitoring_service.check_model_health(model_id, time_window_minutes)


@router.post("/monitoring/drift", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.detect_drift")
async def detect_drift(
    model_id: str,
    reference_data: List[Any],
    current_data: List[Any]
):
    """Detectar drift"""
    return monitoring_service.detect_drift(model_id, reference_data, current_data)


@router.post("/monitoring/alerts", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.generate_alert")
async def generate_alert(
    model_id: str,
    alert_type: str,
    message: str,
    level: str = AlertLevel.WARNING.value
):
    """Generar alerta"""
    return monitoring_service.generate_alert(model_id, alert_type, message, level)


@router.get("/monitoring/dashboard", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("ml_ops.get_dashboard")
async def get_dashboard(model_id: Optional[str] = None):
    """Obtener dashboard de monitoreo"""
    return monitoring_service.get_dashboard(model_id)




