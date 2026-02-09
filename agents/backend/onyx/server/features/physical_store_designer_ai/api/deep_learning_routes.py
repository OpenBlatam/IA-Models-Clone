"""
Deep Learning API Routes
"""

from fastapi import APIRouter
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from ..services.deep_learning_service import DeepLearningService
from ..services.diffusion_service import DiffusionService
from ..services.llm_finetuning_service import LLMFineTuningService
from ..services.embeddings_service import EmbeddingsService
from ..services.experiment_tracking_service import ExperimentTrackingService, ExperimentStatus
from ..services.gradio_integration_service import GradioIntegrationService
from ..core.route_helpers import handle_route_errors, track_route_metrics

router = APIRouter()

# Initialize services
dl_service = DeepLearningService()
diffusion_service = DiffusionService()
finetuning_service = LLMFineTuningService()
embeddings_service = EmbeddingsService()
experiment_service = ExperimentTrackingService()
gradio_service = GradioIntegrationService()


# Deep Learning Models
class CreateModelRequest(BaseModel):
    model_name: str
    model_type: str = "store_design"
    config: Optional[Dict[str, Any]] = None


class TrainModelRequest(BaseModel):
    training_data: List[Dict[str, Any]]
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    use_mixed_precision: bool = True


class PredictRequest(BaseModel):
    input_data: List[float]


@router.post("/deep-learning/models", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.create_model")
async def create_model(request: CreateModelRequest):
    """Crear modelo de deep learning"""
    return dl_service.create_model(
        request.model_name,
        request.model_type,
        request.config
    )


@router.post("/deep-learning/models/{model_id}/train", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.train")
async def train_model(model_id: str, request: TrainModelRequest):
    """Entrenar modelo"""
    return await dl_service.train_model(
        model_id,
        request.training_data,
        request.epochs,
        request.batch_size,
        request.learning_rate,
        request.use_mixed_precision
    )


@router.post("/deep-learning/models/{model_id}/predict", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.predict")
async def predict(model_id: str, request: PredictRequest):
    """Hacer predicción"""
    return await dl_service.predict(model_id, request.input_data)


@router.post("/deep-learning/models/{model_id}/checkpoints", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.save_checkpoint")
async def save_checkpoint(model_id: str, checkpoint_name: str):
    """Guardar checkpoint"""
    return dl_service.save_checkpoint(model_id, checkpoint_name)


# Diffusion Models
class CreatePipelineRequest(BaseModel):
    pipeline_name: str = "stable_diffusion"
    model_id: str = "runwayml/stable-diffusion-v1-5"
    scheduler_type: str = "ddim"


class GenerateImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512


@router.post("/diffusion/pipelines", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.create_pipeline")
async def create_pipeline(request: CreatePipelineRequest):
    """Crear pipeline de difusión"""
    return diffusion_service.create_diffusion_pipeline(
        request.pipeline_name,
        request.model_id,
        request.scheduler_type
    )


@router.post("/diffusion/pipelines/{pipeline_id}/generate", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.generate_image")
async def generate_image(pipeline_id: str, request: GenerateImageRequest):
    """Generar imagen de tienda"""
    return await diffusion_service.generate_store_image(
        pipeline_id,
        request.prompt,
        request.negative_prompt,
        request.num_inference_steps,
        request.guidance_scale,
        request.width,
        request.height
    )


@router.post("/diffusion/pipelines/{pipeline_id}/variations", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.generate_variations")
async def generate_variations(pipeline_id: str, base_prompt: str, num_variations: int = 4):
    """Generar variaciones de diseño"""
    return await diffusion_service.generate_store_variations(
        pipeline_id,
        base_prompt,
        num_variations
    )


# LLM Fine-tuning
class PrepareDatasetRequest(BaseModel):
    dataset_name: str
    data: List[Dict[str, str]]
    split_ratio: float = 0.8


class CreateFinetuningJobRequest(BaseModel):
    base_model: str = "gpt2"
    dataset_id: Optional[str] = None
    method: str = "lora"
    config: Optional[Dict[str, Any]] = None


@router.post("/llm-finetuning/datasets", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.prepare_dataset")
async def prepare_dataset(request: PrepareDatasetRequest):
    """Preparar dataset para fine-tuning"""
    return finetuning_service.prepare_dataset(
        request.dataset_name,
        request.data,
        request.split_ratio
    )


@router.post("/llm-finetuning/jobs", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.create_finetuning_job")
async def create_finetuning_job(request: CreateFinetuningJobRequest):
    """Crear job de fine-tuning"""
    return finetuning_service.create_finetuning_job(
        request.base_model,
        request.dataset_id,
        request.method,
        request.config
    )


@router.post("/llm-finetuning/jobs/{job_id}/start", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.start_finetuning")
async def start_finetuning(job_id: str):
    """Iniciar fine-tuning"""
    return await finetuning_service.start_finetuning(job_id)


@router.get("/llm-finetuning/jobs/{job_id}/model", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.get_finetuned_model")
async def get_finetuned_model(job_id: str):
    """Obtener modelo fine-tuneado"""
    model = finetuning_service.get_finetuned_model(job_id)
    if not model:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Modelo no encontrado o no completado")
    return model


# Embeddings
class GenerateEmbeddingRequest(BaseModel):
    text: str
    document_id: Optional[str] = None


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/embeddings/generate", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.generate_embedding")
async def generate_embedding(request: GenerateEmbeddingRequest):
    """Generar embedding"""
    return await embeddings_service.generate_embedding(
        request.text,
        request.document_id
    )


@router.post("/embeddings/search", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.semantic_search")
async def semantic_search(request: SemanticSearchRequest):
    """Búsqueda semántica"""
    return await embeddings_service.semantic_search(
        request.query,
        request.top_k
    )


@router.post("/embeddings/similar-designs", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.find_similar_designs")
async def find_similar_designs(design_description: str, top_k: int = 5):
    """Encontrar diseños similares"""
    return await embeddings_service.find_similar_designs(
        design_description,
        top_k
    )


# Experiment Tracking
class CreateExperimentRequest(BaseModel):
    experiment_name: str
    description: str
    hyperparameters: Dict[str, Any]
    tags: Optional[List[str]] = None


class CreateRunRequest(BaseModel):
    run_name: str
    config: Optional[Dict[str, Any]] = None


class LogMetricRequest(BaseModel):
    metric_name: str
    value: float
    step: Optional[int] = None


@router.post("/experiments", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.create_experiment")
async def create_experiment(request: CreateExperimentRequest):
    """Crear experimento"""
    return experiment_service.create_experiment(
        request.experiment_name,
        request.description,
        request.hyperparameters,
        request.tags
    )


@router.post("/experiments/{experiment_id}/runs", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.create_run")
async def create_run(experiment_id: str, request: CreateRunRequest):
    """Crear run"""
    return experiment_service.create_run(
        experiment_id,
        request.run_name,
        request.config
    )


@router.post("/experiments/runs/{run_id}/metrics", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.log_metric")
async def log_metric(run_id: str, request: LogMetricRequest):
    """Registrar métrica"""
    return experiment_service.log_metric(
        run_id,
        request.metric_name,
        request.value,
        request.step
    )


@router.post("/experiments/runs/{run_id}/complete", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.complete_run")
async def complete_run(run_id: str, final_metrics: Optional[Dict[str, float]] = None):
    """Completar run"""
    success = experiment_service.complete_run(run_id, final_metrics)
    return {"run_id": run_id, "completed": success}


@router.get("/experiments/{experiment_id}/results", response_model=Dict[str, Any])
async def get_experiment_results(experiment_id: str):
    """Obtener resultados del experimento"""
    try:
        results = experiment_service.get_experiment_results(experiment_id)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        from ..core.exceptions import ServiceError
        raise ServiceError("experiment_service", str(e))


# Gradio Integration
class CreateDemoAppRequest(BaseModel):
    app_name: str
    description: str
    functions: List[Dict[str, Any]]


@router.post("/gradio/apps", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.create_demo_app")
async def create_demo_app(request: CreateDemoAppRequest):
    """Crear aplicación demo"""
    return gradio_service.create_demo_app(
        request.app_name,
        request.description,
        request.functions
    )


@router.post("/gradio/store-designer-demo", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.create_store_demo")
async def create_store_designer_demo(store_id: str):
    """Crear demo de diseñador de tiendas"""
    return gradio_service.create_store_designer_demo(store_id)


@router.post("/gradio/ml-model-demo", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.create_ml_demo")
async def create_ml_model_demo(model_id: str, model_type: str):
    """Crear demo de modelo ML"""
    return gradio_service.create_ml_model_demo(model_id, model_type)


@router.post("/gradio/apps/{app_id}/launch", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("deep_learning.launch_app")
async def launch_app(app_id: str, share: bool = False, server_port: int = 7860):
    """Lanzar aplicación Gradio"""
    return gradio_service.launch_app(app_id, share, server_port)




