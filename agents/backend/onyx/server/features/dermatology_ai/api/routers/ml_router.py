"""
ML/AI Router - Handles advanced ML and AI analysis endpoints
Enhanced with ML Model Manager and optimizations
"""

from fastapi import APIRouter, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import Optional
import json
import numpy as np

from ...api.services_locator import get_service
from ...core.ml_model_manager import MLModelManager, ModelConfig, ModelType
from ...core.ml_optimizer import MLOptimizer, OptimizationConfig
from ...core.experiment_tracker import ExperimentTracker, ExperimentConfig, ExperimentMetrics
from ...utils.logger import logger

router = APIRouter(prefix="/dermatology", tags=["ml-ai"])

# Initialize ML components
_ml_model_manager: Optional[MLModelManager] = None
_ml_optimizer: Optional[MLOptimizer] = None
_experiment_tracker: Optional[ExperimentTracker] = None


def get_ml_model_manager() -> MLModelManager:
    """Get or create ML Model Manager"""
    global _ml_model_manager
    if _ml_model_manager is None:
        _ml_model_manager = MLModelManager()
    return _ml_model_manager


def get_ml_optimizer() -> MLOptimizer:
    """Get or create ML Optimizer"""
    global _ml_optimizer
    if _ml_optimizer is None:
        _ml_optimizer = MLOptimizer()
    return _ml_optimizer


def get_experiment_tracker() -> ExperimentTracker:
    """Get or create Experiment Tracker"""
    global _experiment_tracker
    if _experiment_tracker is None:
        _experiment_tracker = ExperimentTracker()
    return _experiment_tracker


@router.post("/ml/predict")
async def ml_predict(
    model_type: str = Form(...),
    features: str = Form(...)
):
    """Predicción usando modelos ML"""
    try:
        enhanced_ml = get_service("enhanced_ml")
        features_array = np.array(json.loads(features))
        prediction = enhanced_ml.predict(model_type, features_array)
        return JSONResponse(content={
            "success": True,
            "prediction": prediction.to_dict() if hasattr(prediction, 'to_dict') else prediction
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ml/stats")
async def get_ml_stats():
    """Obtiene estadísticas de modelos ML"""
    try:
        enhanced_ml = get_service("enhanced_ml")
        stats = enhanced_ml.get_model_stats()
        return JSONResponse(content={"success": True, "stats": stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ml-advanced/add-data")
async def add_ml_data(
    user_id: str = Form(...),
    data_type: str = Form(...),
    data: str = Form(...)
):
    """Agrega datos para análisis ML avanzado"""
    try:
        advanced_ml_analysis = get_service("advanced_ml_analysis")
        data_dict = json.loads(data)
        result = advanced_ml_analysis.add_data(user_id, data_type, data_dict)
        return JSONResponse(content={"success": True, "result": result.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ml-advanced/analyze/{user_id}")
async def analyze_ml_advanced(user_id: str):
    """Análisis ML avanzado"""
    try:
        advanced_ml_analysis = get_service("advanced_ml_analysis")
        report = advanced_ml_analysis.analyze(user_id)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/conditions/predict")
async def predict_condition(
    user_id: str = Form(...),
    analysis_data: str = Form(...)
):
    """Predice condiciones de piel"""
    try:
        condition_predictor = get_service("condition_predictor")
        data_dict = json.loads(analysis_data)
        prediction = condition_predictor.predict_condition(user_id, data_dict)
        return JSONResponse(content={
            "success": True,
            "prediction": prediction.to_dict() if hasattr(prediction, 'to_dict') else prediction
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/future-prediction/add-data")
async def add_future_prediction_data(
    user_id: str = Form(...),
    date: str = Form(...),
    metrics: str = Form(...)
):
    """Agrega datos para predicción futura"""
    try:
        future_prediction = get_service("future_prediction")
        metrics_dict = json.loads(metrics)
        result = future_prediction.add_data(user_id, date, metrics_dict)
        return JSONResponse(content={"success": True, "result": result.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/future-prediction/generate/{user_id}")
async def generate_future_prediction(user_id: str, days: int = Query(30)):
    """Genera predicción futura"""
    try:
        future_prediction = get_service("future_prediction")
        report = future_prediction.generate_prediction(user_id, days)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ml/models/register")
async def register_ml_model(
    model_id: str = Form(...),
    model_type: str = Form(...),
    model_path: Optional[str] = Form(None),
    device: str = Form("cpu"),
    batch_size: int = Form(32),
    use_cache: bool = Form(True)
):
    """Registra un modelo ML"""
    try:
        manager = get_ml_model_manager()
        config = ModelConfig(
            model_id=model_id,
            model_type=ModelType(model_type),
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            use_cache=use_cache
        )
        manager.register_model(config)
        return JSONResponse(content={"success": True, "model_id": model_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ml/models/predict")
async def ml_model_predict(
    model_id: str = Form(...),
    input_data: str = Form(...),
    use_cache: Optional[bool] = Form(None)
):
    """Ejecuta predicción con modelo ML"""
    try:
        manager = get_ml_model_manager()
        input_array = np.array(json.loads(input_data))
        result = manager.predict(model_id, input_array, use_cache=use_cache)
        return JSONResponse(content={
            "success": True,
            "prediction": result.prediction.tolist() if hasattr(result.prediction, "tolist") else result.prediction,
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "cached": result.cached
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ml/models/batch-predict")
async def ml_model_batch_predict(
    model_id: str = Form(...),
    input_batch: str = Form(...),
    batch_size: Optional[int] = Form(None)
):
    """Ejecuta predicción en batch"""
    try:
        manager = get_ml_model_manager()
        batch_array = [np.array(item) for item in json.loads(input_batch)]
        results = manager.batch_predict(model_id, batch_array, batch_size)
        return JSONResponse(content={
            "success": True,
            "results": [
                {
                    "prediction": r.prediction.tolist() if hasattr(r.prediction, "tolist") else r.prediction,
                    "confidence": r.confidence,
                    "processing_time": r.processing_time
                }
                for r in results
            ]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ml/models/stats")
async def get_ml_model_stats():
    """Obtiene estadísticas de modelos ML"""
    try:
        manager = get_ml_model_manager()
        stats = manager.get_stats()
        return JSONResponse(content={"success": True, "stats": stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ml/experiments/create")
async def create_experiment(
    experiment_id: str = Form(...),
    name: str = Form(...),
    description: str = Form(...),
    model_type: str = Form(...),
    hyperparameters: str = Form(...),
    dataset_info: Optional[str] = Form(None)
):
    """Crea un nuevo experimento"""
    try:
        tracker = get_experiment_tracker()
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            model_type=model_type,
            hyperparameters=json.loads(hyperparameters),
            dataset_info=json.loads(dataset_info) if dataset_info else {}
        )
        exp_id = tracker.create_experiment(config)
        return JSONResponse(content={"success": True, "experiment_id": exp_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ml/experiments/log-metrics")
async def log_experiment_metrics(
    experiment_id: str = Form(...),
    epoch: int = Form(...),
    train_loss: float = Form(...),
    val_loss: Optional[float] = Form(None),
    train_accuracy: Optional[float] = Form(None),
    val_accuracy: Optional[float] = Form(None),
    learning_rate: Optional[float] = Form(None)
):
    """Registra métricas de experimento"""
    try:
        tracker = get_experiment_tracker()
        tracker.current_experiment = experiment_id
        metrics = ExperimentMetrics(
            experiment_id=experiment_id,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            learning_rate=learning_rate
        )
        tracker.log_metrics(metrics)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ml/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Obtiene información de un experimento"""
    try:
        tracker = get_experiment_tracker()
        summary = tracker.get_experiment_summary(experiment_id)
        return JSONResponse(content={"success": True, "experiment": summary})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/ml/experiments")
async def list_experiments():
    """Lista todos los experimentos"""
    try:
        tracker = get_experiment_tracker()
        experiments = tracker.list_experiments()
        return JSONResponse(content={"success": True, "experiments": experiments})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

