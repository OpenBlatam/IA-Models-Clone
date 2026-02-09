"""
Machine Learning Routes
Real, working ML endpoints for AI document processing
"""

import logging
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
from machine_learning_system import ml_system

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/ml", tags=["Machine Learning"])

@router.post("/train-model")
async def train_model(
    model_name: str = Form(...),
    training_data: List[dict] = Form(...),
    model_type: str = Form("classification"),
    algorithm: str = Form("random_forest")
):
    """Train a machine learning model"""
    try:
        result = await ml_system.train_model(model_name, training_data, model_type, algorithm)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def predict(
    model_name: str = Form(...),
    text: str = Form(...)
):
    """Make prediction using trained model"""
    try:
        result = await ml_system.predict(model_name, text)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-predict")
async def batch_predict(
    model_name: str = Form(...),
    texts: List[str] = Form(...)
):
    """Make batch predictions"""
    try:
        result = await ml_system.batch_predict(model_name, texts)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error making batch predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/auto-ml")
async def auto_ml(
    training_data: List[dict] = Form(...),
    model_type: str = Form("classification")
):
    """Automated machine learning"""
    try:
        result = await ml_system.auto_ml(training_data, model_type)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in auto ML: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save-model")
async def save_model(
    model_name: str = Form(...),
    file_path: str = Form(...)
):
    """Save trained model to file"""
    try:
        result = await ml_system.save_model(model_name, file_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load-model")
async def load_model(
    model_name: str = Form(...),
    file_path: str = Form(...)
):
    """Load model from file"""
    try:
        result = await ml_system.load_model(model_name, file_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_models():
    """Get all models"""
    try:
        result = ml_system.get_models()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-metrics/{model_name}")
async def get_model_metrics(model_name: str):
    """Get model metrics"""
    try:
        result = ml_system.get_model_metrics(model_name)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/all-model-metrics")
async def get_all_model_metrics():
    """Get all model metrics"""
    try:
        result = ml_system.get_model_metrics()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting all model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ml-stats")
async def get_ml_stats():
    """Get ML statistics"""
    try:
        result = ml_system.get_ml_stats()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting ML stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-info/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed model information"""
    try:
        models = ml_system.get_models()
        
        if model_name not in models["models"]:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = models["models"][model_name]
        metrics = ml_system.get_model_metrics(model_name)
        
        return JSONResponse(content={
            "model_name": model_name,
            "model_type": model_info["type"],
            "trained": model_info["trained"],
            "accuracy": model_info["accuracy"],
            "created_at": model_info["created_at"],
            "metrics": metrics if "error" not in metrics else {},
            "features": {
                "prediction": True,
                "batch_prediction": True,
                "probability_prediction": True,
                "model_saving": True,
                "model_loading": True
            }
        })
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-algorithms")
async def get_available_algorithms():
    """Get available ML algorithms"""
    try:
        algorithms = {
            "classification": [
                "random_forest",
                "logistic_regression",
                "svm",
                "naive_bayes",
                "gradient_boosting"
            ],
            "regression": [
                "random_forest",
                "linear_regression"
            ],
            "clustering": [
                "kmeans",
                "dbscan"
            ]
        }
        
        return JSONResponse(content={
            "algorithms": algorithms,
            "model_types": list(algorithms.keys())
        })
    except Exception as e:
        logger.error(f"Error getting available algorithms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ml-dashboard")
async def get_ml_dashboard():
    """Get comprehensive ML dashboard"""
    try:
        # Get all ML data
        models = ml_system.get_models()
        stats = ml_system.get_ml_stats()
        all_metrics = ml_system.get_model_metrics()
        
        # Calculate additional metrics
        total_models = len(models["models"])
        trained_models = len([m for m in models["models"].values() if m["trained"]])
        untrained_models = total_models - trained_models
        
        # Calculate average accuracy
        accuracies = [m["accuracy"] for m in models["models"].values() if m["trained"]]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        # Get model types distribution
        model_types = {}
        for model in models["models"].values():
            model_type = model["type"]
            model_types[model_type] = model_types.get(model_type, 0) + 1
        
        # Get algorithm distribution
        algorithms = {}
        for model_name, model in models["models"].items():
            if model["trained"]:
                # This would need to be stored during training
                algorithms["unknown"] = algorithms.get("unknown", 0) + 1
        
        dashboard_data = {
            "timestamp": stats["uptime_seconds"],
            "overview": {
                "total_models": total_models,
                "trained_models": trained_models,
                "untrained_models": untrained_models,
                "average_accuracy": round(avg_accuracy, 3),
                "total_predictions": stats["stats"]["predictions_made"],
                "uptime_hours": stats["uptime_hours"]
            },
            "model_metrics": {
                "total_models": stats["stats"]["total_models"],
                "trained_models": stats["stats"]["trained_models"],
                "failed_models": stats["stats"]["failed_models"],
                "predictions_made": stats["stats"]["predictions_made"]
            },
            "model_types_distribution": model_types,
            "algorithms_distribution": algorithms,
            "models": models["models"],
            "all_metrics": all_metrics
        }
        
        return JSONResponse(content=dashboard_data)
    except Exception as e:
        logger.error(f"Error getting ML dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-performance")
async def get_model_performance():
    """Get model performance analysis"""
    try:
        models = ml_system.get_models()
        all_metrics = ml_system.get_model_metrics()
        
        performance_data = {
            "timestamp": ml_system.get_ml_stats()["uptime_seconds"],
            "model_performance": {},
            "best_models": {
                "classification": None,
                "regression": None,
                "clustering": None
            },
            "performance_summary": {
                "total_models": len(models["models"]),
                "trained_models": len([m for m in models["models"].values() if m["trained"]]),
                "models_with_metrics": len(all_metrics)
            }
        }
        
        # Analyze each model
        for model_name, model_info in models["models"].items():
            if model_info["trained"]:
                metrics = all_metrics.get(model_name, {})
                
                performance_data["model_performance"][model_name] = {
                    "model_type": model_info["type"],
                    "accuracy": model_info["accuracy"],
                    "metrics": metrics,
                    "created_at": model_info["created_at"]
                }
                
                # Track best models by type
                model_type = model_info["type"]
                current_best = performance_data["best_models"][model_type]
                
                if not current_best or model_info["accuracy"] > current_best["accuracy"]:
                    performance_data["best_models"][model_type] = {
                        "model_name": model_name,
                        "accuracy": model_info["accuracy"],
                        "metrics": metrics
                    }
        
        return JSONResponse(content=performance_data)
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-ml")
async def health_check_ml():
    """ML system health check"""
    try:
        stats = ml_system.get_ml_stats()
        models = ml_system.get_models()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Machine Learning System",
            "version": "1.0.0",
            "features": {
                "model_training": True,
                "model_prediction": True,
                "batch_prediction": True,
                "auto_ml": True,
                "model_saving": True,
                "model_loading": True,
                "model_metrics": True,
                "performance_analysis": True
            },
            "ml_stats": stats["stats"],
            "system_status": {
                "total_models": stats["models_count"],
                "trained_models": stats["trained_models_count"],
                "model_metrics": stats["model_metrics_count"],
                "uptime_hours": stats["uptime_hours"]
            },
            "available_algorithms": {
                "classification": ["random_forest", "logistic_regression", "svm", "naive_bayes", "gradient_boosting"],
                "regression": ["random_forest", "linear_regression"],
                "clustering": ["kmeans", "dbscan"]
            }
        })
    except Exception as e:
        logger.error(f"Error in ML health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))













