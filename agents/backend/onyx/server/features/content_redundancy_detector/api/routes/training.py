"""
Training Router - Model training endpoints
"""

import time
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import JSONResponse

try:
    from custom_training_engine import custom_training_engine, TrainingDataset, TrainingConfig
except ImportError:
    logging.warning("custom_training_engine module not available")
    custom_training_engine = None
    TrainingDataset = None
    TrainingConfig = None

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/training", tags=["Training"])


@router.post("/create-job", response_model=Dict[str, Any])
async def create_training_job(
    dataset: Dict[str, Any],
    config: Dict[str, Any]
) -> JSONResponse:
    """Create a new model training job"""
    logger.info(f"Creating training job for dataset: {dataset.get('name', 'unknown')}")
    
    if not custom_training_engine:
        raise HTTPException(status_code=503, detail="Training engine not available")
    
    try:
        if TrainingDataset and TrainingConfig:
            dataset_obj = TrainingDataset(**dataset)
            config_obj = TrainingConfig(**config)
            job_id = await custom_training_engine.create_training_job(dataset_obj, config_obj)
        else:
            job_id = await custom_training_engine.create_training_job(dataset, config)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "job_id": job_id,
                "status": "created",
                "timestamp": time.time()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error creating training job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/start/{job_id}", response_model=Dict[str, Any])
async def start_training_job(job_id: str = Path(...)) -> JSONResponse:
    """Start a training job"""
    logger.info(f"Starting training job: {job_id}")
    
    if not custom_training_engine:
        raise HTTPException(status_code=503, detail="Training engine not available")
    
    try:
        started = await custom_training_engine.start_training(job_id)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "job_id": job_id,
                "started": started,
                "timestamp": time.time()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error starting training job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/jobs", response_model=Dict[str, Any])
async def list_training_jobs() -> JSONResponse:
    """List all training jobs"""
    logger.info("Listing training jobs")
    
    if not custom_training_engine:
        raise HTTPException(status_code=503, detail="Training engine not available")
    
    try:
        jobs = await custom_training_engine.list_training_jobs()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "jobs": [
                    job.model_dump() if hasattr(job, 'model_dump') else job
                    for job in jobs
                ],
                "count": len(jobs),
                "timestamp": time.time()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error listing training jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_training_job_status(job_id: str = Path(...)) -> JSONResponse:
    """Get training job status"""
    logger.info(f"Getting training job status: {job_id}")
    
    if not custom_training_engine:
        raise HTTPException(status_code=503, detail="Training engine not available")
    
    try:
        job = await custom_training_engine.get_training_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        return JSONResponse(content={
            "success": True,
            "data": job.model_dump() if hasattr(job, 'model_dump') else job,
            "error": None
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training job status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/deploy/{job_id}", response_model=Dict[str, Any])
async def deploy_trained_model(
    job_id: str = Path(...),
    model_name: str = None,
    deploy_data: Dict[str, Any] = None
) -> JSONResponse:
    """Deploy a trained model"""
    logger.info(f"Deploying model from job: {job_id}")
    
    if not custom_training_engine:
        raise HTTPException(status_code=503, detail="Training engine not available")
    
    try:
        if not model_name and deploy_data:
            model_name = deploy_data.get("model_name")
        
        if not model_name:
            raise ValueError("Model name is required")
        
        deployed = await custom_training_engine.deploy_model(job_id, model_name)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "job_id": job_id,
                "model_name": model_name,
                "deployed": deployed,
                "timestamp": time.time()
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/models", response_model=Dict[str, Any])
async def list_deployed_models() -> JSONResponse:
    """List deployed models"""
    logger.info("Listing deployed models")
    
    if not custom_training_engine:
        raise HTTPException(status_code=503, detail="Training engine not available")
    
    try:
        models = await custom_training_engine.list_deployed_models()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "models": models,
                "count": len(models),
                "timestamp": time.time()
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error listing deployed models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/predict/{model_name}", response_model=Dict[str, Any])
async def predict_with_custom_model(
    model_name: str = Path(...),
    prediction_data: Dict[str, Any] = None
) -> JSONResponse:
    """Make prediction with custom trained model"""
    logger.info(f"Making prediction with model: {model_name}")
    
    if not custom_training_engine:
        raise HTTPException(status_code=503, detail="Training engine not available")
    
    try:
        text = prediction_data.get("text", "") if prediction_data else ""
        if not text:
            raise ValueError("Text is required")
        
        result = await custom_training_engine.predict_with_custom_model(model_name, text)
        
        return JSONResponse(content={
            "success": True,
            "data": result,
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")






