from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import asyncio
import logging
from pathlib import Path
import json
from datetime import datetime

from model_training import (
from config_loader import (
    import uvicorn
from typing import Any, List, Dict, Optional
    create_model_trainer, 
    DeviceManager,
    quick_train_transformer,
    gradient_accumulation_train_transformer,
    multi_gpu_train_transformer,
    ultra_optimized_train_transformer
)
    load_config_from_yaml,
    create_experiment_config,
    save_experiment_config,
    validate_config
)

# Pydantic models for API
class TrainingRequest(BaseModel):
    model_name: str = Field(..., description="Model name (e.g., distilbert-base-uncased)")
    dataset_path: str = Field(..., description="Path to dataset")
    num_epochs: int = Field(default=5, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(default=16, ge=1, le=128, description="Batch size")
    learning_rate: float = Field(default=2e-5, gt=0, description="Learning rate")
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=32, description="Gradient accumulation steps")
    experiment_id: Optional[str] = Field(None, description="Custom experiment ID")
    description: Optional[str] = Field("", description="Experiment description")

class InferenceRequest(BaseModel):
    text: str = Field(..., description="Input text for inference")
    model_path: str = Field(..., description="Path to trained model")
    task_type: str = Field(default="classification", description="Task type")

class ExperimentStatus(BaseModel):
    experiment_id: str
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    current_loss: Optional[float]
    best_accuracy: Optional[float]
    start_time: datetime
    estimated_completion: Optional[datetime]

class TrainingResponse(BaseModel):
    experiment_id: str
    status: str
    message: str
    estimated_duration: Optional[int]

# FastAPI app setup
app = FastAPI(
    title="Blatam Academy NLP API",
    description="Production-ready NLP training and inference API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global state for tracking experiments
experiments: Dict[str, Dict[str, Any]] = {}
device_manager = DeviceManager()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token verification - replace with your auth logic"""
    if credentials.credentials != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return credentials.credentials

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Blatam Academy NLP API", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_experiments": len(experiments),
        "gpu_available": device_manager.gpu_available
    }

@app.post("/train/quick", response_model=TrainingResponse)
async def quick_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Quick training with default settings"""
    try:
        experiment_id = request.experiment_id or f"quick_{request.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize experiment tracking
        experiments[experiment_id] = {
            "status": "starting",
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": request.num_epochs,
            "start_time": datetime.now(),
            "request": request.dict()
        }
        
        # Start training in background
        background_tasks.add_task(
            run_quick_training,
            experiment_id,
            request.model_name,
            request.dataset_path,
            request.num_epochs
        )
        
        return TrainingResponse(
            experiment_id=experiment_id,
            status="started",
            message="Training started successfully",
            estimated_duration=request.num_epochs * 300  # Rough estimate: 5 min per epoch
        )
        
    except Exception as e:
        logger.error(f"Quick training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/advanced", response_model=TrainingResponse)
async def advanced_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Advanced training with gradient accumulation"""
    try:
        experiment_id = request.experiment_id or f"advanced_{request.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiments[experiment_id] = {
            "status": "starting",
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": request.num_epochs,
            "start_time": datetime.now(),
            "request": request.dict()
        }
        
        background_tasks.add_task(
            run_advanced_training,
            experiment_id,
            request
        )
        
        return TrainingResponse(
            experiment_id=experiment_id,
            status="started",
            message="Advanced training started successfully",
            estimated_duration=request.num_epochs * 400
        )
        
    except Exception as e:
        logger.error(f"Advanced training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/config", response_model=TrainingResponse)
async def config_training(
    config_path: str,
    experiment_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    token: str = Depends(verify_token)
):
    """Training using YAML configuration file"""
    try:
        if not Path(config_path).exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")
        
        config = load_config_from_yaml(config_path)
        if not validate_config(config):
            raise HTTPException(status_code=400, detail="Invalid configuration")
        
        experiment_id = experiment_id or f"config_{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiments[experiment_id] = {
            "status": "starting",
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": config.num_epochs,
            "start_time": datetime.now(),
            "config_path": config_path
        }
        
        if background_tasks:
            background_tasks.add_task(run_config_training, experiment_id, config)
        
        return TrainingResponse(
            experiment_id=experiment_id,
            status="started",
            message="Config-based training started successfully",
            estimated_duration=config.num_epochs * 350
        )
        
    except Exception as e:
        logger.error(f"Config training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments/{experiment_id}", response_model=ExperimentStatus)
async def get_experiment_status(experiment_id: str, token: str = Depends(verify_token)):
    """Get experiment status and progress"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    exp = experiments[experiment_id]
    return ExperimentStatus(
        experiment_id=experiment_id,
        status=exp["status"],
        progress=exp["progress"],
        current_epoch=exp["current_epoch"],
        total_epochs=exp["total_epochs"],
        current_loss=exp.get("current_loss"),
        best_accuracy=exp.get("best_accuracy"),
        start_time=exp["start_time"],
        estimated_completion=exp.get("estimated_completion")
    )

@app.get("/experiments")
async def list_experiments(token: str = Depends(verify_token)):
    """List all experiments"""
    return {
        "experiments": [
            {
                "experiment_id": exp_id,
                "status": exp["status"],
                "progress": exp["progress"],
                "start_time": exp["start_time"].isoformat()
            }
            for exp_id, exp in experiments.items()
        ]
    }

@app.post("/inference")
async def inference(
    request: InferenceRequest,
    token: str = Depends(verify_token)
):
    """Run inference on trained model"""
    try:
        # This would integrate with your model loading and inference logic
        # For now, return a placeholder response
        return {
            "text": request.text,
            "prediction": "positive",  # Placeholder
            "confidence": 0.85,  # Placeholder
            "model_path": request.model_path,
            "task_type": request.task_type
        }
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def run_quick_training(experiment_id: str, model_name: str, dataset_path: str, num_epochs: int):
    """Background task for quick training"""
    try:
        experiments[experiment_id]["status"] = "training"
        
        result = await quick_train_transformer(
            model_name=model_name,
            dataset_path=dataset_path,
            num_epochs=num_epochs
        )
        
        experiments[experiment_id].update({
            "status": "completed",
            "progress": 100.0,
            "result": result
        })
        
        logger.info(f"Quick training completed for {experiment_id}")
        
    except Exception as e:
        experiments[experiment_id].update({
            "status": "failed",
            "error": str(e)
        })
        logger.error(f"Quick training failed for {experiment_id}: {e}")

async def run_advanced_training(experiment_id: str, request: TrainingRequest):
    """Background task for advanced training"""
    try:
        experiments[experiment_id]["status"] = "training"
        
        result = await gradient_accumulation_train_transformer(
            model_name=request.model_name,
            dataset_path=request.dataset_path,
            num_epochs=request.num_epochs,
            gradient_accumulation_steps=request.gradient_accumulation_steps
        )
        
        experiments[experiment_id].update({
            "status": "completed",
            "progress": 100.0,
            "result": result
        })
        
        logger.info(f"Advanced training completed for {experiment_id}")
        
    except Exception as e:
        experiments[experiment_id].update({
            "status": "failed",
            "error": str(e)
        })
        logger.error(f"Advanced training failed for {experiment_id}: {e}")

async def run_config_training(experiment_id: str, config):
    """Background task for config-based training"""
    try:
        experiments[experiment_id]["status"] = "training"
        
        trainer = await create_model_trainer(device_manager)
        result = await trainer.train(config)
        
        experiments[experiment_id].update({
            "status": "completed",
            "progress": 100.0,
            "result": result
        })
        
        logger.info(f"Config training completed for {experiment_id}")
        
    except Exception as e:
        experiments[experiment_id].update({
            "status": "failed",
            "error": str(e)
        })
        logger.error(f"Config training failed for {experiment_id}: {e}")

match __name__:
    case "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 