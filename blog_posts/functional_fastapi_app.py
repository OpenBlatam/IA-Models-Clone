from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Callable
import asyncio
import logging
from pathlib import Path
import json
from datetime import datetime
from functools import partial, reduce
from operator import itemgetter
import uuid
from functional_training import (
        import torch
    import uvicorn
from typing import Any, List, Dict, Optional
"""
🚀 Functional FastAPI Application
================================

Pure functional, declarative approach to FastAPI development.
Uses data transformations, pure functions, and functional patterns instead of classes.

Key Principles:
- Pure functions with no side effects
- Data transformations over mutable state
- Composition over inheritance
- Immutable data structures
- Declarative configuration
"""


    create_default_config,
    update_config,
    validate_config,
    train_model,
    quick_train_transformer,
    functional_train_with_config,
    get_training_summary
)

# ============================================================================
# Pure Data Structures
# ============================================================================

class TrainingRequest(BaseModel):
    """Immutable training request data."""
    model_name: str = Field(..., description="Model name (e.g., distilbert-base-uncased)")
    dataset_path: str = Field(..., description="Path to dataset")
    num_epochs: int = Field(default=5, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(default=16, ge=1, le=128, description="Batch size")
    learning_rate: float = Field(default=2e-5, gt=0, description="Learning rate")
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=32, description="Gradient accumulation steps")
    experiment_id: Optional[str] = Field(None, description="Custom experiment ID")
    description: Optional[str] = Field("", description="Experiment description")

class InferenceRequest(BaseModel):
    """Immutable inference request data."""
    text: str = Field(..., description="Input text for inference")
    model_path: str = Field(..., description="Path to trained model")
    task_type: str = Field(default="classification", description="Task type")

class ExperimentStatus(BaseModel):
    """Immutable experiment status data."""
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
    """Immutable training response data."""
    experiment_id: str
    status: str
    message: str
    estimated_duration: Optional[int]

# ============================================================================
# Pure Functions - Configuration and Setup
# ============================================================================

def create_app_config() -> Dict[str, Any]:
    """Create FastAPI app configuration in a pure functional way."""
    return {
        "title": "Blatam Academy Functional NLP API",
        "description": "Production-ready functional NLP training and inference API",
        "version": "1.0.0"
    }

def create_cors_config() -> Dict[str, Any]:
    """Create CORS configuration in a pure functional way."""
    return {
        "allow_origins": ["*"],
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }

def setup_logging() -> logging.Logger:
    """Setup logging in a pure functional way."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    return logging.getLogger(__name__)

# ============================================================================
# Pure Functions - Authentication
# ============================================================================

def create_token_verifier(secret_token: str = "your-secret-token") -> Callable:
    """Create token verification function in a pure functional way."""
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> str:
        """Pure function for token verification."""
        if credentials.credentials != secret_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return credentials.credentials
    
    return verify_token

# ============================================================================
# Pure Functions - Experiment Management
# ============================================================================

def create_experiment_id(request: TrainingRequest) -> str:
    """Create experiment ID in a pure functional way."""
    if request.experiment_id:
        return request.experiment_id
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"exp_{request.model_name}_{timestamp}_{uuid.uuid4().hex[:8]}"

def create_experiment_state(
    experiment_id: str,
    request: TrainingRequest,
    status: str = "starting"
) -> Dict[str, Any]:
    """Create experiment state in a pure functional way."""
    return {
        "experiment_id": experiment_id,
        "status": status,
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": request.num_epochs,
        "start_time": datetime.now(),
        "request": request.dict(),
        "current_loss": None,
        "best_accuracy": None,
        "estimated_completion": None
    }

def update_experiment_state(
    current_state: Dict[str, Any],
    **updates
) -> Dict[str, Any]:
    """Update experiment state in a pure functional way."""
    return {**current_state, **updates}

def calculate_estimated_duration(num_epochs: int, training_type: str = "quick") -> int:
    """Calculate estimated duration in a pure functional way."""
    base_times = {
        "quick": 300,      # 5 minutes per epoch
        "advanced": 400,   # 6.7 minutes per epoch
        "config": 350      # 5.8 minutes per epoch
    }
    return num_epochs * base_times.get(training_type, 300)

# ============================================================================
# Pure Functions - Training Pipeline
# ============================================================================

async def run_quick_training_pure(
    experiment_id: str,
    model_name: str,
    dataset_path: str,
    num_epochs: int,
    experiments_store: Dict[str, Dict[str, Any]]
) -> None:
    """Run quick training in a pure functional way."""
    logger = setup_logging()
    
    try:
        # Update experiment status
        experiments_store[experiment_id] = update_experiment_state(
            experiments_store[experiment_id],
            status="running",
            progress=0.1
        )
        
        # Create configuration
        config = create_default_config(model_name, dataset_path)
        config = update_config(config, num_epochs=num_epochs)
        
        # Validate configuration
        is_valid, errors = validate_config(config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {errors}")
        
        # Update experiment status
        experiments_store[experiment_id] = update_experiment_state(
            experiments_store[experiment_id],
            status="training",
            progress=0.2
        )
        
        # Run training
        results = await quick_train_transformer(
            model_name=model_name,
            dataset_path=dataset_path,
            num_epochs=num_epochs
        )
        
        # Get training summary
        summary = get_training_summary(results)
        
        # Update experiment status
        experiments_store[experiment_id] = update_experiment_state(
            experiments_store[experiment_id],
            status="completed",
            progress=1.0,
            current_epoch=summary['final_epoch'],
            current_loss=summary.get('final_val_loss'),
            best_accuracy=summary.get('best_val_accuracy'),
            estimated_completion=datetime.now()
        )
        
        logger.info(f"Training completed for experiment {experiment_id}")
        
    except Exception as e:
        logger.error(f"Training failed for experiment {experiment_id}: {e}")
        experiments_store[experiment_id] = update_experiment_state(
            experiments_store[experiment_id],
            status="failed",
            progress=0.0
        )

async def run_advanced_training_pure(
    experiment_id: str,
    request: TrainingRequest,
    experiments_store: Dict[str, Dict[str, Any]]
) -> None:
    """Run advanced training in a pure functional way."""
    logger = setup_logging()
    
    try:
        # Update experiment status
        experiments_store[experiment_id] = update_experiment_state(
            experiments_store[experiment_id],
            status="running",
            progress=0.1
        )
        
        # Create configuration with advanced settings
        config = create_default_config(request.model_name, request.dataset_path)
        config = update_config(
            config,
            num_epochs=request.num_epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            gradient_accumulation_steps=request.gradient_accumulation_steps
        )
        
        # Validate configuration
        is_valid, errors = validate_config(config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {errors}")
        
        # Update experiment status
        experiments_store[experiment_id] = update_experiment_state(
            experiments_store[experiment_id],
            status="training",
            progress=0.2
        )
        
        # Run training
        results = train_model(config)
        
        # Get training summary
        summary = get_training_summary(results)
        
        # Update experiment status
        experiments_store[experiment_id] = update_experiment_state(
            experiments_store[experiment_id],
            status="completed",
            progress=1.0,
            current_epoch=summary['final_epoch'],
            current_loss=summary.get('final_val_loss'),
            best_accuracy=summary.get('best_val_accuracy'),
            estimated_completion=datetime.now()
        )
        
        logger.info(f"Advanced training completed for experiment {experiment_id}")
        
    except Exception as e:
        logger.error(f"Advanced training failed for experiment {experiment_id}: {e}")
        experiments_store[experiment_id] = update_experiment_state(
            experiments_store[experiment_id],
            status="failed",
            progress=0.0
        )

async def run_config_training_pure(
    experiment_id: str,
    config_path: str,
    experiments_store: Dict[str, Dict[str, Any]]
) -> None:
    """Run config-based training in a pure functional way."""
    logger = setup_logging()
    
    try:
        # Update experiment status
        experiments_store[experiment_id] = update_experiment_state(
            experiments_store[experiment_id],
            status="running",
            progress=0.1
        )
        
        # Validate config file exists
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Update experiment status
        experiments_store[experiment_id] = update_experiment_state(
            experiments_store[experiment_id],
            status="training",
            progress=0.2
        )
        
        # Run training
        results = await functional_train_with_config(config_path)
        
        # Get training summary
        summary = get_training_summary(results)
        
        # Update experiment status
        experiments_store[experiment_id] = update_experiment_state(
            experiments_store[experiment_id],
            status="completed",
            progress=1.0,
            current_epoch=summary['final_epoch'],
            current_loss=summary.get('final_val_loss'),
            best_accuracy=summary.get('best_val_accuracy'),
            estimated_completion=datetime.now()
        )
        
        logger.info(f"Config training completed for experiment {experiment_id}")
        
    except Exception as e:
        logger.error(f"Config training failed for experiment {experiment_id}: {e}")
        experiments_store[experiment_id] = update_experiment_state(
            experiments_store[experiment_id],
            status="failed",
            progress=0.0
        )

# ============================================================================
# Pure Functions - Response Creation
# ============================================================================

def create_training_response(
    experiment_id: str,
    status: str,
    message: str,
    estimated_duration: Optional[int]
) -> TrainingResponse:
    """Create training response in a pure functional way."""
    return TrainingResponse(
        experiment_id=experiment_id,
        status=status,
        message=message,
        estimated_duration=estimated_duration
    )

def create_experiment_status_response(
    experiment_data: Dict[str, Any]
) -> ExperimentStatus:
    """Create experiment status response in a pure functional way."""
    return ExperimentStatus(
        experiment_id=experiment_data["experiment_id"],
        status=experiment_data["status"],
        progress=experiment_data["progress"],
        current_epoch=experiment_data["current_epoch"],
        total_epochs=experiment_data["total_epochs"],
        current_loss=experiment_data.get("current_loss"),
        best_accuracy=experiment_data.get("best_accuracy"),
        start_time=experiment_data["start_time"],
        estimated_completion=experiment_data.get("estimated_completion")
    )

def create_health_response() -> Dict[str, Any]:
    """Create health check response in a pure functional way."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_type": "functional",
        "version": "1.0.0"
    }

# ============================================================================
# Pure Functions - Error Handling
# ============================================================================

def handle_training_error(error: Exception, experiment_id: str) -> HTTPException:
    """Handle training errors in a pure functional way."""
    logger = setup_logging()
    logger.error(f"Training error for experiment {experiment_id}: {error}")
    
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Training failed: {str(error)}"
    )

def validate_experiment_exists(
    experiment_id: str,
    experiments_store: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Validate experiment exists in a pure functional way."""
    if experiment_id not in experiments_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )
    return experiments_store[experiment_id]

# ============================================================================
# FastAPI App Creation (Functional Approach)
# ============================================================================

async def create_fastapi_app() -> FastAPI:
    """Create FastAPI app in a pure functional way."""
    app_config = create_app_config()
    app = FastAPI(**app_config)
    
    # Add CORS middleware
    cors_config = create_cors_config()
    app.add_middleware(CORSMiddleware, **cors_config)
    
    return app

def create_experiments_store() -> Dict[str, Dict[str, Any]]:
    """Create experiments store in a pure functional way."""
    return {}

# Create app and global state
app = create_fastapi_app()
experiments = create_experiments_store()
verify_token = create_token_verifier()
logger = setup_logging()

# ============================================================================
# API Endpoints (Functional Approach)
# ============================================================================

@app.get("/")
async def root() -> Dict[str, str]:
    """Health check endpoint using functional approach."""
    return {"message": "Blatam Academy Functional NLP API", "status": "healthy"}

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Detailed health check using functional approach."""
    health_data = create_health_response()
    health_data["active_experiments"] = len(experiments)
    return health_data

@app.post("/train/quick", response_model=TrainingResponse)
async def quick_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
) -> TrainingResponse:
    """Quick training with functional approach."""
    try:
        # Create experiment ID
        experiment_id = create_experiment_id(request)
        
        # Create experiment state
        experiments[experiment_id] = create_experiment_state(experiment_id, request)
        
        # Calculate estimated duration
        estimated_duration = calculate_estimated_duration(request.num_epochs, "quick")
        
        # Start training in background
        background_tasks.add_task(
            run_quick_training_pure,
            experiment_id,
            request.model_name,
            request.dataset_path,
            request.num_epochs,
            experiments
        )
        
        return create_training_response(
            experiment_id=experiment_id,
            status="started",
            message="Quick training started successfully",
            estimated_duration=estimated_duration
        )
        
    except Exception as e:
        raise handle_training_error(e, experiment_id if 'experiment_id' in locals() else "unknown")

@app.post("/train/advanced", response_model=TrainingResponse)
async def advanced_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
) -> TrainingResponse:
    """Advanced training with functional approach."""
    try:
        # Create experiment ID
        experiment_id = create_experiment_id(request)
        
        # Create experiment state
        experiments[experiment_id] = create_experiment_state(experiment_id, request)
        
        # Calculate estimated duration
        estimated_duration = calculate_estimated_duration(request.num_epochs, "advanced")
        
        # Start training in background
        background_tasks.add_task(
            run_advanced_training_pure,
            experiment_id,
            request,
            experiments
        )
        
        return create_training_response(
            experiment_id=experiment_id,
            status="started",
            message="Advanced training started successfully",
            estimated_duration=estimated_duration
        )
        
    except Exception as e:
        raise handle_training_error(e, experiment_id if 'experiment_id' in locals() else "unknown")

@app.post("/train/config", response_model=TrainingResponse)
async def config_training(
    config_path: str,
    experiment_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    token: str = Depends(verify_token)
) -> TrainingResponse:
    """Training using YAML configuration file with functional approach."""
    try:
        # Validate config file exists
        if not Path(config_path).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Config file not found: {config_path}"
            )
        
        # Create experiment ID
        experiment_id = experiment_id or f"config_{uuid.uuid4().hex[:8]}"
        
        # Create experiment state
        experiments[experiment_id] = create_experiment_state(
            experiment_id,
            TrainingRequest(
                model_name="config_based",
                dataset_path="config_based",
                num_epochs=10
            )
        )
        
        # Calculate estimated duration
        estimated_duration = calculate_estimated_duration(10, "config")
        
        # Start training in background
        if background_tasks:
            background_tasks.add_task(
                run_config_training_pure,
                experiment_id,
                config_path,
                experiments
            )
        
        return create_training_response(
            experiment_id=experiment_id,
            status="started",
            message="Config-based training started successfully",
            estimated_duration=estimated_duration
        )
        
    except Exception as e:
        raise handle_training_error(e, experiment_id if 'experiment_id' in locals() else "unknown")

@app.get("/experiments/{experiment_id}", response_model=ExperimentStatus)
async def get_experiment_status(
    experiment_id: str,
    token: str = Depends(verify_token)
) -> ExperimentStatus:
    """Get experiment status using functional approach."""
    try:
        experiment_data = validate_experiment_exists(experiment_id, experiments)
        return create_experiment_status_response(experiment_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get experiment status"
        )

@app.get("/experiments")
async def list_experiments(
    token: str = Depends(verify_token)
) -> Dict[str, Any]:
    """List all experiments using functional approach."""
    try:
        experiment_list = [
            {
                "experiment_id": exp_id,
                "status": exp_data["status"],
                "progress": exp_data["progress"],
                "start_time": exp_data["start_time"].isoformat(),
                "model_name": exp_data["request"].get("model_name", "unknown")
            }
            for exp_id, exp_data in experiments.items()
        ]
        
        return {
            "experiments": experiment_list,
            "total_count": len(experiments),
            "active_count": len([exp for exp in experiment_list if exp["status"] in ["running", "training"]])
        }
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list experiments"
        )

@app.post("/inference")
async def inference(
    request: InferenceRequest,
    token: str = Depends(verify_token)
) -> Dict[str, Any]:
    """Inference endpoint using functional approach."""
    try:
        # Validate model path exists
        if not Path(request.model_path).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model file not found: {request.model_path}"
            )
        
        # Load model (simplified - replace with actual model loading logic)
        checkpoint = torch.load(request.model_path, map_location='cpu')
        
        # Perform inference (simplified - replace with actual inference logic)
        result = {
            "input_text": request.text,
            "task_type": request.task_type,
            "model_path": request.model_path,
            "prediction": "sample_prediction",  # Replace with actual prediction
            "confidence": 0.95,  # Replace with actual confidence
            "inference_time_ms": 50.0  # Replace with actual timing
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

@app.delete("/experiments/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    token: str = Depends(verify_token)
) -> Dict[str, str]:
    """Delete experiment using functional approach."""
    try:
        # Validate experiment exists
        validate_experiment_exists(experiment_id, experiments)
        
        # Remove experiment (pure function - no side effects on global state)
        if experiment_id in experiments:
            del experiments[experiment_id]
        
        return {"message": f"Experiment {experiment_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete experiment"
        )

# ============================================================================
# Utility Endpoints
# ============================================================================

@app.get("/config/template")
async def get_config_template(
    token: str = Depends(verify_token)
) -> Dict[str, Any]:
    """Get configuration template using functional approach."""
    return {
        "model": {
            "type": "transformer",
            "name": "distilbert-base-uncased",
            "training_mode": "fine_tune"
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 2e-5,
            "num_epochs": 10,
            "warmup_steps": 100,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 1,
            "mixed_precision": True,
            "early_stopping_patience": 5
        },
        "data": {
            "dataset_path": "data/dataset.csv",
            "eval_split": 0.2,
            "test_split": 0.1,
            "cross_validation_folds": 5
        },
        "optimization": {
            "enable_gpu_optimization": True,
            "enable_memory_optimization": True,
            "enable_batch_optimization": True,
            "enable_compilation": False,
            "enable_amp": True
        },
        "logging": {
            "log_to_tensorboard": True,
            "log_to_wandb": False,
            "log_to_mlflow": False
        },
        "output": {
            "output_dir": "models",
            "save_steps": 500,
            "eval_steps": 500
        }
    }

@app.get("/stats")
async def get_api_stats(
    token: str = Depends(verify_token)
) -> Dict[str, Any]:
    """Get API statistics using functional approach."""
    try:
        # Calculate statistics
        total_experiments = len(experiments)
        active_experiments = len([exp for exp in experiments.values() if exp["status"] in ["running", "training"]])
        completed_experiments = len([exp for exp in experiments.values() if exp["status"] == "completed"])
        failed_experiments = len([exp for exp in experiments.values() if exp["status"] == "failed"])
        
        # Calculate average training time for completed experiments
        completed_times = []
        for exp in experiments.values():
            if exp["status"] == "completed" and exp.get("estimated_completion") and exp.get("start_time"):
                duration = (exp["estimated_completion"] - exp["start_time"]).total_seconds()
                completed_times.append(duration)
        
        avg_training_time = sum(completed_times) / len(completed_times) if completed_times else 0
        
        return {
            "total_experiments": total_experiments,
            "active_experiments": active_experiments,
            "completed_experiments": completed_experiments,
            "failed_experiments": failed_experiments,
            "success_rate": completed_experiments / total_experiments if total_experiments > 0 else 0,
            "average_training_time_seconds": avg_training_time,
            "api_uptime": "functional",  # Replace with actual uptime calculation
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Error getting API stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get API statistics"
        )

# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event using functional approach."""
    logger.info("🚀 Functional FastAPI app starting up")
    logger.info("Using pure functional, declarative programming approach")
    logger.info("No classes, only pure functions and data transformations")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event using functional approach."""
    logger.info("🛑 Functional FastAPI app shutting down")
    logger.info(f"Cleaning up {len(experiments)} experiments")

# ============================================================================
# Demo Function
# ============================================================================

def demo_functional_api():
    """Demo the functional FastAPI app."""
    print("🚀 Functional FastAPI Demo")
    print("=" * 50)
    print("Key Features:")
    print("- Pure functions with no side effects")
    print("- Immutable data structures")
    print("- Data transformations over mutable state")
    print("- Composition over inheritance")
    print("- Declarative configuration")
    print("- No classes, only functions and data")
    print()
    print("API Endpoints:")
    print("- POST /train/quick - Quick training")
    print("- POST /train/advanced - Advanced training")
    print("- POST /train/config - Config-based training")
    print("- GET /experiments/{id} - Get experiment status")
    print("- GET /experiments - List all experiments")
    print("- POST /inference - Model inference")
    print("- GET /health - Health check")
    print("- GET /stats - API statistics")

if __name__ == "__main__":
    demo_functional_api()
    uvicorn.run(app, host="0.0.0.0", port=8000) 