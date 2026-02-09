"""
Experiments Router - Endpoints for experiment management.

This module provides REST API endpoints for creating,
managing, and monitoring experiments.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials

from ..models import ExperimentRequest, ExperimentResponse, ErrorResponse
from ..auth import verify_token
from core.experiments import ExperimentManager, ExperimentConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])

# Initialize manager (should be injected in production)
experiment_manager = ExperimentManager()


@router.get("", response_model=dict)
async def list_experiments(
    status: Optional[str] = None,
    token: str = Depends(verify_token),
):
    """
    List experiments.
    
    Args:
        status: Filter by status (optional)
        token: Authentication token
    
    Returns:
        Dictionary with experiments list
    """
    try:
        experiments = experiment_manager.list_experiments()
        if status:
            experiments = [e for e in experiments if e.status.value == status]
        return {"experiments": [e.to_dict() if hasattr(e, 'to_dict') else e for e in experiments]}
    except Exception as e:
        logger.exception("Error listing experiments")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=dict)
async def create_experiment(
    request: ExperimentRequest,
    token: str = Depends(verify_token),
):
    """
    Create a new experiment.
    
    Args:
        request: Experiment request data
        token: Authentication token
    
    Returns:
        Experiment dictionary
    """
    try:
        config = ExperimentConfig(
            name=request.name,
            description=request.description,
            model_name=request.model_name,
            benchmark_name=request.benchmark_name,
            hyperparameters=request.hyperparameters,
            tags=request.tags,
        )
        experiment = experiment_manager.create_experiment(config)
        return experiment.to_dict() if hasattr(experiment, 'to_dict') else experiment
    except Exception as e:
        logger.exception("Error creating experiment")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}", response_model=dict)
async def get_experiment(
    experiment_id: str,
    token: str = Depends(verify_token),
):
    """
    Get experiment by ID.
    
    Args:
        experiment_id: Experiment identifier
        token: Authentication token
    
    Returns:
        Experiment dictionary
    
    Raises:
        HTTPException: If experiment not found
    """
    try:
        experiment = experiment_manager.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return experiment.to_dict() if hasattr(experiment, 'to_dict') else experiment
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting experiment")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{experiment_id}/start", response_model=dict)
async def start_experiment(
    experiment_id: str,
    token: str = Depends(verify_token),
):
    """
    Start an experiment.
    
    Args:
        experiment_id: Experiment identifier
        token: Authentication token
    
    Returns:
        Experiment dictionary
    """
    try:
        experiment = experiment_manager.start_experiment(experiment_id)
        return experiment.to_dict() if hasattr(experiment, 'to_dict') else experiment
    except Exception as e:
        logger.exception("Error starting experiment")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{experiment_id}/complete", response_model=dict)
async def complete_experiment(
    experiment_id: str,
    results: Dict[str, Any],
    token: str = Depends(verify_token),
):
    """
    Complete an experiment.
    
    Args:
        experiment_id: Experiment identifier
        results: Experiment results
        token: Authentication token
    
    Returns:
        Experiment dictionary
    """
    try:
        experiment = experiment_manager.complete_experiment(experiment_id, results)
        return experiment.to_dict() if hasattr(experiment, 'to_dict') else experiment
    except Exception as e:
        logger.exception("Error completing experiment")
        raise HTTPException(status_code=500, detail=str(e))












