"""
Federated Learning API Routes - Advanced federated learning and distributed AI endpoints
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.federated_learning_engine import (
    get_federated_learning_engine, FederatedConfig, 
    FederatedClient, FederatedRound, FederatedModel
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/federated-learning", tags=["Federated Learning"])


# Request/Response Models
class ClientRegistrationRequest(BaseModel):
    """Client registration request model"""
    name: str = Field(..., description="Client name", min_length=1)
    data_size: int = Field(..., description="Client data size", gt=0)
    data_distribution: Dict[str, Any] = Field(default={}, description="Data distribution type")
    model_architecture: str = Field(default="simple_nn", description="Model architecture")
    local_epochs: int = Field(default=5, description="Local training epochs", gt=0)
    learning_rate: float = Field(default=0.01, description="Learning rate", gt=0)
    batch_size: int = Field(default=32, description="Batch size", gt=0)
    device_type: str = Field(default="cpu", description="Device type (cpu, gpu, tpu)")
    network_bandwidth: float = Field(default=100.0, description="Network bandwidth", gt=0)
    compute_power: float = Field(default=1.0, description="Compute power", gt=0)
    privacy_level: str = Field(default="medium", description="Privacy level (low, medium, high)")
    participation_rate: float = Field(default=1.0, description="Participation rate", ge=0, le=1)
    capabilities: List[str] = Field(default=["training"], description="Client capabilities")


class FederatedRoundRequest(BaseModel):
    """Federated round request model"""
    round_number: int = Field(..., description="Round number", gt=0)
    selected_clients: Optional[List[str]] = Field(default=None, description="Selected client IDs")
    aggregation_method: Optional[str] = Field(default=None, description="Aggregation method")
    min_clients: Optional[int] = Field(default=None, description="Minimum clients per round", gt=0)


class ClientUpdateRequest(BaseModel):
    """Client update request model"""
    client_id: str = Field(..., description="Client ID", min_length=1)
    round_id: str = Field(..., description="Round ID", min_length=1)
    model_parameters: Dict[str, Any] = Field(..., description="Updated model parameters")
    training_metrics: Dict[str, Any] = Field(default={}, description="Training metrics")
    privacy_cost: float = Field(default=0.0, description="Privacy cost", ge=0)
    compression_ratio: float = Field(default=1.0, description="Compression ratio", gt=0, le=1)
    quantization_bits: int = Field(default=32, description="Quantization bits", gt=0, le=32)
    sparsification_ratio: float = Field(default=1.0, description="Sparsification ratio", gt=0, le=1)


class FederatedModelRequest(BaseModel):
    """Federated model request model"""
    name: str = Field(..., description="Model name", min_length=1)
    architecture: str = Field(..., description="Model architecture", min_length=1)
    version: str = Field(default="1.0.0", description="Model version")
    parameters: Dict[str, Any] = Field(default={}, description="Model parameters")
    weights: Dict[str, Any] = Field(default={}, description="Model weights")
    accuracy: float = Field(default=0.0, description="Model accuracy", ge=0, le=1)
    loss: float = Field(default=0.0, description="Model loss", ge=0)
    training_rounds: int = Field(default=0, description="Training rounds", ge=0)
    privacy_budget_used: float = Field(default=0.0, description="Privacy budget used", ge=0)
    compression_ratio: float = Field(default=1.0, description="Compression ratio", gt=0, le=1)
    quantization_bits: int = Field(default=32, description="Quantization bits", gt=0, le=32)
    sparsification_ratio: float = Field(default=1.0, description="Sparsification ratio", gt=0, le=1)
    validation_metrics: Dict[str, float] = Field(default={}, description="Validation metrics")
    performance_metrics: Dict[str, Any] = Field(default={}, description="Performance metrics")


class FederatedConfigRequest(BaseModel):
    """Federated learning configuration request model"""
    enable_federated_learning: bool = Field(default=True, description="Enable federated learning")
    enable_secure_aggregation: bool = Field(default=True, description="Enable secure aggregation")
    enable_differential_privacy: bool = Field(default=True, description="Enable differential privacy")
    enable_federated_analytics: bool = Field(default=True, description="Enable federated analytics")
    enable_horizontal_fl: bool = Field(default=True, description="Enable horizontal federated learning")
    enable_vertical_fl: bool = Field(default=True, description="Enable vertical federated learning")
    enable_federated_transfer_learning: bool = Field(default=True, description="Enable federated transfer learning")
    enable_federated_meta_learning: bool = Field(default=True, description="Enable federated meta learning")
    enable_federated_reinforcement_learning: bool = Field(default=True, description="Enable federated reinforcement learning")
    enable_federated_gan: bool = Field(default=True, description="Enable federated GAN")
    enable_federated_nas: bool = Field(default=True, description="Enable federated neural architecture search")
    enable_federated_optimization: bool = Field(default=True, description="Enable federated optimization")
    enable_federated_compression: bool = Field(default=True, description="Enable federated compression")
    enable_federated_quantization: bool = Field(default=True, description="Enable federated quantization")
    enable_federated_sparsification: bool = Field(default=True, description="Enable federated sparsification")
    max_clients: int = Field(default=100, description="Maximum clients", gt=0)
    min_clients_per_round: int = Field(default=10, description="Minimum clients per round", gt=0)
    max_rounds: int = Field(default=1000, description="Maximum rounds", gt=0)
    learning_rate: float = Field(default=0.01, description="Learning rate", gt=0)
    batch_size: int = Field(default=32, description="Batch size", gt=0)
    epochs_per_round: int = Field(default=5, description="Epochs per round", gt=0)
    aggregation_method: str = Field(default="fedavg", description="Aggregation method")
    privacy_budget: float = Field(default=1.0, description="Privacy budget", gt=0)
    noise_multiplier: float = Field(default=1.1, description="Noise multiplier", gt=0)
    l2_norm_clip: float = Field(default=1.0, description="L2 norm clip", gt=0)
    secure_aggregation_threshold: int = Field(default=3, description="Secure aggregation threshold", gt=0)
    compression_ratio: float = Field(default=0.1, description="Compression ratio", gt=0, le=1)
    quantization_bits: int = Field(default=8, description="Quantization bits", gt=0, le=32)
    sparsification_ratio: float = Field(default=0.1, description="Sparsification ratio", gt=0, le=1)
    enable_model_validation: bool = Field(default=True, description="Enable model validation")
    enable_anomaly_detection: bool = Field(default=True, description="Enable anomaly detection")
    enable_poisoning_detection: bool = Field(default=True, description="Enable poisoning detection")
    enable_backdoor_detection: bool = Field(default=True, description="Enable backdoor detection")
    enable_byzantine_robustness: bool = Field(default=True, description="Enable Byzantine robustness")


# Dependency to get federated learning engine
async def get_federated_engine():
    """Get federated learning engine dependency"""
    engine = await get_federated_learning_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Federated Learning Engine not available")
    return engine


# Federated Learning Routes
@router.post("/register-client", response_model=Dict[str, Any])
async def register_client(
    request: ClientRegistrationRequest,
    engine: FederatedLearningEngine = Depends(get_federated_engine)
):
    """Register a new federated learning client"""
    try:
        start_time = time.time()
        
        # Register client
        client = await engine.register_client({
            "name": request.name,
            "data_size": request.data_size,
            "data_distribution": request.data_distribution,
            "model_architecture": request.model_architecture,
            "local_epochs": request.local_epochs,
            "learning_rate": request.learning_rate,
            "batch_size": request.batch_size,
            "device_type": request.device_type,
            "network_bandwidth": request.network_bandwidth,
            "compute_power": request.compute_power,
            "privacy_level": request.privacy_level,
            "participation_rate": request.participation_rate,
            "capabilities": request.capabilities
        })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "client": {
                "client_id": client.client_id,
                "timestamp": client.timestamp.isoformat(),
                "name": client.name,
                "data_size": client.data_size,
                "data_distribution": client.data_distribution,
                "model_architecture": client.model_architecture,
                "local_epochs": client.local_epochs,
                "learning_rate": client.learning_rate,
                "batch_size": client.batch_size,
                "device_type": client.device_type,
                "network_bandwidth": client.network_bandwidth,
                "compute_power": client.compute_power,
                "privacy_level": client.privacy_level,
                "participation_rate": client.participation_rate,
                "status": client.status,
                "capabilities": client.capabilities
            },
            "processing_time_ms": processing_time,
            "message": f"Client {request.name} registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Error registering client: {e}")
        raise HTTPException(status_code=500, detail=f"Client registration failed: {str(e)}")


@router.post("/start-round", response_model=Dict[str, Any])
async def start_federated_round(
    request: FederatedRoundRequest,
    engine: FederatedLearningEngine = Depends(get_federated_engine)
):
    """Start a new federated learning round"""
    try:
        start_time = time.time()
        
        # Start federated round
        round_data = await engine.start_federated_round(
            round_number=request.round_number,
            selected_clients=request.selected_clients
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "federated_round": {
                "round_id": round_data.round_id,
                "timestamp": round_data.timestamp.isoformat(),
                "round_number": round_data.round_number,
                "selected_clients": round_data.selected_clients,
                "global_model_version": round_data.global_model_version,
                "aggregation_method": round_data.aggregation_method,
                "total_samples": round_data.total_samples,
                "status": round_data.status
            },
            "processing_time_ms": processing_time,
            "message": f"Federated round {request.round_number} started successfully"
        }
        
    except Exception as e:
        logger.error(f"Error starting federated round: {e}")
        raise HTTPException(status_code=500, detail=f"Round start failed: {str(e)}")


@router.post("/submit-update", response_model=Dict[str, Any])
async def submit_client_update(
    request: ClientUpdateRequest,
    engine: FederatedLearningEngine = Depends(get_federated_engine)
):
    """Submit client model update"""
    try:
        start_time = time.time()
        
        # Submit client update
        client_update = {
            "client_id": request.client_id,
            "round_id": request.round_id,
            "parameters": request.model_parameters,
            "training_metrics": request.training_metrics,
            "privacy_cost": request.privacy_cost,
            "compression_ratio": request.compression_ratio,
            "quantization_bits": request.quantization_bits,
            "sparsification_ratio": request.sparsification_ratio
        }
        
        # Aggregate updates (simplified for demonstration)
        aggregated_model = await engine.aggregate_client_updates(
            round_id=request.round_id,
            client_updates={request.client_id: client_update}
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "client_update": {
                "client_id": request.client_id,
                "round_id": request.round_id,
                "privacy_cost": request.privacy_cost,
                "compression_ratio": request.compression_ratio,
                "quantization_bits": request.quantization_bits,
                "sparsification_ratio": request.sparsification_ratio,
                "timestamp": datetime.now().isoformat()
            },
            "aggregated_model": aggregated_model,
            "processing_time_ms": processing_time,
            "message": f"Client update from {request.client_id} submitted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error submitting client update: {e}")
        raise HTTPException(status_code=500, detail=f"Client update failed: {str(e)}")


@router.post("/create-model", response_model=Dict[str, Any])
async def create_federated_model(
    request: FederatedModelRequest,
    engine: FederatedLearningEngine = Depends(get_federated_engine)
):
    """Create a new federated learning model"""
    try:
        start_time = time.time()
        
        # Create federated model
        model = await engine.create_federated_model({
            "name": request.name,
            "architecture": request.architecture,
            "version": request.version,
            "parameters": request.parameters,
            "weights": request.weights,
            "accuracy": request.accuracy,
            "loss": request.loss,
            "training_rounds": request.training_rounds,
            "privacy_budget_used": request.privacy_budget_used,
            "compression_ratio": request.compression_ratio,
            "quantization_bits": request.quantization_bits,
            "sparsification_ratio": request.sparsification_ratio,
            "validation_metrics": request.validation_metrics,
            "performance_metrics": request.performance_metrics
        })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "federated_model": {
                "model_id": model.model_id,
                "timestamp": model.timestamp.isoformat(),
                "name": model.name,
                "architecture": model.architecture,
                "version": model.version,
                "accuracy": model.accuracy,
                "loss": model.loss,
                "training_rounds": model.training_rounds,
                "total_clients": model.total_clients,
                "privacy_budget_used": model.privacy_budget_used,
                "compression_ratio": model.compression_ratio,
                "quantization_bits": model.quantization_bits,
                "sparsification_ratio": model.sparsification_ratio,
                "deployment_status": model.deployment_status
            },
            "processing_time_ms": processing_time,
            "message": f"Federated model {request.name} created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating federated model: {e}")
        raise HTTPException(status_code=500, detail=f"Model creation failed: {str(e)}")


@router.get("/clients", response_model=Dict[str, Any])
async def get_federated_clients(
    status: Optional[str] = None,
    engine: FederatedLearningEngine = Depends(get_federated_engine)
):
    """Get federated learning clients"""
    try:
        # Get clients
        clients = list(engine.clients.values())
        
        # Filter by status if provided
        if status:
            clients = [c for c in clients if c.status == status]
        
        # Format clients
        formatted_clients = []
        for client in clients:
            formatted_clients.append({
                "client_id": client.client_id,
                "timestamp": client.timestamp.isoformat(),
                "name": client.name,
                "data_size": client.data_size,
                "data_distribution": client.data_distribution,
                "model_architecture": client.model_architecture,
                "device_type": client.device_type,
                "network_bandwidth": client.network_bandwidth,
                "compute_power": client.compute_power,
                "privacy_level": client.privacy_level,
                "participation_rate": client.participation_rate,
                "rounds_participated": client.rounds_participated,
                "model_accuracy": client.model_accuracy,
                "training_loss": client.training_loss,
                "validation_loss": client.validation_loss,
                "status": client.status,
                "capabilities": client.capabilities
            })
        
        return {
            "success": True,
            "federated_clients": formatted_clients,
            "total_count": len(formatted_clients),
            "filter": {"status": status},
            "message": "Federated clients retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting federated clients: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get federated clients: {str(e)}")


@router.get("/rounds", response_model=Dict[str, Any])
async def get_federated_rounds(
    status: Optional[str] = None,
    engine: FederatedLearningEngine = Depends(get_federated_engine)
):
    """Get federated learning rounds"""
    try:
        # Get rounds
        rounds = list(engine.rounds.values())
        
        # Filter by status if provided
        if status:
            rounds = [r for r in rounds if r.status == status]
        
        # Format rounds
        formatted_rounds = []
        for round_data in rounds:
            formatted_rounds.append({
                "round_id": round_data.round_id,
                "timestamp": round_data.timestamp.isoformat(),
                "round_number": round_data.round_number,
                "selected_clients": round_data.selected_clients,
                "global_model_version": round_data.global_model_version,
                "aggregation_method": round_data.aggregation_method,
                "total_samples": round_data.total_samples,
                "training_time": round_data.training_time,
                "aggregation_time": round_data.aggregation_time,
                "communication_time": round_data.communication_time,
                "model_accuracy": round_data.model_accuracy,
                "model_loss": round_data.model_loss,
                "convergence_metric": round_data.convergence_metric,
                "privacy_cost": round_data.privacy_cost,
                "compression_ratio": round_data.compression_ratio,
                "quantization_bits": round_data.quantization_bits,
                "sparsification_ratio": round_data.sparsification_ratio,
                "status": round_data.status
            })
        
        return {
            "success": True,
            "federated_rounds": formatted_rounds,
            "total_count": len(formatted_rounds),
            "filter": {"status": status},
            "message": "Federated rounds retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting federated rounds: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get federated rounds: {str(e)}")


@router.get("/models", response_model=Dict[str, Any])
async def get_federated_models(
    deployment_status: Optional[str] = None,
    engine: FederatedLearningEngine = Depends(get_federated_engine)
):
    """Get federated learning models"""
    try:
        # Get models
        models = list(engine.models.values())
        
        # Filter by deployment status if provided
        if deployment_status:
            models = [m for m in models if m.deployment_status == deployment_status]
        
        # Format models
        formatted_models = []
        for model in models:
            formatted_models.append({
                "model_id": model.model_id,
                "timestamp": model.timestamp.isoformat(),
                "name": model.name,
                "architecture": model.architecture,
                "version": model.version,
                "accuracy": model.accuracy,
                "loss": model.loss,
                "training_rounds": model.training_rounds,
                "total_clients": model.total_clients,
                "privacy_budget_used": model.privacy_budget_used,
                "compression_ratio": model.compression_ratio,
                "quantization_bits": model.quantization_bits,
                "sparsification_ratio": model.sparsification_ratio,
                "validation_metrics": model.validation_metrics,
                "deployment_status": model.deployment_status
            })
        
        return {
            "success": True,
            "federated_models": formatted_models,
            "total_count": len(formatted_models),
            "filter": {"deployment_status": deployment_status},
            "message": "Federated models retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting federated models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get federated models: {str(e)}")


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_federated_capabilities(
    engine: FederatedLearningEngine = Depends(get_federated_engine)
):
    """Get federated learning capabilities"""
    try:
        # Get capabilities
        capabilities = await engine.get_federated_capabilities()
        
        return {
            "success": True,
            "federated_capabilities": capabilities,
            "message": "Federated learning capabilities retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting federated capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get federated capabilities: {str(e)}")


@router.get("/performance-metrics", response_model=Dict[str, Any])
async def get_federated_performance_metrics(
    engine: FederatedLearningEngine = Depends(get_federated_engine)
):
    """Get federated learning performance metrics"""
    try:
        # Get performance metrics
        metrics = await engine.get_federated_performance_metrics()
        
        return {
            "success": True,
            "performance_metrics": metrics,
            "message": "Federated learning performance metrics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting federated performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get federated performance metrics: {str(e)}")


@router.post("/configure", response_model=Dict[str, Any])
async def configure_federated_learning(
    request: FederatedConfigRequest,
    engine: FederatedLearningEngine = Depends(get_federated_engine)
):
    """Configure federated learning settings"""
    try:
        # Update configuration
        config = FederatedConfig(
            enable_federated_learning=request.enable_federated_learning,
            enable_secure_aggregation=request.enable_secure_aggregation,
            enable_differential_privacy=request.enable_differential_privacy,
            enable_federated_analytics=request.enable_federated_analytics,
            enable_horizontal_fl=request.enable_horizontal_fl,
            enable_vertical_fl=request.enable_vertical_fl,
            enable_federated_transfer_learning=request.enable_federated_transfer_learning,
            enable_federated_meta_learning=request.enable_federated_meta_learning,
            enable_federated_reinforcement_learning=request.enable_federated_reinforcement_learning,
            enable_federated_gan=request.enable_federated_gan,
            enable_federated_nas=request.enable_federated_nas,
            enable_federated_optimization=request.enable_federated_optimization,
            enable_federated_compression=request.enable_federated_compression,
            enable_federated_quantization=request.enable_federated_quantization,
            enable_federated_sparsification=request.enable_federated_sparsification,
            max_clients=request.max_clients,
            min_clients_per_round=request.min_clients_per_round,
            max_rounds=request.max_rounds,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            epochs_per_round=request.epochs_per_round,
            aggregation_method=request.aggregation_method,
            privacy_budget=request.privacy_budget,
            noise_multiplier=request.noise_multiplier,
            l2_norm_clip=request.l2_norm_clip,
            secure_aggregation_threshold=request.secure_aggregation_threshold,
            compression_ratio=request.compression_ratio,
            quantization_bits=request.quantization_bits,
            sparsification_ratio=request.sparsification_ratio,
            enable_model_validation=request.enable_model_validation,
            enable_anomaly_detection=request.enable_anomaly_detection,
            enable_poisoning_detection=request.enable_poisoning_detection,
            enable_backdoor_detection=request.enable_backdoor_detection,
            enable_byzantine_robustness=request.enable_byzantine_robustness
        )
        
        # Update engine configuration
        engine.config = config
        
        return {
            "success": True,
            "configuration": {
                "enable_federated_learning": config.enable_federated_learning,
                "enable_secure_aggregation": config.enable_secure_aggregation,
                "enable_differential_privacy": config.enable_differential_privacy,
                "enable_federated_analytics": config.enable_federated_analytics,
                "enable_horizontal_fl": config.enable_horizontal_fl,
                "enable_vertical_fl": config.enable_vertical_fl,
                "enable_federated_transfer_learning": config.enable_federated_transfer_learning,
                "enable_federated_meta_learning": config.enable_federated_meta_learning,
                "enable_federated_reinforcement_learning": config.enable_federated_reinforcement_learning,
                "enable_federated_gan": config.enable_federated_gan,
                "enable_federated_nas": config.enable_federated_nas,
                "enable_federated_optimization": config.enable_federated_optimization,
                "enable_federated_compression": config.enable_federated_compression,
                "enable_federated_quantization": config.enable_federated_quantization,
                "enable_federated_sparsification": config.enable_federated_sparsification,
                "max_clients": config.max_clients,
                "min_clients_per_round": config.min_clients_per_round,
                "max_rounds": config.max_rounds,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs_per_round": config.epochs_per_round,
                "aggregation_method": config.aggregation_method,
                "privacy_budget": config.privacy_budget,
                "noise_multiplier": config.noise_multiplier,
                "l2_norm_clip": config.l2_norm_clip,
                "secure_aggregation_threshold": config.secure_aggregation_threshold,
                "compression_ratio": config.compression_ratio,
                "quantization_bits": config.quantization_bits,
                "sparsification_ratio": config.sparsification_ratio,
                "enable_model_validation": config.enable_model_validation,
                "enable_anomaly_detection": config.enable_anomaly_detection,
                "enable_poisoning_detection": config.enable_poisoning_detection,
                "enable_backdoor_detection": config.enable_backdoor_detection,
                "enable_byzantine_robustness": config.enable_byzantine_robustness
            },
            "message": "Federated learning configuration updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error configuring federated learning: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@router.get("/aggregation-methods", response_model=Dict[str, Any])
async def get_aggregation_methods():
    """Get available aggregation methods"""
    try:
        aggregation_methods = {
            "fedavg": {
                "name": "Federated Averaging",
                "description": "Standard federated averaging algorithm",
                "advantages": ["Simple", "Effective", "Widely used"],
                "disadvantages": ["May not handle non-IID data well"],
                "use_cases": ["IID data", "Homogeneous clients", "Simple scenarios"]
            },
            "fedprox": {
                "name": "FedProx",
                "description": "Federated learning with proximal term",
                "advantages": ["Handles non-IID data", "Better convergence", "Robust"],
                "disadvantages": ["More complex", "Requires tuning"],
                "use_cases": ["Non-IID data", "Heterogeneous clients", "Unstable networks"]
            },
            "fednova": {
                "name": "FedNova",
                "description": "Normalized averaging for heterogeneous data",
                "advantages": ["Handles heterogeneity", "Better convergence", "Adaptive"],
                "disadvantages": ["Complex implementation", "Requires more communication"],
                "use_cases": ["Heterogeneous data", "Variable client participation", "Complex scenarios"]
            },
            "scaffold": {
                "name": "SCAFFOLD",
                "description": "Control variates for better convergence",
                "advantages": ["Fast convergence", "Handles non-IID data", "Theoretically sound"],
                "disadvantages": ["Requires more storage", "Complex implementation"],
                "use_cases": ["Non-IID data", "Fast convergence needed", "Research applications"]
            }
        }
        
        return {
            "success": True,
            "aggregation_methods": aggregation_methods,
            "message": "Aggregation methods retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting aggregation methods: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get aggregation methods: {str(e)}")


@router.get("/privacy-methods", response_model=Dict[str, Any])
async def get_privacy_methods():
    """Get available privacy methods"""
    try:
        privacy_methods = {
            "differential_privacy": {
                "name": "Differential Privacy",
                "description": "Add noise to protect individual privacy",
                "advantages": ["Strong privacy guarantee", "Theoretically sound", "Widely studied"],
                "disadvantages": ["May reduce model accuracy", "Requires careful tuning"],
                "use_cases": ["Sensitive data", "Privacy-critical applications", "Regulatory compliance"]
            },
            "secure_aggregation": {
                "name": "Secure Aggregation",
                "description": "Cryptographic aggregation without revealing individual updates",
                "advantages": ["Strong privacy", "No accuracy loss", "Cryptographically secure"],
                "disadvantages": ["Requires more communication", "Complex implementation"],
                "use_cases": ["High privacy requirements", "Cryptographic security", "Research applications"]
            },
            "homomorphic_encryption": {
                "name": "Homomorphic Encryption",
                "description": "Compute on encrypted data",
                "advantages": ["Strong privacy", "No data exposure", "Theoretically perfect"],
                "disadvantages": ["Very slow", "Limited operations", "Not practical for large models"],
                "use_cases": ["Research", "Small models", "Perfect privacy requirements"]
            }
        }
        
        return {
            "success": True,
            "privacy_methods": privacy_methods,
            "message": "Privacy methods retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting privacy methods: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get privacy methods: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    engine: FederatedLearningEngine = Depends(get_federated_engine)
):
    """Federated Learning Engine health check"""
    try:
        # Check engine components
        components_status = {
            "aggregator": engine.aggregator is not None,
            "secure_aggregation": engine.aggregator.secure_aggregation is not None,
            "differential_privacy": engine.aggregator.differential_privacy is not None,
            "model_compression": engine.aggregator.model_compression is not None
        }
        
        # Get capabilities
        capabilities = await engine.get_federated_capabilities()
        
        # Get performance metrics
        metrics = await engine.get_federated_performance_metrics()
        
        # Determine overall health
        all_healthy = all(components_status.values())
        
        overall_health = "healthy" if all_healthy else "degraded"
        
        return {
            "status": overall_health,
            "timestamp": datetime.now().isoformat(),
            "components": components_status,
            "capabilities": capabilities,
            "performance_metrics": metrics,
            "message": "Federated Learning Engine is operational" if overall_health == "healthy" else "Some federated learning components may not be available"
        }
        
    except Exception as e:
        logger.error(f"Error in Federated Learning health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "message": "Federated Learning Engine health check failed"
        }

















