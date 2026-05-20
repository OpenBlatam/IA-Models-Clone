"""
Neural Interface API - Ultimate Advanced Implementation
====================================================

FastAPI endpoints for neural interface operations.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..services.neural_interface_service import (
    neural_interface_service,
    NeuralInterfaceType,
    BrainSignalType,
    CognitiveStateType
)

logger = logging.getLogger(__name__)

# Pydantic models
class NeuralDeviceRegistration(BaseModel):
    device_id: str = Field(..., description="Unique device identifier")
    device_type: NeuralInterfaceType = Field(..., description="Type of neural interface device")
    device_name: str = Field(..., description="Name of the device")
    capabilities: List[str] = Field(..., description="Device capabilities")
    location: Dict[str, float] = Field(..., description="Device location coordinates")
    device_info: Dict[str, Any] = Field(default_factory=dict, description="Additional device information")

class NeuralSessionCreation(BaseModel):
    device_id: str = Field(..., description="Device ID for the session")
    session_name: str = Field(..., description="Name of the neural session")
    session_config: Dict[str, Any] = Field(default_factory=dict, description="Session configuration")
    user_id: str = Field(..., description="ID of the user")

class BrainSignalCapture(BaseModel):
    signal_type: BrainSignalType = Field(..., description="Type of brain signal")
    signal_data: Dict[str, Any] = Field(..., description="Brain signal data")

class CognitiveStateAnalysis(BaseModel):
    state_type: CognitiveStateType = Field(..., description="Type of cognitive state")
    analysis_data: Dict[str, Any] = Field(..., description="Cognitive state analysis data")

class NeuralModelCreation(BaseModel):
    model_name: str = Field(..., description="Name of the neural model")
    model_type: str = Field(..., description="Type of neural model")
    model_config: Dict[str, Any] = Field(..., description="Model configuration")
    training_data: List[Dict[str, Any]] = Field(..., description="Training data")

class NeuralModelTraining(BaseModel):
    model_id: str = Field(..., description="ID of the neural model")
    training_config: Dict[str, Any] = Field(..., description="Training configuration")

class NeuralModelPrediction(BaseModel):
    model_id: str = Field(..., description="ID of the neural model")
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")

class NeuralWorkflowCreation(BaseModel):
    workflow_name: str = Field(..., description="Name of the neural workflow")
    workflow_type: str = Field(..., description="Type of workflow")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    triggers: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow triggers")
    conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow conditions")

class NeuralWorkflowExecution(BaseModel):
    workflow_id: str = Field(..., description="ID of the workflow to execute")
    session_id: str = Field(..., description="ID of the neural session")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")

# Create router
router = APIRouter(prefix="/neural", tags=["Neural Interface"])

@router.post("/devices/register")
async def register_neural_device(device_data: NeuralDeviceRegistration) -> Dict[str, Any]:
    """Register a new neural interface device"""
    try:
        device_id = await neural_interface_service.register_neural_device(
            device_id=device_data.device_id,
            device_type=device_data.device_type,
            device_name=device_data.device_name,
            capabilities=device_data.capabilities,
            location=device_data.location,
            device_info=device_data.device_info
        )
        
        return {
            "success": True,
            "device_id": device_id,
            "message": "Neural device registered successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to register neural device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/create")
async def create_neural_session(session_data: NeuralSessionCreation) -> Dict[str, Any]:
    """Create a new neural interface session"""
    try:
        session_id = await neural_interface_service.create_neural_session(
            device_id=session_data.device_id,
            session_name=session_data.session_name,
            session_config=session_data.session_config,
            user_id=session_data.user_id
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Neural session created successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to create neural session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/signals")
async def capture_brain_signal(
    session_id: str,
    signal_data: BrainSignalCapture
) -> Dict[str, Any]:
    """Capture brain signal data"""
    try:
        signal_id = await neural_interface_service.capture_brain_signal(
            session_id=session_id,
            signal_type=signal_data.signal_type,
            signal_data=signal_data.signal_data
        )
        
        return {
            "success": True,
            "signal_id": signal_id,
            "message": "Brain signal captured successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to capture brain signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/cognitive-states")
async def analyze_cognitive_state(
    session_id: str,
    state_data: CognitiveStateAnalysis
) -> Dict[str, Any]:
    """Analyze cognitive state from brain signals"""
    try:
        state_id = await neural_interface_service.analyze_cognitive_state(
            session_id=session_id,
            state_type=state_data.state_type,
            analysis_data=state_data.analysis_data
        )
        
        return {
            "success": True,
            "state_id": state_id,
            "message": "Cognitive state analyzed successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to analyze cognitive state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/create")
async def create_neural_model(model_data: NeuralModelCreation) -> Dict[str, Any]:
    """Create a neural network model"""
    try:
        model_id = await neural_interface_service.create_neural_model(
            model_name=model_data.model_name,
            model_type=model_data.model_type,
            model_config=model_data.model_config,
            training_data=model_data.training_data
        )
        
        return {
            "success": True,
            "model_id": model_id,
            "message": "Neural model created successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to create neural model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/train")
async def train_neural_model(training_data: NeuralModelTraining) -> Dict[str, Any]:
    """Train a neural network model"""
    try:
        result = await neural_interface_service.train_neural_model(
            model_id=training_data.model_id,
            training_config=training_data.training_config
        )
        
        return {
            "success": True,
            "result": result,
            "message": "Neural model trained successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to train neural model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/predict")
async def predict_with_neural_model(prediction_data: NeuralModelPrediction) -> Dict[str, Any]:
    """Make predictions with a trained neural model"""
    try:
        prediction = await neural_interface_service.predict_with_neural_model(
            model_id=prediction_data.model_id,
            input_data=prediction_data.input_data
        )
        
        return {
            "success": True,
            "prediction": prediction,
            "message": "Neural model prediction completed successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to predict with neural model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/create")
async def create_neural_workflow(workflow_data: NeuralWorkflowCreation) -> Dict[str, Any]:
    """Create a neural interface workflow"""
    try:
        workflow_id = await neural_interface_service.create_neural_workflow(
            workflow_name=workflow_data.workflow_name,
            workflow_type=workflow_data.workflow_type,
            steps=workflow_data.steps,
            triggers=workflow_data.triggers,
            conditions=workflow_data.conditions
        )
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "message": "Neural workflow created successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to create neural workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/execute")
async def execute_neural_workflow(execution_data: NeuralWorkflowExecution) -> Dict[str, Any]:
    """Execute a neural interface workflow"""
    try:
        result = await neural_interface_service.execute_neural_workflow(
            workflow_id=execution_data.workflow_id,
            session_id=execution_data.session_id,
            context=execution_data.context
        )
        
        return {
            "success": True,
            "result": result,
            "message": "Neural workflow executed successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to execute neural workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/end")
async def end_neural_session(session_id: str) -> Dict[str, Any]:
    """End neural interface session"""
    try:
        result = await neural_interface_service.end_neural_session(session_id=session_id)
        
        return {
            "success": True,
            "result": result,
            "message": "Neural session ended successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to end neural session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/analytics")
async def get_neural_session_analytics(session_id: str) -> Dict[str, Any]:
    """Get neural session analytics"""
    try:
        analytics = await neural_interface_service.get_neural_session_analytics(session_id=session_id)
        
        if analytics is None:
            raise HTTPException(status_code=404, detail="Neural session not found")
        
        return {
            "success": True,
            "analytics": analytics,
            "message": "Neural session analytics retrieved successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get neural session analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_neural_stats() -> Dict[str, Any]:
    """Get neural interface service statistics"""
    try:
        stats = await neural_interface_service.get_neural_stats()
        
        return {
            "success": True,
            "stats": stats,
            "message": "Neural interface statistics retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get neural stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/devices")
async def get_neural_devices() -> Dict[str, Any]:
    """Get all registered neural devices"""
    try:
        devices = list(neural_interface_service.neural_devices.values())
        
        return {
            "success": True,
            "devices": devices,
            "count": len(devices),
            "message": "Neural devices retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get neural devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions")
async def get_neural_sessions() -> Dict[str, Any]:
    """Get all neural sessions"""
    try:
        sessions = list(neural_interface_service.neural_sessions.values())
        
        return {
            "success": True,
            "sessions": sessions,
            "count": len(sessions),
            "message": "Neural sessions retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get neural sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_neural_models() -> Dict[str, Any]:
    """Get all neural models"""
    try:
        models = list(neural_interface_service.neural_models.values())
        
        return {
            "success": True,
            "models": models,
            "count": len(models),
            "message": "Neural models retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get neural models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows")
async def get_neural_workflows() -> Dict[str, Any]:
    """Get all neural workflows"""
    try:
        workflows = list(neural_interface_service.neural_workflows.values())
        
        return {
            "success": True,
            "workflows": workflows,
            "count": len(workflows),
            "message": "Neural workflows retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get neural workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signals")
async def get_brain_signals() -> Dict[str, Any]:
    """Get all brain signals"""
    try:
        signals = list(neural_interface_service.brain_signals.values())
        
        return {
            "success": True,
            "signals": signals,
            "count": len(signals),
            "message": "Brain signals retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get brain signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cognitive-states")
async def get_cognitive_states() -> Dict[str, Any]:
    """Get all cognitive states"""
    try:
        states = list(neural_interface_service.cognitive_states.values())
        
        return {
            "success": True,
            "states": states,
            "count": len(states),
            "message": "Cognitive states retrieved successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to get cognitive states: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def neural_health_check() -> Dict[str, Any]:
    """Neural interface service health check"""
    try:
        stats = await neural_interface_service.get_neural_stats()
        
        return {
            "success": True,
            "status": "healthy",
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Neural interface service is healthy"
        }
    
    except Exception as e:
        logger.error(f"Neural interface service health check failed: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Neural interface service is unhealthy"
        }
