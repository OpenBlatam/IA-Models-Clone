"""
Neural Interface API routes for Facebook Posts API
Brain-computer interface, neural networks, and cognitive computing
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from ..core.config import get_settings
from ..api.schemas import ErrorResponse
from ..api.dependencies import get_request_id
from ..services.neural_interface_service import (
    get_neural_interface_service, NeuralInterfaceType, CognitiveState, NeuralPattern,
    NeuralSignal, CognitiveProfile, NeuralCommand, BrainComputerInterface
)
from ..services.security_service import get_security_service
from ..infrastructure.monitoring import get_monitor, timed

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/neural", tags=["Neural Interface"])

# Security scheme
security = HTTPBearer()


# BCI Device Management Routes

@router.post(
    "/devices/connect",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "BCI device connected successfully"},
        400: {"description": "Invalid device data"},
        401: {"description": "Unauthorized"},
        500: {"description": "BCI device connection error"}
    },
    summary="Connect BCI device",
    description="Connect a brain-computer interface device"
)
@timed("neural_connect_device")
async def connect_bci_device(
    user_id: str = Query(..., description="User ID"),
    device_model: str = Query(..., description="Device model"),
    interface_type: str = Query("mock", description="Neural interface type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Connect BCI device"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not user_id or not device_model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID and device model are required"
            )
        
        # Validate interface type
        try:
            neural_interface_type = NeuralInterfaceType(interface_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interface type. Valid types: {[t.value for t in NeuralInterfaceType]}"
            )
        
        # Get neural interface service
        neural_service = get_neural_interface_service()
        
        # Connect BCI device
        bci_device = await neural_service.connect_bci_device(user_id, device_model, neural_interface_type)
        
        logger.info(
            "BCI device connected",
            device_id=bci_device.id,
            user_id=user_id,
            device_model=device_model,
            interface_type=interface_type,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "BCI device connected successfully",
            "device": {
                "id": bci_device.id,
                "user_id": bci_device.user_id,
                "interface_type": bci_device.interface_type.value,
                "device_model": bci_device.device_model,
                "sampling_rate": bci_device.sampling_rate,
                "channels": bci_device.channels,
                "resolution": bci_device.resolution,
                "is_connected": bci_device.is_connected,
                "created_at": bci_device.created_at.isoformat()
            },
            "request_id": request_id,
            "connected_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "BCI device connection failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"BCI device connection failed: {str(e)}"
        )


@router.post(
    "/devices/disconnect",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "BCI device disconnected successfully"},
        400: {"description": "Invalid disconnect data"},
        401: {"description": "Unauthorized"},
        500: {"description": "BCI device disconnection error"}
    },
    summary="Disconnect BCI device",
    description="Disconnect a brain-computer interface device"
)
@timed("neural_disconnect_device")
async def disconnect_bci_device(
    user_id: str = Query(..., description="User ID"),
    interface_type: str = Query("mock", description="Neural interface type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Disconnect BCI device"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required"
            )
        
        # Validate interface type
        try:
            neural_interface_type = NeuralInterfaceType(interface_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interface type. Valid types: {[t.value for t in NeuralInterfaceType]}"
            )
        
        # Get neural interface service
        neural_service = get_neural_interface_service()
        
        # Disconnect BCI device
        success = await neural_service.disconnect_bci_device(user_id, neural_interface_type)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="BCI device not found or already disconnected"
            )
        
        logger.info(
            "BCI device disconnected",
            user_id=user_id,
            interface_type=interface_type,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "BCI device disconnected successfully",
            "user_id": user_id,
            "interface_type": interface_type,
            "request_id": request_id,
            "disconnected_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "BCI device disconnection failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"BCI device disconnection failed: {str(e)}"
        )


# Neural Recording Routes

@router.post(
    "/recording/start",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Neural recording started successfully"},
        400: {"description": "Invalid recording parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Neural recording start error"}
    },
    summary="Start neural recording",
    description="Start recording neural signals"
)
@timed("neural_start_recording")
async def start_neural_recording(
    user_id: str = Query(..., description="User ID"),
    interface_type: str = Query("mock", description="Neural interface type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Start neural recording"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required"
            )
        
        # Validate interface type
        try:
            neural_interface_type = NeuralInterfaceType(interface_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interface type. Valid types: {[t.value for t in NeuralInterfaceType]}"
            )
        
        # Get neural interface service
        neural_service = get_neural_interface_service()
        
        # Start neural recording
        success = await neural_service.start_neural_recording(user_id, neural_interface_type)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start neural recording"
            )
        
        logger.info(
            "Neural recording started",
            user_id=user_id,
            interface_type=interface_type,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Neural recording started successfully",
            "user_id": user_id,
            "interface_type": interface_type,
            "request_id": request_id,
            "started_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Neural recording start failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Neural recording start failed: {str(e)}"
        )


@router.post(
    "/recording/stop",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Neural recording stopped successfully"},
        400: {"description": "Invalid recording parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Neural recording stop error"}
    },
    summary="Stop neural recording",
    description="Stop recording neural signals"
)
@timed("neural_stop_recording")
async def stop_neural_recording(
    user_id: str = Query(..., description="User ID"),
    interface_type: str = Query("mock", description="Neural interface type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Stop neural recording"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required"
            )
        
        # Validate interface type
        try:
            neural_interface_type = NeuralInterfaceType(interface_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interface type. Valid types: {[t.value for t in NeuralInterfaceType]}"
            )
        
        # Get neural interface service
        neural_service = get_neural_interface_service()
        
        # Stop neural recording
        success = await neural_service.stop_neural_recording(user_id, neural_interface_type)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to stop neural recording"
            )
        
        logger.info(
            "Neural recording stopped",
            user_id=user_id,
            interface_type=interface_type,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Neural recording stopped successfully",
            "user_id": user_id,
            "interface_type": interface_type,
            "request_id": request_id,
            "stopped_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Neural recording stop failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Neural recording stop failed: {str(e)}"
        )


@router.post(
    "/recording/signal",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Neural signal recorded successfully"},
        400: {"description": "Invalid signal parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Neural signal recording error"}
    },
    summary="Record neural signal",
    description="Record a neural signal"
)
@timed("neural_record_signal")
async def record_neural_signal(
    user_id: str = Query(..., description="User ID"),
    duration: float = Query(1.0, description="Recording duration in seconds", ge=0.1, le=60.0),
    interface_type: str = Query("mock", description="Neural interface type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Record neural signal"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required"
            )
        
        # Validate interface type
        try:
            neural_interface_type = NeuralInterfaceType(interface_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interface type. Valid types: {[t.value for t in NeuralInterfaceType]}"
            )
        
        # Get neural interface service
        neural_service = get_neural_interface_service()
        
        # Record neural signal
        signal = await neural_service.record_neural_signal(user_id, duration, neural_interface_type)
        
        logger.info(
            "Neural signal recorded",
            signal_id=signal.id,
            user_id=user_id,
            duration=duration,
            interface_type=interface_type,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Neural signal recorded successfully",
            "signal": {
                "id": signal.id,
                "user_id": signal.user_id,
                "interface_type": signal.interface_type.value,
                "timestamp": signal.timestamp.isoformat(),
                "frequency_bands": signal.frequency_bands,
                "amplitude": signal.amplitude,
                "phase": signal.phase,
                "coherence": signal.coherence,
                "power_spectrum_length": len(signal.power_spectrum),
                "raw_data_length": len(signal.raw_data)
            },
            "request_id": request_id,
            "recorded_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Neural signal recording failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Neural signal recording failed: {str(e)}"
        )


# Neural Signal Processing Routes

@router.post(
    "/signals/process",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Neural signal processed successfully"},
        400: {"description": "Invalid signal data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Neural signal processing error"}
    },
    summary="Process neural signal",
    description="Process and analyze neural signal"
)
@timed("neural_process_signal")
async def process_neural_signal(
    signal_id: str = Query(..., description="Signal ID"),
    user_id: str = Query(..., description="User ID"),
    interface_type: str = Query("mock", description="Neural interface type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Process neural signal"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not signal_id or not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Signal ID and user ID are required"
            )
        
        # Validate interface type
        try:
            neural_interface_type = NeuralInterfaceType(interface_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interface type. Valid types: {[t.value for t in NeuralInterfaceType]}"
            )
        
        # Get neural interface service
        neural_service = get_neural_interface_service()
        
        # Create mock signal for processing (in real implementation, retrieve from database)
        signal = NeuralSignal(
            id=signal_id,
            user_id=user_id,
            interface_type=neural_interface_type,
            timestamp=datetime.now(),
            frequency_bands={"alpha": 10.5, "beta": 20.3, "theta": 6.2, "gamma": 35.1},
            amplitude=75.5,
            phase=1.57,
            coherence=0.85,
            power_spectrum=[10, 20, 30, 40, 50],
            raw_data=[1, 2, 3, 4, 5]
        )
        
        # Process neural signal
        processed_features = await neural_service.process_neural_signal(signal, neural_interface_type)
        
        logger.info(
            "Neural signal processed",
            signal_id=signal_id,
            user_id=user_id,
            interface_type=interface_type,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Neural signal processed successfully",
            "processed_features": processed_features,
            "request_id": request_id,
            "processed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Neural signal processing failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Neural signal processing failed: {str(e)}"
        )


# Cognitive State Analysis Routes

@router.post(
    "/cognitive/analyze",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Cognitive state analyzed successfully"},
        400: {"description": "Invalid analysis parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Cognitive state analysis error"}
    },
    summary="Analyze cognitive state",
    description="Analyze cognitive state from neural signals"
)
@timed("neural_analyze_cognitive_state")
async def analyze_cognitive_state(
    user_id: str = Query(..., description="User ID"),
    duration: float = Query(10.0, description="Analysis duration in seconds", ge=1.0, le=300.0),
    interface_type: str = Query("mock", description="Neural interface type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Analyze cognitive state"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required"
            )
        
        # Validate interface type
        try:
            neural_interface_type = NeuralInterfaceType(interface_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interface type. Valid types: {[t.value for t in NeuralInterfaceType]}"
            )
        
        # Get neural interface service
        neural_service = get_neural_interface_service()
        
        # Analyze cognitive state
        profile = await neural_service.analyze_cognitive_state(user_id, duration, neural_interface_type)
        
        logger.info(
            "Cognitive state analyzed",
            profile_id=profile.id,
            user_id=user_id,
            cognitive_state=profile.cognitive_state.value,
            duration=duration,
            interface_type=interface_type,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Cognitive state analyzed successfully",
            "profile": {
                "id": profile.id,
                "user_id": profile.user_id,
                "cognitive_state": profile.cognitive_state.value,
                "attention_level": profile.attention_level,
                "memory_activation": profile.memory_activation,
                "emotional_valence": profile.emotional_valence,
                "arousal_level": profile.arousal_level,
                "creativity_index": profile.creativity_index,
                "analytical_thinking": profile.analytical_thinking,
                "neural_patterns": profile.neural_patterns,
                "created_at": profile.created_at.isoformat()
            },
            "request_id": request_id,
            "analyzed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Cognitive state analysis failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cognitive state analysis failed: {str(e)}"
        )


@router.post(
    "/cognitive/predict",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Cognitive state predicted successfully"},
        400: {"description": "Invalid prediction parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Cognitive state prediction error"}
    },
    summary="Predict cognitive state",
    description="Predict future cognitive state"
)
@timed("neural_predict_cognitive_state")
async def predict_cognitive_state(
    user_id: str = Query(..., description="User ID"),
    time_horizon: float = Query(60.0, description="Prediction time horizon in seconds", ge=10.0, le=3600.0),
    interface_type: str = Query("mock", description="Neural interface type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Predict cognitive state"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required"
            )
        
        # Validate interface type
        try:
            neural_interface_type = NeuralInterfaceType(interface_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interface type. Valid types: {[t.value for t in NeuralInterfaceType]}"
            )
        
        # Get neural interface service
        neural_service = get_neural_interface_service()
        
        # Predict cognitive state
        predictions = await neural_service.predict_cognitive_state(user_id, time_horizon, neural_interface_type)
        
        logger.info(
            "Cognitive state predicted",
            user_id=user_id,
            time_horizon=time_horizon,
            interface_type=interface_type,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Cognitive state predicted successfully",
            "predictions": predictions,
            "request_id": request_id,
            "predicted_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Cognitive state prediction failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cognitive state prediction failed: {str(e)}"
        )


@router.get(
    "/cognitive/profile/{user_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Cognitive profile retrieved successfully"},
        404: {"description": "Cognitive profile not found"},
        401: {"description": "Unauthorized"},
        500: {"description": "Cognitive profile retrieval error"}
    },
    summary="Get cognitive profile",
    description="Get cognitive profile for user"
)
@timed("neural_get_cognitive_profile")
async def get_cognitive_profile(
    user_id: str = Path(..., description="User ID"),
    interface_type: str = Query("mock", description="Neural interface type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get cognitive profile"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate interface type
        try:
            neural_interface_type = NeuralInterfaceType(interface_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interface type. Valid types: {[t.value for t in NeuralInterfaceType]}"
            )
        
        # Get neural interface service
        neural_service = get_neural_interface_service()
        
        # Get cognitive profile
        profile = await neural_service.get_cognitive_profile(user_id, neural_interface_type)
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Cognitive profile not found"
            )
        
        logger.info(
            "Cognitive profile retrieved",
            profile_id=profile.id,
            user_id=user_id,
            interface_type=interface_type,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Cognitive profile retrieved successfully",
            "profile": {
                "id": profile.id,
                "user_id": profile.user_id,
                "cognitive_state": profile.cognitive_state.value,
                "attention_level": profile.attention_level,
                "memory_activation": profile.memory_activation,
                "emotional_valence": profile.emotional_valence,
                "arousal_level": profile.arousal_level,
                "creativity_index": profile.creativity_index,
                "analytical_thinking": profile.analytical_thinking,
                "neural_patterns": profile.neural_patterns,
                "created_at": profile.created_at.isoformat()
            },
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Cognitive profile retrieval failed",
            user_id=user_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cognitive profile retrieval failed: {str(e)}"
        )


# Neural Command Routes

@router.post(
    "/commands/decode",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Neural command decoded successfully"},
        400: {"description": "Invalid command parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Neural command decoding error"}
    },
    summary="Decode neural command",
    description="Decode neural command from signals"
)
@timed("neural_decode_command")
async def decode_neural_command(
    user_id: str = Query(..., description="User ID"),
    duration: float = Query(5.0, description="Decoding duration in seconds", ge=1.0, le=30.0),
    interface_type: str = Query("mock", description="Neural interface type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Decode neural command"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required"
            )
        
        # Validate interface type
        try:
            neural_interface_type = NeuralInterfaceType(interface_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interface type. Valid types: {[t.value for t in NeuralInterfaceType]}"
            )
        
        # Get neural interface service
        neural_service = get_neural_interface_service()
        
        # Decode neural command
        command = await neural_service.decode_neural_command(user_id, duration, neural_interface_type)
        
        logger.info(
            "Neural command decoded",
            command_id=command.id,
            user_id=user_id,
            command_type=command.command_type,
            confidence=command.confidence,
            duration=duration,
            interface_type=interface_type,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Neural command decoded successfully",
            "command": {
                "id": command.id,
                "user_id": command.user_id,
                "command_type": command.command_type,
                "intent": command.intent,
                "confidence": command.confidence,
                "execution_time": command.execution_time,
                "success_rate": command.success_rate,
                "created_at": command.created_at.isoformat()
            },
            "request_id": request_id,
            "decoded_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Neural command decoding failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Neural command decoding failed: {str(e)}"
        )


@router.post(
    "/commands/execute",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Neural command executed successfully"},
        400: {"description": "Invalid command data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Neural command execution error"}
    },
    summary="Execute neural command",
    description="Execute a neural command"
)
@timed("neural_execute_command")
async def execute_neural_command(
    command_id: str = Query(..., description="Command ID"),
    user_id: str = Query(..., description="User ID"),
    interface_type: str = Query("mock", description="Neural interface type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Execute neural command"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not command_id or not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Command ID and user ID are required"
            )
        
        # Validate interface type
        try:
            neural_interface_type = NeuralInterfaceType(interface_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interface type. Valid types: {[t.value for t in NeuralInterfaceType]}"
            )
        
        # Get neural interface service
        neural_service = get_neural_interface_service()
        
        # Create mock command for execution (in real implementation, retrieve from database)
        command = NeuralCommand(
            id=command_id,
            user_id=user_id,
            command_type="create_content",
            intent="generate_post",
            confidence=0.85,
            execution_time=1.2,
            success_rate=0.92
        )
        
        # Execute neural command
        result = await neural_service.execute_neural_command(command, neural_interface_type)
        
        logger.info(
            "Neural command executed",
            command_id=command_id,
            user_id=user_id,
            success=result["success"],
            interface_type=interface_type,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Neural command executed successfully",
            "execution_result": result,
            "command_id": command_id,
            "user_id": user_id,
            "request_id": request_id,
            "executed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Neural command execution failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Neural command execution failed: {str(e)}"
        )


@router.get(
    "/commands/{user_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Neural commands retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "Neural commands retrieval error"}
    },
    summary="Get neural commands",
    description="Get neural commands for user"
)
@timed("neural_get_commands")
async def get_neural_commands(
    user_id: str = Path(..., description="User ID"),
    interface_type: str = Query("mock", description="Neural interface type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get neural commands"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate interface type
        try:
            neural_interface_type = NeuralInterfaceType(interface_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interface type. Valid types: {[t.value for t in NeuralInterfaceType]}"
            )
        
        # Get neural interface service
        neural_service = get_neural_interface_service()
        
        # Get neural commands
        commands = await neural_service.get_neural_commands(user_id, neural_interface_type)
        
        commands_data = []
        for command in commands:
            commands_data.append({
                "id": command.id,
                "user_id": command.user_id,
                "command_type": command.command_type,
                "intent": command.intent,
                "confidence": command.confidence,
                "execution_time": command.execution_time,
                "success_rate": command.success_rate,
                "created_at": command.created_at.isoformat()
            })
        
        logger.info(
            "Neural commands retrieved",
            user_id=user_id,
            commands_count=len(commands),
            interface_type=interface_type,
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Neural commands retrieved successfully",
            "commands": commands_data,
            "total_count": len(commands),
            "user_id": user_id,
            "interface_type": interface_type,
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Neural commands retrieval failed",
            user_id=user_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Neural commands retrieval failed: {str(e)}"
        )


# Export router
__all__ = ["router"]





























