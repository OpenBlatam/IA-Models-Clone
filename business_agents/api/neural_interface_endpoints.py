"""
Neural Interface API Endpoints
==============================

REST API endpoints for neural interface integration,
brain-computer interfaces, and neural signal processing.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.neural_interface_service import (
    NeuralInterfaceService, NeuralDevice, NeuralSignal, NeuralCommand, NeuralPattern,
    NeuralSignalType, BrainRegion, NeuralCommandType, NeuralInterfaceType
)
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/neural", tags=["Neural Interface"])

# Pydantic models
class NeuralDeviceCreateRequest(BaseModel):
    name: str = Field(..., description="Device name")
    interface_type: str = Field(..., description="Interface type")
    signal_types: List[str] = Field(..., description="Signal types")
    channels: int = Field(..., description="Number of channels")
    sampling_rate: float = Field(..., description="Sampling rate")
    resolution: float = Field(..., description="Resolution")
    calibration_data: Dict[str, Any] = Field(default_factory=dict, description="Calibration data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Device metadata")

class NeuralCommandCreateRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    command_type: str = Field(..., description="Command type")
    intent: str = Field(..., description="Command intent")
    confidence: float = Field(..., description="Confidence level")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Command parameters")

class BrainStateAnalysisRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    analysis_type: str = Field("comprehensive", description="Analysis type")
    time_window: int = Field(300, description="Time window in seconds")

class NeuralSignalFilter(BaseModel):
    device_id: Optional[str] = Field(None, description="Device ID")
    signal_type: Optional[str] = Field(None, description="Signal type")
    brain_region: Optional[str] = Field(None, description="Brain region")
    time_range: Optional[Dict[str, str]] = Field(None, description="Time range")

# Global neural interface service instance
neural_service = None

def get_neural_service() -> NeuralInterfaceService:
    """Get global neural interface service instance."""
    global neural_service
    if neural_service is None:
        neural_service = NeuralInterfaceService({
            "neural_interface": {
                "max_devices": 100,
                "max_signals_per_device": 1000,
                "signal_processing_enabled": True,
                "command_classification_enabled": True,
                "pattern_recognition_enabled": True,
                "real_time_processing": True,
                "calibration_required": True
            }
        })
    return neural_service

# API Endpoints

@router.post("/initialize", response_model=Dict[str, str])
async def initialize_neural_service(
    current_user: User = Depends(require_permission("neural:manage"))
):
    """Initialize the neural interface service."""
    
    neural_service = get_neural_service()
    
    try:
        await neural_service.initialize()
        return {"message": "Neural Interface Service initialized successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize neural service: {str(e)}")

@router.get("/status", response_model=Dict[str, Any])
async def get_neural_status(
    current_user: User = Depends(require_permission("neural:view"))
):
    """Get neural interface service status."""
    
    neural_service = get_neural_service()
    
    try:
        status = await neural_service.get_service_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get neural status: {str(e)}")

@router.post("/devices/register", response_model=Dict[str, Any])
async def register_neural_device(
    request: NeuralDeviceCreateRequest,
    current_user: User = Depends(require_permission("neural:create"))
):
    """Register a new neural device."""
    
    neural_service = get_neural_service()
    
    try:
        # Convert strings to enums
        interface_type = NeuralInterfaceType(request.interface_type)
        signal_types = [NeuralSignalType(st) for st in request.signal_types]
        
        # Create neural device
        device = NeuralDevice(
            device_id="",  # Will be generated
            name=request.name,
            interface_type=interface_type,
            signal_types=signal_types,
            channels=request.channels,
            sampling_rate=request.sampling_rate,
            resolution=request.resolution,
            status="active",
            calibration_data=request.calibration_data,
            last_calibration=datetime.utcnow(),
            metadata=request.metadata
        )
        
        # Register device
        device_id = await neural_service.register_neural_device(device)
        
        return {
            "device_id": device_id,
            "name": device.name,
            "interface_type": device.interface_type.value,
            "signal_types": [st.value for st in device.signal_types],
            "channels": device.channels,
            "sampling_rate": device.sampling_rate,
            "resolution": device.resolution,
            "status": device.status,
            "calibration_data": device.calibration_data,
            "last_calibration": device.last_calibration.isoformat(),
            "metadata": device.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register neural device: {str(e)}")

@router.get("/devices", response_model=List[Dict[str, Any]])
async def get_neural_devices(
    interface_type: Optional[str] = Query(None, description="Filter by interface type"),
    current_user: User = Depends(require_permission("neural:view"))
):
    """Get neural devices."""
    
    neural_service = get_neural_service()
    
    try:
        # Convert string to enum if provided
        interface_type_enum = NeuralInterfaceType(interface_type) if interface_type else None
        
        # Get devices
        devices = await neural_service.get_neural_devices(interface_type_enum)
        
        result = []
        for device in devices:
            device_dict = {
                "device_id": device.device_id,
                "name": device.name,
                "interface_type": device.interface_type.value,
                "signal_types": [st.value for st in device.signal_types],
                "channels": device.channels,
                "sampling_rate": device.sampling_rate,
                "resolution": device.resolution,
                "status": device.status,
                "calibration_data": device.calibration_data,
                "last_calibration": device.last_calibration.isoformat(),
                "metadata": device.metadata
            }
            result.append(device_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get neural devices: {str(e)}")

@router.get("/devices/{device_id}", response_model=Dict[str, Any])
async def get_neural_device(
    device_id: str,
    current_user: User = Depends(require_permission("neural:view"))
):
    """Get specific neural device."""
    
    neural_service = get_neural_service()
    
    try:
        device = await neural_service.get_neural_device(device_id)
        
        if not device:
            raise HTTPException(status_code=404, detail="Neural device not found")
        
        return {
            "device_id": device.device_id,
            "name": device.name,
            "interface_type": device.interface_type.value,
            "signal_types": [st.value for st in device.signal_types],
            "channels": device.channels,
            "sampling_rate": device.sampling_rate,
            "resolution": device.resolution,
            "status": device.status,
            "calibration_data": device.calibration_data,
            "last_calibration": device.last_calibration.isoformat(),
            "metadata": device.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get neural device: {str(e)}")

@router.get("/devices/{device_id}/signals", response_model=List[Dict[str, Any]])
async def get_neural_signals(
    device_id: str,
    signal_type: Optional[str] = Query(None, description="Filter by signal type"),
    limit: int = Query(100, description="Maximum number of signals"),
    current_user: User = Depends(require_permission("neural:view"))
):
    """Get neural signals from device."""
    
    neural_service = get_neural_service()
    
    try:
        # Convert string to enum if provided
        signal_type_enum = NeuralSignalType(signal_type) if signal_type else None
        
        # Get signals
        signals = await neural_service.get_neural_signals(device_id, signal_type_enum, limit)
        
        result = []
        for signal in signals:
            signal_dict = {
                "signal_id": signal.signal_id,
                "device_id": signal.device_id,
                "signal_type": signal.signal_type.value,
                "brain_region": signal.brain_region.value,
                "data_shape": signal.data.shape,
                "timestamp": signal.timestamp.isoformat(),
                "quality": signal.quality,
                "artifacts": signal.artifacts,
                "features": signal.features,
                "metadata": signal.metadata
            }
            result.append(signal_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get neural signals: {str(e)}")

@router.get("/commands", response_model=List[Dict[str, Any]])
async def get_neural_commands(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    command_type: Optional[str] = Query(None, description="Filter by command type"),
    limit: int = Query(100, description="Maximum number of commands"),
    current_user: User = Depends(require_permission("neural:view"))
):
    """Get neural commands."""
    
    neural_service = get_neural_service()
    
    try:
        # Convert string to enum if provided
        command_type_enum = NeuralCommandType(command_type) if command_type else None
        
        # Get commands
        commands = await neural_service.get_neural_commands(user_id, command_type_enum, limit)
        
        result = []
        for command in commands:
            command_dict = {
                "command_id": command.command_id,
                "user_id": command.user_id,
                "command_type": command.command_type.value,
                "intent": command.intent,
                "confidence": command.confidence,
                "parameters": command.parameters,
                "timestamp": command.timestamp.isoformat(),
                "execution_status": command.execution_status,
                "result": command.result,
                "metadata": command.metadata
            }
            result.append(command_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get neural commands: {str(e)}")

@router.post("/commands/create", response_model=Dict[str, Any])
async def create_neural_command(
    request: NeuralCommandCreateRequest,
    current_user: User = Depends(require_permission("neural:create"))
):
    """Create a neural command."""
    
    neural_service = get_neural_service()
    
    try:
        # Convert string to enum
        command_type = NeuralCommandType(request.command_type)
        
        # Create neural command
        command = NeuralCommand(
            command_id=f"cmd_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}",
            user_id=request.user_id,
            command_type=command_type,
            intent=request.intent,
            confidence=request.confidence,
            parameters=request.parameters,
            timestamp=datetime.utcnow(),
            execution_status="pending",
            result=None,
            metadata={"created_by": "api"}
        )
        
        # Store command
        neural_service.neural_commands[command.command_id] = command
        
        return {
            "command_id": command.command_id,
            "user_id": command.user_id,
            "command_type": command.command_type.value,
            "intent": command.intent,
            "confidence": command.confidence,
            "parameters": command.parameters,
            "timestamp": command.timestamp.isoformat(),
            "execution_status": command.execution_status,
            "result": command.result,
            "metadata": command.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create neural command: {str(e)}")

@router.post("/commands/{command_id}/execute", response_model=Dict[str, str])
async def execute_neural_command(
    command_id: str,
    current_user: User = Depends(require_permission("neural:execute"))
):
    """Execute a neural command."""
    
    neural_service = get_neural_service()
    
    try:
        success = await neural_service.execute_neural_command(command_id)
        
        if success:
            return {"message": f"Neural command {command_id} executed successfully"}
        else:
            raise HTTPException(status_code=404, detail="Neural command not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute neural command: {str(e)}")

@router.get("/patterns", response_model=List[Dict[str, Any]])
async def get_neural_patterns(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    limit: int = Query(100, description="Maximum number of patterns"),
    current_user: User = Depends(require_permission("neural:view"))
):
    """Get neural patterns."""
    
    neural_service = get_neural_service()
    
    try:
        # Get patterns
        patterns = await neural_service.get_neural_patterns(user_id, pattern_type, limit)
        
        result = []
        for pattern in patterns:
            pattern_dict = {
                "pattern_id": pattern.pattern_id,
                "user_id": pattern.user_id,
                "pattern_type": pattern.pattern_type,
                "brain_region": pattern.brain_region.value,
                "features": pattern.features,
                "classification": pattern.classification,
                "confidence": pattern.confidence,
                "created_at": pattern.created_at.isoformat(),
                "metadata": pattern.metadata
            }
            result.append(pattern_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get neural patterns: {str(e)}")

@router.post("/brain-state/analyze", response_model=Dict[str, Any])
async def analyze_brain_state(
    request: BrainStateAnalysisRequest,
    current_user: User = Depends(require_permission("neural:view"))
):
    """Analyze brain state."""
    
    neural_service = get_neural_service()
    
    try:
        analysis = await neural_service.get_brain_state_analysis(request.user_id)
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze brain state: {str(e)}")

@router.get("/signal-types", response_model=List[Dict[str, Any]])
async def get_signal_types(
    current_user: User = Depends(require_permission("neural:view"))
):
    """Get available signal types."""
    
    try:
        signal_types = [
            {
                "type": "eeg",
                "name": "Electroencephalography",
                "description": "Electrical activity of the brain",
                "frequency_range": "0.5-100 Hz",
                "applications": ["brain_state", "emotion", "attention", "fatigue"]
            },
            {
                "type": "emg",
                "name": "Electromyography",
                "description": "Electrical activity of muscles",
                "frequency_range": "20-500 Hz",
                "applications": ["movement", "gesture", "control", "rehabilitation"]
            },
            {
                "type": "eog",
                "name": "Electrooculography",
                "description": "Eye movement and position",
                "frequency_range": "0.1-10 Hz",
                "applications": ["eye_tracking", "attention", "fatigue", "navigation"]
            },
            {
                "type": "ecg",
                "name": "Electrocardiography",
                "description": "Heart electrical activity",
                "frequency_range": "0.5-40 Hz",
                "applications": ["stress", "arousal", "health", "emotion"]
            },
            {
                "type": "meg",
                "name": "Magnetoencephalography",
                "description": "Magnetic fields produced by brain activity",
                "frequency_range": "0.1-100 Hz",
                "applications": ["brain_mapping", "epilepsy", "research", "clinical"]
            },
            {
                "type": "fmri",
                "name": "Functional Magnetic Resonance Imaging",
                "description": "Blood flow changes in the brain",
                "frequency_range": "0.01-0.1 Hz",
                "applications": ["brain_mapping", "research", "clinical", "cognitive"]
            },
            {
                "type": "nirs",
                "name": "Near-Infrared Spectroscopy",
                "description": "Blood oxygenation in the brain",
                "frequency_range": "0.01-0.1 Hz",
                "applications": ["brain_activity", "research", "portable", "clinical"]
            },
            {
                "type": "spike",
                "name": "Neural Spikes",
                "description": "Action potentials from individual neurons",
                "frequency_range": "0.1-1000 Hz",
                "applications": ["neural_coding", "research", "invasive", "precision"]
            },
            {
                "type": "lfp",
                "name": "Local Field Potential",
                "description": "Local electrical activity in brain tissue",
                "frequency_range": "0.1-500 Hz",
                "applications": ["neural_activity", "research", "invasive", "local"]
            }
        ]
        
        return signal_types
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get signal types: {str(e)}")

@router.get("/brain-regions", response_model=List[Dict[str, Any]])
async def get_brain_regions(
    current_user: User = Depends(require_permission("neural:view"))
):
    """Get available brain regions."""
    
    try:
        brain_regions = [
            {
                "region": "frontal",
                "name": "Frontal Lobe",
                "description": "Executive functions, decision making, motor control",
                "functions": ["executive_control", "motor_control", "decision_making", "planning"]
            },
            {
                "region": "parietal",
                "name": "Parietal Lobe",
                "description": "Sensory processing, spatial awareness, attention",
                "functions": ["sensory_processing", "spatial_awareness", "attention", "numerical_cognition"]
            },
            {
                "region": "temporal",
                "name": "Temporal Lobe",
                "description": "Auditory processing, memory, language",
                "functions": ["auditory_processing", "memory", "language", "emotion"]
            },
            {
                "region": "occipital",
                "name": "Occipital Lobe",
                "description": "Visual processing and perception",
                "functions": ["visual_processing", "visual_perception", "object_recognition", "motion_detection"]
            },
            {
                "region": "cerebellum",
                "name": "Cerebellum",
                "description": "Motor coordination, balance, learning",
                "functions": ["motor_coordination", "balance", "motor_learning", "timing"]
            },
            {
                "region": "brainstem",
                "name": "Brainstem",
                "description": "Basic life functions, arousal, consciousness",
                "functions": ["arousal", "consciousness", "basic_life_functions", "sensory_relay"]
            },
            {
                "region": "limbic",
                "name": "Limbic System",
                "description": "Emotion, memory, motivation",
                "functions": ["emotion", "memory", "motivation", "learning"]
            },
            {
                "region": "motor",
                "name": "Motor Cortex",
                "description": "Voluntary movement control",
                "functions": ["voluntary_movement", "motor_planning", "motor_execution", "fine_motor_control"]
            },
            {
                "region": "sensory",
                "name": "Sensory Cortex",
                "description": "Sensory information processing",
                "functions": ["sensory_processing", "tactile_perception", "proprioception", "sensory_integration"]
            },
            {
                "region": "visual",
                "name": "Visual Cortex",
                "description": "Visual information processing",
                "functions": ["visual_processing", "visual_perception", "object_recognition", "visual_attention"]
            },
            {
                "region": "auditory",
                "name": "Auditory Cortex",
                "description": "Auditory information processing",
                "functions": ["auditory_processing", "sound_perception", "speech_recognition", "auditory_attention"]
            }
        ]
        
        return brain_regions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get brain regions: {str(e)}")

@router.get("/command-types", response_model=List[Dict[str, Any]])
async def get_command_types(
    current_user: User = Depends(require_permission("neural:view"))
):
    """Get available command types."""
    
    try:
        command_types = [
            {
                "type": "movement",
                "name": "Movement Control",
                "description": "Control movement and navigation",
                "examples": ["move_left", "move_right", "move_forward", "move_backward", "rotate"]
            },
            {
                "type": "selection",
                "name": "Selection Control",
                "description": "Select items and make choices",
                "examples": ["select_item", "choose_option", "click_button", "select_menu"]
            },
            {
                "type": "navigation",
                "name": "Navigation Control",
                "description": "Navigate through interfaces and spaces",
                "examples": ["navigate_menu", "scroll", "zoom", "pan", "switch_view"]
            },
            {
                "type": "communication",
                "name": "Communication Control",
                "description": "Control communication and messaging",
                "examples": ["send_message", "make_call", "start_chat", "end_call"]
            },
            {
                "type": "control",
                "name": "System Control",
                "description": "Control system functions and settings",
                "examples": ["start_app", "close_app", "change_settings", "restart_system"]
            },
            {
                "type": "creation",
                "name": "Content Creation",
                "description": "Create and edit content",
                "examples": ["create_document", "edit_text", "draw", "compose_music"]
            },
            {
                "type": "analysis",
                "name": "Data Analysis",
                "description": "Analyze data and generate insights",
                "examples": ["analyze_data", "generate_report", "create_chart", "find_patterns"]
            },
            {
                "type": "optimization",
                "name": "System Optimization",
                "description": "Optimize system performance and efficiency",
                "examples": ["optimize_workflow", "improve_performance", "reduce_costs", "enhance_quality"]
            }
        ]
        
        return command_types
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get command types: {str(e)}")

@router.get("/interface-types", response_model=List[Dict[str, Any]])
async def get_interface_types(
    current_user: User = Depends(require_permission("neural:view"))
):
    """Get available interface types."""
    
    try:
        interface_types = [
            {
                "type": "non_invasive",
                "name": "Non-Invasive Interface",
                "description": "External sensors that don't penetrate the skin",
                "examples": ["EEG", "EMG", "EOG", "ECG", "MEG", "fMRI", "NIRS"],
                "advantages": ["safe", "portable", "easy_to_use", "no_surgery"],
                "disadvantages": ["lower_resolution", "signal_noise", "limited_precision"]
            },
            {
                "type": "invasive",
                "name": "Invasive Interface",
                "description": "Sensors that penetrate the skin or skull",
                "examples": ["Neural implants", "Microelectrodes", "Deep brain stimulation"],
                "advantages": ["high_resolution", "precise", "direct_neural_access", "clear_signals"],
                "disadvantages": ["surgery_required", "infection_risk", "tissue_damage", "rejection"]
            },
            {
                "type": "hybrid",
                "name": "Hybrid Interface",
                "description": "Combination of invasive and non-invasive methods",
                "examples": ["EEG + EMG", "fMRI + EEG", "MEG + NIRS"],
                "advantages": ["best_of_both", "redundancy", "validation", "comprehensive"],
                "disadvantages": ["complex", "expensive", "multiple_systems", "integration_challenges"]
            },
            {
                "type": "optical",
                "name": "Optical Interface",
                "description": "Light-based neural interfaces",
                "examples": ["Optogenetics", "Two-photon imaging", "Optical coherence tomography"],
                "advantages": ["high_resolution", "precise", "minimal_damage", "real_time"],
                "disadvantages": ["limited_depth", "light_scattering", "complex_setup", "expensive"]
            },
            {
                "type": "electrical",
                "name": "Electrical Interface",
                "description": "Electrical signal-based interfaces",
                "examples": ["EEG", "EMG", "ECG", "Neural implants", "TMS"],
                "advantages": ["direct_electrical_access", "real_time", "portable", "established"],
                "disadvantages": ["signal_noise", "artifacts", "limited_resolution", "interference"]
            },
            {
                "type": "magnetic",
                "name": "Magnetic Interface",
                "description": "Magnetic field-based interfaces",
                "examples": ["MEG", "TMS", "Magnetic resonance imaging"],
                "advantages": ["non_invasive", "high_resolution", "real_time", "safe"],
                "disadvantages": ["expensive", "bulky", "limited_portability", "complex_setup"]
            },
            {
                "type": "ultrasound",
                "name": "Ultrasound Interface",
                "description": "Ultrasound-based neural interfaces",
                "examples": ["Focused ultrasound", "Ultrasound imaging", "Doppler ultrasound"],
                "advantages": ["non_invasive", "portable", "real_time", "affordable"],
                "disadvantages": ["limited_resolution", "depth_limitations", "tissue_heating", "artifacts"]
            }
        ]
        
        return interface_types
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get interface types: {str(e)}")

@router.get("/analytics", response_model=Dict[str, Any])
async def get_neural_analytics(
    current_user: User = Depends(require_permission("neural:view"))
):
    """Get neural interface analytics."""
    
    neural_service = get_neural_service()
    
    try:
        # Get service status
        status = await neural_service.get_service_status()
        
        # Get devices
        devices = await neural_service.get_neural_devices()
        
        # Get commands
        commands = await neural_service.get_neural_commands()
        
        # Get patterns
        patterns = await neural_service.get_neural_patterns()
        
        # Calculate analytics
        analytics = {
            "total_devices": status.get("total_devices", 0),
            "active_devices": status.get("active_devices", 0),
            "total_signals": status.get("total_signals", 0),
            "total_commands": status.get("total_commands", 0),
            "total_patterns": status.get("total_patterns", 0),
            "interface_types": {},
            "signal_types": {},
            "command_types": {},
            "pattern_types": {},
            "average_signal_quality": 0.0,
            "command_success_rate": 0.0,
            "pattern_confidence": 0.0,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Calculate interface types
        for device in devices:
            interface_type = device.interface_type.value
            if interface_type not in analytics["interface_types"]:
                analytics["interface_types"][interface_type] = 0
            analytics["interface_types"][interface_type] += 1
            
        # Calculate signal types
        for device in devices:
            for signal_type in device.signal_types:
                signal_type_str = signal_type.value
                if signal_type_str not in analytics["signal_types"]:
                    analytics["signal_types"][signal_type_str] = 0
                analytics["signal_types"][signal_type_str] += 1
                
        # Calculate command types
        for command in commands:
            command_type = command.command_type.value
            if command_type not in analytics["command_types"]:
                analytics["command_types"][command_type] = 0
            analytics["command_types"][command_type] += 1
            
        # Calculate pattern types
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            if pattern_type not in analytics["pattern_types"]:
                analytics["pattern_types"][pattern_type] = 0
            analytics["pattern_types"][pattern_type] += 1
            
        # Calculate average signal quality
        total_quality = 0
        quality_count = 0
        for device_id, signals in neural_service.neural_signals.items():
            for signal in signals:
                total_quality += signal.quality
                quality_count += 1
                
        if quality_count > 0:
            analytics["average_signal_quality"] = total_quality / quality_count
            
        # Calculate command success rate
        successful_commands = len([c for c in commands if c.execution_status == "completed"])
        if len(commands) > 0:
            analytics["command_success_rate"] = successful_commands / len(commands)
            
        # Calculate pattern confidence
        total_confidence = sum(p.confidence for p in patterns)
        if len(patterns) > 0:
            analytics["pattern_confidence"] = total_confidence / len(patterns)
            
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get neural analytics: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def neural_health_check():
    """Neural interface service health check."""
    
    neural_service = get_neural_service()
    
    try:
        # Check if service is initialized
        initialized = hasattr(neural_service, 'neural_devices') and len(neural_service.neural_devices) >= 0
        
        # Get service status
        status = await neural_service.get_service_status()
        
        return {
            "status": "healthy" if initialized else "initializing",
            "initialized": initialized,
            "total_devices": status.get("total_devices", 0),
            "active_devices": status.get("active_devices", 0),
            "total_signals": status.get("total_signals", 0),
            "total_commands": status.get("total_commands", 0),
            "total_patterns": status.get("total_patterns", 0),
            "signal_processors": status.get("signal_processors", 0),
            "command_classifiers": status.get("command_classifiers", 0),
            "pattern_recognizers": status.get("pattern_recognizers", 0),
            "brain_state_analyzers": status.get("brain_state_analyzers", 0),
            "real_time_processing": status.get("real_time_processing", False),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }



























