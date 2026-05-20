"""
AR/VR Types and Definitions
===========================

Type definitions for Augmented Reality and Virtual Reality interfaces.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
import uuid

class DeviceType(Enum):
    """AR/VR device types."""
    AR_GLASSES = "ar_glasses"
    VR_HEADSET = "vr_headset"
    MIXED_REALITY = "mixed_reality"
    MOBILE_AR = "mobile_ar"
    WEB_AR = "web_ar"
    HAPTIC_DEVICE = "haptic_device"
    CONTROLLER = "controller"
    TRACKER = "tracker"
    CAMERA = "camera"
    SENSOR = "sensor"

class InteractionType(Enum):
    """Interaction types."""
    GESTURE = "gesture"
    VOICE = "voice"
    EYE_TRACKING = "eye_tracking"
    HAND_TRACKING = "hand_tracking"
    CONTROLLER = "controller"
    TOUCH = "touch"
    GAZE = "gaze"
    BRAIN_COMPUTER = "brain_computer"
    HAPTIC = "haptic"
    SPATIAL = "spatial"

class SceneType(Enum):
    """Scene types."""
    AR_OVERLAY = "ar_overlay"
    VR_IMMERSIVE = "vr_immersive"
    MIXED_REALITY = "mixed_reality"
    VIRTUAL_WORLD = "virtual_world"
    AUGMENTED_WORLD = "augmented_world"
    PHOTOREALISTIC = "photorealistic"
    STYLIZED = "stylized"
    MINIMALIST = "minimalist"
    FUTURISTIC = "futuristic"
    HISTORICAL = "historical"

class TrackingType(Enum):
    """Tracking types."""
    SIX_DOF = "six_dof"  # 6 degrees of freedom
    THREE_DOF = "three_dof"  # 3 degrees of freedom
    HAND_TRACKING = "hand_tracking"
    EYE_TRACKING = "eye_tracking"
    FACE_TRACKING = "face_tracking"
    BODY_TRACKING = "body_tracking"
    OBJECT_TRACKING = "object_tracking"
    PLANE_TRACKING = "plane_tracking"
    IMAGE_TRACKING = "image_tracking"
    MARKER_TRACKING = "marker_tracking"

class RenderingEngine(Enum):
    """Rendering engines."""
    UNITY = "unity"
    UNREAL_ENGINE = "unreal_engine"
    WEBXR = "webxr"
    OPENXR = "openxr"
    ARKIT = "arkit"
    ARCORE = "arcore"
    CUSTOM = "custom"

@dataclass
class ARAnchor:
    """AR anchor for spatial positioning."""
    id: str
    position: Tuple[float, float, float]  # x, y, z
    rotation: Tuple[float, float, float, float]  # quaternion
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    anchor_type: str = "world"  # world, image, plane, face
    confidence: float = 1.0
    tracking_state: str = "tracked"  # tracked, limited, not_tracked
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VRController:
    """VR controller definition."""
    id: str
    controller_type: str = "handheld"  # handheld, hand_tracking, eye_tracking
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    buttons: Dict[str, bool] = field(default_factory=dict)
    triggers: Dict[str, float] = field(default_factory=dict)
    joysticks: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    haptic_feedback: bool = False
    battery_level: float = 100.0
    connected: bool = True
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class SpatialMapping:
    """Spatial mapping data."""
    id: str
    mesh_data: bytes = b""
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    triangles: List[Tuple[int, int, int]] = field(default_factory=list)
    normals: List[Tuple[float, float, float]] = field(default_factory=list)
    textures: List[bytes] = field(default_factory=list)
    resolution: float = 0.01  # meters
    coverage_area: float = 0.0  # square meters
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class HapticFeedback:
    """Haptic feedback definition."""
    id: str
    feedback_type: str = "vibration"  # vibration, force, temperature, texture
    intensity: float = 0.5  # 0.0 - 1.0
    duration: float = 0.1  # seconds
    pattern: List[float] = field(default_factory=list)  # vibration pattern
    frequency: float = 200.0  # Hz
    amplitude: float = 1.0
    direction: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spatial_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SceneObject:
    """3D object in AR/VR scene."""
    id: str
    name: str
    object_type: str = "mesh"  # mesh, primitive, particle_system, light, camera
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    mesh_data: Optional[bytes] = None
    texture_data: Optional[bytes] = None
    material_properties: Dict[str, Any] = field(default_factory=dict)
    physics_properties: Dict[str, Any] = field(default_factory=dict)
    animation_data: Dict[str, Any] = field(default_factory=dict)
    interactive: bool = False
    visible: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ARDevice:
    """Augmented Reality device."""
    id: str
    name: str
    device_type: DeviceType
    manufacturer: str = ""
    model: str = ""
    firmware_version: str = ""
    capabilities: List[str] = field(default_factory=list)
    tracking_types: List[TrackingType] = field(default_factory=list)
    display_resolution: Tuple[int, int] = (1920, 1080)
    field_of_view: Tuple[float, float] = (90.0, 90.0)  # horizontal, vertical degrees
    refresh_rate: float = 60.0  # Hz
    latency: float = 20.0  # milliseconds
    battery_level: float = 100.0
    temperature: float = 25.0
    connected: bool = True
    status: str = "active"  # active, standby, error, maintenance
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_healthy(self) -> bool:
        """Check if device is healthy."""
        return (self.connected and 
                self.status == "active" and
                self.battery_level > 10.0 and
                self.temperature < 60.0 and
                (datetime.now() - self.last_heartbeat).total_seconds() < 30)

@dataclass
class VRDevice(ARDevice):
    """Virtual Reality device (extends ARDevice)."""
    vr_specific_capabilities: List[str] = field(default_factory=list)
    room_scale: bool = False
    seated_experience: bool = True
    hand_tracking: bool = False
    eye_tracking: bool = False
    haptic_feedback: bool = False
    controllers: List[VRController] = field(default_factory=list)
    base_stations: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class MixedRealityDevice(ARDevice):
    """Mixed Reality device (extends ARDevice)."""
    passthrough_capability: bool = False
    spatial_mapping: bool = False
    hand_tracking: bool = False
    eye_tracking: bool = False
    voice_commands: bool = False
    spatial_anchors: List[ARAnchor] = field(default_factory=list)
    spatial_mapping_data: Optional[SpatialMapping] = None

@dataclass
class HapticDevice:
    """Haptic feedback device."""
    id: str
    name: str
    device_type: DeviceType
    haptic_types: List[str] = field(default_factory=list)  # vibration, force, temperature
    max_intensity: float = 1.0
    frequency_range: Tuple[float, float] = (1.0, 1000.0)  # Hz
    spatial_resolution: float = 0.001  # meters
    latency: float = 5.0  # milliseconds
    battery_level: float = 100.0
    connected: bool = True
    status: str = "active"
    last_heartbeat: datetime = field(default_factory=datetime.now)

@dataclass
class ARScene:
    """Augmented Reality scene."""
    id: str
    name: str
    description: str
    scene_type: SceneType
    objects: List[SceneObject] = field(default_factory=list)
    anchors: List[ARAnchor] = field(default_factory=list)
    lighting: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_object(self, obj: SceneObject):
        """Add object to scene."""
        self.objects.append(obj)
        self.updated_at = datetime.now()
    
    def remove_object(self, object_id: str) -> bool:
        """Remove object from scene."""
        for i, obj in enumerate(self.objects):
            if obj.id == object_id:
                del self.objects[i]
                self.updated_at = datetime.now()
                return True
        return False
    
    def add_anchor(self, anchor: ARAnchor):
        """Add anchor to scene."""
        self.anchors.append(anchor)
        self.updated_at = datetime.now()

@dataclass
class VRScene(ARScene):
    """Virtual Reality scene (extends ARScene)."""
    vr_specific_properties: Dict[str, Any] = field(default_factory=dict)
    teleport_points: List[Tuple[float, float, float]] = field(default_factory=list)
    boundaries: List[Tuple[float, float, float]] = field(default_factory=list)
    comfort_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MixedRealityScene(ARScene):
    """Mixed Reality scene (extends ARScene)."""
    real_world_objects: List[Dict[str, Any]] = field(default_factory=list)
    occlusion_handling: bool = True
    lighting_estimation: bool = True
    spatial_mapping: Optional[SpatialMapping] = None

@dataclass
class GestureRecognition:
    """Gesture recognition system."""
    id: str
    gesture_type: str = "hand_gesture"  # hand_gesture, body_gesture, face_gesture
    recognition_algorithm: str = "neural_network"
    confidence_threshold: float = 0.8
    supported_gestures: List[str] = field(default_factory=list)
    real_time: bool = True
    latency: float = 50.0  # milliseconds
    accuracy: float = 0.95
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class VoiceCommand:
    """Voice command system."""
    id: str
    language: str = "en-US"
    supported_commands: List[str] = field(default_factory=list)
    wake_word: str = "Hey Assistant"
    noise_cancellation: bool = True
    speaker_identification: bool = False
    real_time: bool = True
    latency: float = 200.0  # milliseconds
    accuracy: float = 0.92
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class EyeTracking:
    """Eye tracking system."""
    id: str
    tracking_accuracy: float = 0.5  # degrees
    sampling_rate: float = 60.0  # Hz
    supported_metrics: List[str] = field(default_factory=lambda: ["gaze_point", "pupil_size", "blink_rate"])
    calibration_required: bool = True
    latency: float = 16.0  # milliseconds
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class HandTracking:
    """Hand tracking system."""
    id: str
    tracking_accuracy: float = 0.01  # meters
    supported_hands: int = 2  # number of hands
    finger_tracking: bool = True
    gesture_recognition: bool = True
    latency: float = 20.0  # milliseconds
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ARVRMetrics:
    """AR/VR performance metrics."""
    total_devices: int = 0
    active_devices: int = 0
    total_sessions: int = 0
    active_sessions: int = 0
    average_session_duration: float = 0.0
    frame_rate: float = 0.0
    latency: float = 0.0
    tracking_accuracy: float = 0.0
    rendering_performance: float = 0.0
    battery_usage: float = 0.0
    thermal_status: float = 0.0
    user_comfort_score: float = 0.0
    interaction_success_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ARVRSession:
    """AR/VR user session."""
    id: str
    user_id: str
    device_id: str
    scene_id: str
    session_type: str = "ar"  # ar, vr, mixed_reality
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: float = 0.0  # seconds
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    user_feedback: Dict[str, Any] = field(default_factory=dict)
    comfort_level: float = 5.0  # 1-10 scale
    motion_sickness: bool = False
    completed: bool = False
