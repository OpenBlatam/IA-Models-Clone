"""
AR/VR Interface Package
=======================

Augmented Reality (AR) and Virtual Reality (VR) interfaces for immersive experiences.
"""

from .manager import ARVRManager, ARManager, VRManager
from .devices import ARDevice, VRDevice, MixedRealityDevice, HapticDevice
from .scenes import ARScene, VRScene, MixedRealityScene, SceneObject
from .interactions import GestureRecognition, VoiceCommand, EyeTracking, HandTracking
from .types import (
    DeviceType, InteractionType, SceneType, TrackingType,
    ARAnchor, VRController, SpatialMapping, HapticFeedback
)

__all__ = [
    "ARVRManager",
    "ARManager",
    "VRManager",
    "ARDevice",
    "VRDevice",
    "MixedRealityDevice",
    "HapticDevice",
    "ARScene",
    "VRScene",
    "MixedRealityScene",
    "SceneObject",
    "GestureRecognition",
    "VoiceCommand",
    "EyeTracking",
    "HandTracking",
    "DeviceType",
    "InteractionType",
    "SceneType",
    "TrackingType",
    "ARAnchor",
    "VRController",
    "SpatialMapping",
    "HapticFeedback"
]
