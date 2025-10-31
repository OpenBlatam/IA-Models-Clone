"""
AR/VR System Module

This module provides comprehensive Augmented and Virtual Reality capabilities for the AI History Comparison System.
"""

from .ar_vr_system import (
    AdvancedARVRManager,
    ARVRType,
    DeviceType,
    TrackingType,
    InteractionType,
    Vector3D,
    Quaternion,
    Transform,
    ARVRObject,
    TrackingData,
    InteractionEvent,
    ARVRScene,
    BaseARVRRenderer,
    OpenGLRenderer,
    VulkanRenderer,
    TrackingSystem,
    InteractionSystem,
    GestureRecognizer,
    VoiceRecognizer,
    HapticSystem,
    SceneManager,
    get_arvr_manager,
    initialize_arvr,
    shutdown_arvr,
    create_arvr_scene,
    render_arvr_scene
)

__all__ = [
    "AdvancedARVRManager",
    "ARVRType",
    "DeviceType",
    "TrackingType",
    "InteractionType",
    "Vector3D",
    "Quaternion",
    "Transform",
    "ARVRObject",
    "TrackingData",
    "InteractionEvent",
    "ARVRScene",
    "BaseARVRRenderer",
    "OpenGLRenderer",
    "VulkanRenderer",
    "TrackingSystem",
    "InteractionSystem",
    "GestureRecognizer",
    "VoiceRecognizer",
    "HapticSystem",
    "SceneManager",
    "get_arvr_manager",
    "initialize_arvr",
    "shutdown_arvr",
    "create_arvr_scene",
    "render_arvr_scene"
]





















