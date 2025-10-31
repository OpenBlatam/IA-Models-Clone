"""
Biometric Systems Module

This module provides advanced biometric authentication and identification capabilities including:
- Facial recognition and analysis
- Fingerprint identification
- Iris and retina scanning
- Voice recognition and analysis
- Behavioral biometrics
- Gait recognition
- Heart rate variability analysis
- DNA sequence analysis
- Multi-modal biometric fusion
- Biometric template management
"""

from .biometric_system import (
    BiometricSystemManager,
    FacialRecognitionService,
    FingerprintService,
    IrisRecognitionService,
    VoiceRecognitionService,
    BehavioralBiometricsService,
    GaitRecognitionService,
    HeartRateAnalysisService,
    DNASequenceService,
    MultiModalBiometricsService,
    BiometricTemplateManager,
    get_biometric_system_manager,
    initialize_biometric_systems,
    shutdown_biometric_systems
)

__all__ = [
    "BiometricSystemManager",
    "FacialRecognitionService",
    "FingerprintService",
    "IrisRecognitionService",
    "VoiceRecognitionService",
    "BehavioralBiometricsService",
    "GaitRecognitionService",
    "HeartRateAnalysisService",
    "DNASequenceService",
    "MultiModalBiometricsService",
    "BiometricTemplateManager",
    "get_biometric_system_manager",
    "initialize_biometric_systems",
    "shutdown_biometric_systems"
]





















