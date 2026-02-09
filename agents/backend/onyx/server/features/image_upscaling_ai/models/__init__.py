"""
Advanced Upscaling Models
=========================

This package provides advanced image upscaling functionality with
a modular architecture based on mixins.

Main Classes:
- AdvancedUpscalingV2: Refactored version using all mixins (recommended)
- AdvancedUpscalingCompat: Compatibility version
- AdvancedUpscaling: Original version (maintained for compatibility)
"""

# Import the recommended version by default
from .advanced_upscaling_v2 import AdvancedUpscalingV2

# Also export as AdvancedUpscaling for easy migration
AdvancedUpscaling = AdvancedUpscalingV2

# Export other versions
try:
    from .advanced_upscaling_compat import AdvancedUpscalingCompat
except ImportError:
    AdvancedUpscalingCompat = None

# Export Real-ESRGAN availability
try:
    from .advanced_upscaling_v2 import REALESRGAN_AVAILABLE, RealESRGANUpscaler
except ImportError:
    REALESRGAN_AVAILABLE = False
    RealESRGANUpscaler = None

# Export mixins for advanced usage
from .mixins import (
    CoreUpscalingMixin,
    EnhancementMixin,
    MLAIMixin,
    AnalysisMixin,
    PipelineMixin,
    AdvancedMethodsMixin,
    BatchProcessingMixin,
    CacheManagementMixin,
    OptimizationMixin,
    QualityAssuranceMixin,
    UtilityMixin,
    SpecializedMixin,
    ExportMixin,
    ConfigurationMixin,
    BenchmarkMixin,
    ValidationMixin,
    MonitoringMixin,
    LearningMixin,
    IntegrationMixin,
    SecurityMixin,
    CompressionMixin,
    PerformanceMixin,
    WorkflowMixin,
    ExperimentationMixin,
    StreamingMixin,
    BackupMixin,
    AllUpscalingMixins,
)

__all__ = [
    # Main classes
    "AdvancedUpscaling",
    "AdvancedUpscalingV2",
    "AdvancedUpscalingCompat",
    
    # Real-ESRGAN
    "REALESRGAN_AVAILABLE",
    "RealESRGANUpscaler",
    
    # Mixins
    "CoreUpscalingMixin",
    "EnhancementMixin",
    "MLAIMixin",
    "AnalysisMixin",
    "PipelineMixin",
    "AdvancedMethodsMixin",
    "BatchProcessingMixin",
    "CacheManagementMixin",
    "OptimizationMixin",
    "QualityAssuranceMixin",
    "UtilityMixin",
    "SpecializedMixin",
    "ExportMixin",
    "ConfigurationMixin",
    "BenchmarkMixin",
    "ValidationMixin",
    "MonitoringMixin",
    "LearningMixin",
    "IntegrationMixin",
    "SecurityMixin",
    "CompressionMixin",
    "PerformanceMixin",
    "WorkflowMixin",
    "ExperimentationMixin",
    "StreamingMixin",
    "BackupMixin",
    "AllUpscalingMixins",
]
