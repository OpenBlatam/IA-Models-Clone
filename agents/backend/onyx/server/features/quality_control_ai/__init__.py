"""
Quality Control AI - Sistema de Control de Calidad con Detección de Defectos por Cámara
Enhanced with PyTorch, Transformers, and Diffusion Models

Refactored with Clean Architecture and Domain-Driven Design principles.

Version: 2.3.0
Author: Blatam Academy
"""

__version__ = "2.3.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered quality control system with defect detection via camera"

# Domain Layer (Clean Architecture)
from .domain import (
    # Entities
    Inspection,
    Defect,
    DefectType,
    DefectSeverity,
    Anomaly,
    AnomalyType,
    AnomalySeverity,
    QualityScore,
    QualityStatus,
    Camera,
    CameraStatus,
    # Value Objects
    ImageMetadata,
    DetectionResult,
    QualityMetrics,
    # Services
    InspectionService,
    QualityAssessmentService,
    DefectClassificationService,
    # Exceptions
    QualityControlException,
    InspectionException,
    ModelException,
    CameraException,
    ConfigurationException,
)

# Application Layer (Clean Architecture)
from .application import (
    # Use Cases
    InspectImageUseCase,
    InspectBatchUseCase,
    StartInspectionStreamUseCase,
    StopInspectionStreamUseCase,
    TrainModelUseCase,
    GenerateReportUseCase,
    # DTOs
    InspectionRequest,
    InspectionResponse,
    DefectDTO,
    AnomalyDTO,
    QualityMetricsDTO,
    BatchInspectionRequest,
    BatchInspectionResponse,
    # Application Services
    InspectionApplicationService,
    ModelTrainingApplicationService,
)

# Factories (Dependency Injection)
from .infrastructure.factories import (
    ServiceFactory,
    UseCaseFactory,
)
from .application.factories import (
    ApplicationServiceFactory,
)

# Infrastructure Layer (Clean Architecture)
from .infrastructure import (
    # Repositories
    InspectionRepository,
    ModelRepository,
    ConfigurationRepository,
    # Adapters
    CameraAdapter,
    MLModelLoader,
    StorageAdapter,
    # ML Services
    AnomalyDetectionService,
    ObjectDetectionService,
    DefectClassificationService,
)

# Presentation Layer (Clean Architecture)
from .presentation import (
    create_app,
)

# Configuration
from .config.app_settings import AppSettings, get_settings

# Constants and Type Aliases
from .utils.constants import (
    QUALITY_EXCELLENT,
    QUALITY_GOOD,
    QUALITY_ACCEPTABLE,
    QUALITY_POOR,
    QUALITY_REJECTED,
    SUPPORTED_IMAGE_FORMATS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_CONFIDENCE_THRESHOLD,
)

# Core components (Legacy - Backward Compatible)
from .core import (
    CameraController,
    ObjectDetector,
    AnomalyDetector,
    EnhancedAnomalyDetector,
    DefectClassifier,
    VideoAnalyzer,
    FastQualityInspector,
)

# PyTorch models
from .core.models import (
    AnomalyAutoencoder,
    create_autoencoder,
    DefectViTClassifier,
    create_defect_classifier,
    DiffusionAnomalyDetector,
    create_diffusion_detector,
    create_fast_autoencoder,
    optimize_for_inference,
)

# Services (Legacy - Backward Compatible)
from .services import (
    QualityInspector,
    DefectAnalyzer,
    AlertSystem,
    Alert,
    AlertLevel,
)

# Utils
from .utils import (
    QualityVisualizer,
    ReportGenerator,
    PerformanceOptimizer,
    measure_time,
    QualityControlGradioInterface,
    create_gradio_app,
    FastPreprocessor,
    BatchProcessor,
    PerformanceBenchmark,
    # New utilities
    export_to_json,
    export_to_csv,
    export_to_dict,
    format_number,
    format_percentage,
    format_currency,
    format_datetime_human,
    format_list,
)

# Training
from .training import ModelTrainer, train_autoencoder, train_classifier

# Configuration (Legacy)
from .config import Config, create_default_config_file

__all__ = [
    # Domain Layer
    "Inspection",
    "Defect",
    "DefectType",
    "DefectSeverity",
    "Anomaly",
    "AnomalyType",
    "AnomalySeverity",
    "QualityScore",
    "QualityStatus",
    "Camera",
    "CameraStatus",
    "ImageMetadata",
    "DetectionResult",
    "QualityMetrics",
    "InspectionService",
    "QualityAssessmentService",
    "DefectClassificationService",
    "QualityControlException",
    "InspectionException",
    "ModelException",
    "CameraException",
    "ConfigurationException",
    # Application Layer
    "InspectImageUseCase",
    "InspectBatchUseCase",
    "StartInspectionStreamUseCase",
    "StopInspectionStreamUseCase",
    "TrainModelUseCase",
    "GenerateReportUseCase",
    "InspectionRequest",
    "InspectionResponse",
    "DefectDTO",
    "AnomalyDTO",
    "QualityMetricsDTO",
    "BatchInspectionRequest",
    "BatchInspectionResponse",
    "InspectionApplicationService",
    "ModelTrainingApplicationService",
    # Factories
    "ServiceFactory",
    "UseCaseFactory",
    "ApplicationServiceFactory",
    # Infrastructure Layer
    "InspectionRepository",
    "ModelRepository",
    "ConfigurationRepository",
    "CameraAdapter",
    "MLModelLoader",
    "StorageAdapter",
    "AnomalyDetectionService",
    "ObjectDetectionService",
    "DefectClassificationService",
    # Presentation Layer
    "create_app",
    # Configuration
    "AppSettings",
    "get_settings",
    # Constants
    "QUALITY_EXCELLENT",
    "QUALITY_GOOD",
    "QUALITY_ACCEPTABLE",
    "QUALITY_POOR",
    "QUALITY_REJECTED",
    "SUPPORTED_IMAGE_FORMATS",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    # Legacy (Backward Compatible)
    "CameraController",
    "ObjectDetector",
    "AnomalyDetector",
    "EnhancedAnomalyDetector",
    "DefectClassifier",
    "VideoAnalyzer",
    "QualityInspector",
    "DefectAnalyzer",
    "AlertSystem",
    "Alert",
    "AlertLevel",
    "QualityVisualizer",
    "ReportGenerator",
    "PerformanceOptimizer",
    "measure_time",
    "QualityControlGradioInterface",
    "create_gradio_app",
    "AnomalyAutoencoder",
    "create_autoencoder",
    "DefectViTClassifier",
    "create_defect_classifier",
    "DiffusionAnomalyDetector",
    "create_diffusion_detector",
    "ModelTrainer",
    "train_autoencoder",
    "train_classifier",
    "Config",
    "create_default_config_file",
    "FastQualityInspector",
    "create_fast_autoencoder",
    "optimize_for_inference",
    "FastPreprocessor",
    "BatchProcessor",
    "PerformanceBenchmark",
    # Export utilities
    "export_to_json",
    "export_to_csv",
    "export_to_dict",
    # Format utilities
    "format_number",
    "format_percentage",
    "format_currency",
    "format_datetime_human",
    "format_list",
]
