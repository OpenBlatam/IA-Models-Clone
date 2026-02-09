"""
Deep Learning Pipelines for Robot Movement AI (optimizado)
===========================================================

Módulo principal que exporta pipelines profesionales para entrenamiento e inferencia.
Optimizado siguiendo mejores prácticas de PyTorch, Transformers, y Deep Learning.

Características:
- Mixed precision training (FP16/BF16)
- Gradient accumulation y clipping
- Early stopping y learning rate scheduling
- Experiment tracking (WandB, TensorBoard)
- Model compilation (torch.compile)
- Efficient data loading con DataLoader optimizado
- Support para Transformers y Diffusers
- Proper error handling y logging

Estructura:
- pipelines_config.py: Configuraciones (TrainingConfig, InferenceConfig, etc.)
- pipelines_datasets.py: Datasets (TrajectoryDataset, SequenceDataset, etc.)
- pipelines_training.py: Pipeline de entrenamiento (TrainingPipeline, EarlyStopping)
- pipelines_inference.py: Pipeline de inferencia (InferencePipeline)
- pipelines_transformers.py: Support para modelos Transformers
- pipelines_diffusion.py: Support para modelos Diffusion
"""

import logging
from typing import TYPE_CHECKING, Optional, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .pipelines_config import (
        TrainingConfig,
        InferenceConfig,
        OptimizerConfig,
        SchedulerConfig,
        OptimizerType,
        SchedulerType,
        LossType,
    )
    from .pipelines_datasets import (
        BaseTrajectoryDataset,
        TrajectoryDataset,
        SequenceDataset,
    )
    from .pipelines_training import TrainingPipeline, EarlyStopping
    from .pipelines_inference import InferencePipeline

# Lazy imports para mejor rendimiento
try:
    from .pipelines_config import (
        TrainingConfig,
        InferenceConfig,
        OptimizerConfig,
        SchedulerConfig,
        OptimizerType,
        SchedulerType,
        LossType,
    )
    from .pipelines_datasets import (
        BaseTrajectoryDataset,
        TrajectoryDataset,
        SequenceDataset,
        create_dataloader,
    )
    from .pipelines_training import TrainingPipeline, EarlyStopping
    from .pipelines_inference import InferencePipeline
    
    # Optional imports para transformers y diffusion
    try:
        from .pipelines_transformers import (
            TransformerTrainingPipeline,
            TransformerInferencePipeline,
        )
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        TransformerTrainingPipeline = None
        TransformerInferencePipeline = None
    
    try:
        from .pipelines_diffusion import DiffusionPipelineWrapper
        DIFFUSION_AVAILABLE = True
    except ImportError:
        DIFFUSION_AVAILABLE = False
        DiffusionPipelineWrapper = None
    
    # Importar módulos adicionales
    try:
        from .pipelines_utils import (
            validate_tensor, check_gradients, detect_anomaly,
            profile_training, count_parameters, get_model_summary,
            set_seed, freeze_model, load_checkpoint, get_device_info,
            time_function, GradientMonitor
        )
        UTILS_AVAILABLE = True
    except ImportError:
        UTILS_AVAILABLE = False
        validate_tensor = None
        check_gradients = None
        detect_anomaly = None
        profile_training = None
        count_parameters = None
        get_model_summary = None
        set_seed = None
        freeze_model = None
        load_checkpoint = None
        get_device_info = None
        time_function = None
        GradientMonitor = None
    
    try:
        from .pipelines_gradio import GradioModelInterface, create_model_comparison_interface
        GRADIO_AVAILABLE = True
    except ImportError:
        GRADIO_AVAILABLE = False
        GradioModelInterface = None
        create_model_comparison_interface = None
    
    try:
        from .pipelines_metrics import MetricsCalculator, ConfusionMatrix
        METRICS_AVAILABLE = True
    except ImportError:
        METRICS_AVAILABLE = False
        MetricsCalculator = None
        ConfusionMatrix = None
    
    try:
        from .pipelines_callbacks import (
            Callback, LearningRateSchedulerCallback, ModelCheckpointCallback,
            ProgressBarCallback, TimingCallback, CallbackList
        )
        CALLBACKS_AVAILABLE = True
    except ImportError:
        CALLBACKS_AVAILABLE = False
        Callback = None
        LearningRateSchedulerCallback = None
        ModelCheckpointCallback = None
        ProgressBarCallback = None
        TimingCallback = None
        CallbackList = None
    
    try:
        from .pipelines_distributed import (
            DistributedTrainingSetup,
            setup_distributed_training
        )
        DISTRIBUTED_AVAILABLE = True
    except ImportError:
        DISTRIBUTED_AVAILABLE = False
        DistributedTrainingSetup = None
        setup_distributed_training = None
    
    try:
        from .pipelines_ensemble import (
            ModelEnsemble,
            StackingEnsemble
        )
        ENSEMBLE_AVAILABLE = True
    except ImportError:
        ENSEMBLE_AVAILABLE = False
        ModelEnsemble = None
        StackingEnsemble = None
    
    try:
        from .pipelines_hyperparameter import (
            HyperparameterSpace,
            HyperparameterTuner
        )
        HYPERPARAMETER_AVAILABLE = True
    except ImportError:
        HYPERPARAMETER_AVAILABLE = False
        HyperparameterSpace = None
        HyperparameterTuner = None
    
    try:
        from .pipelines_export import (
            ModelExporter,
            ModelQuantizer,
            ModelPruner
        )
        EXPORT_AVAILABLE = True
    except ImportError:
        EXPORT_AVAILABLE = False
        ModelExporter = None
        ModelQuantizer = None
        ModelPruner = None
    
    try:
        from .pipelines_serving import (
            ModelServer,
            InferenceOptimizer
        )
        SERVING_AVAILABLE = True
    except ImportError:
        SERVING_AVAILABLE = False
        ModelServer = None
        InferenceOptimizer = None
    
    try:
        from .pipelines_augmentation import (
            DataAugmentation,
            AdvancedAugmentation
        )
        AUGMENTATION_AVAILABLE = True
    except ImportError:
        AUGMENTATION_AVAILABLE = False
        DataAugmentation = None
        AdvancedAugmentation = None
    
    try:
        from .pipelines_visualization import (
            TrainingVisualizer,
            ModelVisualizer,
            InteractiveVisualizer
        )
        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False
        TrainingVisualizer = None
        ModelVisualizer = None
        InteractiveVisualizer = None
    
    try:
        from .pipelines_monitoring import (
            ModelMetrics,
            DataDriftDetector,
            PerformanceMonitor,
            ModelRegistry
        )
        MONITORING_AVAILABLE = True
    except ImportError:
        MONITORING_AVAILABLE = False
        ModelMetrics = None
        DataDriftDetector = None
        PerformanceMonitor = None
        ModelRegistry = None
    
    try:
        from .pipelines_optimization import (
            KnowledgeDistillation,
            NeuralArchitectureSearch,
            ModelCompression,
            GradientBasedOptimization
        )
        OPTIMIZATION_AVAILABLE = True
    except ImportError:
        OPTIMIZATION_AVAILABLE = False
        KnowledgeDistillation = None
        NeuralArchitectureSearch = None
        ModelCompression = None
        GradientBasedOptimization = None
    
    try:
        from .pipelines_transfer import (
            TransferLearningManager,
            PretrainedModelLoader,
            DomainAdaptation
        )
        TRANSFER_AVAILABLE = True
    except ImportError:
        TRANSFER_AVAILABLE = False
        TransferLearningManager = None
        PretrainedModelLoader = None
        DomainAdaptation = None
    
    try:
        from .pipelines_active import (
            ActiveLearningStrategy,
            ActiveLearningLoop
        )
        ACTIVE_LEARNING_AVAILABLE = True
    except ImportError:
        ACTIVE_LEARNING_AVAILABLE = False
        ActiveLearningStrategy = None
        ActiveLearningLoop = None
    
    try:
        from .pipelines_advanced import (
            ReinforcementLearningWrapper,
            MetaLearningWrapper,
            ContinualLearning,
            AdvancedRegularization
        )
        ADVANCED_AVAILABLE = True
    except ImportError:
        ADVANCED_AVAILABLE = False
        ReinforcementLearningWrapper = None
        MetaLearningWrapper = None
        ContinualLearning = None
        AdvancedRegularization = None
    
    try:
        from .pipelines_automl import (
            AutoMLConfig,
            AutoFeatureEngineering,
            AutoModelSelection,
            AutoMLPipeline
        )
        AUTOML_AVAILABLE = True
    except ImportError:
        AUTOML_AVAILABLE = False
        AutoMLConfig = None
        AutoFeatureEngineering = None
        AutoModelSelection = None
        AutoMLPipeline = None
    
    logger.info("Pipeline modules loaded successfully")
    
except ImportError as e:
    logger.error(f"Failed to import pipeline components: {e}", exc_info=True)
    raise ImportError(
        f"Failed to import pipeline components: {e}. "
        "Ensure all pipeline modules are properly installed and PyTorch is available."
    ) from e

# Alias para compatibilidad
PipelineConfig = TrainingConfig

__all__ = [
    # Config
    "TrainingConfig",
    "InferenceConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "OptimizerType",
    "SchedulerType",
    "LossType",
    "PipelineConfig",
    # Datasets
    "BaseTrajectoryDataset",
    "TrajectoryDataset",
    "SequenceDataset",
    "create_dataloader",
    # Training
    "TrainingPipeline",
    "EarlyStopping",
    # Inference
    "InferencePipeline",
    # Transformers (optional)
    "TransformerTrainingPipeline",
    "TransformerInferencePipeline",
    "TRANSFORMERS_AVAILABLE",
    # Diffusion (optional)
    "DiffusionPipelineWrapper",
    "DIFFUSION_AVAILABLE",
    # Utils (optional)
    "validate_tensor", "check_gradients", "detect_anomaly",
    "profile_training", "count_parameters", "get_model_summary",
    "set_seed", "freeze_model", "load_checkpoint", "get_device_info",
    "time_function", "GradientMonitor", "UTILS_AVAILABLE",
    # Gradio (optional)
    "GradioModelInterface", "create_model_comparison_interface", "GRADIO_AVAILABLE",
    # Metrics (optional)
    "MetricsCalculator", "ConfusionMatrix", "METRICS_AVAILABLE",
    # Callbacks (optional)
    "Callback", "LearningRateSchedulerCallback", "ModelCheckpointCallback",
    "ProgressBarCallback", "TimingCallback", "CallbackList", "CALLBACKS_AVAILABLE",
    # Distributed (optional)
    "DistributedTrainingSetup", "setup_distributed_training", "DISTRIBUTED_AVAILABLE",
    # Ensemble (optional)
    "ModelEnsemble", "StackingEnsemble", "ENSEMBLE_AVAILABLE",
    # Hyperparameter Tuning (optional)
    "HyperparameterSpace", "HyperparameterTuner", "HYPERPARAMETER_AVAILABLE",
    # Export & Optimization (optional)
    "ModelExporter", "ModelQuantizer", "ModelPruner", "EXPORT_AVAILABLE",
    # Serving (optional)
    "ModelServer", "InferenceOptimizer", "SERVING_AVAILABLE",
    # Augmentation (optional)
    "DataAugmentation", "AdvancedAugmentation", "AUGMENTATION_AVAILABLE",
    # Visualization (optional)
    "TrainingVisualizer", "ModelVisualizer", "InteractiveVisualizer", "VISUALIZATION_AVAILABLE",
    # Monitoring (optional)
    "ModelMetrics", "DataDriftDetector", "PerformanceMonitor", "ModelRegistry", "MONITORING_AVAILABLE",
    # Advanced Optimization (optional)
    "KnowledgeDistillation", "NeuralArchitectureSearch", "ModelCompression",
    "GradientBasedOptimization", "OPTIMIZATION_AVAILABLE",
    # Transfer Learning (optional)
    "TransferLearningManager", "PretrainedModelLoader", "DomainAdaptation", "TRANSFER_AVAILABLE",
    # Active Learning (optional)
    "ActiveLearningStrategy", "ActiveLearningLoop", "ACTIVE_LEARNING_AVAILABLE",
    # Advanced Features (optional)
    "ReinforcementLearningWrapper", "MetaLearningWrapper", "ContinualLearning",
    "AdvancedRegularization", "ADVANCED_AVAILABLE",
    # AutoML (optional)
    "AutoMLConfig", "AutoFeatureEngineering", "AutoModelSelection", "AutoMLPipeline",
    "AUTOML_AVAILABLE",
]


def get_pipeline_info() -> dict:
    """
    Obtener información sobre pipelines disponibles (optimizado).
    
    Returns:
        Diccionario con información de pipelines y dependencias
    """
    try:
        import torch
        torch_available = True
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
    except ImportError:
        torch_available = False
        torch_version = None
        cuda_available = False
        cuda_version = None
    
    try:
        import transformers
        transformers_available = True
        transformers_version = transformers.__version__
    except ImportError:
        transformers_available = False
        transformers_version = None
    
    try:
        import diffusers
        diffusers_available = True
        diffusers_version = diffusers.__version__
    except ImportError:
        diffusers_available = False
        diffusers_version = None
    
    try:
        import gradio
        gradio_available = True
        gradio_version = gradio.__version__
    except ImportError:
        gradio_available = False
        gradio_version = None
    
    return {
        "torch": {
            "available": torch_available,
            "version": torch_version,
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
        },
        "transformers": {
            "available": transformers_available,
            "version": transformers_version,
        },
        "diffusers": {
            "available": diffusers_available,
            "version": diffusers_version,
        },
        "gradio": {
            "available": gradio_available,
            "version": gradio_version,
        },
        "pipelines": {
            "training": "TrainingPipeline",
            "inference": "InferencePipeline",
            "transformers_training": "TransformerTrainingPipeline" if TRANSFORMERS_AVAILABLE else None,
            "transformers_inference": "TransformerInferencePipeline" if TRANSFORMERS_AVAILABLE else None,
            "diffusion": "DiffusionPipelineWrapper" if DIFFUSION_AVAILABLE else None,
        },
        "modules": {
            "utils": UTILS_AVAILABLE,
            "gradio": GRADIO_AVAILABLE,
            "metrics": METRICS_AVAILABLE,
            "callbacks": CALLBACKS_AVAILABLE,
            "distributed": DISTRIBUTED_AVAILABLE,
            "ensemble": ENSEMBLE_AVAILABLE,
            "hyperparameter": HYPERPARAMETER_AVAILABLE,
            "export": EXPORT_AVAILABLE,
            "serving": SERVING_AVAILABLE,
            "augmentation": AUGMENTATION_AVAILABLE,
            "visualization": VISUALIZATION_AVAILABLE,
            "monitoring": MONITORING_AVAILABLE,
            "optimization": OPTIMIZATION_AVAILABLE,
            "transfer": TRANSFER_AVAILABLE,
            "active_learning": ACTIVE_LEARNING_AVAILABLE,
            "advanced": ADVANCED_AVAILABLE,
            "automl": AUTOML_AVAILABLE,
        }
    }
