"""
Deep Learning Module - Modular Deep Learning Code Generation
============================================================

Organized into specialized sub-modules following best practices:
- models: Model architectures (Transformer, CNN, RNN, etc.) with nn.Module
- training: Training utilities with mixed precision, gradient accumulation
- data: Data loaders and datasets with functional programming patterns
- evaluation: Metrics and model evaluation
- inference: Model inference and Gradio integration
- config: YAML configuration management
- utils: Device management, experiment tracking, debugging

Legacy generators for code generation:
- generator_config: Centralized generator configuration
- generator_registry: Registry and Factory for generators
- generation_strategy: Generation strategies
"""

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Core imports - siempre disponibles
from .models import BaseModel, TransformerModel, create_model, initialize_weights
from .data import BaseDataset, TextDataset, ImageDataset, create_dataloader, train_val_test_split
from .training import Trainer, TrainingConfig, EarlyStopping, create_optimizer, create_scheduler
from .evaluation import Metrics, compute_classification_metrics, compute_regression_metrics, evaluate_model
from .inference import InferenceEngine, create_gradio_app, batch_inference
from .config import ConfigManager, load_config, save_config, merge_configs
from .utils.device_utils import get_device, set_seed, enable_anomaly_detection
from .utils.experiment_tracking import ExperimentTracker

__all__ = [
    "BaseModel", "TransformerModel", "create_model", "initialize_weights",
    "BaseDataset", "TextDataset", "ImageDataset", "create_dataloader", "train_val_test_split",
    "Trainer", "TrainingConfig", "EarlyStopping", "create_optimizer", "create_scheduler",
    "Metrics", "compute_classification_metrics", "compute_regression_metrics", "evaluate_model",
    "InferenceEngine", "create_gradio_app", "batch_inference",
    "ConfigManager", "load_config", "save_config", "merge_configs",
    "get_device", "set_seed", "enable_anomaly_detection", "ExperimentTracker",
]

# Importar módulos opcionales usando el sistema mejorado
try:
    from ._import_utils import ImportManager, create_import_groups
    
    _import_manager = ImportManager(globals(), verbose=False)
    _optional_groups = create_import_groups()
    _import_results = _import_manager.import_all_groups(_optional_groups)
    
    # Agregar símbolos importados a __all__
    for group in _optional_groups:
        for symbol in group.symbols:
            if symbol not in __all__:
                __all__.append(symbol)
    
except ImportError:
    logger.debug("Import utilities not available, using fallback import method")
    
    def _try_import(module_path: str, attributes: list):
        """Fallback helper para imports condicionales."""
        try:
            module = __import__(module_path, fromlist=attributes, level=1)
            for attr in attributes:
                if hasattr(module, attr):
                    globals()[attr] = getattr(module, attr)
                    if attr not in __all__:
                        __all__.append(attr)
                else:
                    globals()[attr] = None
            return True
        except (ImportError, AttributeError):
            for attr in attributes:
                globals()[attr] = None
            return False
    
    _optional_modules = [
        ('.models', ['CNNModel', 'RNNModel', 'TransformersModelWrapper', 'create_transformers_model',
                     'DiffusionModelWrapper', 'create_diffusion_model']),
        ('.data', ['get_image_augmentation', 'Mixup', 'CutMix']),
        ('.training', ['ModelCheckpoint', 'MetricsLogger', 'CallbackList']),
        ('.pipelines', ['TrainingPipeline', 'InferencePipeline']),
        ('.helpers', ['count_parameters', 'get_model_summary', 'freeze_layers', 'unfreeze_layers',
                      'plot_training_curves', 'plot_confusion_matrix']),
        ('.core', ['BaseComponent', 'ComponentRegistry', 'Factory']),
        ('.presets', ['get_model_preset', 'get_training_preset', 'get_optimizer_preset',
                      'get_data_preset', 'list_presets']),
        ('.templates', ['get_training_template', 'get_inference_template', 'get_config_template',
                        'generate_project_structure']),
        ('.integration', ['HuggingFaceHubIntegration', 'MLflowIntegration']),
        ('.architecture', ['ModelBuilder', 'TrainingBuilder', 'TrainingStrategy', 'StandardTrainingStrategy',
                           'FastTrainingStrategy', 'DataStrategy', 'StandardDataStrategy',
                           'CrossValidationDataStrategy', 'EventPublisher', 'TrainingObserver']),
        ('.services', ['ModelService', 'TrainingService', 'InferenceService', 'DataService']),
        ('.losses', ['FocalLoss', 'LabelSmoothingLoss', 'DiceLoss', 'CombinedLoss', 'create_loss']),
        ('.optimization', ['quantize_model', 'quantize_model_for_mobile', 'prune_model',
                           'get_pruning_sparsity', 'iterative_pruning', 'KnowledgeDistillation',
                           'DistillationTrainer']),
        ('.transformers', ['load_pretrained_model', 'load_tokenizer', 'setup_lora', 'TokenizedDataset']),
        ('.diffusion', ['create_diffusion_pipeline', 'DiffusionPipelineWrapper']),
        ('.deployment', ['export_to_onnx', 'load_onnx_model', 'export_to_torchscript', 'create_model_api']),
        ('.testing', ['ModelTester', 'create_test_suite', 'benchmark_model']),
        ('.monitoring', ['ModelMonitor', 'DriftDetector', 'PerformanceMonitor']),
        ('.visualization', ['plot_training_history', 'visualize_model_architecture', 'plot_attention_weights',
                            'visualize_feature_maps', 'plot_confusion_matrix_advanced']),
        ('.security', ['validate_inputs', 'sanitize_inputs', 'detect_adversarial', 'ModelEncryption']),
        ('.experimentation', ['HyperparameterSearch', 'ExperimentComparator', 'ABTester', 'create_experiment_config']),
        ('.accelerators', ['setup_accelerator', 'optimize_for_gpu', 'setup_multi_gpu', 'get_accelerator_info',
                             'enable_mixed_precision', 'optimize_batch_size']),
        ('.documentation', ['generate_model_docs', 'generate_training_docs', 'generate_api_docs',
                            'create_project_readme']),
        ('.benchmarking', ['benchmark_inference', 'benchmark_training', 'compare_models', 'BenchmarkSuite']),
        ('.serialization', ['save_model_complete', 'load_model_complete', 'serialize_config',
                            'deserialize_config', 'ModelVersionManager']),
        ('.logging', ['setup_logging', 'TrainingLogger', 'PerformanceLogger', 'ErrorTracker']),
        ('.reporting', ['generate_training_report', 'generate_model_report', 'generate_experiment_report',
                        'ReportGenerator']),
        ('.conversion', ['convert_model_format', 'convert_data_format', 'convert_config_format',
                         'FormatConverter']),
        ('.validation', ['validate_dataset', 'validate_model_config', 'validate_training_config',
                         'ValidationSuite']),
        ('.postprocessing', ['format_predictions', 'aggregate_predictions', 'apply_threshold',
                              'PostProcessor']),
        ('.workflow', ['Workflow', 'Pipeline', 'Task', 'TaskStatus', 'WorkflowExecutor']),
        ('.automation', ['AutoML', 'AutoTrainer', 'AutoPipeline', 'automated_model_selection']),
    ]
    
    for module_path, attributes in _optional_modules:
        _try_import(module_path, attributes)

# Importar generadores de código (legacy)
try:
    from .model_generator import ModelGenerator
    from .training_generator import TrainingGenerator
    from .data_generator import DataGenerator
    from .evaluation_generator import EvaluationGenerator
    from .interface_generator import InterfaceGenerator
    from .config_generator import ConfigGenerator
    from .performance_generator import PerformanceGenerator
    from .utils_generator import UtilsGenerator
    from .deployment_generator import DeploymentGenerator
    from .testing_generator import TestingGenerator
    from .conversion_generator import ConversionGenerator
    from .monitoring_generator import MonitoringGenerator
    from .analysis_generator import AnalysisGenerator
    from .experimentation_generator import ExperimentationGenerator
    from .serialization_generator import SerializationGenerator
    from .validation_generator import ValidationGenerator
    from .security_generator import SecurityGenerator
    from .data_io_generator import DataIOGenerator
    from .reporting_generator import ReportingGenerator
    from .preprocessing_generator import PreprocessingGenerator
    from .postprocessing_generator import PostprocessingGenerator
    
    from .generator_config import GENERATOR_MAP, TRAINING_UTILS, GENERATION_GROUPS
    from .generator_registry import GeneratorRegistry, GeneratorFactory
    from .generation_strategy import (
        GenerationStrategy, CoreModelStrategy, TrainingStrategy,
        InterfaceStrategy, ConfigStrategy, StrategyOrchestrator,
    )
    
    __all__.extend([
        "ModelGenerator", "TrainingGenerator", "DataGenerator", "EvaluationGenerator",
        "InterfaceGenerator", "ConfigGenerator", "PerformanceGenerator", "UtilsGenerator",
        "DeploymentGenerator", "TestingGenerator", "ConversionGenerator", "MonitoringGenerator",
        "AnalysisGenerator", "ExperimentationGenerator", "SerializationGenerator",
        "ValidationGenerator", "SecurityGenerator", "DataIOGenerator", "ReportingGenerator",
        "PreprocessingGenerator", "PostprocessingGenerator",
        "GENERATOR_MAP", "TRAINING_UTILS", "GENERATION_GROUPS",
        "GeneratorRegistry", "GeneratorFactory", "GenerationStrategy",
        "CoreModelStrategy", "TrainingStrategy", "InterfaceStrategy",
        "ConfigStrategy", "StrategyOrchestrator",
    ])
except (ImportError, AttributeError) as e:
    logger.debug(f"Some generators not available: {e}")


# Funciones de utilidad públicas
def get_import_status() -> dict:
    """
    Obtener estado de los imports del módulo.
    
    Returns:
        Diccionario con estadísticas de imports.
    """
    try:
        if '_import_manager' in globals():
            return _import_manager.get_import_status()
    except NameError:
        pass
    return {"status": "unknown"}


def check_imports() -> dict:
    """
    Verificar qué componentes están disponibles.
    
    Returns:
        Diccionario con estado de cada componente (True = disponible).
    """
    try:
        if '_import_manager' in globals():
            return {
                symbol: _import_manager.check_symbol(symbol)
                for symbol in __all__
            }
    except NameError:
        pass
    return {}


def get_available_features() -> list:
    """
    Obtener lista de características disponibles.
    
    Returns:
        Lista de nombres de características disponibles.
    """
    try:
        if '_import_manager' in globals():
            return _import_manager.get_available_symbols()
    except NameError:
        pass
    return []


__all__.extend(["get_import_status", "check_imports", "get_available_features"])
