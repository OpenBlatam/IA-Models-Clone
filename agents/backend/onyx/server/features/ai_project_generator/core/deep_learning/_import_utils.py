"""
Import Utilities - Gestión mejorada de imports condicionales
============================================================

Utilidades para gestionar imports condicionales de forma organizada
y eficiente, con mejor manejo de errores y logging.

Este módulo proporciona:
- ImportGroup: Dataclass para organizar grupos de imports relacionados
- ImportManager: Clase principal para gestionar imports condicionales
- create_import_groups: Función para crear grupos de imports predefinidos

Ejemplo de uso:
    >>> from ._import_utils import ImportManager, create_import_groups
    >>> manager = ImportManager(globals())
    >>> groups = create_import_groups()
    >>> results = manager.import_all_groups(groups)
    >>> status = manager.get_import_status()
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ImportGroup:
    """
    Grupo de imports relacionados.
    
    Attributes:
        name: Nombre identificador del grupo
        module_path: Ruta del módulo relativa (ej: ".models")
        symbols: Lista de nombres de símbolos a importar
        description: Descripción del grupo (opcional)
        is_optional: Si True, el grupo es opcional y los errores se ignoran
        priority: Prioridad de importación (mayor = más prioritario)
    
    Example:
        >>> group = ImportGroup(
        ...     name="models",
        ...     module_path=".models",
        ...     symbols=["CNNModel", "RNNModel"],
        ...     description="Model architectures",
        ...     priority=10
        ... )
    """
    name: str
    module_path: str
    symbols: List[str]
    description: str = ""
    is_optional: bool = True
    priority: int = 0


class ImportManager:
    """
    Gestor de imports condicionales mejorado.
    
    Proporciona una forma organizada de gestionar imports opcionales
    con mejor manejo de errores, logging y estadísticas.
    """
    
    def __init__(self, namespace: Dict[str, Any], verbose: bool = False):
        """
        Inicializar gestor de imports.
        
        Args:
            namespace: Namespace donde asignar los imports (típicamente globals()).
            verbose: Si True, loguea información detallada de imports.
        """
        self.namespace = namespace
        self.verbose = verbose
        self._imported_symbols: Dict[str, Any] = {}
        self._failed_imports: Set[str] = set()
        self._successful_imports: Set[str] = set()
        self._import_groups: Dict[str, ImportGroup] = {}
    
    def register_group(self, group: ImportGroup) -> None:
        """
        Registrar un grupo de imports.
        
        Args:
            group: Grupo de imports a registrar.
        """
        self._import_groups[group.name] = group
    
    def import_group(self, group: ImportGroup) -> Dict[str, bool]:
        """
        Importar un grupo de símbolos.
        
        Args:
            group: Grupo de imports a importar.
            
        Returns:
            Diccionario con estado de cada símbolo (True = importado exitosamente).
        """
        results = {}
        
        try:
            fromlist = group.symbols
            module = __import__(group.module_path, fromlist=fromlist, level=1)
            
            imported_count = 0
            for symbol_name in group.symbols:
                try:
                    if hasattr(module, symbol_name):
                        symbol_value = getattr(module, symbol_name)
                        self.namespace[symbol_name] = symbol_value
                        self._imported_symbols[symbol_name] = symbol_value
                        self._successful_imports.add(symbol_name)
                        results[symbol_name] = True
                        imported_count += 1
                    else:
                        if self.verbose:
                            logger.debug(
                                f"Symbol '{symbol_name}' not found in '{group.module_path}'"
                            )
                        self.namespace[symbol_name] = None
                        self._imported_symbols[symbol_name] = None
                        self._failed_imports.add(symbol_name)
                        results[symbol_name] = False
                except AttributeError as e:
                    if self.verbose:
                        logger.debug(
                            f"Failed to import '{symbol_name}' from '{group.module_path}': {e}"
                        )
                    self.namespace[symbol_name] = None
                    self._imported_symbols[symbol_name] = None
                    self._failed_imports.add(symbol_name)
                    results[symbol_name] = False
            
            if imported_count > 0 and self.verbose:
                logger.debug(
                    f"Imported {imported_count}/{len(group.symbols)} symbols "
                    f"from {group.module_path} ({group.name})"
                )
                
        except ImportError as e:
            if self.verbose:
                logger.debug(f"{group.name} not available ({group.module_path}): {e}")
            
            for symbol_name in group.symbols:
                self.namespace[symbol_name] = None
                self._imported_symbols[symbol_name] = None
                self._failed_imports.add(symbol_name)
                results[symbol_name] = False
                
        except Exception as e:
            logger.warning(
                f"Error importing {group.name} ({group.module_path}): {e}",
                exc_info=self.verbose
            )
            
            for symbol_name in group.symbols:
                self.namespace[symbol_name] = None
                self._imported_symbols[symbol_name] = None
                self._failed_imports.add(symbol_name)
                results[symbol_name] = False
        
        return results
    
    def import_all_groups(self, groups: List[ImportGroup]) -> Dict[str, Dict[str, bool]]:
        """
        Importar múltiples grupos de imports.
        
        Args:
            groups: Lista de grupos de imports a importar.
            
        Returns:
            Diccionario con resultados por grupo.
        """
        results = {}
        
        sorted_groups = sorted(groups, key=lambda g: g.priority, reverse=True)
        
        for group in sorted_groups:
            results[group.name] = self.import_group(group)
        
        return results
    
    def get_import_status(self) -> Dict[str, Any]:
        """
        Obtener estado de los imports.
        
        Returns:
            Diccionario con estadísticas de imports.
        """
        total = len(self._imported_symbols)
        successful = len(self._successful_imports)
        failed = len(self._failed_imports)
        
        return {
            "total_symbols": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0.0,
            "successful_symbols": list(self._successful_imports),
            "failed_symbols": list(self._failed_imports),
        }
    
    def check_symbol(self, symbol_name: str) -> bool:
        """
        Verificar si un símbolo está disponible.
        
        Args:
            symbol_name: Nombre del símbolo a verificar.
            
        Returns:
            True si el símbolo está disponible, False en caso contrario.
        """
        return symbol_name in self._successful_imports
    
    def get_available_symbols(self) -> List[str]:
        """
        Obtener lista de símbolos disponibles.
        
        Returns:
            Lista de nombres de símbolos disponibles.
        """
        return list(self._successful_imports)
    
    def get_missing_symbols(self) -> List[str]:
        """
        Obtener lista de símbolos no disponibles.
        
        Returns:
            Lista de nombres de símbolos faltantes.
        """
        return list(self._failed_imports)


def create_import_groups() -> List[ImportGroup]:
    """
    Crear grupos de imports para el módulo deep_learning.
    
    Esta función define todos los grupos de imports opcionales del módulo,
    organizados por categoría y prioridad. Los grupos se ordenan por
    prioridad antes de importarse.
    
    Returns:
        Lista de grupos de imports organizados por categoría.
        
    Note:
        Los grupos se ordenan por prioridad (mayor = más prioritario)
        antes de ser importados. Esto permite importar dependencias
        críticas antes que dependencias opcionales.
        
    Example:
        >>> groups = create_import_groups()
        >>> manager = ImportManager(globals())
        >>> results = manager.import_all_groups(groups)
    """
    groups = [
        # Extended Models
        ImportGroup(
            name="extended_models",
            module_path=".models",
            symbols=["CNNModel", "RNNModel", "TransformersModelWrapper", 
                    "create_transformers_model", "DiffusionModelWrapper", "create_diffusion_model"],
            description="Extended model architectures",
            priority=10,
        ),
        # Data Augmentation
        ImportGroup(
            name="augmentation",
            module_path=".data",
            symbols=["get_image_augmentation", "Mixup", "CutMix"],
            description="Data augmentation utilities",
            priority=9,
        ),
        # Training Callbacks
        ImportGroup(
            name="callbacks",
            module_path=".training",
            symbols=["ModelCheckpoint", "MetricsLogger", "CallbackList"],
            description="Training callbacks",
            priority=9,
        ),
        # Pipelines
        ImportGroup(
            name="pipelines",
            module_path=".pipelines",
            symbols=["TrainingPipeline", "InferencePipeline"],
            description="Training and inference pipelines",
            priority=8,
        ),
        # Helpers
        ImportGroup(
            name="helpers",
            module_path=".helpers",
            symbols=["count_parameters", "get_model_summary", "freeze_layers", 
                    "unfreeze_layers", "plot_training_curves", "plot_confusion_matrix"],
            description="Model helper utilities",
            priority=8,
        ),
        # Core Components
        ImportGroup(
            name="core",
            module_path=".core",
            symbols=["BaseComponent", "ComponentRegistry", "Factory"],
            description="Core component abstractions",
            priority=7,
        ),
        # Presets
        ImportGroup(
            name="presets",
            module_path=".presets",
            symbols=["get_model_preset", "get_training_preset", "get_optimizer_preset",
                    "get_data_preset", "list_presets"],
            description="Configuration presets",
            priority=7,
        ),
        # Templates
        ImportGroup(
            name="templates",
            module_path=".templates",
            symbols=["get_training_template", "get_inference_template", 
                    "get_config_template", "generate_project_structure"],
            description="Code templates",
            priority=7,
        ),
        # Integrations
        ImportGroup(
            name="integrations",
            module_path=".integration",
            symbols=["HuggingFaceHubIntegration", "MLflowIntegration"],
            description="Third-party integrations",
            priority=6,
        ),
        # Architecture Patterns
        ImportGroup(
            name="architecture",
            module_path=".architecture",
            symbols=["ModelBuilder", "TrainingBuilder", "TrainingStrategy", 
                    "StandardTrainingStrategy", "FastTrainingStrategy", "DataStrategy",
                    "StandardDataStrategy", "CrossValidationDataStrategy", 
                    "EventPublisher", "TrainingObserver"],
            description="Architecture patterns",
            priority=6,
        ),
        # Services
        ImportGroup(
            name="services",
            module_path=".services",
            symbols=["ModelService", "TrainingService", "InferenceService", "DataService"],
            description="Service layer components",
            priority=6,
        ),
        # Losses
        ImportGroup(
            name="losses",
            module_path=".losses",
            symbols=["FocalLoss", "LabelSmoothingLoss", "DiceLoss", 
                    "CombinedLoss", "create_loss"],
            description="Custom loss functions",
            priority=5,
        ),
        # Optimization
        ImportGroup(
            name="optimization",
            module_path=".optimization",
            symbols=["quantize_model", "quantize_model_for_mobile", "prune_model",
                    "get_pruning_sparsity", "iterative_pruning", "KnowledgeDistillation",
                    "DistillationTrainer"],
            description="Model optimization utilities",
            priority=5,
        ),
        # Transformers
        ImportGroup(
            name="transformers_utils",
            module_path=".transformers",
            symbols=["load_pretrained_model", "load_tokenizer", "setup_lora", "TokenizedDataset"],
            description="Hugging Face Transformers utilities",
            priority=4,
        ),
        # Diffusion
        ImportGroup(
            name="diffusion_utils",
            module_path=".diffusion",
            symbols=["create_diffusion_pipeline", "DiffusionPipelineWrapper"],
            description="Diffusion model utilities",
            priority=4,
        ),
        # Deployment
        ImportGroup(
            name="deployment",
            module_path=".deployment",
            symbols=["export_to_onnx", "load_onnx_model", "export_to_torchscript", "create_model_api"],
            description="Model deployment utilities",
            priority=5,
        ),
        # Testing
        ImportGroup(
            name="testing",
            module_path=".testing",
            symbols=["ModelTester", "create_test_suite", "benchmark_model"],
            description="Testing utilities",
            priority=5,
        ),
        # Monitoring
        ImportGroup(
            name="monitoring",
            module_path=".monitoring",
            symbols=["ModelMonitor", "DriftDetector", "PerformanceMonitor"],
            description="Model monitoring utilities",
            priority=5,
        ),
        # Visualization
        ImportGroup(
            name="visualization",
            module_path=".visualization",
            symbols=["plot_training_history", "visualize_model_architecture", 
                    "plot_attention_weights", "visualize_feature_maps", 
                    "plot_confusion_matrix_advanced"],
            description="Visualization utilities",
            priority=4,
        ),
        # Security
        ImportGroup(
            name="security",
            module_path=".security",
            symbols=["validate_inputs", "sanitize_inputs", "detect_adversarial", "ModelEncryption"],
            description="Security utilities",
            priority=4,
        ),
        # Experimentation
        ImportGroup(
            name="experimentation",
            module_path=".experimentation",
            symbols=["HyperparameterSearch", "ExperimentComparator", "ABTester", 
                    "create_experiment_config"],
            description="Experimentation utilities",
            priority=4,
        ),
        # Accelerators
        ImportGroup(
            name="accelerators",
            module_path=".accelerators",
            symbols=["setup_accelerator", "optimize_for_gpu", "setup_multi_gpu", 
                    "get_accelerator_info", "enable_mixed_precision", "optimize_batch_size"],
            description="Accelerator utilities",
            priority=3,
        ),
        # Documentation
        ImportGroup(
            name="documentation",
            module_path=".documentation",
            symbols=["generate_model_docs", "generate_training_docs", 
                    "generate_api_docs", "create_project_readme"],
            description="Documentation generation",
            priority=3,
        ),
        # Benchmarking
        ImportGroup(
            name="benchmarking",
            module_path=".benchmarking",
            symbols=["benchmark_inference", "benchmark_training", "compare_models", "BenchmarkSuite"],
            description="Benchmarking utilities",
            priority=3,
        ),
        # Serialization
        ImportGroup(
            name="serialization",
            module_path=".serialization",
            symbols=["save_model_complete", "load_model_complete", "serialize_config",
                    "deserialize_config", "ModelVersionManager"],
            description="Serialization utilities",
            priority=3,
        ),
        # Logging
        ImportGroup(
            name="logging",
            module_path=".logging",
            symbols=["setup_logging", "TrainingLogger", "PerformanceLogger", "ErrorTracker"],
            description="Logging utilities",
            priority=3,
        ),
        # Reporting
        ImportGroup(
            name="reporting",
            module_path=".reporting",
            symbols=["generate_training_report", "generate_model_report", 
                    "generate_experiment_report", "ReportGenerator"],
            description="Report generation",
            priority=3,
        ),
        # Conversion
        ImportGroup(
            name="conversion",
            module_path=".conversion",
            symbols=["convert_model_format", "convert_data_format", "convert_config_format",
                    "FormatConverter"],
            description="Format conversion utilities",
            priority=2,
        ),
        # Validation
        ImportGroup(
            name="validation",
            module_path=".validation",
            symbols=["validate_dataset", "validate_model_config", "validate_training_config",
                    "ValidationSuite"],
            description="Validation utilities",
            priority=2,
        ),
        # Postprocessing
        ImportGroup(
            name="postprocessing",
            module_path=".postprocessing",
            symbols=["format_predictions", "aggregate_predictions", "apply_threshold",
                    "PostProcessor"],
            description="Postprocessing utilities",
            priority=2,
        ),
        # Workflow
        ImportGroup(
            name="workflow",
            module_path=".workflow",
            symbols=["Workflow", "Pipeline", "Task", "TaskStatus", "WorkflowExecutor"],
            description="Workflow management",
            priority=2,
        ),
        # Automation
        ImportGroup(
            name="automation",
            module_path=".automation",
            symbols=["AutoML", "AutoTrainer", "AutoPipeline", "automated_model_selection"],
            description="Automation utilities",
            priority=1,
        ),
    ]
    
    return groups

