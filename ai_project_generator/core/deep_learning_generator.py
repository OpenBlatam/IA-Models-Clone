"""
Deep Learning Generator - Generador especializado para proyectos de Deep Learning
==================================================================================

Genera código especializado para proyectos que usan PyTorch, Transformers, 
Diffusers, y LLMs siguiendo mejores prácticas de Deep Learning.
Optimizado siguiendo mejores prácticas de Python y FastAPI.

Refactorizado con arquitectura ultra-modular:
- Separación de responsabilidades
- Factory pattern
- Validación robusta
- Integración modular
"""

import logging
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# Import from modular sub-packages
try:
    from .deep_learning_generator.constants import (
        SUPPORTED_FRAMEWORKS,
        SUPPORTED_MODEL_TYPES,
        DEFAULT_CONFIG,
        CAPABILITIES,
        VERSION,
        MODULE_NAME
    )
    from .deep_learning_generator.validators import (
        validate_framework,
        validate_model_type,
        validate_generator_config as _validate_generator_config,
        ValidationError
    )
    from .deep_learning_generator.factory import get_factory
    from .deep_learning_generator.integration import AdvancedFeaturesIntegrator
    _MODULAR_IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Modular imports not available: {e}")
    _MODULAR_IMPORTS_AVAILABLE = False

# Fallback to direct imports
try:
    from .deep_learning.core import DeepLearningGenerator
    _GENERATOR_AVAILABLE = True
except ImportError:
    _GENERATOR_AVAILABLE = False
    DeepLearningGenerator = None


def create_generator(
    framework: str = "pytorch",
    model_type: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    enable_advanced_features: bool = True
) -> Optional[Any]:
    """
    Crea una instancia de DeepLearningGenerator con configuración mejorada.
    
    Args:
        framework: Framework a usar (pytorch, tensorflow, jax)
        model_type: Tipo de modelo (transformer, cnn, rnn, etc.)
        config: Configuración adicional opcional
        enable_advanced_features: Habilitar características avanzadas (pipelines, etc.)
        
    Returns:
        Instancia de DeepLearningGenerator o None si no está disponible
        
    Raises:
        ImportError: Si el generador no está disponible
        ValueError: Si los parámetros son inválidos
        ValidationError: Si la validación falla
    """
    if _MODULAR_IMPORTS_AVAILABLE:
        # Use modular factory pattern
        factory = get_factory()
        
        # Validate inputs
        validate_framework(framework, SUPPORTED_FRAMEWORKS)
        if model_type:
            validate_model_type(model_type, SUPPORTED_MODEL_TYPES)
        
        # Create generator using factory
        return factory.create(
            framework=framework,
            model_type=model_type,
            config=config,
            enable_advanced_features=enable_advanced_features
        )
    else:
        # Fallback to old implementation
        if not _GENERATOR_AVAILABLE:
            raise ImportError(
                "DeepLearningGenerator is not available. "
                "Check that deep_learning.core module is properly installed."
            )
        
        if config is None:
            config = {}
        
        config.setdefault("framework", framework)
        if model_type:
            config.setdefault("model_type", model_type)
        
        try:
            return DeepLearningGenerator(**config)
        except Exception as e:
            logger.error(f"Error creating DeepLearningGenerator: {e}", exc_info=True)
            raise


def get_supported_frameworks() -> List[str]:
    """
    Retorna la lista de frameworks soportados.
    
    Returns:
        Lista de nombres de frameworks
    """
    if _MODULAR_IMPORTS_AVAILABLE:
        return SUPPORTED_FRAMEWORKS.copy()
    else:
        return ["pytorch", "tensorflow", "jax", "onnx"]


def get_supported_model_types() -> List[str]:
    """
    Retorna la lista de tipos de modelos soportados.
    
    Returns:
        Lista de tipos de modelos
    """
    if _MODULAR_IMPORTS_AVAILABLE:
        return SUPPORTED_MODEL_TYPES.copy()
    else:
        return [
            "transformer", "cnn", "rnn", "lstm", "gru",
            "gan", "vae", "diffusion", "llm", "vision_transformer"
        ]


def validate_generator_config(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Valida una configuración de generador.
    
    Args:
        config: Configuración a validar
        
    Returns:
        Tupla (is_valid, error_message)
    """
    if _MODULAR_IMPORTS_AVAILABLE:
        return _validate_generator_config(
            config,
            SUPPORTED_FRAMEWORKS,
            SUPPORTED_MODEL_TYPES
        )
    else:
        # Fallback validation
        if not isinstance(config, dict):
            return False, "Config must be a dictionary"
        return True, None


def get_generator_info() -> Dict[str, Any]:
    """
    Retorna información sobre el generador.
    
    Returns:
        Diccionario con información del generador
    """
    if _MODULAR_IMPORTS_AVAILABLE:
        integrator = AdvancedFeaturesIntegrator()
        advanced_features = integrator.get_pipeline_info()
        
        factory = get_factory()
        return {
            "available": factory.is_available,
            "supported_frameworks": get_supported_frameworks(),
            "supported_model_types": get_supported_model_types(),
            "version": VERSION,
            "module": MODULE_NAME,
            "advanced_features": advanced_features,
            "capabilities": CAPABILITIES,
            "default_config": DEFAULT_CONFIG
        }
    else:
        return {
            "available": _GENERATOR_AVAILABLE,
            "supported_frameworks": get_supported_frameworks(),
            "supported_model_types": get_supported_model_types(),
            "version": "2.1.0",
            "module": "deep_learning_generator",
            "advanced_features": {},
            "capabilities": [
                "model_generation",
                "training_pipelines",
                "evaluation_metrics",
                "optimization",
                "transfer_learning"
            ]
        }


# Importar helpers si están disponibles
try:
    from .deep_learning_generator_helpers import (
        detect_framework_from_code,
        detect_model_type_from_code,
        analyze_code_file,
        suggest_generator_config,
        get_framework_info,
        get_model_type_info,
        generate_config_template
    )
    _HELPERS_AVAILABLE = True
except ImportError:
    _HELPERS_AVAILABLE = False
    logger.debug("Helper functions not available")


# Import additional utilities if available
try:
    from .deep_learning_generator import (
        ConfigBuilder,
        create_config_builder,
        get_preset,
        list_presets,
        get_metrics,
        record_generator_creation,
        optimize_config,
        save_config,
        load_config,
        create_tester,
        get_plugin_manager,
        create_benchmark_runner,
        recommend_config,
        compare_configs,
        merge_configs,
        get_version_manager
    )
    _EXTENDED_FEATURES_AVAILABLE = True
except ImportError:
    _EXTENDED_FEATURES_AVAILABLE = False

# Exportar todo
__all__ = [
    "create_generator",
    "get_supported_frameworks",
    "get_supported_model_types",
    "validate_generator_config",
    "get_generator_info",
]

# Export extended features if available
if _EXTENDED_FEATURES_AVAILABLE:
    __all__.extend([
        "ConfigBuilder",
        "create_config_builder",
        "get_preset",
        "list_presets",
        "get_metrics",
        "record_generator_creation",
        "optimize_config",
        "save_config",
        "load_config",
        "create_tester",
        "get_plugin_manager",
        "create_benchmark_runner",
        "recommend_config",
        "compare_configs",
        "merge_configs",
        "get_version_manager"
    ])

# Exportar helpers si están disponibles
if _HELPERS_AVAILABLE:
    __all__.extend([
        "detect_framework_from_code",
        "detect_model_type_from_code",
        "analyze_code_file",
        "suggest_generator_config",
        "get_framework_info",
        "get_model_type_info",
        "generate_config_template"
    ])

# Exportar DeepLearningGenerator if available
if _MODULAR_IMPORTS_AVAILABLE:
    factory = get_factory()
    if factory.is_available and factory.generator_class:
        DeepLearningGenerator = factory.generator_class
        __all__.append("DeepLearningGenerator")
elif _GENERATOR_AVAILABLE:
    __all__.append("DeepLearningGenerator")
