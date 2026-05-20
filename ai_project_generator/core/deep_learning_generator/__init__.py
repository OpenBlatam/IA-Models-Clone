"""
Deep Learning Generator Module

Ultra-modular generator for deep learning projects.
Organized into sub-modules:
- constants: Configuration constants
- validators: Validation functions
- factory: Factory pattern for generator creation
- integration: External system integration
- config_builder: Fluent API for building configurations
- presets: Pre-configured presets for common use cases
- cache: Caching utilities
- monitoring: Metrics and monitoring
"""

from .constants import (
    SUPPORTED_FRAMEWORKS,
    SUPPORTED_MODEL_TYPES,
    DEFAULT_CONFIG,
    CAPABILITIES,
    VERSION,
    MODULE_NAME
)

from .validators import (
    validate_config_dict,
    validate_framework,
    validate_model_type,
    validate_generator_config,
    ValidationError
)

from .factory import (
    GeneratorFactory,
    get_factory
)

from .integration import (
    AdvancedFeaturesIntegrator
)

from .config_builder import (
    ConfigBuilder,
    create_config_builder
)

from .presets import (
    PresetManager,
    get_preset,
    list_presets
)

from .cache import (
    GeneratorCache,
    cached_generator,
    clear_all_caches
)

from .monitoring import (
    GeneratorMetrics,
    get_metrics,
    record_generator_creation
)

from .optimizer import (
    ConfigOptimizer,
    optimize_config
)

from .serialization import (
    ConfigSerializer,
    save_config,
    load_config
)

from .testing import (
    GeneratorTester,
    create_tester
)

from .plugins import (
    GeneratorPlugin,
    PluginManager,
    get_plugin_manager,
    register_plugin,
    unregister_plugin
)

from .benchmark import (
    BenchmarkRunner,
    BenchmarkResult,
    create_benchmark_runner
)

from .recommender import (
    ConfigRecommender,
    recommend_config
)

from .comparator import (
    ConfigComparator,
    compare_configs,
    merge_configs
)

from .versioning import (
    ConfigVersion,
    ConfigVersionManager,
    get_version_manager,
    save_config_version,
    get_config_versions,
    get_latest_config_version
)

__all__ = [
    # Constants
    "SUPPORTED_FRAMEWORKS",
    "SUPPORTED_MODEL_TYPES",
    "DEFAULT_CONFIG",
    "CAPABILITIES",
    "VERSION",
    "MODULE_NAME",
    # Validators
    "validate_config_dict",
    "validate_framework",
    "validate_model_type",
    "validate_generator_config",
    "ValidationError",
    # Factory
    "GeneratorFactory",
    "get_factory",
    # Integration
    "AdvancedFeaturesIntegrator",
    # Config Builder
    "ConfigBuilder",
    "create_config_builder",
    # Presets
    "PresetManager",
    "get_preset",
    "list_presets",
    # Cache
    "GeneratorCache",
    "cached_generator",
    "clear_all_caches",
    # Monitoring
    "GeneratorMetrics",
    "get_metrics",
    "record_generator_creation",
    # Optimizer
    "ConfigOptimizer",
    "optimize_config",
    # Serialization
    "ConfigSerializer",
    "save_config",
    "load_config",
    # Testing
    "GeneratorTester",
    "create_tester",
    # Plugins
    "GeneratorPlugin",
    "PluginManager",
    "get_plugin_manager",
    "register_plugin",
    "unregister_plugin",
    # Benchmark
    "BenchmarkRunner",
    "BenchmarkResult",
    "create_benchmark_runner",
    # Recommender
    "ConfigRecommender",
    "recommend_config",
    # Comparator
    "ConfigComparator",
    "compare_configs",
    "merge_configs",
    # Versioning
    "ConfigVersion",
    "ConfigVersionManager",
    "get_version_manager",
    "save_config_version",
    "get_config_versions",
    "get_latest_config_version",
]

