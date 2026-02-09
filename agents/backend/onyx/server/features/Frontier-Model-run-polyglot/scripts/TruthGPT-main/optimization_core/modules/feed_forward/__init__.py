"""
Feed-Forward module for TruthGPT Optimization Core
Contains feed-forward network implementations with various activation functions
"""

from __future__ import annotations
import logging

_logger = logging.getLogger(__name__)

# Core feed-forward
try:
    from .feed_forward import (
        FeedForward,
        GatedFeedForward,
        SwiGLU,
        create_feed_forward
    )
except ImportError as e:
    _logger.debug(f"Could not import feed_forward: {e}")

# PiMoE router
try:
    from .pimoe_router import (
        PiMoESystem,
        TokenLevelRouter,
        PiMoEExpert,
        ExpertType,
        RoutingDecision,
        create_pimoe_system
    )
except ImportError as e:
    _logger.debug(f"Could not import pimoe_router: {e}")

# Enhanced PiMoE integration
try:
    from .enhanced_pimoe_integration import (
        EnhancedPiMoEIntegration,
        AdaptivePiMoE,
        OptimizationMetrics,
        create_enhanced_pimoe_integration
    )
except ImportError as e:
    _logger.debug(f"Could not import enhanced_pimoe_integration: {e}")

# Advanced PiMoE routing
try:
    from .advanced_pimoe_routing import (
        AdvancedPiMoESystem,
        RoutingStrategy,
        AdvancedRoutingConfig,
        AttentionBasedRouter,
        HierarchicalRouter,
        DynamicExpertScaler,
        CrossExpertCommunicator,
        NeuralArchitectureSearchRouter,
    )
except ImportError as e:
    _logger.debug(f"Could not import advanced_pimoe_routing: {e}")

# Performance optimizer
try:
    from .pimoe_performance_optimizer import (
        PiMoEPerformanceOptimizer,
        PerformanceConfig,
        OptimizationLevel,
        MemoryOptimizer,
        ComputationalOptimizer,
        ParallelProcessor,
        CacheManager,
        HardwareOptimizer,
        create_performance_optimizer
    )
except ImportError as e:
    _logger.debug(f"Could not import pimoe_performance_optimizer: {e}")

# Production PiMoE system
try:
    from .production_pimoe_system import (
        ProductionPiMoESystem,
        ProductionConfig,
        ProductionMode,
        ProductionLogger,
        ProductionMonitor,
        ProductionErrorHandler,
        ProductionRequestQueue,
        create_production_pimoe_system,
    )
except ImportError as e:
    _logger.debug(f"Could not import production_pimoe_system: {e}")

# Production API server
try:
    from .production_api_server import (
        ProductionAPIServer,
        PiMoERequest,
        PiMoEResponse,
        HealthResponse,
        MetricsResponse,
        WebSocketMessage,
        create_production_api_server,
        run_production_api_demo
    )
except ImportError as e:
    _logger.debug(f"Could not import production_api_server: {e}")

# Production deployment
try:
    from .production_deployment import (
        ProductionDeployment,
        DeploymentEnvironment,
        ScalingStrategy,
        DockerConfig,
        KubernetesConfig,
        MonitoringConfig,
        LoadBalancerConfig,
        create_production_deployment,
    )
except ImportError as e:
    _logger.debug(f"Could not import production_deployment: {e}")

# Refactored base
try:
    from .refactored_pimoe_base import (
        SystemConfig,
        LoggerProtocol,
        MonitorProtocol,
        ErrorHandlerProtocol,
        RequestQueueProtocol,
        PiMoEProcessorProtocol,
        BaseService,
        BaseConfig,
        ServiceFactory,
        DIContainer,
        EventBus,
        ResourceManager,
        MetricsCollector,
        HealthChecker,
        BasePiMoESystem,
    )
except ImportError as e:
    _logger.debug(f"Could not import refactored_pimoe_base: {e}")

# Refactored production system
try:
    from .refactored_production_system import (
        RefactoredProductionPiMoESystem,
        create_refactored_production_system,
    )
except ImportError as e:
    _logger.debug(f"Could not import refactored_production_system: {e}")

# Refactored config manager
try:
    from .refactored_config_manager import (
        ConfigurationManager,
        ConfigurationFactory,
        ConfigTemplates,
        ConfigValidators,
        EnvironmentConfigBuilder,
        ConfigSource,
        ConfigFormat,
        ConfigValidationRule,
        ConfigSourceInfo,
    )
except ImportError as e:
    _logger.debug(f"Could not import refactored_config_manager: {e}")
