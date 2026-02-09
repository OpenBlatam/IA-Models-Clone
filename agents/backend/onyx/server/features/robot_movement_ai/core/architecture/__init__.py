"""
Architecture Layer
==================

Capa de arquitectura con interfaces, protocolos y patrones de diseño.
"""

from .interfaces import (
    IRouteModel,
    IRouteStrategy,
    IRouteRepository,
    IRouteService,
    IDataProcessor,
    ITrainingPipeline,
    IInferenceEngine
)

from .domain import (
    Route,
    RouteRequest,
    RouteResponse,
    RouteMetrics,
    Node,
    Edge,
    Graph
)

from .services import (
    RouteService,
    ModelService,
    TrainingService,
    InferenceService
)

from .repositories import (
    RouteRepository,
    ModelRepository,
    DataRepository
)

from .factories import (
    RouteModelFactory,
    StrategyFactory,
    PipelineFactory
)

from .builders import (
    RouteServiceBuilder,
    TrainingPipelineBuilder,
    InferencePipelineBuilder
)

from .plugins import (
    PluginManager,
    RouteStrategyPlugin,
    ModelPlugin
)

from .events import (
    EventBus,
    EventHandler,
    TrainingEvent,
    InferenceEvent
)

from .dependency_injection import (
    Container,
    ServiceProvider,
    register_service,
    resolve_service
)

from .pipelines import (
    PipelineStage,
    Pipeline,
    ParallelPipeline,
    ConditionalPipeline,
    PipelineBuilder
)

__all__ = [
    # Interfaces
    "IRouteModel",
    "IRouteStrategy",
    "IRouteRepository",
    "IRouteService",
    "IDataProcessor",
    "ITrainingPipeline",
    "IInferenceEngine",
    # Domain
    "Route",
    "RouteRequest",
    "RouteResponse",
    "RouteMetrics",
    "Node",
    "Edge",
    "Graph",
    # Services
    "RouteService",
    "ModelService",
    "TrainingService",
    "InferenceService",
    # Repositories
    "RouteRepository",
    "ModelRepository",
    "DataRepository",
    # Factories
    "RouteModelFactory",
    "StrategyFactory",
    "PipelineFactory",
    # Builders
    "RouteServiceBuilder",
    "TrainingPipelineBuilder",
    "InferencePipelineBuilder",
    # Plugins
    "PluginManager",
    "RouteStrategyPlugin",
    "ModelPlugin",
    # Events
    "EventBus",
    "EventHandler",
    "TrainingEvent",
    "InferenceEvent",
    # DI
    "Container",
    "ServiceProvider",
    "register_service",
    "resolve_service",
    # Pipelines
    "PipelineStage",
    "Pipeline",
    "ParallelPipeline",
    "ConditionalPipeline",
    "PipelineBuilder"
]

