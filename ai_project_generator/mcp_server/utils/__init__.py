"""
MCP Server Utilities
=====================

Utilidades compartidas para el servidor MCP.
"""

# ... existing imports ...

# Dependency Management (nuevo)
from .dependency_manager import (
    DependencyManager,
    Dependency,
    DependencyType,
    DependencyStatus,
    DependencyConflict,
    analyze_dependencies,
    check_dependency_health,
)
from .requirements_optimizer import RequirementsOptimizer
from .testing_advanced import (
    MockResponse,
    TestDataFactory,
    AsyncTestHelper,
    MockHelper,
    TestAssertions,
    TestFixtureManager,
    patch_config,
    with_timeout,
)
from .serialization_utils import (
    JSONEncoder,
    serialize_json,
    deserialize_json,
    serialize_yaml,
    deserialize_yaml,
    serialize_msgpack,
    deserialize_msgpack,
    serialize_pickle,
    deserialize_pickle,
    serialize_base64,
    deserialize_base64,
    Serializer,
    to_dict,
    from_dict,
)
from .structured_logging import (
    StructuredFormatter,
    ContextLogger,
    setup_structured_logging,
    log_with_context,
    create_logger,
)
from .validation_advanced import (
    ValidationError,
    Validator,
    SchemaValidator,
    required,
    not_empty,
    min_length,
    max_length,
    min_value,
    max_value,
    pattern,
    email,
    url,
    one_of,
    custom,
    combine,
)
from .observability_utils import (
    Metric,
    MetricsCollector,
    TraceSpan,
    Tracer,
    get_metrics_collector,
    get_tracer,
    measure_time,
)
from .cache_advanced import (
    CacheStrategy,
    CacheEntry,
    AdvancedCache,
    make_cache_key,
)
from .event_system import (
    EventPriority,
    Event,
    EventHandler,
    EventBus,
    get_event_bus,
)
from .scheduler_utils import (
    TaskStatus,
    ScheduledTask,
    Scheduler,
    get_scheduler,
    schedule_task,
)
from .pipeline_utils import (
    PipelineStage,
    Pipeline,
    ParallelPipeline,
    ConditionalPipeline,
    map_stage,
    filter_stage,
    validate_stage,
)

# ... existing __all__ ...

__all__ = [
    # ... existing exports ...
    # Dependency Management
    "DependencyManager",
    "Dependency",
    "DependencyType",
    "DependencyStatus",
    "DependencyConflict",
    "analyze_dependencies",
    "check_dependency_health",
    "RequirementsOptimizer",
    # Advanced Testing
    "MockResponse",
    "TestDataFactory",
    "AsyncTestHelper",
    "MockHelper",
    "TestAssertions",
    "TestFixtureManager",
    "patch_config",
    "with_timeout",
    # Serialization
    "JSONEncoder",
    "serialize_json",
    "deserialize_json",
    "serialize_yaml",
    "deserialize_yaml",
    "serialize_msgpack",
    "deserialize_msgpack",
    "serialize_pickle",
    "deserialize_pickle",
    "serialize_base64",
    "deserialize_base64",
    "Serializer",
    "to_dict",
    "from_dict",
    # Structured Logging
    "StructuredFormatter",
    "ContextLogger",
    "setup_structured_logging",
    "log_with_context",
    "create_logger",
    # Advanced Validation
    "ValidationError",
    "Validator",
    "SchemaValidator",
    "required",
    "not_empty",
    "min_length",
    "max_length",
    "min_value",
    "max_value",
    "pattern",
    "email",
    "url",
    "one_of",
    "custom",
    "combine",
    # Observability
    "Metric",
    "MetricsCollector",
    "TraceSpan",
    "Tracer",
    "get_metrics_collector",
    "get_tracer",
    "measure_time",
    # Advanced Cache
    "CacheStrategy",
    "CacheEntry",
    "AdvancedCache",
    "make_cache_key",
    # Event System
    "EventPriority",
    "Event",
    "EventHandler",
    "EventBus",
    "get_event_bus",
    # Scheduler
    "TaskStatus",
    "ScheduledTask",
    "Scheduler",
    "get_scheduler",
    "schedule_task",
    # Pipeline
    "PipelineStage",
    "Pipeline",
    "ParallelPipeline",
    "ConditionalPipeline",
    "map_stage",
    "filter_stage",
    "validate_stage",
]
