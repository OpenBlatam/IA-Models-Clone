"""
TruthGPT Continuous Agent Implementation

Refactored architecture with separated concerns:
- models.py: Data models (enums, dataclasses)
- task_manager.py: Task queue and processing management
- reflection_planner.py: Reflection and planning logic (Generative Agents)
- metrics_manager.py: Metrics tracking
- maintenance_manager.py: Maintenance operations
- callback_manager.py: Callback handling
- paper_strategies.py: Paper-specific strategies (ReAct, LATS, ToT, ToM, Personality, Toolformer)
- openrouter_client.py: OpenRouter/TruthGPT client
- turtlegpt_continuous_agent.py: Main agent class

Integrates 10 research papers:
1. Generative Agents
2. ReAct
3. LATS
4. LLM to Autonomous
5. Self-Initiated Learning
6. Tree of Thoughts
7. Theory of Mind
8. Personality-Driven
9. Toolformer
10. Sparks of AGI
"""

from .turtlegpt_continuous_agent import TurtleGPTContinuousAgent
from .models import (
    TaskStatus,
    AgentTask,
    AgentMetrics,
    ContinuousAgentConfig
)
from .paper_strategies import (
    ReactStrategy,
    LATSStrategy,
    TreeOfThoughtsStrategy,
    TheoryOfMindStrategy,
    PersonalityStrategy,
    ToolformerStrategy
)
from .component_factory import ComponentFactory
from .config_validator import ConfigValidator
from .event_system import EventBus, EventType, Event
from .health_monitor import HealthMonitor, HealthStatus, ComponentHealth
from .agent_integrator import AgentIntegrator
from .task_processor import TaskProcessor, TaskExecutionResult
from .hook_manager import HookManager, HookType, create_hook_manager
from .memory_operations import MemoryOperations, create_memory_operations
from .resilient_operations import CircuitBreaker, CircuitState, create_circuit_breaker, resilient_call, resilient_call_async
from .periodic_scheduler import PeriodicScheduler, PeriodicTask, PeriodicTaskStatus, create_periodic_scheduler
from .test_generator import (
    UnitTestGenerator, TestFramework, TestComplexity,
    TestObjective, TestCase, FunctionAnalysis, TestSuite,
    create_test_generator
)
from .config_manager import ConfigManager, ConfigSnapshot, create_config_manager
from .logging_manager import LoggingManager, LogConfig, LogLevel, LogEntry, create_logging_manager
from .performance_profiler import (
    PerformanceProfiler, ProfilerMode, PerformanceMetric, FunctionProfile,
    create_performance_profiler
)

__all__ = [
    "TurtleGPTContinuousAgent",
    "TaskStatus",
    "AgentTask",
    "AgentMetrics",
    "ContinuousAgentConfig",
    # Paper strategies
    "ReactStrategy",
    "LATSStrategy",
    "TreeOfThoughtsStrategy",
    "TheoryOfMindStrategy",
    "PersonalityStrategy",
    "ToolformerStrategy",
    # Managers
    "StrategySelector",
    "LearningManager",
    "PromptBuilder",
    "AGICapabilitiesManager",
    # Infrastructure
    "ComponentFactory",
    "ConfigValidator",
    "EventBus",
    "EventType",
    "Event",
    "HealthMonitor",
    "HealthStatus",
    "ComponentHealth",
    "AgentIntegrator",
    # Client
    "OpenRouterTruthGPTClient",
    # Constants
    "STRATEGY_LATS",
    "STRATEGY_TOT",
    "STRATEGY_REACT",
    "STRATEGY_STANDARD",
    "PRIORITY_LATS_THRESHOLD",
    "PRIORITY_TOT_THRESHOLD",
    "PRIORITY_REACT_THRESHOLD",
    "PAPERS",
    # Config Helpers
    "get_config_value",
    "get_bool_config",
    "get_int_config",
    "get_float_config",
    "get_str_config",
    # Utils
    "format_context",
    "validate_priority",
    "format_duration",
    "calculate_success_rate",
    # Error Handling
    "AgentError",
    "TaskProcessingError",
    "StrategyExecutionError",
    "LLMError",
    "MemoryError",
    "ErrorSeverity",
    "handle_errors",
    "safe_execute",
    "ErrorRecoveryStrategy",
    # Decorators
    "log_execution",
    "time_execution",
    "cache_result",
    "rate_limit",
    "validate_input",
    # Middleware
    "Middleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "ValidationMiddleware",
    "MiddlewarePipeline",
    # Task Processor
    "TaskProcessor",
    "TaskExecutionResult",
    # Loop Coordinator
    "LoopCoordinator",
    "LoopPhase",
    "LoopOperation",
    "LoopPhaseBuilder",
    # Async Utils
    "AsyncTaskManager",
    "run_with_timeout",
    "run_with_retry",
    "gather_with_limit",
    "run_periodically",
    "async_retry",
    "async_timeout",
    # Memory Context
    "MemoryContextBuilder",
    "create_memory_context_builder",
    # Agent Operations
    "AgentOperations",
    # LLM Service
    "LLMService",
    "LLMCallTracker",
    # Status Builder
    "StatusBuilder",
    "build_agent_status",
    # Event Publisher
    "EventPublisher",
    "create_event_publisher",
    "publish_on_success",
    # Signal Handler
    "SignalHandler",
    "SignalType",
    "create_signal_handler",
    # Agent Lifecycle
    "AgentLifecycle",
    "create_agent_lifecycle",
    # Startup Logger
    "StartupLogger",
    "create_startup_logger",
    # Loop Configurator
    "LoopConfigurator",
    "create_loop_configurator",
    # Task Validator
    "TaskValidator",
    "create_task_validator",
    # Strategy Manager
    "StrategyManager",
    "create_strategy_manager",
    # Task Executor
    "TaskExecutor",
    "create_task_executor",
    # Hook Manager
    "HookManager",
    "HookType",
    "create_hook_manager",
    # Memory Operations
    "MemoryOperations",
    "create_memory_operations",
    # Resilient Operations
    "CircuitBreaker",
    "CircuitState",
    "create_circuit_breaker",
    "resilient_call",
    "resilient_call_async",
    # Tool Executor
    "ToolExecutor",
    "create_tool_executor",
    # State Manager
    "StateManager",
    "StateTransition",
    "create_state_manager",
    # Metrics Tracker
    "MetricsTracker",
    "create_metrics_tracker",
    # Task Submitter
    "TaskSubmitter",
    "create_task_submitter",
    # Component Registry
    "ComponentRegistry",
    "create_component_registry",
    # Periodic Scheduler
    "PeriodicScheduler",
    "PeriodicTask",
    "PeriodicTaskStatus",
    "create_periodic_scheduler",
    # Test Generator
    "UnitTestGenerator",
    "TestFramework",
    "TestComplexity",
    "TestObjective",
    "TestCase",
    "FunctionAnalysis",
    "TestSuite",
    "create_test_generator",
    # Config Manager
    "ConfigManager",
    "ConfigSnapshot",
    "create_config_manager",
    # Logging Manager
    "LoggingManager",
    "LogConfig",
    "LogLevel",
    "LogEntry",
    "create_logging_manager",
    # Performance Profiler
    "PerformanceProfiler",
    "ProfilerMode",
    "PerformanceMetric",
    "FunctionProfile",
    "create_performance_profiler",
    # State Persistence
    "StatePersistence",
    "SerializationFormat",
    "StateSnapshot",
    "create_state_persistence",
    # Report Generator
    "ReportGenerator",
    "ReportType",
    "ReportFormat",
    "Report",
    "ReportSection",
    "create_report_generator",
]



