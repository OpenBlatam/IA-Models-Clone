"""
Component Initializer - Inicialización de componentes
====================================================

Gestiona la inicialización de todos los componentes del agente
(core y opcionales) de forma centralizada y con manejo robusto de errores.
"""

import logging
from typing import Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path

from ..error_handling import safe_initialize, log_component_status

if TYPE_CHECKING:
    from ..agent import CursorAgent
    from ..devin_persona import DevinPersona, AgentMode

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Registry de componentes inicializados.
    
    Mantiene referencias a todos los componentes (core y opcionales)
    que fueron inicializados exitosamente.
    """
    
    def __init__(self) -> None:
        """Inicializar registry vacío."""
        # Core components (always initialized)
        self.security_manager: Optional[Any] = None
        self.devin: Optional[Any] = None
        self.devin_commands: Optional[Any] = None
        self.tool_manager: Optional[Any] = None
        self.change_verifier: Optional[Any] = None
        self.test_runner: Optional[Any] = None
        self.reference_tracker: Optional[Any] = None
        self.parallel_executor: Optional[Any] = None
        self.completion_verifier: Optional[Any] = None
        self.iteration_manager: Optional[Any] = None
        self.reasoning_trigger: Optional[Any] = None
        self.intent_verifier: Optional[Any] = None
        self.browser_integration: Optional[Any] = None
        self.planning_verifier: Optional[Any] = None
        self.state_manager: Optional[Any] = None
        self.task_processor: Optional[Any] = None
        
        # Optional components (may fail to initialize)
        self.code_understanding: Optional[Any] = None
        self.code_conventions: Optional[Any] = None
        self.context_analyzer: Optional[Any] = None
        self.critical_verifier: Optional[Any] = None
        self.test_protector: Optional[Any] = None
        self.ci_integration: Optional[Any] = None
        self.git_manager: Optional[Any] = None
        self.multi_location_verifier: Optional[Any] = None
        self.notifications: Optional[Any] = None
        self.metrics: Optional[Any] = None
        self.rate_limiter: Optional[Any] = None
        self.plugin_manager: Optional[Any] = None
        self.scheduler: Optional[Any] = None
        self.backup_manager: Optional[Any] = None
        self.command_cache: Optional[Any] = None
        self.template_manager: Optional[Any] = None
        self.validator: Optional[Any] = None
        self.event_bus: Optional[Any] = None
        self.cluster_manager: Optional[Any] = None
        self.config_manager: Optional[Any] = None
        self.alert_manager: Optional[Any] = None
        self.ai_processor: Optional[Any] = None
        self.embedding_store: Optional[Any] = None
        self.pattern_learner: Optional[Any] = None
        self.file_watcher: Optional[Any] = None
    
    def get_all_components(self) -> Dict[str, Optional[Any]]:
        """
        Obtener todos los componentes como diccionario.
        
        Returns:
            Diccionario con todos los componentes.
        """
        return {
            # Core components
            "security_manager": self.security_manager,
            "devin": self.devin,
            "devin_commands": self.devin_commands,
            "tool_manager": self.tool_manager,
            "change_verifier": self.change_verifier,
            "test_runner": self.test_runner,
            "reference_tracker": self.reference_tracker,
            "parallel_executor": self.parallel_executor,
            "completion_verifier": self.completion_verifier,
            "iteration_manager": self.iteration_manager,
            "reasoning_trigger": self.reasoning_trigger,
            "intent_verifier": self.intent_verifier,
            "browser_integration": self.browser_integration,
            "planning_verifier": self.planning_verifier,
            "state_manager": self.state_manager,
            "task_processor": self.task_processor,
            # Optional components
            "code_understanding": self.code_understanding,
            "code_conventions": self.code_conventions,
            "context_analyzer": self.context_analyzer,
            "critical_verifier": self.critical_verifier,
            "test_protector": self.test_protector,
            "ci_integration": self.ci_integration,
            "git_manager": self.git_manager,
            "multi_location_verifier": self.multi_location_verifier,
            "notifications": self.notifications,
            "metrics": self.metrics,
            "rate_limiter": self.rate_limiter,
            "plugin_manager": self.plugin_manager,
            "scheduler": self.scheduler,
            "backup_manager": self.backup_manager,
            "command_cache": self.command_cache,
            "template_manager": self.template_manager,
            "validator": self.validator,
            "event_bus": self.event_bus,
            "cluster_manager": self.cluster_manager,
            "config_manager": self.config_manager,
            "alert_manager": self.alert_manager,
            "ai_processor": self.ai_processor,
            "embedding_store": self.embedding_store,
            "pattern_learner": self.pattern_learner,
            "file_watcher": self.file_watcher,
        }
    
    def get_available_components(self) -> list[str]:
        """
        Obtener lista de componentes disponibles.
        
        Returns:
            Lista de nombres de componentes que están inicializados.
        """
        return [
            name
            for name, component in self.get_all_components().items()
            if component is not None
        ]


class ComponentInitializer:
    """
    Inicializador de componentes (core y opcionales).
    
    Gestiona la inicialización de todos los componentes del agente
    con manejo robusto de errores. Los componentes core deben
    inicializarse exitosamente, mientras que los opcionales pueden
    fallar sin afectar el funcionamiento del agente.
    """
    
    def __init__(self, agent: "CursorAgent") -> None:
        """
        Inicializar el inicializador de componentes.
        
        Args:
            agent: Instancia del agente para pasar a componentes que lo requieren.
        """
        self.agent = agent
        self.registry = ComponentRegistry()
        self._workspace_root = self._get_workspace_root()
    
    def _get_workspace_root(self) -> str:
        """Obtener ruta del workspace root."""
        try:
            return str(Path(self.agent.config.storage_path).parent.parent)
        except Exception:
            return "."
    
    def initialize_all(self) -> ComponentRegistry:
        """
        Inicializar todos los componentes (core y opcionales).
        
        Returns:
            Registry con todos los componentes inicializados.
        
        Raises:
            RuntimeError: Si hay error crítico durante la inicialización de componentes core.
        """
        logger.debug("Initializing all components...")
        
        # Inicializar componentes core (deben tener éxito)
        self._initialize_core_components()
        
        # Inicializar componentes opcionales (pueden fallar)
        self._initialize_optional_components()
        
        available = self.registry.get_available_components()
        logger.info(f"Initialized {len(available)} components: {', '.join(available)}")
        
        return self.registry
    
    def _initialize_core_components(self) -> None:
        """Inicializar componentes core (críticos)."""
        logger.debug("Initializing core components...")
        
        # Security Manager (siempre requerido)
        from ..security import SecurityManager
        self.registry.security_manager = SecurityManager()
        
        # Devin Persona (si está habilitado)
        if self.agent.config.enable_devin_persona:
            from ..devin_persona import DevinPersona, AgentMode
            self.registry.devin = DevinPersona(agent=self.agent)
            self.registry.devin.set_language(self.agent.config.devin_language)
            mode = AgentMode.PLANNING if self.agent.config.devin_mode == "planning" else AgentMode.STANDARD
            self.registry.devin.set_mode(mode)
        
        # Devin Commands
        from ..devin_commands import DevinCommandExecutor
        self.registry.devin_commands = DevinCommandExecutor(agent=self.agent)
        
        # Tool Manager
        from ..tool_manager import ToolManager
        self.registry.tool_manager = ToolManager()
        
        # Change Verifier
        from ..change_verifier import ChangeVerifier
        self.registry.change_verifier = ChangeVerifier(self._workspace_root)
        
        # Test Runner
        from ..test_runner import TestRunner
        self.registry.test_runner = TestRunner(self._workspace_root)
        
        # Reference Tracker
        from ..reference_tracker import ReferenceTracker
        self.registry.reference_tracker = ReferenceTracker(self._workspace_root)
        
        # Parallel Executor
        from ..parallel_executor import ParallelExecutor
        self.registry.parallel_executor = ParallelExecutor(
            max_concurrent=self.agent.config.max_concurrent_tasks
        )
        
        # Completion Verifier
        from ..completion_verifier import CompletionVerifier
        self.registry.completion_verifier = CompletionVerifier()
        
        # Iteration Manager
        from ..iteration_manager import IterationManager
        self.registry.iteration_manager = IterationManager()
        
        # Reasoning Trigger
        from ..reasoning_trigger import ReasoningTriggerSystem
        self.registry.reasoning_trigger = ReasoningTriggerSystem()
        self.registry.reasoning_trigger.set_agent(self.agent)
        
        # Intent Verifier
        from ..intent_verifier import IntentVerifier
        self.registry.intent_verifier = IntentVerifier()
        
        # Browser Integration
        from ..browser_integration import BrowserIntegration
        self.registry.browser_integration = BrowserIntegration()
        
        # Planning Verifier
        from ..planning_verifier import PlanningVerifier
        self.registry.planning_verifier = PlanningVerifier()
        
        # State Manager
        from ..state import StateManager
        self.registry.state_manager = StateManager(self.agent, self.agent.config.storage_path)
        
        # Task Processor
        from ..task import TaskProcessor
        self.registry.task_processor = TaskProcessor(self.agent)
        
        logger.debug("Core components initialized")
    
    def _initialize_optional_components(self) -> None:
        """Inicializar componentes opcionales (pueden fallar)."""
        logger.debug("Initializing optional components...")
        
        # Code Understanding
        self._initialize_code_understanding()
        
        # Code Conventions
        self._initialize_code_conventions()
        
        # Context Analyzer
        self._initialize_context_analyzer()
        
        # Critical Verifier
        self._initialize_critical_verifier()
        
        # Test Protector
        self._initialize_test_protector()
        
        # CI Integration
        self._initialize_ci_integration()
        
        # Git Manager
        self._initialize_git_manager()
        
        # Multi-location Verifier
        self._initialize_multi_location_verifier()
        
        # Optional infrastructure components
        self._initialize_notifications_and_metrics()
        self._initialize_rate_limiter()
        self._initialize_plugin_manager()
        self._initialize_scheduler()
        self._initialize_backup_manager()
        self._initialize_cache()
        self._initialize_template_manager()
        self._initialize_validator()
        self._initialize_event_bus()
        self._initialize_cluster_manager()
        self._initialize_config_manager()
        self._initialize_alert_manager()
        self._initialize_ai_processor()
        self._initialize_embedding_store()
        self._initialize_pattern_learner()
    
    def _initialize_code_understanding(self) -> None:
        """Inicializar code understanding."""
        def init():
            from ..code_understanding import CodeUnderstanding
            return CodeUnderstanding(self._workspace_root)
        
        self.registry.code_understanding = safe_initialize(
            "code_understanding",
            init,
            logger_instance=logger
        )
        log_component_status("code_understanding", self.registry.code_understanding is not None, logger)
    
    def _initialize_code_conventions(self) -> None:
        """Inicializar code conventions analyzer."""
        def init():
            from ..code_conventions import CodeConventionsAnalyzer
            return CodeConventionsAnalyzer(self._workspace_root)
        
        self.registry.code_conventions = safe_initialize(
            "code_conventions",
            init,
            logger_instance=logger
        )
        log_component_status("code_conventions", self.registry.code_conventions is not None, logger)
    
    def _initialize_context_analyzer(self) -> None:
        """Inicializar context analyzer."""
        def init():
            from ..context_analyzer import ContextAnalyzer
            return ContextAnalyzer(self._workspace_root)
        
        self.registry.context_analyzer = safe_initialize(
            "context_analyzer",
            init,
            logger_instance=logger
        )
        log_component_status("context_analyzer", self.registry.context_analyzer is not None, logger)
    
    def _initialize_critical_verifier(self) -> None:
        """Inicializar critical verifier."""
        def init():
            from ..critical_verifier import CriticalVerifier
            return CriticalVerifier(self._workspace_root)
        
        self.registry.critical_verifier = safe_initialize(
            "critical_verifier",
            init,
            logger_instance=logger
        )
        log_component_status("critical_verifier", self.registry.critical_verifier is not None, logger)
    
    def _initialize_test_protector(self) -> None:
        """Inicializar test protector."""
        def init():
            from ..test_protector import TestProtector
            return TestProtector(self._workspace_root)
        
        self.registry.test_protector = safe_initialize(
            "test_protector",
            init,
            logger_instance=logger
        )
        log_component_status("test_protector", self.registry.test_protector is not None, logger)
    
    def _initialize_ci_integration(self) -> None:
        """Inicializar CI integration."""
        def init():
            from ..ci_integration import CIIntegration
            return CIIntegration(self._workspace_root)
        
        self.registry.ci_integration = safe_initialize(
            "ci_integration",
            init,
            logger_instance=logger
        )
        log_component_status("ci_integration", self.registry.ci_integration is not None, logger)
    
    def _initialize_git_manager(self) -> None:
        """Inicializar git manager."""
        def init():
            from ..git_manager import GitManager
            return GitManager(self._workspace_root)
        
        self.registry.git_manager = safe_initialize(
            "git_manager",
            init,
            logger_instance=logger
        )
        log_component_status("git_manager", self.registry.git_manager is not None, logger)
    
    def _initialize_multi_location_verifier(self) -> None:
        """Inicializar multi-location verifier."""
        def init():
            from ..multi_location_verifier import MultiLocationVerifier
            return MultiLocationVerifier(self._workspace_root)
        
        self.registry.multi_location_verifier = safe_initialize(
            "multi_location_verifier",
            init,
            logger_instance=logger
        )
        log_component_status("multi_location_verifier", self.registry.multi_location_verifier is not None, logger)
    
    def _initialize_notifications_and_metrics(self) -> None:
        """Inicializar sistema de notificaciones y métricas."""
        def init():
            from ..notifications import NotificationManager
            from ..metrics import MetricsCollector
            return NotificationManager(), MetricsCollector()
        
        result = safe_initialize(
            "notifications_and_metrics",
            init,
            logger_instance=logger
        )
        if result:
            self.registry.notifications, self.registry.metrics = result
        log_component_status("notifications", self.registry.notifications is not None, logger)
        log_component_status("metrics", self.registry.metrics is not None, logger)
    
    def _initialize_rate_limiter(self) -> None:
        """Inicializar rate limiter."""
        def init():
            from ..rate_limiter import TaskRateLimiter
            return TaskRateLimiter(
                max_tasks_per_minute=60,
                max_concurrent_tasks=self.agent.config.max_concurrent_tasks
            )
        
        self.registry.rate_limiter = safe_initialize(
            "rate_limiter",
            init,
            logger_instance=logger
        )
        log_component_status("rate_limiter", self.registry.rate_limiter is not None, logger)
    
    def _initialize_plugin_manager(self) -> None:
        """Inicializar plugin manager."""
        def init():
            from ..plugins import PluginManager
            return PluginManager(self.agent)
        
        self.registry.plugin_manager = safe_initialize(
            "plugin_manager",
            init,
            logger_instance=logger
        )
        log_component_status("plugin_manager", self.registry.plugin_manager is not None, logger)
    
    def _initialize_scheduler(self) -> None:
        """Inicializar scheduler."""
        def init():
            from ..scheduler import TaskScheduler
            return TaskScheduler(self.agent)
        
        self.registry.scheduler = safe_initialize(
            "scheduler",
            init,
            logger_instance=logger
        )
        log_component_status("scheduler", self.registry.scheduler is not None, logger)
    
    def _initialize_backup_manager(self) -> None:
        """Inicializar backup manager."""
        def init():
            from ..backup import BackupManager
            return BackupManager(self.agent)
        
        self.registry.backup_manager = safe_initialize(
            "backup_manager",
            init,
            logger_instance=logger
        )
        log_component_status("backup_manager", self.registry.backup_manager is not None, logger)
    
    def _initialize_cache(self) -> None:
        """Inicializar command cache."""
        def init():
            from ..cache import CommandCache
            return CommandCache()
        
        self.registry.command_cache = safe_initialize(
            "command_cache",
            init,
            logger_instance=logger
        )
        log_component_status("command_cache", self.registry.command_cache is not None, logger)
    
    def _initialize_template_manager(self) -> None:
        """Inicializar template manager."""
        def init():
            from ..templates import TemplateManager
            return TemplateManager()
        
        self.registry.template_manager = safe_initialize(
            "template_manager",
            init,
            logger_instance=logger
        )
        log_component_status("template_manager", self.registry.template_manager is not None, logger)
    
    def _initialize_validator(self) -> None:
        """Inicializar command validator."""
        try:
            from ..validators import CommandValidator
            self.registry.validator = CommandValidator()
            logger.debug("Command validator initialized")
        except ImportError as e:
            logger.debug(f"Optional component not available: validator ({e})")
        except Exception as e:
            logger.warning(f"Error initializing validator: {e}")
    
    def _initialize_event_bus(self) -> None:
        """Inicializar event bus."""
        def init():
            from ..event_bus import EventBus
            return EventBus()
        
        self.registry.event_bus = safe_initialize(
            "event_bus",
            init,
            logger_instance=logger
        )
        log_component_status("event_bus", self.registry.event_bus is not None, logger)
    
    def _initialize_cluster_manager(self) -> None:
        """Inicializar cluster manager."""
        def init():
            from ..cluster import ClusterManager
            return ClusterManager(self.agent)
        
        self.registry.cluster_manager = safe_initialize(
            "cluster_manager",
            init,
            logger_instance=logger
        )
        log_component_status("cluster_manager", self.registry.cluster_manager is not None, logger)
    
    def _initialize_config_manager(self) -> None:
        """Inicializar config manager."""
        def init():
            from ..config_manager import ConfigManager
            config_path = Path(self.agent.config.storage_path).parent / "config.json"
            return ConfigManager(str(config_path))
        
        self.registry.config_manager = safe_initialize(
            "config_manager",
            init,
            logger_instance=logger
        )
        log_component_status("config_manager", self.registry.config_manager is not None, logger)
    
    def _initialize_alert_manager(self) -> None:
        """Inicializar alert manager."""
        def init():
            from ..alerting import AlertManager
            return AlertManager(self.agent)
        
        self.registry.alert_manager = safe_initialize(
            "alert_manager",
            init,
            logger_instance=logger
        )
        log_component_status("alert_manager", self.registry.alert_manager is not None, logger)
    
    def _initialize_ai_processor(self) -> None:
        """Inicializar AI processor."""
        def init():
            from ..ai_processor import AIProcessor
            return AIProcessor(use_local=True)
        
        self.registry.ai_processor = safe_initialize(
            "ai_processor",
            init,
            logger_instance=logger
        )
        log_component_status("ai_processor", self.registry.ai_processor is not None, logger)
    
    def _initialize_embedding_store(self) -> None:
        """Inicializar embedding store."""
        def init():
            from ..embeddings import EmbeddingStore
            return EmbeddingStore()
        
        self.registry.embedding_store = safe_initialize(
            "embedding_store",
            init,
            logger_instance=logger
        )
        log_component_status("embedding_store", self.registry.embedding_store is not None, logger)
    
    def _initialize_pattern_learner(self) -> None:
        """Inicializar pattern learner."""
        def init():
            from ..pattern_learner import PatternLearner
            return PatternLearner()
        
        self.registry.pattern_learner = safe_initialize(
            "pattern_learner",
            init,
            logger_instance=logger
        )
        log_component_status("pattern_learner", self.registry.pattern_learner is not None, logger)

