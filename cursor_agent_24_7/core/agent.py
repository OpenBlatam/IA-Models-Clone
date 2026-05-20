"""
Cursor Agent - Agente principal 24/7
=====================================

Agente persistente que escucha y ejecuta comandos desde Cursor.
Gestiona tareas, procesamiento con IA, métricas, y persistencia de estado.
"""

import asyncio
import logging
from enum import Enum
from typing import (
    Optional,
    Dict,
    Any,
    List,
    Callable,
    Union,
    TYPE_CHECKING
)
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from .validation_utils import validate_not_empty, validate_positive
from .error_handling import safe_async_call
from .task_utils import format_task_command

if TYPE_CHECKING:
    from .notifications import NotificationManager
    from .metrics import MetricsCollector
    from .rate_limiter import TaskRateLimiter
    from .plugins import PluginManager
    from .scheduler import TaskScheduler
    from .backup import BackupManager
    from .cache import CommandCache
    from .templates import TemplateManager
    from .validators import CommandValidator
    from .event_bus import EventBus
    from .cluster import ClusterManager
    from .config_manager import ConfigManager
    from .alerting import AlertManager
    from .ai_processor import AIProcessor
    from .embeddings import EmbeddingStore
    from .pattern_learner import PatternLearner
    from .file_watcher import FileWatcher
    from .devin_persona import DevinPersona, AgentMode

# Usar orjson para JSON más rápido
try:
    import orjson as json
    HAS_ORJSON = True
except ImportError:
    import json  # Fallback a json estándar
    HAS_ORJSON = False

# Usar structlog para logging estructurado
try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Estados del agente"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AgentConfig:
    """
    Configuración del agente.
    
    Attributes:
        check_interval: Segundos entre checks de comandos (default: 1.0).
        max_concurrent_tasks: Máximo de tareas simultáneas (default: 5).
        task_timeout: Timeout por tarea en segundos (default: 300.0).
        auto_restart: Reiniciar automáticamente en caso de error (default: True).
        persistent_storage: Guardar estado en disco (default: True).
        storage_path: Ruta del archivo de estado (default: "./data/agent_state.json").
        command_file: Archivo para recibir comandos (opcional).
        watch_directory: Directorio a monitorear (opcional).
        enable_devin_persona: Habilitar personalidad Devin (default: True).
        devin_mode: Modo inicial de Devin (default: "standard").
        devin_language: Idioma de comunicación (default: "es").
    """
    check_interval: float = 1.0
    max_concurrent_tasks: int = 5
    task_timeout: float = 300.0
    auto_restart: bool = True
    persistent_storage: bool = True
    storage_path: str = "./data/agent_state.json"
    command_file: Optional[str] = None
    watch_directory: Optional[str] = None
    enable_devin_persona: bool = True
    devin_mode: str = "standard"
    devin_language: str = "es"
    
    def __post_init__(self) -> None:
        """Validar configuración después de inicialización"""
        # Importar aquí para evitar circular imports
        from .validation_utils import validate_positive
        
        validate_positive(self.check_interval, "check_interval")
        validate_positive(self.max_concurrent_tasks, "max_concurrent_tasks")
        validate_positive(self.task_timeout, "task_timeout")


@dataclass
class Task:
    """
    Tarea a ejecutar.
    
    Attributes:
        id: Identificador único de la tarea.
        command: Comando a ejecutar.
        timestamp: Fecha y hora de creación.
        status: Estado actual de la tarea.
        result: Resultado de la ejecución (si fue exitosa).
        error: Mensaje de error (si falló).
    """
    id: str
    command: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    result: Optional[str] = None
    error: Optional[str] = None


class CursorAgent:
    """
    Agente principal que escucha y ejecuta comandos.
    
    Gestiona el ciclo de vida completo de tareas: recepción, validación,
    ejecución, y persistencia. Soporta múltiples componentes opcionales
    para procesamiento con IA, métricas, notificaciones, etc.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        """
        Inicializar agente.
        
        Args:
            config: Configuración del agente. Si es None, usa valores por defecto.
        
        Raises:
            ValueError: Si la configuración es inválida.
        """
        self.config: AgentConfig = config or AgentConfig()
        self.status: AgentStatus = AgentStatus.IDLE
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.Queue[Task] = asyncio.Queue()
        self.running: bool = False
        self._listener_task: Optional[asyncio.Task[None]] = None
        self._executor_task: Optional[asyncio.Task[None]] = None
        self._callbacks: List[Callable[[str], Union[None, Any]]] = []
        
        # Estado persistente
        self._state_file: str = self.config.storage_path
        
        # Componentes opcionales (type hints para TYPE_CHECKING)
        self.notifications: Optional["NotificationManager"] = None
        self.metrics: Optional["MetricsCollector"] = None
        self.rate_limiter: Optional["TaskRateLimiter"] = None
        self.plugin_manager: Optional["PluginManager"] = None
        self.scheduler: Optional["TaskScheduler"] = None
        self.backup_manager: Optional["BackupManager"] = None
        self.command_cache: Optional["CommandCache"] = None
        self.template_manager: Optional["TemplateManager"] = None
        self.validator: Optional["CommandValidator"] = None
        self.event_bus: Optional["EventBus"] = None
        self.cluster_manager: Optional["ClusterManager"] = None
        self.config_manager: Optional["ConfigManager"] = None
        self.alert_manager: Optional["AlertManager"] = None
        self.ai_processor: Optional["AIProcessor"] = None
        self.embedding_store: Optional["EmbeddingStore"] = None
        self.pattern_learner: Optional["PatternLearner"] = None
        self.file_watcher: Optional["FileWatcher"] = None
        
        # Core components (type hints)
        self.security_manager: Any = None
        self.devin: Optional["DevinPersona"] = None
        self.devin_commands: Any = None
        self.tool_manager: Any = None
        self.code_understanding: Optional[Any] = None
        self.code_conventions: Optional[Any] = None
        self.change_verifier: Any = None
        self.test_runner: Any = None
        self.reference_tracker: Any = None
        self.parallel_executor: Any = None
        self.context_analyzer: Optional[Any] = None
        self.completion_verifier: Any = None
        self.iteration_manager: Any = None
        self.critical_verifier: Optional[Any] = None
        self.reasoning_trigger: Any = None
        self.intent_verifier: Any = None
        self.test_protector: Optional[Any] = None
        self.ci_integration: Optional[Any] = None
        self.git_manager: Optional[Any] = None
        self.multi_location_verifier: Optional[Any] = None
        self.browser_integration: Any = None
        self.planning_verifier: Any = None
        
        # Inicializar todos los componentes usando ComponentInitializer
        from .components import ComponentInitializer
        component_initializer = ComponentInitializer(self)
        registry = component_initializer.initialize_all()
        
        # Asignar todos los componentes desde registry
        # Core components
        self.security_manager = registry.security_manager
        self.devin = registry.devin
        self.devin_commands = registry.devin_commands
        self.tool_manager = registry.tool_manager
        self.change_verifier = registry.change_verifier
        self.test_runner = registry.test_runner
        self.reference_tracker = registry.reference_tracker
        self.parallel_executor = registry.parallel_executor
        self.completion_verifier = registry.completion_verifier
        self.iteration_manager = registry.iteration_manager
        self.reasoning_trigger = registry.reasoning_trigger
        self.intent_verifier = registry.intent_verifier
        self.browser_integration = registry.browser_integration
        self.planning_verifier = registry.planning_verifier
        self._state_manager = registry.state_manager
        self._task_processor = registry.task_processor
        
        # Optional components
        self.code_understanding = registry.code_understanding
        self.code_conventions = registry.code_conventions
        self.context_analyzer = registry.context_analyzer
        self.critical_verifier = registry.critical_verifier
        self.test_protector = registry.test_protector
        self.ci_integration = registry.ci_integration
        self.git_manager = registry.git_manager
        self.multi_location_verifier = registry.multi_location_verifier
        self.notifications = registry.notifications
        self.metrics = registry.metrics
        self.rate_limiter = registry.rate_limiter
        self.plugin_manager = registry.plugin_manager
        self.scheduler = registry.scheduler
        self.backup_manager = registry.backup_manager
        self.command_cache = registry.command_cache
        self.template_manager = registry.template_manager
        self.validator = registry.validator
        self.event_bus = registry.event_bus
        self.cluster_manager = registry.cluster_manager
        self.config_manager = registry.config_manager
        self.alert_manager = registry.alert_manager
        self.ai_processor = registry.ai_processor
        self.embedding_store = registry.embedding_store
        self.pattern_learner = registry.pattern_learner
    
    async def start(self) -> None:
        """
        Iniciar el agente.
        
        Inicializa todos los componentes, carga el estado persistente,
        y comienza a escuchar y ejecutar comandos.
        
        Raises:
            RuntimeError: Si hay error al iniciar componentes críticos.
        """
        if self.status == AgentStatus.RUNNING:
            logger.warning("Agent is already running")
            return
        
        logger.info("🚀 Starting Cursor Agent 24/7...")
        self.running = True
        self.status = AgentStatus.RUNNING
        
        try:
            # Inicializar componentes de IA usando safe_async_call
            if self.ai_processor:
                await safe_async_call(
                    self.ai_processor.initialize,
                    operation="initializing AI processor",
                    logger_instance=logger,
                    reraise=False
                )
            if self.embedding_store:
                await safe_async_call(
                    self.embedding_store.initialize,
                    operation="initializing embedding store",
                    logger_instance=logger,
                    reraise=False
                )
            if self.pattern_learner:
                await safe_async_call(
                    self.pattern_learner.load,
                    operation="loading pattern learner",
                    logger_instance=logger,
                    reraise=False
                )
            
            # Cargar estado persistente si existe
            if self.config.persistent_storage:
                await safe_async_call(
                    self._state_manager.load,
                    operation="loading persistent state",
                    logger_instance=logger,
                    reraise=False
                )
            
            # Iniciar listener y executor
            self._listener_task = asyncio.create_task(self._listen_loop())
            self._executor_task = asyncio.create_task(self._executor_loop())
            
            # Iniciar componentes opcionales usando utilidades
            from .component_lifecycle import safe_start_component
            
            await safe_start_component("plugin_manager", self.plugin_manager, lambda c: c.start_all(), logger)
            await safe_start_component("scheduler", self.scheduler, None, logger)
            await safe_start_component("backup_manager", self.backup_manager, lambda c: c.start_auto_backup(), logger)
            await safe_start_component("cluster_manager", self.cluster_manager, None, logger)
            await safe_start_component("alert_manager", self.alert_manager, None, logger)
            
            logger.info("✅ Agent started successfully")
            await self._notify_callbacks("started")
            
            # Publicar evento
            if self.event_bus:
                async def publish_start_event():
                    await self.event_bus.publish(
                        self.event_bus.EventType.AGENT_STARTED,
                        {"agent_id": id(self)},
                        source="agent"
                    )
                
                await safe_async_call(
                    publish_start_event,
                    operation="publishing agent started event",
                    logger_instance=logger,
                    reraise=False
                )
            
            # Notificación y métrica
            if self.notifications:
                async def notify_start():
                    await self.notifications.notify(
                        "Agent Started",
                        "Cursor Agent 24/7 is now running",
                        level=self.notifications.NotificationLevel.SUCCESS
                    )
                
                await safe_async_call(
                    notify_start,
                    operation="sending start notification",
                    logger_instance=logger,
                    reraise=False
                )
            
            if self.metrics:
                def update_metrics():
                    self.metrics.increment("agent_starts")
                    self.metrics.set_gauge("agent_running", 1.0)
                
                await safe_async_call(
                    update_metrics,
                    operation="updating start metrics",
                    logger_instance=logger,
                    reraise=False
                )
                
        except Exception as e:
            logger.error(f"Error starting agent: {e}", exc_info=True)
            self.status = AgentStatus.ERROR
            self.running = False
            
            # Reportar problema de entorno si Devin está habilitado
            if self.devin:
                async def report_issue():
                    await self.devin.report_environment_issue(
                        issue_type="agent_startup_failure",
                        description=f"Failed to start agent: {str(e)}",
                        suggestion="Check logs, verify dependencies, and ensure all required services are running",
                        severity="high"
                    )
                
                await safe_async_call(
                    report_issue,
                    operation="reporting environment issue to Devin",
                    logger_instance=logger,
                    reraise=False
                )
            
            raise RuntimeError(f"Failed to start agent: {e}") from e
    
    async def stop(self) -> None:
        """
        Detener el agente.
        
        Cancela todas las tareas en ejecución, detiene los componentes,
        y guarda el estado si está habilitado.
        """
        if self.status == AgentStatus.STOPPED:
            return
        
        logger.info("🛑 Stopping Cursor Agent...")
        self.running = False
        self.status = AgentStatus.STOPPED
        
        # Cancelar tareas
        if self._listener_task:
            self._listener_task.cancel()
        if self._executor_task:
            self._executor_task.cancel()
        
        # Esperar a que terminen
        await asyncio.gather(
            self._listener_task,
            self._executor_task,
            return_exceptions=True
        )
        
        # Detener componentes usando utilidades
        from .component_lifecycle import safe_stop_components
        
        await safe_stop_components([
            ("alert_manager", self.alert_manager, None),
            ("cluster_manager", self.cluster_manager, None),
            ("scheduler", self.scheduler, None),
            ("backup_manager", self.backup_manager, lambda c: c.stop_auto_backup()),
            ("plugin_manager", self.plugin_manager, lambda c: c.stop_all()),
            ("file_watcher", self.file_watcher, None),
        ], logger_instance=logger)
        
        # Publicar evento
        if self.event_bus:
            async def publish_stop_event():
                await self.event_bus.publish(
                    self.event_bus.EventType.AGENT_STOPPED,
                    {"agent_id": id(self)},
                    source="agent"
                )
            
            await safe_async_call(
                publish_stop_event,
                operation="publishing agent stopped event",
                logger_instance=logger,
                reraise=False
            )
        
        # Guardar estado
        if self.config.persistent_storage:
            await safe_async_call(
                self._state_manager.save,
                operation="saving persistent state",
                logger_instance=logger,
                reraise=False
            )
        
        logger.info("✅ Agent stopped")
        await self._notify_callbacks("stopped")
        
        # Notificación y métrica
        if self.notifications:
            async def notify_stop():
                await self.notifications.notify(
                    "Agent Stopped",
                    "Cursor Agent 24/7 has been stopped",
                    level=self.notifications.NotificationLevel.INFO
                )
            
            await safe_async_call(
                notify_stop,
                operation="sending stop notification",
                logger_instance=logger,
                reraise=False
            )
        
        if self.metrics:
            def update_metrics():
                self.metrics.increment("agent_stops")
                self.metrics.set_gauge("agent_running", 0.0)
            
            await safe_async_call(
                update_metrics,
                operation="updating stop metrics",
                logger_instance=logger,
                reraise=False
            )
    
    async def pause(self) -> None:
        """
        Pausar el agente temporalmente.
        
        El agente deja de procesar nuevas tareas pero mantiene
        las tareas en ejecución.
        """
        if self.status == AgentStatus.RUNNING:
            self.status = AgentStatus.PAUSED
            logger.info("⏸️ Agent paused")
            await self._notify_callbacks("paused")
    
    async def resume(self) -> None:
        """
        Reanudar el agente después de una pausa.
        
        El agente vuelve a procesar nuevas tareas.
        """
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.RUNNING
            logger.info("▶️ Agent resumed")
            await self._notify_callbacks("resumed")
    
    async def add_task(self, command: str) -> str:
        """
        Agregar una tarea al queue.
        
        Args:
            command: Comando a ejecutar.
        
        Returns:
            ID de la tarea creada.
        
        Raises:
            ValueError: Si la validación del comando falla.
            RuntimeError: Si se excede el rate limit.
        """
        validate_not_empty(command, "command")
        
        # Validar seguridad del comando
        is_valid, error_msg = self.security_manager.validate_command(command)
        if not is_valid:
            if self.devin:
                await self.devin.message_user(
                    f"Comando bloqueado por seguridad: {error_msg}",
                    level=self.devin.CommunicationLevel.WARNING
                )
            raise ValueError(f"Security validation failed: {error_msg}")
        
        # Escanear comando en busca de secretos
        secrets = self.security_manager.scan_command_for_secrets(command)
        if secrets:
            logger.warning(f"⚠️ Potential secrets detected in command")
            if self.devin:
                await self.devin.message_user(
                    "Se detectaron posibles secretos en el comando. Por favor, usa variables de entorno en su lugar.",
                    level=self.devin.CommunicationLevel.WARNING
                )
        
        # Validar comando
        if self.validator:
            validation = self.validator.validate(command)
            if not validation.valid:
                errors = ', '.join(validation.errors) if validation.errors else "Unknown validation error"
                raise ValueError(f"Command validation failed: {errors}")
            if validation.warnings:
                logger.warning(f"Command warnings: {', '.join(validation.warnings)}")
            # Sanitizar comando
            command = self.validator.sanitize(command)
        
        # Verificar rate limit
        if self.rate_limiter:
            if not await self.rate_limiter.can_add_task():
                raise RuntimeError("Rate limit exceeded. Too many tasks.")
            await self.rate_limiter.add_task()
        
        # Crear tarea usando utilidades
        from .task_utils import create_task_id, count_tasks_by_status
        
        task_id = create_task_id(len(self.tasks))
        task = Task(
            id=task_id,
            command=command,
            status="pending"
        )
        
        self.tasks[task_id] = task
        await self.task_queue.put(task)
        
        logger.info(f"📝 Task added: {task_id} - {format_task_command(command)}")
        
        # Métricas
        if self.metrics:
            self.metrics.increment("tasks_added")
            self.metrics.set_gauge(
                "tasks_pending",
                count_tasks_by_status(self.tasks, "pending")
            )
        
        # Notificar plugins
        if self.plugin_manager:
            async def notify_plugin():
                await self.plugin_manager.notify_task_added(task_id, command)
            
            await safe_async_call(
                notify_plugin,
                operation="notifying plugin manager",
                logger_instance=logger,
                reraise=False
            )
        
        # Publicar evento
        if self.event_bus:
            async def publish_event():
                await self.event_bus.publish(
                    self.event_bus.EventType.TASK_ADDED,
                    {"task_id": task_id, "command": command[:100]},
                    source="agent"
                )
            
            await safe_async_call(
                publish_event,
                operation="publishing task added event",
                logger_instance=logger,
                reraise=False
            )
        
        return task_id
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado del agente.
        
        Returns:
            Diccionario con información del estado del agente.
        """
        # Usar utilidades para contar tareas
        from .task_utils import count_tasks_by_status
        
        status: Dict[str, Any] = {
            "status": self.status.value,
            "running": self.running,
            "tasks_total": len(self.tasks),
            "tasks_pending": count_tasks_by_status(self.tasks, "pending"),
            "tasks_running": count_tasks_by_status(self.tasks, "running"),
            "tasks_completed": count_tasks_by_status(self.tasks, "completed"),
            "tasks_failed": count_tasks_by_status(self.tasks, "failed"),
        }
        
        # Agregar estado de componentes usando utilidades
        from .status_utils import (
            safe_get_status,
            get_component_status_dict,
            get_list_length_status,
            aggregate_status
        )
        
        # Devin
        status["devin"] = safe_get_status(
            "devin",
            self.devin,
            lambda c: c.get_status(),
            logger_instance=logger
        )
        
        # Tool Manager
        status["tools"] = safe_get_status(
            "tools",
            self.tool_manager,
            lambda c: c.get_status(),
            logger_instance=logger
        )
        
        # Devin Commands
        status["devin_commands"] = get_component_status_dict(
            "devin_commands",
            self.devin_commands,
            {
                "command_history_count": lambda c: len(c.command_history),
                "plans_count": lambda c: len(c.plans)
            },
            logger_instance=logger
        )
        
        # Test Runner
        status["tests"] = get_component_status_dict(
            "tests",
            self.test_runner,
            {
                "test_runs": lambda c: len(c.test_results),
                "lint_runs": lambda c: len(c.lint_results)
            },
            logger_instance=logger
        )
        
        # Reference Tracker
        aggregate_status(
            status,
            "references",
            get_list_length_status(
                self.reference_tracker,
                lambda c: c.get_tracked_changes(),
                "tracked_changes"
            )
        )
        
        # Parallel Executor
        status["parallel_executor"] = get_component_status_dict(
            "parallel_executor",
            self.parallel_executor,
            {
                "total_tasks": lambda c: len(c.get_all_tasks()),
                "max_concurrent": lambda c: c.max_concurrent
            },
            logger_instance=logger
        )
        
        # Context Analyzer
        aggregate_status(
            status,
            "context_analyzer",
            get_component_status_dict(
                "context_analyzer",
                self.context_analyzer,
                {"cached_contexts": lambda c: len(c.context_cache)},
                logger_instance=logger
            )
        )
        
        # Completion Verifier
        aggregate_status(
            status,
            "completion_verifier",
            get_list_length_status(
                self.completion_verifier,
                lambda c: c.get_all_tasks(),
                "total_tasks"
            )
        )
        
        # Iteration Manager
        aggregate_status(
            status,
            "iteration_manager",
            get_list_length_status(
                self.iteration_manager,
                lambda c: c.get_all_tasks(),
                "total_tasks"
            )
        )
        
        # Critical Verifier
        aggregate_status(
            status,
            "critical_verifier",
            get_list_length_status(
                self.critical_verifier,
                lambda c: c.get_all_verifications(),
                "total_verifications"
            )
        )
        
        # Reasoning Trigger
        aggregate_status(
            status,
            "reasoning_trigger",
            get_list_length_status(
                self.reasoning_trigger,
                lambda c: c.get_recent_triggers(5),
                "recent_triggers"
            )
        )
        
        # Intent Verifier
        aggregate_status(
            status,
            "intent_verifier",
            get_list_length_status(
                self.intent_verifier,
                lambda c: c.get_all_verifications(),
                "total_verifications"
            )
        )
        
        # Test Protector
        status["test_protector"] = get_component_status_dict(
            "test_protector",
            self.test_protector,
            {
                "test_files_detected": lambda c: len(c.get_test_files()),
                "modification_attempts": lambda c: len(c.get_modification_attempts())
            },
            logger_instance=logger
        )
        
        # CI Integration
        status["ci_integration"] = get_component_status_dict(
            "ci_integration",
            self.ci_integration,
            {
                "ci_systems": lambda c: c.ci_systems,
                "test_requests": lambda c: len(c.get_all_test_requests())
            },
            logger_instance=logger
        )
        
        # Git Manager
        status["git_manager"] = get_component_status_dict(
            "git_manager",
            self.git_manager,
            {
                "branches": lambda c: len(c.get_branches()),
                "operations": lambda c: len(c.get_operations()),
                "current_branch": lambda c: c.get_current_branch()
            },
            logger_instance=logger
        )
        
        # Multi-location Verifier
        aggregate_status(
            status,
            "multi_location_verifier",
            get_list_length_status(
                self.multi_location_verifier,
                lambda c: c.get_all_tasks(),
                "total_tasks"
            )
        )
        
        # Browser Integration
        status["browser_integration"] = get_component_status_dict(
            "browser_integration",
            self.browser_integration,
            {
                "sessions": lambda c: len(c.get_all_sessions()),
                "links_to_visit": lambda c: len(c.get_links_to_visit()),
                "browser_available": lambda c: c.browser_available
            },
            logger_instance=logger
        )
        
        # Planning Verifier
        aggregate_status(
            status,
            "planning_verifier",
            get_list_length_status(
                self.planning_verifier,
                lambda c: c.get_all_verifications(),
                "total_verifications"
            )
        )
        
        # Metrics
        status["metrics"] = safe_get_status(
            "metrics",
            self.metrics,
            lambda c: c.get_summary(),
            logger_instance=logger
        )
        
        # Notifications
        status["notifications"] = safe_get_status(
            "notifications",
            self.notifications,
            lambda c: c.get_stats(),
            logger_instance=logger
        )
        
        return status
    
    async def get_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtener lista de tareas.
        
        Args:
            limit: Número máximo de tareas a retornar (default: 50).
        
        Returns:
            Lista de diccionarios con información de las tareas.
        """
        validate_positive(limit, "limit")
        
        # Usar utilidades para convertir tareas
        from .task_utils import tasks_to_dict_list
        
        return tasks_to_dict_list(list(self.tasks.values()), limit=limit)
    
    def on_event(self, callback: Callable[[str], Union[None, Any]]) -> None:
        """
        Registrar callback para eventos.
        
        Args:
            callback: Función o coroutine a llamar cuando ocurre un evento.
        """
        from .validation_utils import validate_not_none
        validate_not_none(callback, "callback")
        if not callable(callback):
            raise ValueError("Callback must be callable")
        self._callbacks.append(callback)
    
    async def _notify_callbacks(self, event: str) -> None:
        """
        Notificar callbacks de eventos.
        
        Args:
            event: Nombre del evento.
        """
        for callback in self._callbacks:
            async def execute_callback():
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            
            await safe_async_call(
                execute_callback,
                operation=f"executing callback for event: {event}",
                logger_instance=logger,
                reraise=False
            )
    
    async def _listen_loop(self) -> None:
        """
        Loop principal que escucha comandos.
        
        Monitorea archivos o directorios para recibir comandos nuevos.
        """
        logger.info("👂 Starting command listener...")
        
        try:
            # Inicializar file watcher
            from .file_watcher import FileWatcher
            
            # Determinar archivo de comandos
            if self.config.command_file:
                command_file = Path(self.config.command_file)
            else:
                command_file = Path(self.config.storage_path).parent / "commands.txt"
            
            self.file_watcher = FileWatcher(
                command_file=str(command_file) if command_file else None,
                watch_dir=self.config.watch_directory
            )
            self.file_watcher.set_callback(self._on_command_received)
            await self.file_watcher.start()
            
            if command_file:
                logger.info(f"📝 Listening for commands in: {command_file}")
                logger.info(f"💡 Write commands to: {command_file}")
            elif self.config.watch_directory:
                logger.info(f"📁 Monitoring directory: {self.config.watch_directory}")
            
            while self.running:
                try:
                    await asyncio.sleep(self.config.check_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in listener loop: {e}", exc_info=True)
                    if not self.config.auto_restart:
                        break
                    await asyncio.sleep(5)
        
        except Exception as e:
            logger.error(f"Error initializing listener: {e}", exc_info=True)
            raise
        
        finally:
            # Detener file watcher
            if self.file_watcher:
                try:
                    await self.file_watcher.stop()
                except Exception as e:
                    logger.error(f"Error stopping file watcher: {e}")
    
    async def _on_command_received(self, command: str) -> None:
        """
        Callback cuando se recibe un comando.
        
        Args:
            command: Comando recibido.
        """
        if command and command.strip():
            logger.info(f"📨 Command received: {format_task_command(command)}")
            
            async def add_task_safely():
                await self.add_task(command.strip())
            
            async def notify_devin_error(error: str):
                if self.devin and self.devin.should_communicate("environment_issue"):
                    await self.devin.message_user(
                        f"Error al agregar tarea: {error}",
                        level=self.devin.CommunicationLevel.ERROR
                    )
            
            result = await safe_async_call(
                add_task_safely,
                operation="adding task from command",
                logger_instance=logger,
                reraise=False
            )
            
            if result is None:  # Error occurred
                async def notify_error():
                    await notify_devin_error("Unknown error")
                
                await safe_async_call(
                    notify_error,
                    operation="notifying devin of error",
                    logger_instance=logger,
                    reraise=False
                )
    
    async def _executor_loop(self) -> None:
        """
        Loop que ejecuta tareas.
        
        Procesa tareas de la cola respetando el límite de tareas concurrentes.
        """
        logger.info("⚙️ Starting task executor...")
        
        active_tasks: Dict[str, asyncio.Task[None]] = {}
        
        while self.running:
            try:
                # Procesar tareas pendientes
                if len(active_tasks) < self.config.max_concurrent_tasks:
                    try:
                        task = await asyncio.wait_for(
                            self.task_queue.get(),
                            timeout=0.5
                        )
                        
                        if task.status == "pending":
                            task.status = "running"
                            task_executor = asyncio.create_task(
                                self._execute_task(task)
                            )
                            active_tasks[task.id] = task_executor
                    except asyncio.TimeoutError:
                        pass
                
                # Limpiar tareas completadas
                completed = [
                    task_id
                    for task_id, exec_task in active_tasks.items()
                    if exec_task.done()
                ]
                
                for task_id in completed:
                    del active_tasks[task_id]
                
                await asyncio.sleep(0.1)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in executor loop: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        # Esperar a que terminen todas las tareas activas
        from .async_utils import gather_tasks_safely
        if active_tasks:
            await gather_tasks_safely(
                active_tasks,
                return_exceptions=True,
                logger_instance=logger
            )
    
    async def _execute_task(self, task: Task) -> None:
        """
        Ejecutar una tarea usando TaskProcessor.
        
        Args:
            task: Tarea a ejecutar.
        """
        await self._task_processor.process(task)
    
