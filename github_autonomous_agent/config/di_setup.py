"""
Dependency Injection Setup

Configures all services in the DI container.
"""

from config.logging_config import get_logger
from core.di import get_container
from core.storage import TaskStorage
from core.github_client import GitHubClient
from core.task_processor import TaskProcessor
from core.worker import WorkerManager
from core.services import (
    CacheService, MetricsService, RateLimitService, LLMService,
    AuditService, NotificationService, MonitoringService,
    PerformanceProfiler, CacheWarmingService, AuthService, FeatureFlagService,
    QueueService, BatchProcessor, AnalyticsService, SearchService, ValidationService
)
from config.settings import settings

# Import use cases
from application.use_cases.task_use_cases import (
    CreateTaskUseCase,
    GetTaskUseCase,
    ListTasksUseCase
)
from application.use_cases.github_use_cases import (
    GetRepositoryInfoUseCase,
    CloneRepositoryUseCase
)

logger = get_logger(__name__)


def setup_dependencies() -> None:
    """
    Configure all dependencies in the DI container with improved error handling.
    
    This should be called early in the application lifecycle.
    
    Raises:
        ValueError: Si hay un error crítico al configurar dependencias
        RuntimeError: Si el contenedor no está disponible
    """
    try:
        container = get_container()
    except Exception as e:
        logger.error(f"Error al obtener contenedor DI: {e}", exc_info=True)
        raise RuntimeError(f"No se pudo inicializar el contenedor de dependency injection: {e}") from e
    
    logger.info("Setting up dependency injection...")
    
    registered_services = []
    failed_services = []
    
    # ============================================
    # Core Services Layer
    # ============================================
    
    # Register cache service (singleton)
    def create_cache_service():
        try:
            service = CacheService(max_size=1000, default_ttl=300)
            logger.debug("Cache service creado exitosamente")
            return service
        except Exception as e:
            logger.error(f"Error al crear cache service: {e}", exc_info=True)
            raise
    
    try:
        container.register(
            "cache_service",
            create_cache_service,
            singleton=True,
            factory=create_cache_service
        )
        registered_services.append("cache_service")
        logger.debug("Registered: cache_service")
    except Exception as e:
        logger.error(f"Error al registrar cache_service: {e}", exc_info=True)
        failed_services.append(("cache_service", str(e)))
    
    # Register metrics service (singleton)
    def create_metrics_service():
        return MetricsService(use_prometheus=True)
    
    container.register(
        "metrics_service",
        create_metrics_service,
        singleton=True,
        factory=create_metrics_service
    )
    logger.debug("Registered: metrics_service")
    
    # Register rate limit service (singleton)
    def create_rate_limit_service():
        return RateLimitService(
            limit=RateLimitService.GITHUB_RATE_LIMIT_AUTHENTICATED,
            window_seconds=3600
        )
    
    container.register(
        "rate_limit_service",
        create_rate_limit_service,
        singleton=True,
        factory=create_rate_limit_service
    )
    logger.debug("Registered: rate_limit_service")
    
    # Register LLM service (singleton, optional)
    def create_llm_service():
        if not settings.LLM_ENABLED:
            logger.warning("LLM service está deshabilitado en configuración")
            return None
        if not settings.OPENROUTER_API_KEY:
            logger.warning("OpenRouter API key no configurada, LLM service no disponible")
            return None
        
        # Obtener cache service si está disponible
        cache_service = None
        try:
            cache_service = container.get("cache_service")
        except (ValueError, Exception):
            logger.debug("Cache service no disponible para LLM service")
        
        # Obtener metrics service si está disponible
        metrics_service = None
        try:
            metrics_service = container.get("metrics_service")
        except (ValueError, Exception):
            logger.debug("Metrics service no disponible para LLM service")
        
        return LLMService(
            api_key=settings.OPENROUTER_API_KEY,
            default_models=settings.LLM_DEFAULT_MODELS,
            timeout=settings.LLM_TIMEOUT,
            max_parallel_requests=settings.LLM_MAX_PARALLEL_REQUESTS,
            cache_service=cache_service,
            metrics_service=metrics_service,
            enable_cache=True,
            cache_ttl=3600  # 1 hora por defecto
        )
    
    container.register(
        "llm_service",
        create_llm_service,
        singleton=True,
        factory=create_llm_service
    )
    logger.debug("Registered: llm_service")
    
    # Register audit service (singleton)
    def create_audit_service():
        return AuditService()
    
    container.register(
        "audit_service",
        create_audit_service,
        singleton=True,
        factory=create_audit_service
    )
    logger.debug("Registered: audit_service")
    
    # Register notification service (singleton)
    def create_notification_service():
        service = NotificationService()
        # Registrar handler de WebSocket
        try:
            service.register_handler(
                NotificationChannel.WEBSOCKET,
                service.send_to_websocket
            )
        except Exception:
            pass  # WebSocket puede no estar disponible
        return service
    
    container.register(
        "notification_service",
        create_notification_service,
        singleton=True,
        factory=create_notification_service
    )
    logger.debug("Registered: notification_service")
    
    # Register monitoring service (singleton)
    def create_monitoring_service():
        service = MonitoringService()
        # Integrar con notification service
        try:
            notification_service = container.get("notification_service")
            async def alert_handler(alert):
                await notification_service.send(
                    title=f"Alert: {alert['rule_name']}",
                    message=alert['message'],
                    level=NotificationLevel(alert['severity'].upper()),
                    channels=[NotificationChannel.LOG, NotificationChannel.WEBSOCKET],
                    metadata=alert
                )
            service.register_alert_handler(alert_handler)
        except Exception:
            pass
        return service
    
    container.register(
        "monitoring_service",
        create_monitoring_service,
        singleton=True,
        factory=create_monitoring_service
    )
    logger.debug("Registered: monitoring_service")
    
    # Register performance profiler (singleton)
    def create_performance_profiler():
        return PerformanceProfiler()
    
    container.register(
        "performance_profiler",
        create_performance_profiler,
        singleton=True,
        factory=create_performance_profiler
    )
    logger.debug("Registered: performance_profiler")
    
    # Register cache warming service (singleton)
    def create_cache_warming_service():
        return CacheWarmingService()
    
    container.register(
        "cache_warming_service",
        create_cache_warming_service,
        singleton=True,
        factory=create_cache_warming_service
    )
    logger.debug("Registered: cache_warming_service")
    
    # Register auth service (singleton)
    def create_auth_service():
        return AuthService()
    
    container.register(
        "auth_service",
        create_auth_service,
        singleton=True,
        factory=create_auth_service
    )
    logger.debug("Registered: auth_service")
    
    # Register feature flags service (singleton)
    def create_feature_flags_service():
        return FeatureFlagService()
    
    container.register(
        "feature_flags_service",
        create_feature_flags_service,
        singleton=True,
        factory=create_feature_flags_service
    )
    logger.debug("Registered: feature_flags_service")
    
    # Register webhook service (singleton)
    def create_webhook_service():
        service = WebhookService()
        # Integrar con sistema de eventos
        try:
            from core.events import EventBus, EventType
            from core.services.webhook_service import WebhookEvent
            event_bus = EventBus()
            
            async def event_handler(event):
                # Mapear EventType a WebhookEvent
                event_mapping = {
                    EventType.TASK_CREATED: WebhookEvent.TASK_CREATED,
                    EventType.TASK_COMPLETED: WebhookEvent.TASK_COMPLETED,
                    EventType.TASK_FAILED: WebhookEvent.TASK_FAILED,
                    EventType.AGENT_STARTED: WebhookEvent.AGENT_STARTED,
                    EventType.AGENT_STOPPED: WebhookEvent.AGENT_STOPPED
                }
                webhook_event = event_mapping.get(event.event_type)
                if webhook_event:
                    await service.trigger_event(webhook_event, event.data)
            
            # Suscribirse a eventos relevantes
            for event_type in [
                EventType.TASK_CREATED,
                EventType.TASK_COMPLETED,
                EventType.TASK_FAILED,
                EventType.AGENT_STARTED,
                EventType.AGENT_STOPPED
            ]:
                event_bus.subscribe(event_type, event_handler)
        except Exception as e:
            logger.debug(f"Webhook service event integration failed: {e}")
        return service
    
    container.register(
        "webhook_service",
        create_webhook_service,
        singleton=True,
        factory=create_webhook_service
    )
    logger.debug("Registered: webhook_service")
    
    # Register scheduler service (singleton)
    def create_scheduler_service():
        service = SchedulerService()
        # Registrar handler para ejecutar tareas
        async def task_handler(task_data):
            try:
                from config.di_setup import get_service
                from application.use_cases.task_use_cases import CreateTaskUseCase
                storage = get_service("storage")
                use_case = CreateTaskUseCase(storage=storage)
                await use_case.execute(**task_data)
            except Exception as e:
                logger.error(f"Error ejecutando tarea programada: {e}", exc_info=True)
                raise
        
        service.register_task_handler(task_handler)
        return service
    
    container.register(
        "scheduler_service",
        create_scheduler_service,
        singleton=True,
        factory=create_scheduler_service
    )
    logger.debug("Registered: scheduler_service")
    
    # Register queue service (singleton)
    def create_queue_service():
        return QueueService(max_size=1000)
    
    container.register(
        "queue_service",
        create_queue_service,
        singleton=True,
        factory=create_queue_service
    )
    logger.debug("Registered: queue_service")
    
    # Register batch processor (singleton)
    def create_batch_processor():
        return BatchProcessor(max_concurrent=5, batch_size=10)
    
    container.register(
        "batch_processor",
        create_batch_processor,
        singleton=True,
        factory=create_batch_processor
    )
    logger.debug("Registered: batch_processor")
    
    # Register analytics service (singleton)
    def create_analytics_service():
        return AnalyticsService(max_events=10000)
    
    container.register(
        "analytics_service",
        create_analytics_service,
        singleton=True,
        factory=create_analytics_service
    )
    logger.debug("Registered: analytics_service")
    
    # Register search service (singleton)
    def create_search_service():
        return SearchService()
    
    container.register(
        "search_service",
        create_search_service,
        singleton=True,
        factory=create_search_service
    )
    logger.debug("Registered: search_service")
    
    # Register validation service (singleton)
    def create_validation_service():
        service = ValidationService()
        # Agregar reglas comunes
        from core.services.validation_service import is_required, is_min_length, is_max_length
        
        # Reglas para tareas
        service.add_rule(
            "instruction",
            is_required,
            "La instrucción es requerida",
            code="INSTRUCTION_REQUIRED"
        )
        service.add_rule(
            "instruction",
            is_min_length(3),
            "La instrucción debe tener al menos 3 caracteres",
            code="INSTRUCTION_TOO_SHORT"
        )
        service.add_rule(
            "instruction",
            is_max_length(1000),
            "La instrucción no puede exceder 1000 caracteres",
            code="INSTRUCTION_TOO_LONG"
        )
        
        return service
    
    container.register(
        "validation_service",
        create_validation_service,
        singleton=True,
        factory=create_validation_service
    )
    logger.debug("Registered: validation_service")
    
    # Register health checker (singleton)
    def create_health_checker():
        from core.health import HealthChecker, create_database_check, create_disk_space_check, create_memory_check
        
        checker = HealthChecker()
        
        # Registrar checks comunes
        try:
            storage = get_service("storage")
            checker.register_check("database", create_database_check(storage))
        except Exception as e:
            logger.warning(f"No se pudo registrar check de base de datos: {e}")
        
        try:
            checker.register_check("disk_space", create_disk_space_check(threshold_percent=90.0))
        except Exception as e:
            logger.warning(f"No se pudo registrar check de disco: {e}")
        
        try:
            checker.register_check("memory", create_memory_check(threshold_percent=90.0))
        except Exception as e:
            logger.warning(f"No se pudo registrar check de memoria: {e}")
        
        return checker
    
    container.register(
        "health_checker",
        create_health_checker,
        singleton=True,
        factory=create_health_checker
    )
    logger.debug("Registered: health_checker")
    
    # ============================================
    # Core Layer
    # ============================================
    
    # Register storage (singleton)
    container.register(
        "storage",
        TaskStorage,
        singleton=True
    )
    logger.debug("Registered: storage")
    
    # Register GitHub client (singleton)
    def create_github_client():
        if not settings.GITHUB_TOKEN:
            error_msg = "GitHub token not configured. Please set GITHUB_TOKEN environment variable."
            logger.error(error_msg)
            raise ValueError(error_msg)
        try:
            client = GitHubClient()
            logger.debug("GitHub client creado exitosamente")
            return client
        except Exception as e:
            logger.error(f"Error al crear GitHub client: {e}", exc_info=True)
            raise ValueError(f"Error al inicializar GitHub client: {e}") from e
    
    try:
        container.register(
            "github_client",
            create_github_client,
            singleton=True,
            factory=create_github_client
        )
        registered_services.append("github_client")
        logger.debug("Registered: github_client")
    except ValueError:
        # GitHub client es crítico, re-raise
        raise
    except Exception as e:
        logger.error(f"Error al registrar github_client: {e}", exc_info=True)
        failed_services.append(("github_client", str(e)))
        raise ValueError(f"Error crítico al registrar github_client: {e}") from e
    
    # Register task processor (singleton)
    # Dependencies will be auto-detected from constructor (github_client, storage, llm_service)
    def create_task_processor():
        github_client = container.get("github_client")
        storage = container.get("storage")
        llm_service = None
        try:
            llm_service = container.get("llm_service")
        except (ValueError, Exception):
            logger.debug("LLM service no disponible, continuando sin él")
        return TaskProcessor(
            github_client=github_client,
            storage=storage,
            llm_service=llm_service
        )
    
    container.register(
        "task_processor",
        create_task_processor,
        singleton=True,
        factory=create_task_processor
    )
    logger.debug("Registered: task_processor")
    
    # Register worker manager (singleton)
    # Dependencies (storage, task_processor) will be auto-detected from constructor
    container.register(
        "worker_manager",
        WorkerManager,
        singleton=True
    )
    logger.debug("Registered: worker_manager")
    
    # ============================================
    # Application Layer - Use Cases
    # ============================================
    
    # Register task use cases
    # Dependencies will be auto-resolved from constructor signatures
    container.register(
        "create_task_use_case",
        CreateTaskUseCase,
        singleton=True
    )
    logger.debug("Registered: create_task_use_case")
    
    container.register(
        "get_task_use_case",
        GetTaskUseCase,
        singleton=True
    )
    logger.debug("Registered: get_task_use_case")
    
    container.register(
        "list_tasks_use_case",
        ListTasksUseCase,
        singleton=True
    )
    logger.debug("Registered: list_tasks_use_case")
    
    # Register GitHub use cases
    # Dependencies will be auto-resolved from constructor signatures
    container.register(
        "get_repository_info_use_case",
        GetRepositoryInfoUseCase,
        singleton=True
    )
    logger.debug("Registered: get_repository_info_use_case")
    
    container.register(
        "clone_repository_use_case",
        CloneRepositoryUseCase,
        singleton=True
    )
    logger.debug("Registered: clone_repository_use_case")
    
    # Resumen de registro
    logger.info(
        f"Dependency injection setup completed: "
        f"{len(registered_services)} servicios registrados"
    )
    
    if failed_services:
        logger.warning(
            f"Algunos servicios no se pudieron registrar: {len(failed_services)}"
        )
        for service_name, error in failed_services:
            logger.warning(f"  - {service_name}: {error}")
    
    if not registered_services:
        logger.error("No se registró ningún servicio. Verifica la configuración.")
        raise RuntimeError("No se pudo registrar ningún servicio en el contenedor DI")


def get_service(service_name: str) -> Any:
    """
    Get a service from the DI container with improved error handling.
    
    Args:
        service_name: Name of the service
    
    Returns:
        Service instance
        
    Raises:
        ValueError: Si el servicio no está registrado
        RuntimeError: Si el contenedor no está disponible
    """
    try:
        container = get_container()
    except Exception as e:
        logger.error(f"Error al obtener contenedor DI: {e}", exc_info=True)
        raise RuntimeError(f"Contenedor DI no disponible: {e}") from e
    
    try:
        service = container.get(service_name)
        logger.debug(f"Servicio '{service_name}' obtenido exitosamente")
        return service
    except ValueError as e:
        logger.warning(f"Servicio '{service_name}' no encontrado en el contenedor: {e}")
        raise
    except Exception as e:
        logger.error(
            f"Error inesperado al obtener servicio '{service_name}': {e}",
            exc_info=True
        )
        raise RuntimeError(f"Error al obtener servicio '{service_name}': {e}") from e

