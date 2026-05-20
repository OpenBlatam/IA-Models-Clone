"""
Chat API - FastAPI endpoints for continuous chat
=================================================

API REST para el sistema de chat continuo proactivo.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..core.chat_engine import ContinuousChatEngine
from ..core.chat_session import ChatSession, ChatState
from ..core.session_storage import SessionStorage, JSONSessionStorage, RedisSessionStorage
from ..core.rate_limiter import RateLimiter, RateLimitConfig
from ..core.plugins import PluginManager
from ..core.conversation_analyzer import ConversationAnalyzer
from ..core.exporters import ConversationExporter
from ..core.webhooks import WebhookManager, WebhookEvent
from ..core.templates import TemplateManager
from ..core.auth import AuthManager, Role
from ..core.backup_manager import BackupManager, BackupConfig
from ..core.performance_optimizer import PerformanceOptimizer
from ..core.health_monitor import HealthMonitor
from ..core.task_queue import TaskQueue
from ..core.alerting import AlertManager, AlertType, AlertLevel
from ..core.clustering import ClusterManager
from ..core.feature_flags import FeatureFlagManager
from ..core.api_versioning import APIVersionManager
from ..core.advanced_analytics import AdvancedAnalytics
from ..core.recommendations import RecommendationEngine
from ..core.ab_testing import ABTestingFramework, Variant
from ..core.event_system import EventBus, EventType
from ..core.security import SecurityManager
from ..core.i18n import I18nManager, Language
from ..core.workflow import WorkflowEngine, Workflow, WorkflowStep
from ..core.notifications import NotificationManager, NotificationType, NotificationPriority
from ..core.integrations import IntegrationManager, Integration, IntegrationType
from ..core.benchmarking import BenchmarkRunner
from ..core.api_docs import APIDocumentationGenerator, APIDocumentation
from ..core.monitoring import AdvancedMonitoring
from ..core.secrets_manager import SecretsManager
from ..core.ml_optimizer import MLOptimizer
from ..core.deployment import DeploymentManager, DeploymentStatus
from ..core.reports import ReportGenerator, ReportType
from ..core.user_management import UserManager, UserRole, UserStatus
from ..core.search_engine import SearchEngine
from ..core.message_queue import MessageQueue, MessagePriority
from ..core.validation import ValidationEngine, ValidationRule
from ..core.throttling import Throttler, ThrottleConfig
from ..core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from ..core.intelligent_optimizer import IntelligentOptimizer, OptimizationSuggestion
from ..core.adaptive_learning import AdaptiveLearningSystem, LearningPattern
from ..core.demand_predictor import DemandPredictor, DemandForecast
from ..core.intelligent_health import IntelligentHealthChecker, HealthStatus
from ..core.predictive_scaling import PredictiveScaler, ScalingAction
from ..core.cost_optimizer import CostOptimizer, CostOptimizationSuggestion
from ..core.intelligent_alerts import IntelligentAlertSystem, AlertSeverity
from ..core.advanced_observability import AdvancedObservability
from ..core.intelligent_load_balancer import IntelligentLoadBalancer, LoadBalancingAlgorithm
from ..core.resource_manager import ResourceManager, ResourceType
from ..core.disaster_recovery import DisasterRecovery, RecoveryStatus
from ..core.advanced_security import AdvancedSecurity, ThreatLevel, SecurityEventType
from ..core.auto_optimizer import AutoOptimizer, OptimizationResult
from ..core.federated_learning import FederatedLearning, LearningRoundStatus
from ..core.knowledge_manager import KnowledgeManager, KnowledgeType
from ..core.auto_generator import AutoGenerator, GenerationType
from ..core.architecture_recommender import ArchitectureRecommender, ArchitecturePattern
from ..core.mlops_manager import MLOpsManager, ExperimentStatus, ModelStatus
from ..core.dependency_manager import DependencyManager, DependencyStatus, VulnerabilitySeverity
from ..core.cicd_manager import CICDManager, PipelineStatus, StageStatus
from ..core.code_quality import CodeQuality, QualityLevel, CodeSmellType
from ..core.business_metrics import BusinessMetrics, MetricCategory
from ..core.version_control import VersionControl, ChangeType
from ..core.log_analyzer import LogAnalyzer, LogLevel, LogPattern
from ..core.api_performance import APIPerformance, PerformanceMetric
from ..core.advanced_secrets import AdvancedSecrets, SecretType, SecretStatus
from ..core.intelligent_cache import IntelligentCache, CacheStrategy
from ..core.sentiment_analyzer import SentimentAnalyzer, Sentiment, Emotion
from ..core.task_manager import TaskManager, TaskStatus, TaskPriority
from ..core.resource_monitor import ResourceMonitor, ResourceType, AlertLevel
from ..core.push_notifications import PushNotifications, NotificationChannel, NotificationPriority
from ..core.distributed_sync import DistributedSync, SyncStatus, ConflictResolution
from ..core.query_analyzer import QueryAnalyzer, QueryType
from ..core.file_manager import FileManager, FileType, FileStatus
from ..core.data_compression import DataCompression, CompressionAlgorithm
from ..core.incremental_backup import IncrementalBackup, BackupType
from ..core.network_analyzer import NetworkAnalyzer, NetworkEventType
from ..core.config_manager import ConfigManager, ConfigFormat, ConfigStatus
from ..core.mfa_authentication import MFAAuthentication, MFAMethod, MFAStatus
from ..core.advanced_rate_limiter import AdvancedRateLimiter, RateLimitStrategy
from ..core.user_behavior_analyzer import UserBehaviorAnalyzer, BehaviorType
from ..core.event_stream import EventStream, EventType
from ..core.security_analyzer import SecurityAnalyzer, ThreatLevel, ThreatType
from ..core.session_manager import SessionManager, SessionStatus
from ..core.realtime_metrics import RealTimeMetrics, MetricType
from ..core.auto_optimizer import AutoOptimizer, OptimizationTarget
from ..core.predictive_analytics import PredictiveAnalytics, PredictionType
from ..core.policy_engine import PolicyEngine, PolicyType, PolicyStatus
from ..core.audit_system import AuditSystem, AuditEventType, AuditSeverity
from ..core.task_orchestrator import TaskOrchestrator, TaskStatus, TaskPriority
from ..core.resource_allocator import ResourceAllocator, ResourceType as AllocatorResourceType, AllocationStatus
from ..core.service_orchestrator import ServiceOrchestrator, ServiceStatus
from ..core.performance_profiler import PerformanceProfiler, ProfilerScope
from ..core.adaptive_rate_controller import AdaptiveRateController, RateAdjustmentStrategy
from ..core.smart_retry_manager import SmartRetryManager, RetryStrategy, RetryStatus
from ..core.distributed_lock_manager import DistributedLockManager, LockStatus
from ..core.data_pipeline_manager import DataPipelineManager, PipelineStatus, StageType
from ..core.event_scheduler import EventScheduler, ScheduleType, ScheduleStatus
from ..core.graceful_degradation_manager import GracefulDegradationManager, DegradationLevel, ServiceState
from ..core.cache_warmer import CacheWarmer, WarmingStrategy
from ..core.load_shedder import LoadShedder, SheddingStrategy, RequestPriority
from ..core.conflict_resolver import ConflictResolver, ConflictType, ResolutionStrategy, ConflictStatus
from ..core.state_machine import StateMachineManager, TransitionType
from ..core.workflow_engine_v2 import WorkflowEngineV2, StepType, WorkflowStatus
from ..core.event_bus import EventBus, EventPriority
from ..core.feature_toggle import FeatureToggleManager, ToggleType, ToggleStatus
from ..core.rate_limiter_v2 import RateLimiterV2, RateLimitAlgorithm
from ..core.circuit_breaker_v2 import CircuitBreakerV2, FailureStrategy, CircuitState
from ..core.adaptive_optimizer import AdaptiveOptimizer, OptimizationTarget
from ..core.health_checker_v2 import HealthCheckerV2, CheckType, HealthStatus
from ..core.auto_scaler import AutoScaler, ScalingAction
from ..core.batch_processor import BatchProcessor, BatchStrategy
from ..core.performance_monitor import PerformanceMonitor, MetricType
from ..core.resource_pool import ResourcePool, PoolConfig
from ..core.queue_manager import QueueManager, QueuePriority
from ..core.connection_manager import ConnectionManager
from ..core.transaction_manager import TransactionManager, TransactionStatus
from ..core.saga_orchestrator import SagaOrchestrator, SagaStatus
from ..core.distributed_coordinator import DistributedCoordinator, ConsensusAlgorithm, NodeRole
from ..core.service_mesh import ServiceMesh, LoadBalancingStrategy, ServiceStatus
from ..core.adaptive_throttler import AdaptiveThrottler, ThrottleStrategy
from ..core.backpressure_manager import BackpressureManager, BackpressureLevel
from ..api.websocket_api import create_websocket_router
from ..api.graphql_api import create_graphql_router
from ..config.chat_config import ChatConfig

logger = logging.getLogger(__name__)


# Modelos Pydantic para requests/responses
class ChatCreateRequest(BaseModel):
    """Request para crear una nueva sesión de chat."""
    user_id: Optional[str] = None
    initial_message: Optional[str] = None
    auto_continue: bool = True
    response_interval: float = 2.0


class ChatMessageRequest(BaseModel):
    """Request para enviar un mensaje."""
    message: str = Field(..., description="Mensaje del usuario")


class ChatControlRequest(BaseModel):
    """Request para controlar el chat (pausar/reanudar)."""
    action: str = Field(..., description="Acción: 'pause' o 'resume'")
    reason: Optional[str] = None


class ChatResponse(BaseModel):
    """Response de la API."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class ChatSessionResponse(BaseModel):
    """Response con información de la sesión."""
    session_id: str
    state: str
    is_paused: bool
    message_count: int
    auto_continue: bool


def create_chat_app(config: Optional[ChatConfig] = None) -> FastAPI:
    """
    Crear aplicación FastAPI para el chat continuo.
    
    Args:
        config: Configuración del chat (opcional)
    
    Returns:
        FastAPI app configurada
    """
    app_config = config or ChatConfig()
    
    # Crear sistema de almacenamiento
    storage = None
    if app_config.storage_type == "redis" and app_config.redis_url:
        try:
            storage = RedisSessionStorage(
                redis_url=app_config.redis_url,
                ttl=app_config.session_ttl,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Redis storage: {e}, falling back to JSON")
            storage = JSONSessionStorage(storage_path=app_config.storage_path)
    elif app_config.storage_type == "json":
        storage = JSONSessionStorage(storage_path=app_config.storage_path)
    
    # Crear rate limiter
    rate_limiter = RateLimiter(
        RateLimitConfig(
            max_requests=app_config.rate_limit_max_requests,
            time_window=app_config.rate_limit_window,
            max_concurrent=app_config.rate_limit_max_concurrent,
        )
    )
    
    # Crear app FastAPI
    app = FastAPI(
        title="Bulk Chat API",
        description="API para chat continuo proactivo",
        version="2.0.0",
    )
    
    # Crear plugin manager
    plugin_manager = PluginManager()
    # Registrar plugins por defecto
    if app_config.enable_plugins:
        from ..core.plugins import (
            SentimentAnalyzerPlugin,
            ProfanityFilterPlugin,
            ResponseEnhancerPlugin,
        )
        from ..core.plugins import PluginType
        
        plugin_manager.register(SentimentAnalyzerPlugin(), PluginType.ANALYZER)
        plugin_manager.register(ProfanityFilterPlugin(), PluginType.PRE_PROCESSOR)
        plugin_manager.register(ResponseEnhancerPlugin(), PluginType.POST_PROCESSOR)
    
    # Crear servicios adicionales
    analyzer = ConversationAnalyzer()
    exporter = ConversationExporter()
    webhook_manager = WebhookManager()
    template_manager = TemplateManager()
    auth_manager = AuthManager()
    performance_optimizer = PerformanceOptimizer()
    health_monitor = HealthMonitor()
    task_queue = TaskQueue(max_workers=5)
    alert_manager = AlertManager()
    cluster_manager = ClusterManager(
        node_id=f"node_{app_config.api_port}",
        host=app_config.api_host,
        port=app_config.api_port,
    )
    feature_flags = FeatureFlagManager()
    api_versioning = APIVersionManager()
    advanced_analytics = AdvancedAnalytics()
    recommendation_engine = RecommendationEngine()
    ab_testing = ABTestingFramework()
    event_bus = EventBus()
    security_manager = SecurityManager()
    i18n_manager = I18nManager()
    workflow_engine = WorkflowEngine()
    notification_manager = NotificationManager()
    integration_manager = IntegrationManager()
    benchmark_runner = BenchmarkRunner()
    api_docs_generator = APIDocumentationGenerator()
    advanced_monitoring = AdvancedMonitoring()
    secrets_manager = SecretsManager()
    ml_optimizer = MLOptimizer()
    deployment_manager = DeploymentManager()
    report_generator = ReportGenerator()
    user_manager = UserManager()
    search_engine = SearchEngine()
    message_queue = MessageQueue()
    validation_engine = ValidationEngine()
    throttler = Throttler()
    circuit_breaker = CircuitBreaker()
    intelligent_optimizer = IntelligentOptimizer()
    adaptive_learning = AdaptiveLearningSystem()
    demand_predictor = DemandPredictor()
    intelligent_health = IntelligentHealthChecker()
    predictive_scaler = PredictiveScaler()
    cost_optimizer = CostOptimizer()
    intelligent_alerts = IntelligentAlertSystem()
    advanced_observability = AdvancedObservability()
    load_balancer = IntelligentLoadBalancer()
    resource_manager = ResourceManager()
    disaster_recovery = DisasterRecovery()
    advanced_security = AdvancedSecurity()
    auto_optimizer = AutoOptimizer()
    federated_learning = FederatedLearning()
    knowledge_manager = KnowledgeManager()
    auto_generator = AutoGenerator()
    architecture_recommender = ArchitectureRecommender()
    mlops_manager = MLOpsManager()
    dependency_manager = DependencyManager()
    cicd_manager = CICDManager()
    code_quality = CodeQuality()
    business_metrics = BusinessMetrics()
    version_control = VersionControl()
    log_analyzer = LogAnalyzer()
    api_performance = APIPerformance()
    advanced_secrets = AdvancedSecrets()
    intelligent_cache = IntelligentCache(max_size=5000, strategy=CacheStrategy.ADAPTIVE)
    sentiment_analyzer = SentimentAnalyzer()
    task_manager = TaskManager()
    resource_monitor = ResourceMonitor()
    push_notifications = PushNotifications()
    distributed_sync = DistributedSync(node_id="node_1")
    query_analyzer = QueryAnalyzer()
    file_manager = FileManager()
    data_compression = DataCompression()
    incremental_backup = IncrementalBackup()
    network_analyzer = NetworkAnalyzer()
    config_manager = ConfigManager()
    mfa_authentication = MFAAuthentication()
    advanced_rate_limiter = AdvancedRateLimiter()
    user_behavior_analyzer = UserBehaviorAnalyzer()
    event_stream = EventStream()
    security_analyzer = SecurityAnalyzer()
    session_manager = SessionManager()
    realtime_metrics = RealTimeMetrics()
    auto_optimizer = AutoOptimizer()
    predictive_analytics = PredictiveAnalytics()
    policy_engine = PolicyEngine()
    audit_system = AuditSystem()
    task_orchestrator = TaskOrchestrator()
    resource_allocator = ResourceAllocator()
    service_orchestrator = ServiceOrchestrator()
    performance_profiler = PerformanceProfiler()
    adaptive_rate_controller = AdaptiveRateController()
    smart_retry_manager = SmartRetryManager()
    distributed_lock_manager = DistributedLockManager()
    data_pipeline_manager = DataPipelineManager()
    event_scheduler = EventScheduler()
    graceful_degradation_manager = GracefulDegradationManager()
    cache_warmer = CacheWarmer(cache_manager=intelligent_cache)
    load_shedder = LoadShedder()
    conflict_resolver = ConflictResolver()
    state_machine_manager = StateMachineManager()
    workflow_engine_v2 = WorkflowEngineV2()
    event_bus = EventBus()
    feature_toggle_manager = FeatureToggleManager()
    rate_limiter_v2 = RateLimiterV2()
    circuit_breaker_v2 = CircuitBreakerV2()
    adaptive_optimizer = AdaptiveOptimizer()
    health_checker_v2 = HealthCheckerV2()
    auto_scaler = AutoScaler()
    batch_processor = BatchProcessor()
    performance_monitor = PerformanceMonitor()
    queue_manager = QueueManager()
    connection_manager = ConnectionManager()
    transaction_manager = TransactionManager()
    saga_orchestrator = SagaOrchestrator()
    distributed_coordinator = DistributedCoordinator(node_id=f"node_{datetime.now().timestamp()}")
    service_mesh = ServiceMesh()
    adaptive_throttler = AdaptiveThrottler()
    backpressure_manager = BackpressureManager()
    
    # Iniciar scheduler de eventos
    event_scheduler.start_scheduler()
    
    # Iniciar cache warmer
    cache_warmer.start_warming()
    
    # Iniciar event bus
    event_bus.start_processing()
    
    # Iniciar health checker
    health_checker_v2.start_checking()
    
    # Iniciar auto scaler
    auto_scaler.start_scaling()
    
    # Iniciar coordinación distribuida
    distributed_coordinator.start_coordination()
    
    # Iniciar monitoreo de recursos
    asyncio.create_task(resource_monitor.start_monitoring())
    
    # Iniciar task queue
    asyncio.create_task(task_queue.start())
    
    # Iniciar cluster manager
    asyncio.create_task(cluster_manager.start_heartbeat_monitor())
    
    # Configurar alertas automáticas
    async def check_performance_alerts():
        slow_ops = await performance_optimizer.optimize_slow_operations(threshold=2.0)
        if slow_ops:
            await alert_manager.create_alert(
                AlertType.PERFORMANCE,
                AlertLevel.WARNING,
                "Operaciones Lentas Detectadas",
                f"Se detectaron {len(slow_ops)} operaciones lentas",
                {"slow_operations": slow_ops}
            )
    
    # Verificar alertas periódicamente
    async def alert_loop():
        while True:
            try:
                await check_performance_alerts()
                await asyncio.sleep(60)  # Cada minuto
            except Exception as e:
                logger.error(f"Error in alert loop: {e}")
                await asyncio.sleep(60)
    
    asyncio.create_task(alert_loop())
    
    # Crear backup manager
    backup_config = BackupConfig(
        enabled=app_config.enable_backups,
        interval_hours=app_config.backup_interval_hours,
        backup_directory=app_config.backup_directory,
    )
    backup_manager = BackupManager(
        config=backup_config,
        storage_path=app_config.storage_path,
    )
    
    # Iniciar backups en background task
    backup_task = None
    if backup_config.enabled:
        async def start_backups():
            await backup_manager.start()
        backup_task = asyncio.create_task(start_backups())
    
    # Crear motor de chat
    chat_engine = ContinuousChatEngine(
        llm_provider=app_config.get_llm_provider(),
        auto_continue=app_config.auto_continue,
        response_interval=app_config.response_interval,
        max_consecutive_responses=app_config.max_consecutive_responses,
        storage=storage,
        enable_metrics=app_config.enable_metrics,
        auto_save=app_config.auto_save,
        save_interval=app_config.save_interval,
        enable_cache=app_config.enable_cache,
        cache_size=app_config.cache_size,
        cache_ttl=app_config.cache_ttl,
        plugin_manager=plugin_manager,
        webhook_manager=webhook_manager,
    )
    
    # Agregar router de WebSocket
    ws_router = create_websocket_router(chat_engine)
    app.include_router(ws_router)
    
    # Agregar router de GraphQL (setup después de crear app)
    try:
        graphql_router = create_graphql_router(chat_engine)
        app.include_router(graphql_router, prefix="/graphql")
    except Exception as e:
        logger.warning(f"Could not setup GraphQL: {e}")
    
    # Servir dashboard
    try:
        dashboard_path = Path(__file__).parent.parent / "dashboard"
        if dashboard_path.exists():
            app.mount("/dashboard", StaticFiles(directory=str(dashboard_path), html=True), name="dashboard")
    except Exception as e:
        logger.warning(f"Could not mount dashboard: {e}")
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Endpoints
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint básico."""
        return {
            "status": "healthy",
            "service": "bulk_chat",
            "active_sessions": len(chat_engine.sessions),
        }
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Health check detallado."""
        system_health = await health_monitor.get_system_health()
        return system_health
    
    @app.get("/api/v1/performance/metrics")
    async def get_performance_metrics(operation: Optional[str] = None):
        """Obtener métricas de rendimiento."""
        metrics = performance_optimizer.get_metrics(operation)
        return {"metrics": metrics}
    
    @app.get("/api/v1/performance/slow-operations")
    async def get_slow_operations(threshold: float = 1.0):
        """Obtener operaciones lentas."""
        slow_ops = await performance_optimizer.optimize_slow_operations(threshold)
        return {"slow_operations": slow_ops}
    
    @app.get("/api/v1/tasks/queue")
    async def get_task_queue_status():
        """Obtener estado de la cola de tareas."""
        return {
            "queue_size": task_queue.get_queue_size(),
            "active_tasks": len(task_queue.get_active_tasks()),
            "pending_tasks": len(task_queue.get_pending_tasks()),
            "max_workers": task_queue.max_workers,
        }
    
    # Endpoints de alertas
    @app.get("/api/v1/alerts")
    async def get_alerts(
        alert_type: Optional[str] = None,
        level: Optional[str] = None,
        resolved: Optional[bool] = None,
    ):
        """Obtener alertas."""
        from ..core.alerting import AlertType, AlertLevel
        
        alert_type_enum = AlertType(alert_type) if alert_type else None
        level_enum = AlertLevel(level) if level else None
        
        alerts = alert_manager.get_alerts(
            alert_type=alert_type_enum,
            level=level_enum,
            resolved=resolved,
        )
        
        return {
            "alerts": [
                {
                    "id": a.id,
                    "type": a.type.value,
                    "level": a.level.value,
                    "title": a.title,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                    "resolved": a.resolved,
                }
                for a in alerts
            ],
            "total": len(alerts),
        }
    
    @app.post("/api/v1/alerts/{alert_id}/resolve")
    async def resolve_alert(alert_id: str):
        """Resolver alerta."""
        await alert_manager.resolve_alert(alert_id)
        return {"success": True, "message": "Alert resolved"}
    
    # Endpoints de feature flags
    @app.get("/api/v1/feature-flags")
    async def list_feature_flags():
        """Listar feature flags."""
        flags = feature_flags.list_flags()
        return {"feature_flags": flags}
    
    @app.get("/api/v1/feature-flags/{flag_name}")
    async def get_feature_flag(flag_name: str, user_id: Optional[str] = None):
        """Obtener estado de feature flag."""
        enabled = await feature_flags.is_enabled(flag_name, user_id)
        flag = feature_flags.get_flag(flag_name)
        
        if not flag:
            raise HTTPException(status_code=404, detail="Feature flag not found")
        
        return {
            "name": flag_name,
            "enabled": enabled,
            "status": flag.status.value,
            "description": flag.description,
        }
    
    @app.post("/api/v1/feature-flags/{flag_name}/enable")
    async def enable_feature_flag(flag_name: str):
        """Habilitar feature flag."""
        await feature_flags.enable(flag_name)
        return {"success": True, "message": f"Feature flag {flag_name} enabled"}
    
    @app.post("/api/v1/feature-flags/{flag_name}/disable")
    async def disable_feature_flag(flag_name: str):
        """Deshabilitar feature flag."""
        await feature_flags.disable(flag_name)
        return {"success": True, "message": f"Feature flag {flag_name} disabled"}
    
    # Endpoints de versionado
    @app.get("/api/versions")
    async def get_api_versions():
        """Obtener versiones de API disponibles."""
        versions = api_versioning.get_all_versions()
        return {
            "current_version": api_versioning.current_version,
            "supported_versions": api_versioning.get_supported_versions(),
            "versions": versions,
        }
    
    # Endpoints de clustering
    @app.get("/api/v1/cluster/info")
    async def get_cluster_info():
        """Obtener información del cluster."""
        info = cluster_manager.get_cluster_info()
        return info
    
    # Endpoints de analytics avanzado
    @app.get("/api/v1/analytics/patterns")
    async def get_analytics_patterns():
        """Obtener patrones detectados."""
        sessions = list(chat_engine.sessions.values())
        patterns = await advanced_analytics.detect_patterns(sessions)
        
        return {
            "patterns": [
                {
                    "type": p.pattern_type,
                    "frequency": p.frequency,
                    "confidence": p.confidence,
                    "examples": p.examples,
                }
                for p in patterns
            ]
        }
    
    @app.get("/api/v1/analytics/user/{user_id}/behavior")
    async def get_user_behavior(user_id: str):
        """Obtener comportamiento de usuario."""
        sessions = [
            s for s in chat_engine.sessions.values()
            if s.user_id == user_id
        ]
        
        behavior = await advanced_analytics.analyze_user_behavior(user_id, sessions)
        
        return {
            "user_id": user_id,
            "total_sessions": behavior.total_sessions,
            "average_session_duration": behavior.average_session_duration,
            "favorite_topics": behavior.favorite_topics,
            "preferred_time": behavior.preferred_time,
            "engagement_score": behavior.engagement_score,
        }
    
    @app.get("/api/v1/analytics/insights")
    async def get_advanced_insights():
        """Obtener insights avanzados."""
        sessions = list(chat_engine.sessions.values())
        insights = await advanced_analytics.generate_insights(sessions)
        return insights
    
    # Endpoints de recomendaciones
    @app.get("/api/v1/recommendations/{user_id}")
    async def get_recommendations(
        user_id: str,
        item_type: Optional[str] = None,
        limit: int = 10,
    ):
        """Obtener recomendaciones para usuario."""
        recommendations = await recommendation_engine.recommend_items(
            user_id,
            item_type,
            limit,
        )
        return {
            "user_id": user_id,
            "recommendations": [
                {
                    "item_id": r.item_id,
                    "item_type": r.item_type,
                    "score": r.score,
                    "reason": r.reason,
                }
                for r in recommendations
            ],
        }
    
    @app.post("/api/v1/recommendations/interaction")
    async def record_interaction(
        user_id: str,
        item_id: str,
        item_type: str,
        rating: float = 1.0,
    ):
        """Registrar interacción de usuario."""
        await recommendation_engine.record_interaction(
            user_id,
            item_id,
            item_type,
            rating,
        )
        return {"success": True, "message": "Interaction recorded"}
    
    # Endpoints de A/B Testing
    @app.post("/api/v1/ab-testing/experiments")
    async def create_experiment(
        experiment_id: str,
        name: str,
        description: str,
        variants: List[str],
        traffic_split: Optional[Dict[str, float]] = None,
    ):
        """Crear experimento A/B."""
        variant_enums = [Variant(v) for v in variants]
        split_dict = None
        if traffic_split:
            split_dict = {Variant(k): v for k, v in traffic_split.items()}
        
        experiment = ab_testing.create_experiment(
            experiment_id,
            name,
            description,
            variant_enums,
            split_dict,
        )
        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "is_active": experiment.is_active,
        }
    
    @app.get("/api/v1/ab-testing/experiments/{experiment_id}/variant")
    async def get_variant(
        experiment_id: str,
        user_id: str,
    ):
        """Obtener variante asignada a usuario."""
        variant = ab_testing.get_variant(experiment_id, user_id)
        if not variant:
            raise HTTPException(status_code=404, detail="Experiment not found or inactive")
        return {"experiment_id": experiment_id, "user_id": user_id, "variant": variant.value}
    
    @app.get("/api/v1/ab-testing/experiments/{experiment_id}/stats")
    async def get_experiment_stats(experiment_id: str):
        """Obtener estadísticas de experimento."""
        stats = await ab_testing.get_experiment_stats(experiment_id)
        return stats
    
    # Endpoints de eventos
    @app.get("/api/v1/events/history")
    async def get_event_history(
        event_type: Optional[str] = None,
        limit: int = 100,
    ):
        """Obtener historial de eventos."""
        event_type_enum = None
        if event_type:
            event_type_enum = EventType(event_type)
        
        events = event_bus.get_event_history(event_type_enum, limit)
        return {
            "events": [e.to_dict() for e in events],
            "total": len(events),
        }
    
    @app.get("/api/v1/events/subscribers")
    async def get_subscribers():
        """Obtener conteo de suscriptores."""
        return event_bus.get_subscribers_count()
    
    # Endpoints de seguridad
    @app.get("/api/v1/security/audit-logs")
    async def get_audit_logs(
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ):
        """Obtener logs de auditoría."""
        logs = security_manager.get_audit_logs(user_id, action, None, limit)
        return {
            "logs": [
                {
                    "log_id": log.log_id,
                    "user_id": log.user_id,
                    "action": log.action,
                    "resource": log.resource,
                    "timestamp": log.timestamp.isoformat(),
                    "success": log.success,
                }
                for log in logs
            ],
            "total": len(logs),
        }
    
    @app.get("/api/v1/security/stats")
    async def get_security_stats():
        """Obtener estadísticas de seguridad."""
        return security_manager.get_security_stats()
    
    # Endpoints de i18n
    @app.get("/api/v1/i18n/translate")
    async def translate(
        key: str,
        language: Optional[str] = None,
    ):
        """Traducir clave."""
        lang = None
        if language:
            lang = Language(language)
        
        translation = i18n_manager.translate(key, lang)
        return {"key": key, "language": language or i18n_manager.default_language.value, "translation": translation}
    
    @app.get("/api/v1/i18n/languages")
    async def get_supported_languages():
        """Obtener idiomas soportados."""
        return {"languages": i18n_manager.get_supported_languages()}
    
    # Endpoints de workflows
    @app.post("/api/v1/workflows/execute")
    async def execute_workflow(
        workflow_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
    ):
        """Ejecutar workflow."""
        result = await workflow_engine.execute_workflow(workflow_id, initial_context)
        return result
    
    @app.get("/api/v1/workflows")
    async def list_workflows():
        """Listar workflows."""
        workflows = workflow_engine.list_workflows()
        return {"workflows": workflows}
    
    # Endpoints de notificaciones
    @app.post("/api/v1/notifications/send")
    async def send_notification(
        user_id: str,
        title: str,
        message: str,
        notification_type: str = "info",
        priority: str = "medium",
    ):
        """Enviar notificación."""
        notif_type = NotificationType(notification_type)
        notif_priority = NotificationPriority(priority)
        
        notification = await notification_manager.send_notification(
            user_id,
            title,
            message,
            notif_type,
            notif_priority,
        )
        return {
            "notification_id": notification.notification_id,
            "success": True,
        }
    
    @app.get("/api/v1/notifications/{user_id}")
    async def get_notifications(
        user_id: str,
        unread_only: bool = False,
        limit: int = 50,
    ):
        """Obtener notificaciones de usuario."""
        notifications = await notification_manager.get_notifications(
            user_id,
            unread_only,
            limit,
        )
        return {
            "notifications": [
                {
                    "notification_id": n.notification_id,
                    "title": n.title,
                    "message": n.message,
                    "type": n.notification_type.value,
                    "priority": n.priority.value,
                    "created_at": n.created_at.isoformat(),
                    "read_at": n.read_at.isoformat() if n.read_at else None,
                }
                for n in notifications
            ],
            "unread_count": notification_manager.get_unread_count(user_id),
        }
    
    @app.post("/api/v1/notifications/{user_id}/read/{notification_id}")
    async def mark_notification_read(user_id: str, notification_id: str):
        """Marcar notificación como leída."""
        await notification_manager.mark_as_read(user_id, notification_id)
        return {"success": True}
    
    # Endpoints de integraciones
    @app.post("/api/v1/integrations/call")
    async def call_integration(
        integration_id: str,
        data: Dict[str, Any],
    ):
        """Llamar integración."""
        result = await integration_manager.call_integration(integration_id, data)
        return result
    
    @app.get("/api/v1/integrations")
    async def list_integrations():
        """Listar integraciones."""
        integrations = integration_manager.list_integrations()
        return {"integrations": integrations}
    
    # Endpoints de benchmarking
    @app.post("/api/v1/benchmark/run")
    async def run_benchmark(
        benchmark_id: str,
        name: str,
        iterations: int = 10,
        warmup_runs: int = 2,
    ):
        """Ejecutar benchmark (requiere función específica)."""
        # Nota: Este endpoint es un ejemplo. En producción, necesitarías
        # pasar la función a ejecutar de forma segura.
        return {
            "message": "Benchmark endpoint requires function definition",
            "benchmark_id": benchmark_id,
        }
    
    @app.get("/api/v1/benchmark/results")
    async def get_benchmark_results():
        """Obtener resultados de benchmarks."""
        results = benchmark_runner.list_results()
        return {"results": results}
    
    # Endpoints de documentación
    @app.get("/api/v1/docs/openapi")
    async def get_openapi_spec():
        """Obtener especificación OpenAPI."""
        spec = api_docs_generator.generate_openapi_spec()
        return spec
    
    @app.get("/api/v1/docs/markdown")
    async def get_markdown_docs():
        """Obtener documentación en Markdown."""
        docs = api_docs_generator.generate_markdown_docs()
        return {"content": docs, "format": "markdown"}
    
    @app.get("/api/v1/docs/endpoints")
    async def list_endpoints():
        """Listar todos los endpoints documentados."""
        endpoints = api_docs_generator.list_endpoints()
        return {"endpoints": endpoints}
    
    # Endpoints de monitoring
    @app.post("/api/v1/monitoring/metrics")
    async def record_metric(
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Registrar métrica."""
        await advanced_monitoring.record_metric(name, value, tags)
        return {"success": True, "message": f"Metric {name} recorded"}
    
    @app.get("/api/v1/monitoring/metrics/{metric_name}/stats")
    async def get_metric_stats(metric_name: str, window_minutes: int = 60):
        """Obtener estadísticas de métrica."""
        stats = await advanced_monitoring.get_metric_stats(metric_name, window_minutes)
        return stats
    
    @app.get("/api/v1/monitoring/summary")
    async def get_monitoring_summary():
        """Obtener resumen de monitoreo."""
        summary = advanced_monitoring.get_metrics_summary()
        return summary
    
    @app.get("/api/v1/monitoring/alerts")
    async def get_monitoring_alerts(
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
    ):
        """Obtener alertas de monitoreo."""
        alerts = await advanced_monitoring.get_alerts(severity, resolved)
        return {
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "name": a.name,
                    "severity": a.severity,
                    "message": a.message,
                    "metric_name": a.metric_name,
                    "threshold": a.threshold,
                    "current_value": a.current_value,
                    "timestamp": a.timestamp.isoformat(),
                    "resolved": a.resolved,
                }
                for a in alerts
            ]
        }
    
    # Endpoints de secretos
    @app.post("/api/v1/secrets/store")
    async def store_secret(
        secret_id: str,
        name: str,
        value: str,
        secret_type: str = "api_key",
        encrypted: bool = False,
    ):
        """Almacenar secreto."""
        secret = await secrets_manager.store_secret(
            secret_id,
            name,
            value,
            secret_type,
            encrypted,
        )
        return {
            "secret_id": secret.secret_id,
            "name": secret.name,
            "success": True,
        }
    
    @app.get("/api/v1/secrets/{secret_id}")
    async def get_secret(secret_id: str, decrypt: bool = True):
        """Obtener secreto."""
        value = await secrets_manager.get_secret(secret_id, decrypt)
        if value is None:
            raise HTTPException(status_code=404, detail="Secret not found")
        return {"secret_id": secret_id, "value": value}
    
    @app.get("/api/v1/secrets")
    async def list_secrets():
        """Listar secretos."""
        secrets = await secrets_manager.list_secrets()
        return {"secrets": secrets}
    
    # Endpoints de ML Optimizer
    @app.post("/api/v1/ml-optimizer/record")
    async def record_performance(
        parameters: Dict[str, float],
        performance_metric: float,
    ):
        """Registrar datos de rendimiento."""
        await ml_optimizer.record_performance(parameters, performance_metric)
        return {"success": True, "message": "Performance recorded"}
    
    @app.post("/api/v1/ml-optimizer/optimize")
    async def optimize_parameter(
        parameter_name: str,
        min_value: float = 0.0,
        max_value: float = 1.0,
        step: float = 0.1,
    ):
        """Optimizar parámetro."""
        result = await ml_optimizer.optimize_parameter(
            parameter_name,
            min_value,
            max_value,
            step,
        )
        if not result:
            raise HTTPException(status_code=404, detail="Parameter not found or insufficient data")
        
        return {
            "parameter_name": result.parameter_name,
            "optimal_value": result.optimal_value,
            "confidence": result.confidence,
            "improvement": result.improvement,
            "metadata": result.metadata,
        }
    
    @app.post("/api/v1/ml-optimizer/predict")
    async def predict_performance(
        parameter_name: str,
        value: float,
    ):
        """Predecir rendimiento."""
        predicted = await ml_optimizer.predict_performance(parameter_name, value)
        if predicted is None:
            raise HTTPException(status_code=404, detail="Parameter not found or insufficient data")
        
        return {
            "parameter_name": parameter_name,
            "value": value,
            "predicted_performance": predicted,
        }
    
    # Endpoints de deployment
    @app.post("/api/v1/deployment/deploy")
    async def deploy(
        deployment_id: str,
        version: str,
        environment: str,
        rollback_version: Optional[str] = None,
    ):
        """Ejecutar deployment."""
        deployment = await deployment_manager.deploy(
            deployment_id,
            version,
            environment,
            rollback_version,
        )
        return {
            "deployment_id": deployment.deployment_id,
            "version": deployment.version,
            "environment": deployment.environment,
            "status": deployment.status.value,
            "logs": deployment.logs,
        }
    
    @app.post("/api/v1/deployment/{deployment_id}/rollback")
    async def rollback_deployment(
        deployment_id: str,
        version: Optional[str] = None,
    ):
        """Hacer rollback de deployment."""
        deployment = await deployment_manager.rollback(deployment_id, version)
        return {
            "deployment_id": deployment.deployment_id,
            "status": deployment.status.value,
            "logs": deployment.logs,
        }
    
    @app.get("/api/v1/deployment/current")
    async def get_current_version(environment: str):
        """Obtener versión actual en ambiente."""
        version = deployment_manager.get_current_version(environment)
        if not version:
            raise HTTPException(status_code=404, detail="No version found for environment")
        return {"environment": environment, "version": version}
    
    @app.get("/api/v1/deployment")
    async def list_deployments(
        environment: Optional[str] = None,
        limit: int = 50,
    ):
        """Listar deployments."""
        deployments = deployment_manager.list_deployments(environment, limit)
        return {"deployments": deployments}
    
    # Endpoints de reportes
    @app.post("/api/v1/reports/generate")
    async def generate_report(
        report_id: str,
        report_type: str,
        title: str,
        period_start: str,
        period_end: str,
        format: str = "json",
    ):
        """Generar reporte."""
        # Nota: En producción, necesitarías pasar la función data_source
        return {
            "message": "Report generation requires data_source function",
            "report_id": report_id,
        }
    
    @app.get("/api/v1/reports")
    async def list_reports(
        report_type: Optional[str] = None,
        limit: int = 50,
    ):
        """Listar reportes."""
        rtype = ReportType(report_type) if report_type else None
        reports = report_generator.list_reports(rtype, limit)
        return {"reports": reports}
    
    @app.get("/api/v1/reports/{report_id}")
    async def get_report(report_id: str):
        """Obtener reporte."""
        report = report_generator.get_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return {
            "report_id": report.report_id,
            "title": report.title,
            "report_type": report.report_type.value,
            "generated_at": report.generated_at.isoformat(),
            "period_start": report.period_start.isoformat(),
            "period_end": report.period_end.isoformat(),
            "data": report.data,
            "format": report.format,
        }
    
    # Endpoints de gestión de usuarios
    @app.post("/api/v1/users/register")
    async def register_user(
        username: str,
        email: str,
        password: str,
        role: str = "user",
    ):
        """Registrar nuevo usuario."""
        user_role = UserRole(role)
        user = await user_manager.create_user(username, email, password, user_role)
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "status": user.status.value,
        }
    
    @app.post("/api/v1/users/login")
    async def login_user(username: str, password: str):
        """Iniciar sesión."""
        user = await user_manager.authenticate(username, password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        return {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "status": user.status.value,
        }
    
    @app.get("/api/v1/users")
    async def list_users(
        role: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ):
        """Listar usuarios."""
        user_role = UserRole(role) if role else None
        user_status = UserStatus(status) if status else None
        
        users = user_manager.list_users(user_role, user_status, limit)
        return {"users": users}
    
    @app.get("/api/v1/users/{user_id}")
    async def get_user(user_id: str):
        """Obtener usuario."""
        user = user_manager.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "status": user.status.value,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None,
        }
    
    # Endpoints de búsqueda
    @app.get("/api/v1/search")
    async def search(
        q: str,
        item_type: Optional[str] = None,
        limit: int = 20,
    ):
        """Buscar items."""
        results = await search_engine.search(q, item_type, limit)
        return {
            "query": q,
            "results": [
                {
                    "item_id": r.item_id,
                    "item_type": r.item_type,
                    "score": r.score,
                    "title": r.title,
                    "content": r.content,
                    "highlights": r.highlights,
                }
                for r in results
            ],
            "total": len(results),
        }
    
    @app.post("/api/v1/search/index")
    async def index_item(
        item_id: str,
        item_type: str,
        title: str,
        content: str,
    ):
        """Indexar item."""
        await search_engine.index_item(item_id, item_type, title, content)
        return {"success": True, "message": f"Item {item_id} indexed"}
    
    @app.get("/api/v1/search/stats")
    async def get_search_stats():
        """Obtener estadísticas de búsqueda."""
        stats = search_engine.get_index_stats()
        return stats
    
    # Endpoints de cola de mensajes
    @app.post("/api/v1/queue/enqueue")
    async def enqueue_message(
        queue_name: str,
        payload: Dict[str, Any],
        priority: str = "medium",
        max_attempts: int = 3,
    ):
        """Agregar mensaje a la cola."""
        msg_priority = MessagePriority(priority)
        message_id = await message_queue.enqueue(
            queue_name,
            payload,
            msg_priority,
            max_attempts,
        )
        return {
            "message_id": message_id,
            "queue_name": queue_name,
            "success": True,
        }
    
    @app.get("/api/v1/queue/stats")
    async def get_queue_stats():
        """Obtener estadísticas de colas."""
        stats = message_queue.get_queue_stats()
        return stats
    
    @app.get("/api/v1/queue/{queue_name}/size")
    async def get_queue_size(queue_name: str):
        """Obtener tamaño de cola."""
        size = message_queue.get_queue_size(queue_name)
        return {"queue_name": queue_name, "size": size}
    
    # Endpoints de validación
    @app.post("/api/v1/validation/validate")
    async def validate_data(
        context: str,
        data: Dict[str, Any],
    ):
        """Validar datos."""
        errors = await validation_engine.validate(context, data)
        
        if errors:
            return {
                "valid": False,
                "errors": errors,
            }
        
        return {"valid": True, "errors": {}}
    
    @app.post("/api/v1/validation/rules")
    async def register_validation_rule(
        context: str,
        rule_id: str,
        field_name: str,
        validator_type: str,
        error_message: str,
        required: bool = True,
    ):
        """Registrar regla de validación."""
        # Crear validador según tipo
        validator_func = validation_engine.validators.get(validator_type)
        if not validator_func:
            raise HTTPException(status_code=400, detail=f"Unknown validator type: {validator_type}")
        
        rule = ValidationRule(
            rule_id=rule_id,
            field_name=field_name,
            validator=validator_func,
            error_message=error_message,
            required=required,
        )
        
        validation_engine.register_rule(context, rule)
        return {"success": True, "message": f"Rule {rule_id} registered"}
    
    # Endpoints de throttling
    @app.post("/api/v1/throttle/configure")
    async def configure_throttle(
        identifier: str,
        max_requests: int,
        window_seconds: float,
    ):
        """Configurar throttling."""
        throttler.configure(identifier, max_requests, window_seconds)
        return {"success": True, "message": f"Throttling configured for {identifier}"}
    
    @app.get("/api/v1/throttle/status/{identifier}")
    async def get_throttle_status(identifier: str):
        """Obtener estado de throttling."""
        status = await throttler.get_throttle_status(identifier)
        return status
    
    # Endpoints de circuit breaker
    @app.get("/api/v1/circuit-breaker/{identifier}/state")
    async def get_circuit_breaker_state(identifier: str):
        """Obtener estado del circuit breaker."""
        state = await circuit_breaker.get_state(identifier)
        return state
    
    @app.post("/api/v1/circuit-breaker/{identifier}/reset")
    async def reset_circuit_breaker(identifier: str):
        """Resetear circuit breaker."""
        await circuit_breaker.reset(identifier)
        return {"success": True, "message": f"Circuit breaker {identifier} reset"}
    
    # Intelligent Optimizer
    @app.post("/api/v1/optimizer/record-performance")
    async def record_performance(request: Dict[str, Any]):
        """Registrar rendimiento para optimización."""
        await intelligent_optimizer.record_performance(
            operation=request.get("operation", "unknown"),
            parameters=request.get("parameters", {}),
            metrics=request.get("metrics", {}),
        )
        return {"success": True, "message": "Performance recorded"}
    
    @app.post("/api/v1/optimizer/analyze")
    async def analyze_optimization(request: Dict[str, Any]):
        """Analizar y obtener sugerencias de optimización."""
        suggestions = await intelligent_optimizer.analyze_and_suggest(
            operation=request.get("operation", "unknown"),
            current_parameters=request.get("parameters", {}),
        )
        return {
            "suggestions": [
                {
                    "suggestion_id": s.suggestion_id,
                    "parameter": s.parameter,
                    "current_value": s.current_value,
                    "suggested_value": s.suggested_value,
                    "expected_improvement": s.expected_improvement,
                    "confidence": s.confidence,
                    "reason": s.reason,
                }
                for s in suggestions
            ]
        }
    
    @app.post("/api/v1/optimizer/apply/{suggestion_id}")
    async def apply_optimization(suggestion_id: str):
        """Aplicar optimización sugerida."""
        applied = await intelligent_optimizer.apply_optimization(suggestion_id)
        if not applied:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        return {"success": True, "message": "Optimization applied"}
    
    @app.get("/api/v1/optimizer/applied")
    async def get_applied_optimizations():
        """Obtener optimizaciones aplicadas."""
        return intelligent_optimizer.get_applied_optimizations()
    
    @app.get("/api/v1/optimizer/history")
    async def get_optimization_history():
        """Obtener historial de optimizaciones."""
        return {"history": intelligent_optimizer.get_optimization_history()}
    
    # Adaptive Learning
    @app.post("/api/v1/learning/observe")
    async def observe_learning(request: Dict[str, Any]):
        """Observar resultado para aprendizaje."""
        await adaptive_learning.observe(
            context=request.get("context", {}),
            action=request.get("action", ""),
            outcome=request.get("outcome", {}),
        )
        return {"success": True, "message": "Observation recorded"}
    
    @app.post("/api/v1/learning/predict")
    async def predict_action(request: Dict[str, Any]):
        """Predecir mejor acción basado en aprendizaje."""
        prediction = await adaptive_learning.predict_best_action(
            context=request.get("context", {})
        )
        if not prediction:
            return {"prediction": None, "message": "No patterns found for context"}
        return {"prediction": prediction}
    
    @app.get("/api/v1/learning/patterns")
    async def get_learned_patterns(
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.5,
    ):
        """Obtener patrones aprendidos."""
        patterns = adaptive_learning.get_learned_patterns(pattern_type, min_confidence)
        return {"patterns": patterns, "count": len(patterns)}
    
    # Demand Predictor
    @app.post("/api/v1/demand/record")
    async def record_demand(request: Dict[str, Any]):
        """Registrar demanda actual."""
        await demand_predictor.record_demand(
            resource_type=request.get("resource_type", "unknown"),
            value=request.get("value", 0.0),
        )
        return {"success": True, "message": "Demand recorded"}
    
    @app.post("/api/v1/demand/predict")
    async def predict_demand(request: Dict[str, Any]):
        """Predecir demanda futura."""
        forecast = await demand_predictor.predict_demand(
            resource_type=request.get("resource_type", "unknown"),
            time_horizon_minutes=request.get("time_horizon_minutes", 5),
        )
        
        if not forecast:
            return {"forecast": None, "message": "Insufficient data for prediction"}
        
        return {
            "forecast": {
                "forecast_id": forecast.forecast_id,
                "resource_type": forecast.resource_type,
                "predicted_value": forecast.predicted_value,
                "confidence": forecast.confidence,
                "time_horizon_minutes": forecast.time_horizon_minutes,
                "metadata": forecast.metadata,
            }
        }
    
    @app.post("/api/v1/demand/predict-multiple")
    async def predict_multiple_demands(request: Dict[str, Any]):
        """Predecir demanda para múltiples recursos."""
        predictions = await demand_predictor.predict_multiple_resources(
            resource_types=request.get("resource_types", []),
            time_horizon_minutes=request.get("time_horizon_minutes", 5),
        )
        
        return {
            "predictions": {
                resource_type: {
                    "predicted_value": forecast.predicted_value,
                    "confidence": forecast.confidence,
                    "metadata": forecast.metadata,
                }
                for resource_type, forecast in predictions.items()
            }
        }
    
    @app.get("/api/v1/demand/history/{resource_type}")
    async def get_demand_history(resource_type: str, limit: int = 100):
        """Obtener historial de demanda."""
        history = demand_predictor.get_demand_history(resource_type, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/demand/forecasts")
    async def get_forecast_history(resource_type: Optional[str] = None, limit: int = 100):
        """Obtener historial de pronósticos."""
        forecasts = demand_predictor.get_forecast_history(resource_type, limit)
        return {"forecasts": forecasts, "count": len(forecasts)}
    
    # Intelligent Health Checker
    @app.post("/api/v1/health/register-check")
    async def register_health_check(request: Dict[str, Any]):
        """Registrar check de salud."""
        check_id = request.get("check_id")
        # Nota: En producción necesitarías deserializar la función
        return {
            "success": True,
            "message": "Check registration endpoint - implement check function registration",
            "check_id": check_id,
        }
    
    @app.post("/api/v1/health/register-metric")
    async def register_health_metric(request: Dict[str, Any]):
        """Registrar métrica de salud."""
        intelligent_health.register_metric(
            metric_name=request.get("metric_name"),
            threshold_warning=request.get("threshold_warning", 80.0),
            threshold_critical=request.get("threshold_critical", 95.0),
            unit=request.get("unit", ""),
        )
        return {"success": True, "message": "Metric registered"}
    
    @app.post("/api/v1/health/update-metric")
    async def update_health_metric(request: Dict[str, Any]):
        """Actualizar valor de métrica."""
        await intelligent_health.update_metric(
            metric_name=request.get("metric_name"),
            value=request.get("value", 0.0),
        )
        return {"success": True, "message": "Metric updated"}
    
    @app.get("/api/v1/health/check")
    async def run_health_checks():
        """Ejecutar todos los health checks."""
        results = await intelligent_health.run_all_checks()
        return results
    
    @app.get("/api/v1/health/summary")
    async def get_health_summary():
        """Obtener resumen de salud."""
        summary = intelligent_health.get_health_summary()
        return summary
    
    # Intelligent Cache
    @app.post("/api/v1/cache/get")
    async def cache_get(request: Dict[str, Any]):
        """Obtener valor del caché."""
        key = request.get("key")
        default = request.get("default")
        value = await intelligent_cache.get(key, default)
        return {"found": value is not None, "value": value}
    
    @app.post("/api/v1/cache/set")
    async def cache_set(request: Dict[str, Any]):
        """Guardar valor en caché."""
        await intelligent_cache.set(
            key=request.get("key", ""),
            value=request.get("value"),
            ttl=request.get("ttl"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "message": "Value cached"}
    
    @app.post("/api/v1/cache/invalidate")
    async def cache_invalidate(request: Dict[str, Any]):
        """Invalidar entrada."""
        key = request.get("key")
        if key:
            await intelligent_cache.invalidate(key)
        return {"success": True}
    
    @app.post("/api/v1/cache/invalidate-pattern")
    async def cache_invalidate_pattern(request: Dict[str, Any]):
        """Invalidar por patrón."""
        pattern = request.get("pattern", "")
        count = await intelligent_cache.invalidate_pattern(pattern)
        return {"success": True, "invalidated_count": count}
    
    @app.post("/api/v1/cache/prefetch")
    async def cache_prefetch(request: Dict[str, Any]):
        """Pre-cargar entrada."""
        # Nota: En producción, fetch_func necesitaría ser deserializado
        return {
            "success": True,
            "message": "Prefetch endpoint - implement fetch function registration"
        }
    
    @app.get("/api/v1/cache/stats")
    async def get_cache_stats():
        """Obtener estadísticas de caché."""
        stats = intelligent_cache.get_cache_stats()
        return stats
    
    @app.get("/api/v1/cache/patterns")
    async def get_cache_patterns():
        """Analizar patrones de acceso."""
        patterns = intelligent_cache.analyze_access_patterns()
        return {"patterns": patterns}
    
    @app.post("/api/v1/cache/clear")
    async def cache_clear():
        """Limpiar caché."""
        await intelligent_cache.clear()
        return {"success": True, "message": "Cache cleared"}
    
    # Sentiment Analyzer
    @app.post("/api/v1/sentiment/analyze")
    async def analyze_sentiment(request: Dict[str, Any]):
        """Analizar sentimiento."""
        text = request.get("text", "")
        result = sentiment_analyzer.analyze(text)
        
        return {
            "sentiment": result.sentiment.value,
            "polarity": result.polarity,
            "confidence": result.confidence,
            "emotions": result.emotions,
            "keywords": result.keywords,
        }
    
    @app.post("/api/v1/sentiment/analyze-batch")
    async def analyze_sentiment_batch(request: Dict[str, Any]):
        """Analizar lote de textos."""
        texts = request.get("texts", [])
        results = await sentiment_analyzer.analyze_batch(texts)
        
        return {
            "results": [
                {
                    "sentiment": r.sentiment.value,
                    "polarity": r.polarity,
                    "confidence": r.confidence,
                    "emotions": r.emotions,
                    "keywords": r.keywords,
                }
                for r in results
            ],
            "count": len(results),
        }
    
    @app.post("/api/v1/sentiment/summary")
    async def get_sentiment_summary(request: Dict[str, Any]):
        """Obtener resumen de sentimientos."""
        # Asumir que results viene en el request
        results_data = request.get("results", [])
        # Convertir de vuelta a SentimentResult
        from ..core.sentiment_analyzer import SentimentResult as SR
        
        results = [
            SR(
                sentiment=Sentiment(r["sentiment"]),
                polarity=r["polarity"],
                confidence=r["confidence"],
                emotions=r.get("emotions", {}),
                keywords=r.get("keywords", []),
            )
            for r in results_data
        ]
        
        summary = sentiment_analyzer.get_sentiment_summary(results)
        return summary
    
    # Task Manager
    @app.post("/api/v1/tasks/create")
    async def create_task(request: Dict[str, Any]):
        """Crear tarea."""
        priority_str = request.get("priority", "medium")
        priority = TaskPriority(priority_str)
        
        due_date_str = request.get("due_date")
        due_date = datetime.fromisoformat(due_date_str) if due_date_str else None
        
        task_id = task_manager.create_task(
            task_id=request.get("task_id", ""),
            title=request.get("title", ""),
            description=request.get("description", ""),
            priority=priority,
            assignee=request.get("assignee"),
            due_date=due_date,
            dependencies=request.get("dependencies"),
            metadata=request.get("metadata"),
        )
        
        return {"success": True, "task_id": task_id}
    
    @app.post("/api/v1/tasks/{task_id}/update-status")
    async def update_task_status(task_id: str, request: Dict[str, Any]):
        """Actualizar estado de tarea."""
        status_str = request.get("status")
        status = TaskStatus(status_str)
        
        success = task_manager.update_task_status(task_id, status)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {"success": True, "status": status.value}
    
    @app.post("/api/v1/tasks/{task_id}/update-progress")
    async def update_task_progress(task_id: str, request: Dict[str, Any]):
        """Actualizar progreso de tarea."""
        progress = request.get("progress", 0.0)
        
        success = task_manager.update_task_progress(task_id, progress)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {"success": True, "progress": progress}
    
    @app.get("/api/v1/tasks/{task_id}")
    async def get_task(task_id: str):
        """Obtener tarea."""
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task
    
    @app.get("/api/v1/tasks/status/{status}")
    async def get_tasks_by_status(status: str):
        """Obtener tareas por estado."""
        status_enum = TaskStatus(status)
        tasks = task_manager.get_tasks_by_status(status_enum)
        return {"tasks": tasks, "count": len(tasks)}
    
    @app.get("/api/v1/tasks/assignee/{assignee}")
    async def get_tasks_by_assignee(assignee: str):
        """Obtener tareas por asignado."""
        tasks = task_manager.get_tasks_by_assignee(assignee)
        return {"tasks": tasks, "count": len(tasks)}
    
    @app.get("/api/v1/tasks/overdue")
    async def get_overdue_tasks():
        """Obtener tareas vencidas."""
        tasks = task_manager.get_overdue_tasks()
        return {"tasks": tasks, "count": len(tasks)}
    
    @app.post("/api/v1/task-lists/create")
    async def create_task_list(request: Dict[str, Any]):
        """Crear lista de tareas."""
        list_id = task_manager.create_task_list(
            list_id=request.get("list_id", ""),
            name=request.get("name", ""),
            description=request.get("description", ""),
            metadata=request.get("metadata"),
        )
        return {"success": True, "list_id": list_id}
    
    @app.post("/api/v1/task-lists/{list_id}/add-task")
    async def add_task_to_list(list_id: str, request: Dict[str, Any]):
        """Agregar tarea a lista."""
        success = task_manager.add_task_to_list(list_id, request.get("task_id"))
        if not success:
            raise HTTPException(status_code=404, detail="Task list not found")
        return {"success": True}
    
    @app.get("/api/v1/task-lists/{list_id}")
    async def get_task_list(list_id: str):
        """Obtener lista de tareas."""
        task_list = task_manager.get_task_list(list_id)
        if not task_list:
            raise HTTPException(status_code=404, detail="Task list not found")
        return task_list
    
    @app.get("/api/v1/tasks/summary")
    async def get_task_manager_summary():
        """Obtener resumen del gestor de tareas."""
        summary = task_manager.get_task_manager_summary()
        return summary
    
    # Resource Monitor
    @app.get("/api/v1/resources/current")
    async def get_current_resources():
        """Obtener métricas actuales."""
        metrics = resource_monitor.get_current_metrics()
        return metrics
    
    @app.get("/api/v1/resources/history/{resource_type}")
    async def get_resource_history(resource_type: str, limit: int = 100):
        """Obtener historial de recurso."""
        rt = ResourceType(resource_type)
        history = resource_monitor.get_metric_history(rt, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/resources/statistics/{resource_type}")
    async def get_resource_statistics(resource_type: str, period_minutes: int = 60):
        """Obtener estadísticas de recurso."""
        rt = ResourceType(resource_type)
        stats = resource_monitor.get_metric_statistics(rt, period_minutes)
        return stats
    
    @app.get("/api/v1/resources/alerts")
    async def get_resource_alerts(level: Optional[str] = None):
        """Obtener alertas de recursos."""
        alert_level = AlertLevel(level) if level else None
        alerts = resource_monitor.get_active_alerts(alert_level)
        return {"alerts": alerts, "count": len(alerts)}
    
    @app.post("/api/v1/resources/alerts/{alert_id}/resolve")
    async def resolve_resource_alert(alert_id: str):
        """Resolver alerta."""
        success = resource_monitor.resolve_alert(alert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"success": True, "message": "Alert resolved"}
    
    @app.get("/api/v1/resources/summary")
    async def get_resource_monitor_summary():
        """Obtener resumen del monitor."""
        summary = resource_monitor.get_resource_monitor_summary()
        return summary
    
    # Push Notifications
    @app.post("/api/v1/notifications/send")
    async def send_notification(request: Dict[str, Any]):
        """Enviar notificación."""
        channels_str = request.get("channels", [])
        channels = [NotificationChannel(c) for c in channels_str]
        
        priority_str = request.get("priority", "normal")
        priority = NotificationPriority(priority_str)
        
        notification_id = await push_notifications.send_notification(
            user_id=request.get("user_id", ""),
            title=request.get("title", ""),
            message=request.get("message", ""),
            channels=channels,
            priority=priority,
            metadata=request.get("metadata"),
            action_url=request.get("action_url"),
        )
        
        return {"success": True, "notification_id": notification_id}
    
    @app.post("/api/v1/notifications/subscribe")
    async def subscribe_notifications(request: Dict[str, Any]):
        """Suscribir usuario a notificaciones."""
        channels_str = request.get("channels", [])
        channels = [NotificationChannel(c) for c in channels_str]
        
        push_notifications.subscribe_user(
            user_id=request.get("user_id", ""),
            channels=channels,
            preferences=request.get("preferences"),
        )
        
        return {"success": True, "message": "User subscribed"}
    
    @app.post("/api/v1/notifications/unsubscribe")
    async def unsubscribe_notifications(request: Dict[str, Any]):
        """Desuscribir usuario."""
        push_notifications.unsubscribe_user(request.get("user_id", ""))
        return {"success": True, "message": "User unsubscribed"}
    
    @app.get("/api/v1/notifications/user/{user_id}")
    async def get_user_notifications(user_id: str, unread_only: bool = False, limit: int = 50):
        """Obtener notificaciones de usuario."""
        notifications = push_notifications.get_user_notifications(user_id, unread_only, limit)
        return {"notifications": notifications, "count": len(notifications)}
    
    @app.post("/api/v1/notifications/{notification_id}/read")
    async def mark_notification_read(notification_id: str):
        """Marcar notificación como leída."""
        success = push_notifications.mark_as_read(notification_id)
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")
        return {"success": True, "message": "Notification marked as read"}
    
    @app.get("/api/v1/notifications/stats")
    async def get_notification_stats(user_id: Optional[str] = None):
        """Obtener estadísticas de notificaciones."""
        stats = push_notifications.get_notification_stats(user_id)
        return stats
    
    # Distributed Sync
    @app.post("/api/v1/sync/create-resource")
    async def create_sync_resource(request: Dict[str, Any]):
        """Crear recurso sincronizado."""
        operation_id = distributed_sync.create_resource(
            resource_id=request.get("resource_id", ""),
            resource_type=request.get("resource_type", ""),
            data=request.get("data", {}),
        )
        return {"success": True, "operation_id": operation_id}
    
    @app.post("/api/v1/sync/update-resource")
    async def update_sync_resource(request: Dict[str, Any]):
        """Actualizar recurso sincronizado."""
        operation_id = distributed_sync.update_resource(
            resource_id=request.get("resource_id", ""),
            data=request.get("data", {}),
            expected_version=request.get("expected_version"),
        )
        return {"success": True, "operation_id": operation_id}
    
    @app.post("/api/v1/sync/sync-from-remote")
    async def sync_from_remote(request: Dict[str, Any]):
        """Sincronizar desde operaciones remotas."""
        result = await distributed_sync.sync_from_remote(
            remote_operations=request.get("operations", [])
        )
        return result
    
    @app.post("/api/v1/sync/resolve-conflict")
    async def resolve_sync_conflict(request: Dict[str, Any]):
        """Resolver conflicto de sincronización."""
        resolution_str = request.get("resolution", "last_write_wins")
        resolution = ConflictResolution(resolution_str)
        
        success = distributed_sync.resolve_conflict(
            conflict_id=request.get("conflict_id", ""),
            resolution=resolution,
            resolved_data=request.get("resolved_data"),
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Conflict not found")
        
        return {"success": True, "message": "Conflict resolved"}
    
    @app.get("/api/v1/sync/resource/{resource_id}")
    async def get_sync_resource(resource_id: str):
        """Obtener recurso sincronizado."""
        resource = distributed_sync.get_resource(resource_id)
        if not resource:
            raise HTTPException(status_code=404, detail="Resource not found")
        return resource
    
    @app.get("/api/v1/sync/conflicts")
    async def get_sync_conflicts(resolved: Optional[bool] = None):
        """Obtener conflictos de sincronización."""
        conflicts = distributed_sync.get_conflicts(resolved)
        return {"conflicts": conflicts, "count": len(conflicts)}
    
    @app.get("/api/v1/sync/summary")
    async def get_sync_summary():
        """Obtener resumen de sincronización."""
        summary = distributed_sync.get_sync_summary()
        return summary
    
    # Query Analyzer
    @app.post("/api/v1/queries/record")
    async def record_query(request: Dict[str, Any]):
        """Registrar ejecución de query."""
        query_id = query_analyzer.record_query(
            query_text=request.get("query_text", ""),
            execution_time=request.get("execution_time", 0.0),
            rows_affected=request.get("rows_affected", 0),
            error=request.get("error"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "query_id": query_id}
    
    @app.get("/api/v1/queries/slow")
    async def get_slow_queries(threshold: Optional[float] = None, limit: int = 50):
        """Obtener queries lentas."""
        slow = query_analyzer.get_slow_queries(threshold, limit)
        return {"slow_queries": slow, "count": len(slow)}
    
    @app.get("/api/v1/queries/patterns")
    async def get_query_patterns(query_type: Optional[str] = None, limit: int = 50):
        """Obtener patrones de queries."""
        qtype = QueryType(query_type) if query_type else None
        patterns = query_analyzer.get_query_patterns(qtype, limit)
        return {"patterns": patterns, "count": len(patterns)}
    
    @app.get("/api/v1/queries/statistics")
    async def get_query_statistics(
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ):
        """Obtener estadísticas de queries."""
        start = datetime.fromisoformat(start_time) if start_time else None
        end = datetime.fromisoformat(end_time) if end_time else None
        
        stats = query_analyzer.get_query_statistics(start, end)
        return stats
    
    @app.get("/api/v1/queries/summary")
    async def get_query_analyzer_summary():
        """Obtener resumen del analizador."""
        summary = query_analyzer.get_query_analyzer_summary()
        return summary
    
    # File Manager
    @app.post("/api/v1/files/upload")
    async def upload_file(request: Dict[str, Any]):
        """Subir archivo."""
        file_id = request.get("file_id", f"file_{datetime.now().timestamp()}")
        filename = request.get("filename", "")
        data_bytes = request.get("data")
        
        if not filename or not data_bytes:
            raise HTTPException(status_code=400, detail="filename and data required")
        
        # Convertir data a bytes si viene como string (base64)
        if isinstance(data_bytes, str):
            import base64
            data_bytes = base64.b64decode(data_bytes)
        
        file_id = await file_manager.upload_file(
            file_id=file_id,
            filename=filename,
            data=data_bytes,
            mime_type=request.get("mime_type", ""),
            tags=request.get("tags"),
            metadata=request.get("metadata"),
        )
        
        return {"success": True, "file_id": file_id}
    
    @app.get("/api/v1/files/{file_id}")
    async def download_file(file_id: str):
        """Descargar archivo."""
        data = await file_manager.download_file(file_id)
        if data is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        metadata = file_manager.get_file_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        from fastapi.responses import Response
        
        return Response(
            content=data,
            media_type=metadata.get("mime_type", "application/octet-stream"),
            headers={
                "Content-Disposition": f'attachment; filename="{metadata["filename"]}"',
            },
        )
    
    @app.get("/api/v1/files/{file_id}/metadata")
    async def get_file_metadata(file_id: str):
        """Obtener metadatos de archivo."""
        metadata = file_manager.get_file_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        return metadata
    
    @app.get("/api/v1/files/{file_id}/versions")
    async def get_file_versions(file_id: str):
        """Obtener versiones de archivo."""
        versions = file_manager.get_file_versions(file_id)
        return {"versions": versions, "count": len(versions)}
    
    @app.get("/api/v1/files/search")
    async def search_files(
        tags: Optional[str] = None,
        file_type: Optional[str] = None,
        limit: int = 100,
    ):
        """Buscar archivos."""
        tags_list = tags.split(",") if tags else None
        file_type_enum = FileType(file_type) if file_type else None
        
        files = file_manager.search_files(tags_list, file_type_enum, limit)
        return {"files": files, "count": len(files)}
    
    @app.delete("/api/v1/files/{file_id}")
    async def delete_file(file_id: str, permanent: bool = False):
        """Eliminar archivo."""
        success = await file_manager.delete_file(file_id, permanent)
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        return {"success": True, "message": "File deleted"}
    
    @app.post("/api/v1/files/{file_id}/restore")
    async def restore_file(file_id: str):
        """Restaurar archivo eliminado."""
        success = await file_manager.restore_file(file_id)
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        return {"success": True, "message": "File restored"}
    
    @app.get("/api/v1/files/summary")
    async def get_file_manager_summary():
        """Obtener resumen del gestor."""
        summary = file_manager.get_file_manager_summary()
        return summary
    
    # Data Compression
    @app.post("/api/v1/compression/compress")
    async def compress_data(request: Dict[str, Any]):
        """Comprimir datos."""
        data = request.get("data")
        algorithm_str = request.get("algorithm", "gzip")
        algorithm = CompressionAlgorithm(algorithm_str)
        
        if not data:
            raise HTTPException(status_code=400, detail="data required")
        
        # Convertir a bytes si viene como string (base64)
        if isinstance(data, str):
            import base64
            data = base64.b64decode(data)
        
        compressed = data_compression.compress(data, algorithm)
        
        # Convertir a base64 para respuesta
        import base64
        compressed_b64 = base64.b64encode(compressed).decode()
        
        return {
            "success": True,
            "compressed_data": compressed_b64,
            "original_size": len(data),
            "compressed_size": len(compressed),
            "compression_ratio": len(compressed) / len(data) if len(data) > 0 else 1.0,
        }
    
    @app.post("/api/v1/compression/decompress")
    async def decompress_data(request: Dict[str, Any]):
        """Descomprimir datos."""
        compressed_data = request.get("compressed_data")
        algorithm_str = request.get("algorithm", "gzip")
        algorithm = CompressionAlgorithm(algorithm_str)
        
        if not compressed_data:
            raise HTTPException(status_code=400, detail="compressed_data required")
        
        # Convertir de base64
        import base64
        compressed_bytes = base64.b64decode(compressed_data)
        
        data = data_compression.decompress(compressed_bytes, algorithm)
        data_b64 = base64.b64encode(data).decode()
        
        return {
            "success": True,
            "data": data_b64,
            "decompressed_size": len(data),
        }
    
    @app.post("/api/v1/compression/find-best")
    async def find_best_compression(request: Dict[str, Any]):
        """Encontrar mejor algoritmo de compresión."""
        data = request.get("data")
        
        if not data:
            raise HTTPException(status_code=400, detail="data required")
        
        if isinstance(data, str):
            import base64
            data = base64.b64decode(data)
        
        best_algorithm = data_compression.find_best_algorithm(data)
        
        return {
            "success": True,
            "best_algorithm": best_algorithm.value,
        }
    
    @app.get("/api/v1/compression/stats")
    async def get_compression_stats():
        """Obtener estadísticas de compresión."""
        stats = data_compression.get_compression_stats()
        return stats
    
    # Incremental Backup
    @app.post("/api/v1/backup/create")
    async def create_backup(request: Dict[str, Any]):
        """Crear backup."""
        backup_id = request.get("backup_id", f"backup_{datetime.now().timestamp()}")
        data = request.get("data")
        backup_type_str = request.get("backup_type", "incremental")
        backup_type = BackupType(backup_type_str)
        
        if not data:
            raise HTTPException(status_code=400, detail="data required")
        
        if isinstance(data, str):
            import base64
            data = base64.b64decode(data)
        elif isinstance(data, dict):
            data = json.dumps(data).encode()
        
        backup_id = await incremental_backup.create_backup(
            backup_id=backup_id,
            data=data,
            backup_type=backup_type,
            parent_backup_id=request.get("parent_backup_id"),
            metadata=request.get("metadata"),
        )
        
        return {"success": True, "backup_id": backup_id}
    
    @app.post("/api/v1/backup/restore")
    async def restore_backup(request: Dict[str, Any]):
        """Restaurar backup."""
        backup_id = request.get("backup_id")
        
        if not backup_id:
            raise HTTPException(status_code=400, detail="backup_id required")
        
        data = await incremental_backup.restore_backup(backup_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Backup not found")
        
        import base64
        data_b64 = base64.b64encode(data).decode()
        
        return {"success": True, "data": data_b64, "size": len(data)}
    
    @app.get("/api/v1/backup/{backup_id}")
    async def get_backup_info(backup_id: str):
        """Obtener información de backup."""
        backup = incremental_backup.get_backup(backup_id)
        if not backup:
            raise HTTPException(status_code=404, detail="Backup not found")
        return backup
    
    @app.get("/api/v1/backup/list")
    async def list_backups(backup_type: Optional[str] = None, limit: int = 100):
        """Listar backups."""
        btype = BackupType(backup_type) if backup_type else None
        backups = incremental_backup.list_backups(btype, limit)
        return {"backups": backups, "count": len(backups)}
    
    @app.post("/api/v1/backup/create-set")
    async def create_backup_set(request: Dict[str, Any]):
        """Crear conjunto de backups."""
        set_id = await incremental_backup.create_backup_set(
            set_id=request.get("set_id", f"set_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            backup_ids=request.get("backup_ids", []),
            metadata=request.get("metadata"),
        )
        return {"success": True, "set_id": set_id}
    
    @app.get("/api/v1/backup/summary")
    async def get_backup_summary():
        """Obtener resumen de backups."""
        summary = incremental_backup.get_backup_summary()
        return summary
    
    # Network Analyzer
    @app.post("/api/v1/network/record")
    async def record_network_metric(request: Dict[str, Any]):
        """Registrar métrica de red."""
        metric_id = network_analyzer.record_metric(
            endpoint=request.get("endpoint", ""),
            latency=request.get("latency", 0.0),
            bytes_sent=request.get("bytes_sent", 0),
            bytes_received=request.get("bytes_received", 0),
            status_code=request.get("status_code"),
            error=request.get("error"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "metric_id": metric_id}
    
    @app.get("/api/v1/network/endpoint/{endpoint}")
    async def get_network_endpoint_stats(endpoint: str):
        """Obtener estadísticas de endpoint."""
        stats = network_analyzer.get_endpoint_stats(endpoint)
        if not stats:
            raise HTTPException(status_code=404, detail="Endpoint not found")
        return stats
    
    @app.get("/api/v1/network/slow-endpoints")
    async def get_slow_network_endpoints(threshold: float = 1.0, limit: int = 10):
        """Obtener endpoints lentos."""
        slow = network_analyzer.get_slow_endpoints(threshold, limit)
        return {"slow_endpoints": slow, "count": len(slow)}
    
    @app.get("/api/v1/network/events")
    async def get_network_events(
        event_type: Optional[str] = None,
        endpoint: Optional[str] = None,
        limit: int = 100,
    ):
        """Obtener eventos de red."""
        etype = NetworkEventType(event_type) if event_type else None
        events = network_analyzer.get_network_events(etype, endpoint, limit)
        return {"events": events, "count": len(events)}
    
    @app.get("/api/v1/network/summary")
    async def get_network_summary():
        """Obtener resumen de red."""
        summary = network_analyzer.get_network_summary()
        return summary
    
    # Config Manager
    @app.post("/api/v1/config/register")
    async def register_config(request: Dict[str, Any]):
        """Registrar configuración."""
        format_str = request.get("format", "json")
        config_format = ConfigFormat(format_str)
        
        config_id = config_manager.register_config(
            config_id=request.get("config_id", ""),
            name=request.get("name", ""),
            data=request.get("data", {}),
            format=config_format,
            metadata=request.get("metadata"),
        )
        
        return {"success": True, "config_id": config_id}
    
    @app.get("/api/v1/config/{config_id}")
    async def get_config(config_id: str, version: Optional[int] = None):
        """Obtener configuración."""
        config = config_manager.get_config(config_id, version)
        if not config:
            raise HTTPException(status_code=404, detail="Config not found")
        return config
    
    @app.get("/api/v1/config/{config_id}/history")
    async def get_config_history(config_id: str, limit: int = 50):
        """Obtener historial de configuración."""
        history = config_manager.get_config_history(config_id, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/config/changes")
    async def get_config_changes(config_id: Optional[str] = None, limit: int = 100):
        """Obtener cambios de configuración."""
        changes = config_manager.get_config_changes(config_id, limit)
        return {"changes": changes, "count": len(changes)}
    
    @app.post("/api/v1/config/{config_id}/rollback")
    async def rollback_config(config_id: str, request: Dict[str, Any]):
        """Revertir configuración."""
        target_version = request.get("target_version")
        
        if target_version is None:
            raise HTTPException(status_code=400, detail="target_version required")
        
        success = config_manager.rollback_config(config_id, target_version)
        if not success:
            raise HTTPException(status_code=404, detail="Config or version not found")
        
        return {"success": True, "message": f"Config rolled back to version {target_version}"}
    
    @app.post("/api/v1/config/{config_id}/subscribe")
    async def subscribe_config_changes(config_id: str):
        """Suscribirse a cambios de configuración."""
        # Nota: En producción, el listener necesitaría ser registrado de otra manera
        return {
            "success": True,
            "message": "Config change subscription endpoint - implement listener registration"
        }
    
    @app.get("/api/v1/config/summary")
    async def get_config_manager_summary():
        """Obtener resumen del gestor."""
        summary = config_manager.get_config_manager_summary()
        return summary
    
    # MFA Authentication
    @app.post("/api/v1/mfa/setup/totp")
    async def setup_totp(request: Dict[str, Any]):
        """Configurar TOTP para usuario."""
        user_id = request.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")
        
        result = mfa_authentication.setup_totp(user_id)
        return {"success": True, "data": result}
    
    @app.post("/api/v1/mfa/setup/sms")
    async def setup_sms(request: Dict[str, Any]):
        """Configurar SMS para usuario."""
        user_id = request.get("user_id")
        phone = request.get("phone")
        
        if not user_id or not phone:
            raise HTTPException(status_code=400, detail="user_id and phone required")
        
        success = mfa_authentication.setup_sms(user_id, phone)
        return {"success": success}
    
    @app.post("/api/v1/mfa/setup/email")
    async def setup_email(request: Dict[str, Any]):
        """Configurar Email para usuario."""
        user_id = request.get("user_id")
        email = request.get("email")
        
        if not user_id or not email:
            raise HTTPException(status_code=400, detail="user_id and email required")
        
        success = mfa_authentication.setup_email(user_id, email)
        return {"success": success}
    
    @app.post("/api/v1/mfa/initiate")
    async def initiate_mfa(request: Dict[str, Any]):
        """Iniciar proceso de MFA."""
        user_id = request.get("user_id")
        method_str = request.get("method", "totp")
        method = MFAMethod(method_str)
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")
        
        session_id = await mfa_authentication.initiate_mfa(user_id, method)
        return {"success": True, "session_id": session_id}
    
    @app.post("/api/v1/mfa/verify")
    async def verify_mfa(request: Dict[str, Any]):
        """Verificar código MFA."""
        session_id = request.get("session_id")
        code = request.get("code")
        
        if not session_id or not code:
            raise HTTPException(status_code=400, detail="session_id and code required")
        
        verified = await mfa_authentication.verify_mfa(session_id, code)
        return {"success": verified, "verified": verified}
    
    @app.get("/api/v1/mfa/status/{user_id}")
    async def get_mfa_status(user_id: str):
        """Obtener estado MFA del usuario."""
        status = mfa_authentication.get_user_mfa_status(user_id)
        return status
    
    # Advanced Rate Limiter
    @app.post("/api/v1/rate-limit/create-rule")
    async def create_rate_limit_rule(request: Dict[str, Any]):
        """Crear regla de rate limiting."""
        strategy_str = request.get("strategy", "fixed_window")
        strategy = RateLimitStrategy(strategy_str)
        
        rule_id = advanced_rate_limiter.create_rule(
            rule_id=request.get("rule_id", f"rule_{datetime.now().timestamp()}"),
            identifier=request.get("identifier", ""),
            strategy=strategy,
            max_requests=request.get("max_requests", 100),
            window_seconds=request.get("window_seconds", 60),
            burst_size=request.get("burst_size"),
            tokens_per_second=request.get("tokens_per_second"),
            metadata=request.get("metadata"),
        )
        
        return {"success": True, "rule_id": rule_id}
    
    @app.post("/api/v1/rate-limit/check")
    async def check_rate_limit(request: Dict[str, Any]):
        """Verificar rate limit."""
        identifier = request.get("identifier")
        rule_id = request.get("rule_id")
        
        if not identifier:
            raise HTTPException(status_code=400, detail="identifier required")
        
        allowed, info = await advanced_rate_limiter.check_rate_limit(identifier, rule_id)
        
        return {
            "allowed": allowed,
            **info,
        }
    
    @app.post("/api/v1/rate-limit/block")
    async def block_identifier(request: Dict[str, Any]):
        """Bloquear identificador."""
        identifier = request.get("identifier")
        duration_seconds = request.get("duration_seconds", 3600)
        
        if not identifier:
            raise HTTPException(status_code=400, detail="identifier required")
        
        await advanced_rate_limiter.block_identifier(identifier, duration_seconds)
        return {"success": True, "message": f"Blocked {identifier} for {duration_seconds} seconds"}
    
    @app.get("/api/v1/rate-limit/violations")
    async def get_rate_limit_violations(identifier: Optional[str] = None, limit: int = 100):
        """Obtener violaciones de rate limit."""
        violations = advanced_rate_limiter.get_violations(identifier, limit)
        return {"violations": violations, "count": len(violations)}
    
    @app.get("/api/v1/rate-limit/summary")
    async def get_rate_limiter_summary():
        """Obtener resumen del rate limiter."""
        summary = advanced_rate_limiter.get_rate_limiter_summary()
        return summary
    
    # User Behavior Analyzer
    @app.post("/api/v1/behavior/record")
    async def record_user_action(request: Dict[str, Any]):
        """Registrar acción de usuario."""
        action_id = user_behavior_analyzer.record_action(
            user_id=request.get("user_id", ""),
            action_type=request.get("action_type", ""),
            ip_address=request.get("ip_address"),
            user_agent=request.get("user_agent"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "action_id": action_id}
    
    @app.get("/api/v1/behavior/profile/{user_id}")
    async def get_user_profile(user_id: str):
        """Obtener perfil de usuario."""
        profile = user_behavior_analyzer.get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        return profile
    
    @app.get("/api/v1/behavior/high-risk")
    async def get_high_risk_users(threshold: float = 0.7, limit: int = 50):
        """Obtener usuarios de alto riesgo."""
        high_risk = user_behavior_analyzer.get_high_risk_users(threshold, limit)
        return {"users": high_risk, "count": len(high_risk)}
    
    @app.get("/api/v1/behavior/anomalies")
    async def get_behavior_anomalies(user_id: Optional[str] = None, limit: int = 100):
        """Obtener anomalías de comportamiento."""
        anomalies = user_behavior_analyzer.get_anomalies(user_id, limit)
        return {"anomalies": anomalies, "count": len(anomalies)}
    
    @app.get("/api/v1/behavior/summary")
    async def get_behavior_analyzer_summary():
        """Obtener resumen del analizador."""
        summary = user_behavior_analyzer.get_behavior_analyzer_summary()
        return summary
    
    # Event Stream
    @app.post("/api/v1/events/publish")
    async def publish_event(request: Dict[str, Any]):
        """Publicar evento."""
        event_type_str = request.get("event_type", "custom")
        event_type = EventType(event_type_str)
        
        event_id = event_stream.publish(
            event_type=event_type,
            source=request.get("source", "system"),
            data=request.get("data", {}),
            metadata=request.get("metadata"),
        )
        
        return {"success": True, "event_id": event_id}
    
    @app.get("/api/v1/events")
    async def get_events(
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ):
        """Obtener eventos."""
        etype = EventType(event_type) if event_type else None
        start = datetime.fromisoformat(start_time) if start_time else None
        end = datetime.fromisoformat(end_time) if end_time else None
        
        events = event_stream.get_events(etype, source, start, end, limit)
        return {"events": events, "count": len(events)}
    
    @app.post("/api/v1/events/subscribe")
    async def subscribe_events(request: Dict[str, Any]):
        """Suscribirse a eventos."""
        # Nota: En producción, el handler necesitaría ser registrado de otra manera
        return {
            "success": True,
            "message": "Event subscription endpoint - implement handler registration"
        }
    
    @app.get("/api/v1/events/summary")
    async def get_event_stream_summary():
        """Obtener resumen del stream."""
        summary = event_stream.get_event_stream_summary()
        return summary
    
    # Security Analyzer
    @app.post("/api/v1/security/analyze")
    async def analyze_security(request: Dict[str, Any]):
        """Analizar entrada en busca de amenazas."""
        input_data = request.get("input_data", "")
        source = request.get("source", "unknown")
        
        if not input_data:
            raise HTTPException(status_code=400, detail="input_data required")
        
        threats = security_analyzer.analyze_input(input_data, source, request.get("context"))
        
        return {
            "threats_detected": len(threats),
            "threats": [
                {
                    "threat_id": t.threat_id,
                    "threat_type": t.threat_type.value,
                    "threat_level": t.threat_level.value,
                    "description": t.description,
                }
                for t in threats
            ],
        }
    
    @app.post("/api/v1/security/block")
    async def block_source(request: Dict[str, Any]):
        """Bloquear fuente."""
        source = request.get("source")
        duration_seconds = request.get("duration_seconds", 3600)
        
        if not source:
            raise HTTPException(status_code=400, detail="source required")
        
        security_analyzer.block_source(source, duration_seconds)
        return {"success": True, "message": f"Blocked {source} for {duration_seconds} seconds"}
    
    @app.get("/api/v1/security/threats")
    async def get_security_threats(
        threat_type: Optional[str] = None,
        threat_level: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 100,
    ):
        """Obtener amenazas."""
        ttype = ThreatType(threat_type) if threat_type else None
        tlevel = ThreatLevel(threat_level) if threat_level else None
        
        threats = security_analyzer.get_threats(ttype, tlevel, resolved, limit)
        return {"threats": threats, "count": len(threats)}
    
    @app.post("/api/v1/security/threats/{threat_id}/resolve")
    async def resolve_threat(threat_id: str):
        """Resolver amenaza."""
        success = security_analyzer.resolve_threat(threat_id)
        if not success:
            raise HTTPException(status_code=404, detail="Threat not found")
        return {"success": True, "message": "Threat resolved"}
    
    @app.get("/api/v1/security/summary")
    async def get_security_analyzer_summary():
        """Obtener resumen del analizador."""
        summary = security_analyzer.get_security_analyzer_summary()
        return summary
    
    # Session Manager
    @app.post("/api/v1/sessions/create")
    async def create_session_manager(request: Dict[str, Any]):
        """Crear sesión en el gestor."""
        session_id = request.get("session_id", f"session_{datetime.now().timestamp()}")
        user_id = request.get("user_id", "")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")
        
        session_id = session_manager.create_session(session_id, user_id, request.get("metadata"))
        return {"success": True, "session_id": session_id}
    
    @app.post("/api/v1/sessions/{session_id}/activity")
    async def update_session_activity(
        session_id: str,
        request: Dict[str, Any],
    ):
        """Actualizar actividad de sesión."""
        session_manager.update_session_activity(
            session_id,
            message_count=request.get("message_count", 0),
            response_time=request.get("response_time"),
        )
        return {"success": True}
    
    @app.post("/api/v1/sessions/{session_id}/status")
    async def update_session_status(
        session_id: str,
        request: Dict[str, Any],
    ):
        """Actualizar estado de sesión."""
        status_str = request.get("status", "active")
        status = SessionStatus(status_str)
        
        session_manager.update_session_status(session_id, status)
        return {"success": True}
    
    @app.get("/api/v1/sessions/{session_id}")
    async def get_session_info(session_id: str):
        """Obtener información de sesión."""
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    
    @app.get("/api/v1/sessions/{session_id}/analytics")
    async def get_session_analytics(session_id: str):
        """Obtener analíticas de sesión."""
        analytics = session_manager.get_session_analytics(session_id)
        if not analytics:
            raise HTTPException(status_code=404, detail="Session analytics not found")
        return analytics
    
    @app.get("/api/v1/sessions/active")
    async def get_active_sessions(limit: int = 100):
        """Obtener sesiones activas."""
        active = session_manager.get_active_sessions(limit)
        return {"sessions": active, "count": len(active)}
    
    @app.post("/api/v1/sessions/cleanup")
    async def cleanup_sessions(request: Dict[str, Any]):
        """Limpiar sesiones expiradas."""
        max_idle_seconds = request.get("max_idle_seconds", 3600)
        expired = session_manager.cleanup_expired_sessions(max_idle_seconds)
        return {"success": True, "expired_count": len(expired), "expired_sessions": expired}
    
    @app.get("/api/v1/sessions/summary")
    async def get_session_manager_summary():
        """Obtener resumen del gestor."""
        summary = session_manager.get_session_manager_summary()
        return summary
    
    # Real-time Metrics
    @app.post("/api/v1/metrics/record")
    async def record_metric(request: Dict[str, Any]):
        """Registrar métrica."""
        metric_type_str = request.get("metric_type", "gauge")
        metric_type = MetricType(metric_type_str)
        
        metric_id = realtime_metrics.record_metric(
            metric_name=request.get("metric_name", ""),
            value=request.get("value", 0.0),
            metric_type=metric_type,
            labels=request.get("labels"),
            metadata=request.get("metadata"),
        )
        
        return {"success": True, "metric_id": metric_id}
    
    @app.get("/api/v1/metrics/aggregates/{metric_name}")
    async def get_metric_aggregates(metric_name: str, labels: Optional[str] = None):
        """Obtener agregados de métrica."""
        import json
        labels_dict = json.loads(labels) if labels else None
        
        aggregates = realtime_metrics.get_metric_aggregates(metric_name, labels_dict)
        if not aggregates:
            raise HTTPException(status_code=404, detail="Metric aggregates not found")
        return aggregates
    
    @app.get("/api/v1/metrics")
    async def get_metrics(
        metric_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 1000,
    ):
        """Obtener métricas."""
        start = datetime.fromisoformat(start_time) if start_time else None
        end = datetime.fromisoformat(end_time) if end_time else None
        
        metrics = realtime_metrics.get_metrics(metric_name, start, end, limit)
        return {"metrics": metrics, "count": len(metrics)}
    
    @app.post("/api/v1/metrics/alerts/create")
    async def create_metric_alert(request: Dict[str, Any]):
        """Crear alerta de métrica."""
        alert_id = realtime_metrics.create_alert(
            alert_id=request.get("alert_id", f"alert_{datetime.now().timestamp()}"),
            metric_name=request.get("metric_name", ""),
            condition=request.get("condition", "gt"),
            threshold=request.get("threshold", 0.0),
        )
        
        return {"success": True, "alert_id": alert_id}
    
    @app.get("/api/v1/metrics/summary")
    async def get_realtime_metrics_summary():
        """Obtener resumen de métricas."""
        summary = realtime_metrics.get_realtime_metrics_summary()
        return summary
    
    # Auto Optimizer
    @app.post("/api/v1/optimizer/register-parameter")
    async def register_parameter(request: Dict[str, Any]):
        """Registrar parámetro para optimización."""
        target_str = request.get("target", "latency")
        target = OptimizationTarget(target_str)
        
        auto_optimizer.register_parameter(
            parameter_name=request.get("parameter_name", ""),
            min_value=request.get("min_value"),
            max_value=request.get("max_value"),
            current_value=request.get("current_value"),
            target=target,
            step=request.get("step"),
        )
        
        return {"success": True}
    
    @app.post("/api/v1/optimizer/record-performance")
    async def record_performance(request: Dict[str, Any]):
        """Registrar métrica de performance."""
        auto_optimizer.record_performance(
            metric_name=request.get("metric_name", ""),
            value=request.get("value", 0.0),
            context=request.get("context"),
        )
        
        return {"success": True}
    
    @app.get("/api/v1/optimizer/parameter/{parameter_name}")
    async def get_parameter_value(parameter_name: str):
        """Obtener valor de parámetro."""
        value = auto_optimizer.get_parameter_value(parameter_name)
        if value is None:
            raise HTTPException(status_code=404, detail="Parameter not found")
        return {"parameter_name": parameter_name, "value": value}
    
    @app.get("/api/v1/optimizer/optimizations")
    async def get_optimizations(
        parameter_name: Optional[str] = None,
        target: Optional[str] = None,
        limit: int = 100,
    ):
        """Obtener optimizaciones."""
        ttarget = OptimizationTarget(target) if target else None
        
        optimizations = auto_optimizer.get_optimizations(parameter_name, ttarget, limit)
        return {"optimizations": optimizations, "count": len(optimizations)}
    
    @app.get("/api/v1/optimizer/summary")
    async def get_auto_optimizer_summary():
        """Obtener resumen del optimizador."""
        summary = auto_optimizer.get_auto_optimizer_summary()
        return summary
    
    # Adaptive Rate Controller
    @app.post("/api/v1/adaptive-rate/register")
    async def register_adaptive_rate(request: Dict[str, Any]):
        """Registrar identificador para control de tasa adaptativo."""
        identifier = adaptive_rate_controller.register_identifier(
            identifier=request.get("identifier", ""),
            base_rate=request.get("base_rate"),
            min_rate=request.get("min_rate"),
            max_rate=request.get("max_rate"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "identifier": identifier}
    
    @app.post("/api/v1/adaptive-rate/record")
    async def record_adaptive_rate_request(request: Dict[str, Any]):
        """Registrar petición para ajuste de tasa."""
        await adaptive_rate_controller.record_request(
            identifier=request.get("identifier", ""),
            success=request.get("success", True),
            response_time=request.get("response_time", 0.0),
            system_load=request.get("system_load"),
        )
        return {"success": True}
    
    @app.post("/api/v1/adaptive-rate/check")
    async def check_adaptive_rate_limit(request: Dict[str, Any]):
        """Verificar rate limit adaptativo."""
        allowed, info = await adaptive_rate_controller.check_rate_limit(
            request.get("identifier", "")
        )
        return {"allowed": allowed, **info}
    
    @app.get("/api/v1/adaptive-rate/{identifier}")
    async def get_adaptive_rate_limit(identifier: str):
        """Obtener límite de tasa adaptativo."""
        rate_limit = adaptive_rate_controller.get_rate_limit(identifier)
        if not rate_limit:
            raise HTTPException(status_code=404, detail="Identifier not found")
        return rate_limit
    
    @app.get("/api/v1/adaptive-rate/{identifier}/history")
    async def get_adaptive_rate_history(identifier: str, limit: int = 100):
        """Obtener historial de ajustes."""
        history = adaptive_rate_controller.get_adjustment_history(identifier, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/adaptive-rate/summary")
    async def get_adaptive_rate_summary():
        """Obtener resumen del controlador."""
        return adaptive_rate_controller.get_adaptive_rate_controller_summary()
    
    # Smart Retry Manager
    @app.post("/api/v1/retry/create")
    async def create_retry_operation(request: Dict[str, Any]):
        """Crear operación con reintentos."""
        from ..core.smart_retry_manager import RetryConfig, RetryStrategy
        
        retry_strategy_str = request.get("strategy", "exponential_backoff")
        retry_strategy = RetryStrategy(retry_strategy_str)
        
        config = RetryConfig(
            max_attempts=request.get("max_attempts", 3),
            initial_delay=request.get("initial_delay", 1.0),
            max_delay=request.get("max_delay", 60.0),
            backoff_multiplier=request.get("backoff_multiplier", 2.0),
            strategy=retry_strategy,
            retryable_errors=request.get("retryable_errors", []),
        )
        
        operation_id = smart_retry_manager.create_retry_operation(
            operation_id=request.get("operation_id", f"op_{datetime.now().timestamp()}"),
            operation_type=request.get("operation_type", "unknown"),
            config=config,
            metadata=request.get("metadata"),
        )
        return {"success": True, "operation_id": operation_id}
    
    @app.get("/api/v1/retry/{operation_id}")
    async def get_retry_operation(operation_id: str):
        """Obtener información de operación."""
        operation = smart_retry_manager.get_operation(operation_id)
        if not operation:
            raise HTTPException(status_code=404, detail="Operation not found")
        return operation
    
    @app.get("/api/v1/retry/patterns/{operation_type}")
    async def get_retry_patterns(operation_type: str):
        """Obtener patrones aprendidos."""
        patterns = smart_retry_manager.get_learning_patterns(operation_type)
        return patterns
    
    @app.get("/api/v1/retry/summary")
    async def get_retry_manager_summary():
        """Obtener resumen del gestor."""
        return smart_retry_manager.get_smart_retry_manager_summary()
    
    # Distributed Lock Manager
    @app.post("/api/v1/locks/acquire")
    async def acquire_distributed_lock(request: Dict[str, Any]):
        """Adquirir lock distribuido."""
        lock_id = await distributed_lock_manager.acquire_lock(
            resource_id=request.get("resource_id", ""),
            owner_id=request.get("owner_id", ""),
            ttl_seconds=request.get("ttl_seconds"),
            wait_timeout=request.get("wait_timeout"),
            metadata=request.get("metadata"),
        )
        if not lock_id:
            raise HTTPException(status_code=409, detail="Lock not available")
        return {"success": True, "lock_id": lock_id}
    
    @app.post("/api/v1/locks/{lock_id}/release")
    async def release_distributed_lock(lock_id: str, request: Dict[str, Any]):
        """Liberar lock distribuido."""
        success = await distributed_lock_manager.release_lock(
            lock_id=lock_id,
            owner_id=request.get("owner_id"),
        )
        if not success:
            raise HTTPException(status_code=404, detail="Lock not found or already released")
        return {"success": True}
    
    @app.post("/api/v1/locks/{lock_id}/renew")
    async def renew_distributed_lock(lock_id: str, request: Dict[str, Any]):
        """Renovar lock distribuido."""
        success = await distributed_lock_manager.renew_lock(
            lock_id=lock_id,
            owner_id=request.get("owner_id"),
            ttl_seconds=request.get("ttl_seconds"),
        )
        if not success:
            raise HTTPException(status_code=404, detail="Lock not found")
        return {"success": True}
    
    @app.get("/api/v1/locks/{lock_id}")
    async def get_distributed_lock(lock_id: str):
        """Obtener información de lock."""
        lock = distributed_lock_manager.get_lock(lock_id)
        if not lock:
            raise HTTPException(status_code=404, detail="Lock not found")
        return lock
    
    @app.get("/api/v1/locks/resource/{resource_id}")
    async def get_resource_lock(resource_id: str):
        """Obtener lock de recurso."""
        lock = distributed_lock_manager.get_resource_lock(resource_id)
        if not lock:
            raise HTTPException(status_code=404, detail="No lock for resource")
        return lock
    
    @app.post("/api/v1/locks/cleanup")
    async def cleanup_expired_locks():
        """Limpiar locks expirados."""
        expired_count = await distributed_lock_manager.cleanup_expired_locks()
        return {"success": True, "expired_count": expired_count}
    
    @app.get("/api/v1/locks/summary")
    async def get_lock_manager_summary():
        """Obtener resumen del gestor."""
        return distributed_lock_manager.get_distributed_lock_manager_summary()
    
    # Data Pipeline Manager
    @app.post("/api/v1/pipelines/create")
    async def create_data_pipeline(request: Dict[str, Any]):
        """Crear pipeline de datos."""
        pipeline_id = data_pipeline_manager.create_pipeline(
            pipeline_id=request.get("pipeline_id", f"pipeline_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            description=request.get("description", ""),
            metadata=request.get("metadata"),
        )
        return {"success": True, "pipeline_id": pipeline_id}
    
    @app.post("/api/v1/pipelines/{pipeline_id}/add-stage")
    async def add_pipeline_stage(pipeline_id: str, request: Dict[str, Any]):
        """Agregar stage a pipeline."""
        # Nota: En producción, processor necesitaría ser deserializado
        return {
            "success": True,
            "message": "Stage registration endpoint - implement processor function registration"
        }
    
    @app.post("/api/v1/pipelines/{pipeline_id}/execute")
    async def execute_data_pipeline(pipeline_id: str, request: Dict[str, Any]):
        """Ejecutar pipeline."""
        execution_id = await data_pipeline_manager.execute_pipeline(
            pipeline_id=pipeline_id,
            input_data=request.get("input_data"),
            execution_id=request.get("execution_id"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "execution_id": execution_id}
    
    @app.get("/api/v1/pipelines/{pipeline_id}")
    async def get_data_pipeline(pipeline_id: str):
        """Obtener información de pipeline."""
        pipeline = data_pipeline_manager.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        return pipeline
    
    @app.get("/api/v1/pipelines/executions/{execution_id}")
    async def get_pipeline_execution(execution_id: str):
        """Obtener información de ejecución."""
        execution = data_pipeline_manager.get_execution(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        return execution
    
    @app.post("/api/v1/pipelines/executions/{execution_id}/cancel")
    async def cancel_pipeline_execution(execution_id: str):
        """Cancelar ejecución."""
        success = await data_pipeline_manager.cancel_execution(execution_id)
        if not success:
            raise HTTPException(status_code=404, detail="Execution not found or cannot be cancelled")
        return {"success": True}
    
    @app.get("/api/v1/pipelines/executions/history")
    async def get_pipeline_execution_history(pipeline_id: Optional[str] = None, limit: int = 100):
        """Obtener historial de ejecuciones."""
        history = data_pipeline_manager.get_pipeline_execution_history(pipeline_id, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/pipelines/summary")
    async def get_pipeline_manager_summary():
        """Obtener resumen del gestor."""
        return data_pipeline_manager.get_data_pipeline_manager_summary()
    
    # Event Scheduler
    @app.post("/api/v1/scheduler/schedule")
    async def schedule_event(request: Dict[str, Any]):
        """Programar evento."""
        schedule_type_str = request.get("schedule_type", "interval")
        schedule_type = ScheduleType(schedule_type_str)
        
        # Nota: En producción, handler necesitaría ser deserializado
        return {
            "success": True,
            "message": "Event scheduling endpoint - implement handler function registration"
        }
    
    @app.post("/api/v1/scheduler/{event_id}/pause")
    async def pause_scheduled_event(event_id: str):
        """Pausar evento."""
        success = await event_scheduler.pause_event(event_id)
        if not success:
            raise HTTPException(status_code=404, detail="Event not found")
        return {"success": True}
    
    @app.post("/api/v1/scheduler/{event_id}/resume")
    async def resume_scheduled_event(event_id: str):
        """Reanudar evento."""
        success = await event_scheduler.resume_event(event_id)
        if not success:
            raise HTTPException(status_code=404, detail="Event not found")
        return {"success": True}
    
    @app.post("/api/v1/scheduler/{event_id}/cancel")
    async def cancel_scheduled_event(event_id: str):
        """Cancelar evento."""
        success = await event_scheduler.cancel_event(event_id)
        if not success:
            raise HTTPException(status_code=404, detail="Event not found")
        return {"success": True}
    
    @app.get("/api/v1/scheduler/{event_id}")
    async def get_scheduled_event(event_id: str):
        """Obtener información de evento."""
        event = event_scheduler.get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        return event
    
    @app.get("/api/v1/scheduler/history")
    async def get_event_run_history(event_id: Optional[str] = None, limit: int = 100):
        """Obtener historial de ejecuciones."""
        history = event_scheduler.get_event_run_history(event_id, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/scheduler/summary")
    async def get_event_scheduler_summary():
        """Obtener resumen del scheduler."""
        return event_scheduler.get_event_scheduler_summary()
    
    # Graceful Degradation Manager
    @app.post("/api/v1/degradation/register-service")
    async def register_service_degradation(request: Dict[str, Any]):
        """Registrar servicio para degradación."""
        service_id = graceful_degradation_manager.register_service(
            service_id=request.get("service_id", ""),
            initial_state=ServiceState(request.get("initial_state", "healthy")),
            metadata=request.get("metadata"),
        )
        return {"success": True, "service_id": service_id}
    
    @app.post("/api/v1/degradation/register-fallback")
    async def register_fallback_strategy(request: Dict[str, Any]):
        """Registrar estrategia de fallback."""
        # Nota: En producción, handler necesitaría ser deserializado
        return {
            "success": True,
            "message": "Fallback registration endpoint - implement handler function registration"
        }
    
    @app.post("/api/v1/degradation/add-rule")
    async def add_degradation_rule(request: Dict[str, Any]):
        """Agregar regla de degradación."""
        degradation_level_str = request.get("degradation_level", "degraded")
        degradation_level = DegradationLevel(degradation_level_str)
        
        rule_id = graceful_degradation_manager.add_degradation_rule(
            rule_id=request.get("rule_id", f"rule_{datetime.now().timestamp()}"),
            service_id=request.get("service_id", ""),
            metric_name=request.get("metric_name", ""),
            threshold=request.get("threshold", 0.0),
            degradation_level=degradation_level,
            metadata=request.get("metadata"),
        )
        return {"success": True, "rule_id": rule_id}
    
    @app.post("/api/v1/degradation/record-metric")
    async def record_degradation_metric(request: Dict[str, Any]):
        """Registrar métrica para degradación."""
        await graceful_degradation_manager.record_metric(
            service_id=request.get("service_id", ""),
            metric_name=request.get("metric_name", ""),
            value=request.get("value", 0.0),
        )
        return {"success": True}
    
    @app.post("/api/v1/degradation/record-call")
    async def record_service_call(request: Dict[str, Any]):
        """Registrar llamada a servicio."""
        await graceful_degradation_manager.record_service_call(
            service_id=request.get("service_id", ""),
            success=request.get("success", True),
            response_time=request.get("response_time", 0.0),
        )
        return {"success": True}
    
    @app.get("/api/v1/degradation/service/{service_id}")
    async def get_service_health_degradation(service_id: str):
        """Obtener salud de servicio."""
        health = graceful_degradation_manager.get_service_health(service_id)
        if not health:
            raise HTTPException(status_code=404, detail="Service not found")
        return health
    
    @app.get("/api/v1/degradation/status")
    async def get_degradation_status():
        """Obtener estado de degradación."""
        return graceful_degradation_manager.get_degradation_status()
    
    @app.get("/api/v1/degradation/history")
    async def get_degradation_history(limit: int = 100):
        """Obtener historial de degradación."""
        history = graceful_degradation_manager.get_degradation_history(limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/degradation/summary")
    async def get_degradation_manager_summary():
        """Obtener resumen del gestor."""
        return graceful_degradation_manager.get_graceful_degradation_manager_summary()
    
    # Cache Warmer
    @app.post("/api/v1/cache-warmer/register-rule")
    async def register_warming_rule(request: Dict[str, Any]):
        """Registrar regla de precalentamiento."""
        # Nota: En producción, fetch_function necesitaría ser deserializado
        return {
            "success": True,
            "message": "Warming rule registration endpoint - implement fetch function registration"
        }
    
    @app.post("/api/v1/cache-warmer/record-access")
    async def record_cache_access(request: Dict[str, Any]):
        """Registrar acceso a cache."""
        cache_warmer.record_access(
            key=request.get("key", ""),
        )
        return {"success": True}
    
    @app.post("/api/v1/cache-warmer/start")
    async def start_cache_warming():
        """Iniciar precalentamiento."""
        cache_warmer.start_warming()
        return {"success": True, "message": "Cache warming started"}
    
    @app.post("/api/v1/cache-warmer/stop")
    async def stop_cache_warming():
        """Detener precalentamiento."""
        cache_warmer.stop_warming()
        return {"success": True, "message": "Cache warming stopped"}
    
    @app.get("/api/v1/cache-warmer/patterns")
    async def get_cache_access_patterns(limit: int = 50):
        """Obtener patrones de acceso."""
        patterns = cache_warmer.get_access_patterns(limit)
        return {"patterns": patterns, "count": len(patterns)}
    
    @app.get("/api/v1/cache-warmer/statistics")
    async def get_warming_statistics():
        """Obtener estadísticas de precalentamiento."""
        return cache_warmer.get_warming_statistics()
    
    @app.get("/api/v1/cache-warmer/summary")
    async def get_cache_warmer_summary():
        """Obtener resumen del warmer."""
        return cache_warmer.get_cache_warmer_summary()
    
    # Load Shedder
    @app.post("/api/v1/load-shedder/record-metric")
    async def record_load_shedder_metric(request: Dict[str, Any]):
        """Registrar métrica de carga."""
        load_shedder.record_load_metric(
            cpu_usage=request.get("cpu_usage"),
            memory_usage=request.get("memory_usage"),
            request_rate=request.get("request_rate"),
            response_time=request.get("response_time"),
            queue_size=request.get("queue_size"),
            error_rate=request.get("error_rate"),
        )
        return {"success": True}
    
    @app.post("/api/v1/load-shedder/add-rule")
    async def add_shedding_rule(request: Dict[str, Any]):
        """Agregar regla de descarga."""
        strategy_str = request.get("strategy", "priority")
        strategy = SheddingStrategy(strategy_str)
        
        rule_id = load_shedder.add_shedding_rule(
            rule_id=request.get("rule_id", f"rule_{datetime.now().timestamp()}"),
            metric_name=request.get("metric_name", ""),
            threshold=request.get("threshold", 0.0),
            shedding_percentage=request.get("shedding_percentage", 0.1),
            strategy=strategy,
            metadata=request.get("metadata"),
        )
        return {"success": True, "rule_id": rule_id}
    
    @app.post("/api/v1/load-shedder/check-request")
    async def check_should_accept_request(request: Dict[str, Any]):
        """Verificar si se debe aceptar petición."""
        priority_str = request.get("priority", "normal")
        priority = RequestPriority(priority_str)
        
        allowed = await load_shedder.should_accept_request(
            request_id=request.get("request_id", ""),
            priority=priority,
        )
        return {"allowed": allowed}
    
    @app.get("/api/v1/load-shedder/statistics")
    async def get_load_shedder_statistics(window_minutes: int = 5):
        """Obtener estadísticas de carga."""
        stats = load_shedder.get_load_statistics(window_minutes)
        return stats
    
    @app.get("/api/v1/load-shedder/history")
    async def get_shedding_history(limit: int = 100):
        """Obtener historial de descarga."""
        history = load_shedder.get_shedding_history(limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/load-shedder/summary")
    async def get_load_shedder_summary():
        """Obtener resumen del shedder."""
        return load_shedder.get_load_shedder_summary()
    
    # Conflict Resolver
    @app.post("/api/v1/conflicts/register")
    async def register_conflict(request: Dict[str, Any]):
        """Registrar conflicto."""
        conflict_type_str = request.get("conflict_type", "data_update")
        conflict_type = ConflictType(conflict_type_str)
        
        conflict_id = conflict_resolver.register_conflict(
            conflict_id=request.get("conflict_id", f"conflict_{datetime.now().timestamp()}"),
            conflict_type=conflict_type,
            resource_id=request.get("resource_id", ""),
            conflicting_values=request.get("conflicting_values", {}),
            metadata=request.get("metadata"),
        )
        return {"success": True, "conflict_id": conflict_id}
    
    @app.post("/api/v1/conflicts/{conflict_id}/resolve")
    async def resolve_conflict(conflict_id: str, request: Dict[str, Any]):
        """Resolver conflicto."""
        strategy_str = request.get("strategy", "last_write_wins")
        strategy = ResolutionStrategy(strategy_str)
        
        success = await conflict_resolver.apply_resolution(
            conflict_id=conflict_id,
            resolved_value=request.get("resolved_value"),
            strategy=strategy,
            resolver_id=request.get("resolver_id", "user"),
        )
        if not success:
            raise HTTPException(status_code=404, detail="Conflict not found or already resolved")
        return {"success": True}
    
    @app.post("/api/v1/conflicts/register-rule")
    async def register_resolution_rule(request: Dict[str, Any]):
        """Registrar regla de resolución."""
        conflict_type_str = request.get("conflict_type", "data_update")
        conflict_type = ConflictType(conflict_type_str)
        
        strategy_str = request.get("strategy", "last_write_wins")
        strategy = ResolutionStrategy(strategy_str)
        
        conflict_resolver.register_resolution_rule(conflict_type, strategy)
        return {"success": True}
    
    @app.get("/api/v1/conflicts/{conflict_id}")
    async def get_conflict_info(conflict_id: str):
        """Obtener información de conflicto."""
        conflict = conflict_resolver.get_conflict(conflict_id)
        if not conflict:
            raise HTTPException(status_code=404, detail="Conflict not found")
        return conflict
    
    @app.get("/api/v1/conflicts/pending")
    async def get_pending_conflicts(limit: int = 100):
        """Obtener conflictos pendientes."""
        conflicts = conflict_resolver.get_pending_conflicts(limit)
        return {"conflicts": conflicts, "count": len(conflicts)}
    
    @app.get("/api/v1/conflicts/history")
    async def get_resolution_history(limit: int = 100):
        """Obtener historial de resoluciones."""
        history = conflict_resolver.get_resolution_history(limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/conflicts/summary")
    async def get_conflict_resolver_summary():
        """Obtener resumen del resolvedor."""
        return conflict_resolver.get_conflict_resolver_summary()
    
    # State Machine Manager
    @app.post("/api/v1/state-machines/create")
    async def create_state_machine(request: Dict[str, Any]):
        """Crear máquina de estados."""
        machine_id = state_machine_manager.create_state_machine(
            machine_id=request.get("machine_id", f"machine_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            initial_state=request.get("initial_state", ""),
            states=request.get("states", []),
            metadata=request.get("metadata"),
        )
        return {"success": True, "machine_id": machine_id}
    
    @app.post("/api/v1/state-machines/{machine_id}/add-transition")
    async def add_state_transition(machine_id: str, request: Dict[str, Any]):
        """Agregar transición."""
        # Nota: En producción, condition y on_transition necesitarían ser deserializados
        return {
            "success": True,
            "message": "Transition registration endpoint - implement function registration"
        }
    
    @app.post("/api/v1/state-machines/{machine_id}/transition")
    async def transition_state_machine(machine_id: str, request: Dict[str, Any]):
        """Realizar transición."""
        success = await state_machine_manager.transition(
            machine_id=machine_id,
            to_state=request.get("to_state", ""),
            transition_id=request.get("transition_id"),
            metadata=request.get("metadata"),
        )
        if not success:
            raise HTTPException(status_code=400, detail="Transition not allowed")
        return {"success": True}
    
    @app.get("/api/v1/state-machines/{machine_id}")
    async def get_state_machine(machine_id: str):
        """Obtener información de máquina de estados."""
        machine = state_machine_manager.get_state_machine(machine_id)
        if not machine:
            raise HTTPException(status_code=404, detail="State machine not found")
        return machine
    
    @app.get("/api/v1/state-machines/history")
    async def get_state_machine_history(machine_id: Optional[str] = None, limit: int = 100):
        """Obtener historial de estados."""
        history = state_machine_manager.get_state_history(machine_id, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/state-machines/summary")
    async def get_state_machine_manager_summary():
        """Obtener resumen del gestor."""
        return state_machine_manager.get_state_machine_manager_summary()
    
    # Workflow Engine V2
    @app.post("/api/v1/workflows-v2/create")
    async def create_workflow_v2(request: Dict[str, Any]):
        """Crear workflow."""
        workflow_id = workflow_engine_v2.create_workflow(
            workflow_id=request.get("workflow_id", f"workflow_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            description=request.get("description", ""),
            metadata=request.get("metadata"),
        )
        return {"success": True, "workflow_id": workflow_id}
    
    @app.post("/api/v1/workflows-v2/{workflow_id}/add-step")
    async def add_workflow_step_v2(workflow_id: str, request: Dict[str, Any]):
        """Agregar step a workflow."""
        # Nota: En producción, handler necesitaría ser deserializado
        return {
            "success": True,
            "message": "Step registration endpoint - implement handler function registration"
        }
    
    @app.post("/api/v1/workflows-v2/{workflow_id}/execute")
    async def execute_workflow_v2(workflow_id: str, request: Dict[str, Any]):
        """Ejecutar workflow."""
        execution_id = await workflow_engine_v2.execute_workflow(
            workflow_id=workflow_id,
            context=request.get("context", {}),
            execution_id=request.get("execution_id"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "execution_id": execution_id}
    
    @app.get("/api/v1/workflows-v2/{workflow_id}")
    async def get_workflow_v2(workflow_id: str):
        """Obtener información de workflow."""
        workflow = workflow_engine_v2.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return workflow
    
    @app.get("/api/v1/workflows-v2/executions/{execution_id}")
    async def get_workflow_execution_v2(execution_id: str):
        """Obtener información de ejecución."""
        execution = workflow_engine_v2.get_execution(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        return execution
    
    @app.post("/api/v1/workflows-v2/executions/{execution_id}/cancel")
    async def cancel_workflow_execution_v2(execution_id: str):
        """Cancelar ejecución."""
        success = await workflow_engine_v2.cancel_execution(execution_id)
        if not success:
            raise HTTPException(status_code=404, detail="Execution not found or cannot be cancelled")
        return {"success": True}
    
    @app.get("/api/v1/workflows-v2/summary")
    async def get_workflow_engine_v2_summary():
        """Obtener resumen del motor."""
        return workflow_engine_v2.get_workflow_engine_v2_summary()
    
    # Event Bus
    @app.post("/api/v1/events/publish")
    async def publish_event(request: Dict[str, Any]):
        """Publicar evento."""
        priority_str = request.get("priority", "normal")
        priority = EventPriority(priority_str)
        
        event_id = await event_bus.publish(
            event_type=request.get("event_type", ""),
            source=request.get("source", "api"),
            payload=request.get("payload", {}),
            priority=priority,
            metadata=request.get("metadata"),
        )
        return {"success": True, "event_id": event_id}
    
    @app.post("/api/v1/events/subscribe")
    async def subscribe_to_events(request: Dict[str, Any]):
        """Suscribirse a eventos."""
        # Nota: En producción, handler y filter_func necesitarían ser deserializados
        return {
            "success": True,
            "message": "Subscription endpoint - implement handler function registration"
        }
    
    @app.post("/api/v1/events/unsubscribe")
    async def unsubscribe_from_events(request: Dict[str, Any]):
        """Desuscribirse de eventos."""
        success = await event_bus.unsubscribe(request.get("subscription_id", ""))
        if not success:
            raise HTTPException(status_code=404, detail="Subscription not found")
        return {"success": True}
    
    @app.get("/api/v1/events/history")
    async def get_event_history(event_type: Optional[str] = None, source: Optional[str] = None, limit: int = 100):
        """Obtener historial de eventos."""
        history = event_bus.get_event_history(event_type, source, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/events/subscriptions")
    async def get_event_subscriptions(event_type: Optional[str] = None):
        """Obtener suscripciones."""
        subscriptions = event_bus.get_subscriptions(event_type)
        return {"subscriptions": subscriptions, "count": len(subscriptions)}
    
    @app.get("/api/v1/events/summary")
    async def get_event_bus_summary():
        """Obtener resumen del bus."""
        return event_bus.get_event_bus_summary()
    
    # Feature Toggle Manager
    @app.post("/api/v1/feature-toggles/create")
    async def create_feature_toggle(request: Dict[str, Any]):
        """Crear feature toggle."""
        toggle_type_str = request.get("toggle_type", "boolean")
        toggle_type = ToggleType(toggle_type_str)
        
        toggle_id = feature_toggle_manager.create_toggle(
            toggle_id=request.get("toggle_id", f"toggle_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            toggle_type=toggle_type,
            enabled=request.get("enabled", False),
            percentage=request.get("percentage", 0.0),
            target_users=request.get("target_users"),
            target_attributes=request.get("target_attributes"),
            variants=request.get("variants"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "toggle_id": toggle_id}
    
    @app.get("/api/v1/feature-toggles/{toggle_id}/check")
    async def check_feature_toggle(toggle_id: str, user_id: Optional[str] = None):
        """Verificar si toggle está habilitado."""
        user_attributes = {}  # En producción, obtener de request
        enabled = feature_toggle_manager.is_enabled(toggle_id, user_id, user_attributes)
        
        # Registrar evaluación
        variant = feature_toggle_manager.get_variant(toggle_id, user_id)
        feature_toggle_manager.record_evaluation(toggle_id, user_id, enabled, variant)
        
        return {"enabled": enabled, "variant": variant}
    
    @app.post("/api/v1/feature-toggles/{toggle_id}/update")
    async def update_feature_toggle(toggle_id: str, request: Dict[str, Any]):
        """Actualizar feature toggle."""
        status = None
        if request.get("status"):
            status = ToggleStatus(request.get("status"))
        
        success = feature_toggle_manager.update_toggle(
            toggle_id=toggle_id,
            enabled=request.get("enabled"),
            percentage=request.get("percentage"),
            target_users=request.get("target_users"),
            target_attributes=request.get("target_attributes"),
            variants=request.get("variants"),
            status=status,
        )
        if not success:
            raise HTTPException(status_code=404, detail="Toggle not found")
        return {"success": True}
    
    @app.get("/api/v1/feature-toggles/{toggle_id}")
    async def get_feature_toggle(toggle_id: str):
        """Obtener información de toggle."""
        toggle = feature_toggle_manager.get_toggle(toggle_id)
        if not toggle:
            raise HTTPException(status_code=404, detail="Toggle not found")
        return toggle
    
    @app.get("/api/v1/feature-toggles/{toggle_id}/statistics")
    async def get_feature_toggle_statistics(toggle_id: str):
        """Obtener estadísticas de evaluación."""
        stats = feature_toggle_manager.get_evaluation_statistics(toggle_id)
        return stats
    
    @app.get("/api/v1/feature-toggles/summary")
    async def get_feature_toggle_manager_summary():
        """Obtener resumen del gestor."""
        return feature_toggle_manager.get_feature_toggle_manager_summary()
    
    # Rate Limiter V2
    @app.post("/api/v1/rate-limiter-v2/add-rule")
    async def add_rate_limit_rule_v2(request: Dict[str, Any]):
        """Agregar regla de rate limiting."""
        algorithm_str = request.get("algorithm", "fixed_window")
        algorithm = RateLimitAlgorithm(algorithm_str)
        
        rule_id = rate_limiter_v2.add_rule(
            rule_id=request.get("rule_id", f"rule_{datetime.now().timestamp()}"),
            identifier=request.get("identifier", ""),
            algorithm=algorithm,
            limit=request.get("limit", 100),
            window_seconds=request.get("window_seconds", 60.0),
            tokens=request.get("tokens"),
            refill_rate=request.get("refill_rate"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "rule_id": rule_id}
    
    @app.post("/api/v1/rate-limiter-v2/check")
    async def check_rate_limit_v2(request: Dict[str, Any]):
        """Verificar rate limit."""
        allowed, info = await rate_limiter_v2.check_rate_limit(
            identifier=request.get("identifier", ""),
            rule_id=request.get("rule_id"),
        )
        return info
    
    @app.get("/api/v1/rate-limiter-v2/status")
    async def get_rate_limit_status_v2(identifier: str, rule_id: Optional[str] = None):
        """Obtener estado de rate limiting."""
        status = rate_limiter_v2.get_rate_limit_status(identifier, rule_id)
        return status
    
    @app.get("/api/v1/rate-limiter-v2/history")
    async def get_rate_limit_block_history(limit: int = 100):
        """Obtener historial de bloqueos."""
        history = rate_limiter_v2.get_block_history(limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/rate-limiter-v2/summary")
    async def get_rate_limiter_v2_summary():
        """Obtener resumen del limitador."""
        return rate_limiter_v2.get_rate_limiter_v2_summary()
    
    # Circuit Breaker V2
    @app.post("/api/v1/circuit-breakers-v2/create")
    async def create_circuit_breaker_v2(request: Dict[str, Any]):
        """Crear circuit breaker."""
        strategy_str = request.get("failure_strategy", "count_based")
        strategy = FailureStrategy(strategy_str)
        
        circuit_id = circuit_breaker_v2.create_circuit(
            circuit_id=request.get("circuit_id", f"circuit_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            failure_threshold=request.get("failure_threshold", 5),
            failure_percentage=request.get("failure_percentage", 0.5),
            timeout_seconds=request.get("timeout_seconds", 60.0),
            half_open_max_calls=request.get("half_open_max_calls", 3),
            failure_strategy=strategy,
            metadata=request.get("metadata"),
        )
        return {"success": True, "circuit_id": circuit_id}
    
    @app.get("/api/v1/circuit-breakers-v2/{circuit_id}")
    async def get_circuit_breaker_v2(circuit_id: str):
        """Obtener información de circuit breaker."""
        circuit = circuit_breaker_v2.get_circuit(circuit_id)
        if not circuit:
            raise HTTPException(status_code=404, detail="Circuit breaker not found")
        return circuit
    
    @app.post("/api/v1/circuit-breakers-v2/{circuit_id}/reset")
    async def reset_circuit_breaker_v2(circuit_id: str):
        """Resetear circuit breaker."""
        success = await circuit_breaker_v2.reset_circuit(circuit_id)
        if not success:
            raise HTTPException(status_code=404, detail="Circuit breaker not found")
        return {"success": True}
    
    @app.get("/api/v1/circuit-breakers-v2/summary")
    async def get_circuit_breaker_v2_summary():
        """Obtener resumen del gestor."""
        return circuit_breaker_v2.get_circuit_breaker_v2_summary()
    
    # Adaptive Optimizer
    @app.post("/api/v1/optimizer/register-parameter")
    async def register_optimization_parameter(request: Dict[str, Any]):
        """Registrar parámetro a optimizar."""
        parameter_id = adaptive_optimizer.register_parameter(
            parameter_id=request.get("parameter_id", f"param_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            initial_value=request.get("initial_value", 1.0),
            min_value=request.get("min_value", 0.0),
            max_value=request.get("max_value", 10.0),
            step=request.get("step", 0.1),
            metadata=request.get("metadata"),
        )
        return {"success": True, "parameter_id": parameter_id}
    
    @app.post("/api/v1/optimizer/add-goal")
    async def add_optimization_goal(request: Dict[str, Any]):
        """Agregar objetivo de optimización."""
        target_str = request.get("target", "latency")
        target = OptimizationTarget(target_str)
        
        goal_id = adaptive_optimizer.add_goal(
            goal_id=request.get("goal_id", f"goal_{datetime.now().timestamp()}"),
            target=target,
            target_value=request.get("target_value"),
            maximize=request.get("maximize", True),
            weight=request.get("weight", 1.0),
            metadata=request.get("metadata"),
        )
        return {"success": True, "goal_id": goal_id}
    
    @app.post("/api/v1/optimizer/record-metric")
    async def record_optimization_metric(request: Dict[str, Any]):
        """Registrar métrica para optimización."""
        adaptive_optimizer.record_metric(
            metric_name=request.get("metric_name", ""),
            value=request.get("value", 0.0),
        )
        return {"success": True}
    
    @app.post("/api/v1/optimizer/start")
    async def start_adaptive_optimization():
        """Iniciar optimización."""
        adaptive_optimizer.start_optimization()
        return {"success": True, "message": "Optimization started"}
    
    @app.post("/api/v1/optimizer/stop")
    async def stop_adaptive_optimization():
        """Detener optimización."""
        adaptive_optimizer.stop_optimization()
        return {"success": True, "message": "Optimization stopped"}
    
    @app.get("/api/v1/optimizer/parameter/{parameter_id}")
    async def get_optimization_parameter(parameter_id: str):
        """Obtener información de parámetro."""
        param = adaptive_optimizer.get_parameter(parameter_id)
        if not param:
            raise HTTPException(status_code=404, detail="Parameter not found")
        return param
    
    @app.get("/api/v1/optimizer/results")
    async def get_optimization_results(limit: int = 100):
        """Obtener resultados de optimización."""
        results = adaptive_optimizer.get_optimization_results(limit)
        return {"results": results, "count": len(results)}
    
    @app.get("/api/v1/optimizer/summary")
    async def get_adaptive_optimizer_summary():
        """Obtener resumen del optimizador."""
        return adaptive_optimizer.get_adaptive_optimizer_summary()
    
    # Health Checker V2
    @app.post("/api/v1/health-v2/register-check")
    async def register_health_check_v2(request: Dict[str, Any]):
        """Registrar health check."""
        # Nota: En producción, handler necesitaría ser deserializado
        return {
            "success": True,
            "message": "Health check registration endpoint - implement handler function registration"
        }
    
    @app.post("/api/v1/health-v2/{check_id}/run")
    async def run_health_check_v2(check_id: str):
        """Ejecutar health check manualmente."""
        result = await health_checker_v2.run_check(check_id)
        if not result:
            raise HTTPException(status_code=404, detail="Health check not found")
        return result
    
    @app.get("/api/v1/health-v2/overall")
    async def get_overall_health_v2():
        """Obtener salud general."""
        return health_checker_v2.get_overall_health()
    
    @app.get("/api/v1/health-v2/{check_id}/history")
    async def get_health_check_history_v2(check_id: str, limit: int = 100):
        """Obtener historial de checks."""
        history = health_checker_v2.get_check_history(check_id, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/health-v2/summary")
    async def get_health_checker_v2_summary():
        """Obtener resumen del verificador."""
        return health_checker_v2.get_health_checker_v2_summary()
    
    # Auto Scaler
    @app.post("/api/v1/auto-scaler/add-rule")
    async def add_scaling_rule(request: Dict[str, Any]):
        """Agregar regla de escalado."""
        rule_id = auto_scaler.add_rule(
            rule_id=request.get("rule_id", f"rule_{datetime.now().timestamp()}"),
            metric_name=request.get("metric_name", ""),
            threshold_up=request.get("threshold_up", 80.0),
            threshold_down=request.get("threshold_down", 20.0),
            min_instances=request.get("min_instances", 1),
            max_instances=request.get("max_instances", 10),
            scale_up_step=request.get("scale_up_step", 1),
            scale_down_step=request.get("scale_down_step", 1),
            cooldown_seconds=request.get("cooldown_seconds", 300.0),
            metadata=request.get("metadata"),
        )
        return {"success": True, "rule_id": rule_id}
    
    @app.post("/api/v1/auto-scaler/record-metric")
    async def record_scaling_metric(request: Dict[str, Any]):
        """Registrar métrica para escalado."""
        auto_scaler.record_metric(
            metric_name=request.get("metric_name", ""),
            value=request.get("value", 0.0),
        )
        return {"success": True}
    
    @app.post("/api/v1/auto-scaler/set-instances")
    async def set_scaler_instances(request: Dict[str, Any]):
        """Establecer número de instancias manualmente."""
        auto_scaler.set_instances(request.get("instances", 1))
        return {"success": True}
    
    @app.post("/api/v1/auto-scaler/start")
    async def start_auto_scaling():
        """Iniciar auto-escalado."""
        auto_scaler.start_scaling()
        return {"success": True, "message": "Auto-scaling started"}
    
    @app.post("/api/v1/auto-scaler/stop")
    async def stop_auto_scaling():
        """Detener auto-escalado."""
        auto_scaler.stop_scaling()
        return {"success": True, "message": "Auto-scaling stopped"}
    
    @app.get("/api/v1/auto-scaler/status")
    async def get_scaling_status():
        """Obtener estado de escalado."""
        return auto_scaler.get_scaling_status()
    
    @app.get("/api/v1/auto-scaler/history")
    async def get_scaling_history(limit: int = 100):
        """Obtener historial de escalado."""
        history = auto_scaler.get_scaling_history(limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/auto-scaler/summary")
    async def get_auto_scaler_summary():
        """Obtener resumen del escalador."""
        return auto_scaler.get_auto_scaler_summary()
    
    # Batch Processor
    @app.post("/api/v1/batch/add-item")
    async def add_batch_item(request: Dict[str, Any]):
        """Agregar item a batch."""
        item_id = batch_processor.add_item(
            queue_id=request.get("queue_id", "default"),
            item_id=request.get("item_id", f"item_{datetime.now().timestamp()}"),
            data=request.get("data"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "item_id": item_id}
    
    @app.post("/api/v1/batch/register-processor")
    async def register_batch_processor(request: Dict[str, Any]):
        """Registrar procesador para cola."""
        # Nota: En producción, processor necesitaría ser deserializado
        return {
            "success": True,
            "message": "Processor registration endpoint - implement processor function registration"
        }
    
    @app.get("/api/v1/batch/queue-status")
    async def get_batch_queue_status(queue_id: Optional[str] = None):
        """Obtener estado de cola(s)."""
        status = batch_processor.get_queue_status(queue_id)
        return status
    
    @app.get("/api/v1/batch/history")
    async def get_batch_history(queue_id: Optional[str] = None, limit: int = 100):
        """Obtener historial de batches."""
        history = batch_processor.get_batch_history(queue_id, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/batch/summary")
    async def get_batch_processor_summary():
        """Obtener resumen del procesador."""
        return batch_processor.get_batch_processor_summary()
    
    # Performance Monitor
    @app.post("/api/v1/performance/record-metric")
    async def record_performance_metric(request: Dict[str, Any]):
        """Registrar métrica de rendimiento."""
        metric_type_str = request.get("metric_type", "gauge")
        metric_type = MetricType(metric_type_str)
        
        performance_monitor.record_metric(
            metric_name=request.get("metric_name", ""),
            value=request.get("value", 0.0),
            metric_type=metric_type,
            labels=request.get("labels"),
        )
        return {"success": True}
    
    @app.post("/api/v1/performance/record-latency")
    async def record_performance_latency(request: Dict[str, Any]):
        """Registrar latencia."""
        performance_monitor.record_latency(
            operation_name=request.get("operation_name", ""),
            latency_seconds=request.get("latency_seconds", 0.0),
        )
        return {"success": True}
    
    @app.post("/api/v1/performance/create-snapshot")
    async def create_performance_snapshot():
        """Crear snapshot de rendimiento."""
        snapshot_id = performance_monitor.create_snapshot()
        return {"success": True, "snapshot_id": snapshot_id}
    
    @app.get("/api/v1/performance/summary")
    async def get_performance_summary(window_minutes: int = 5):
        """Obtener resumen de rendimiento."""
        summary = performance_monitor.get_performance_summary(window_minutes)
        return summary
    
    @app.get("/api/v1/performance/metric/{metric_name}")
    async def get_metric_history(metric_name: str, limit: int = 100):
        """Obtener historial de métrica."""
        history = performance_monitor.get_metric_history(metric_name, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/performance/monitor-summary")
    async def get_performance_monitor_summary():
        """Obtener resumen del monitor."""
        return performance_monitor.get_performance_monitor_summary()
    
    # Queue Manager
    @app.post("/api/v1/queues/create")
    async def create_queue(request: Dict[str, Any]):
        """Crear cola."""
        queue_name = queue_manager.create_queue(
            queue_name=request.get("queue_name", ""),
            max_size=request.get("max_size", 10000),
            visibility_timeout=request.get("visibility_timeout", 30.0),
            message_retention=request.get("message_retention", 86400.0),
            dead_letter_queue=request.get("dead_letter_queue"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "queue_name": queue_name}
    
    @app.post("/api/v1/queues/{queue_name}/enqueue")
    async def enqueue_message(queue_name: str, request: Dict[str, Any]):
        """Encolar mensaje."""
        priority_str = request.get("priority", "normal")
        priority = QueuePriority[priority_str.upper()]
        
        message_id = queue_manager.enqueue(
            queue_name=queue_name,
            payload=request.get("payload"),
            priority=priority,
            message_id=request.get("message_id"),
            max_attempts=request.get("max_attempts", 3),
            metadata=request.get("metadata"),
        )
        return {"success": True, "message_id": message_id}
    
    @app.post("/api/v1/queues/{queue_name}/dequeue")
    async def dequeue_message(queue_name: str, timeout: Optional[float] = None):
        """Desencolar mensaje."""
        message = await queue_manager.dequeue(queue_name, timeout)
        if not message:
            raise HTTPException(status_code=404, detail="No messages available")
        
        return {
            "message_id": message.message_id,
            "payload": message.payload,
            "priority": message.priority.value,
            "attempts": message.attempts,
            "created_at": message.created_at.isoformat(),
        }
    
    @app.post("/api/v1/queues/messages/{message_id}/ack")
    async def acknowledge_message(message_id: str):
        """Confirmar mensaje."""
        await queue_manager.acknowledge(message_id)
        return {"success": True}
    
    @app.post("/api/v1/queues/messages/{message_id}/nack")
    async def nack_message(message_id: str, requeue: bool = True):
        """Negar mensaje."""
        await queue_manager.nack(message_id, requeue)
        return {"success": True}
    
    @app.get("/api/v1/queues/{queue_name}/status")
    async def get_queue_status(queue_name: str):
        """Obtener estado de cola."""
        status = queue_manager.get_queue_status(queue_name)
        if not status:
            raise HTTPException(status_code=404, detail="Queue not found")
        return status
    
    @app.get("/api/v1/queues/summary")
    async def get_queue_manager_summary():
        """Obtener resumen del gestor."""
        return queue_manager.get_queue_manager_summary()
    
    # Connection Manager
    @app.post("/api/v1/connections/register")
    async def register_connection(request: Dict[str, Any]):
        """Registrar conexión."""
        connection_id = connection_manager.register_connection(
            connection_id=request.get("connection_id", f"conn_{datetime.now().timestamp()}"),
            connection_type=request.get("connection_type", ""),
            metadata=request.get("metadata"),
        )
        return {"success": True, "connection_id": connection_id}
    
    @app.post("/api/v1/connections/{connection_type}/acquire")
    async def acquire_connection(connection_type: str, timeout: Optional[float] = None):
        """Adquirir conexión."""
        connection_id = await connection_manager.acquire_connection(connection_type, timeout)
        if not connection_id:
            raise HTTPException(status_code=404, detail="No connection available")
        return {"success": True, "connection_id": connection_id}
    
    @app.post("/api/v1/connections/{connection_id}/release")
    async def release_connection(connection_id: str):
        """Liberar conexión."""
        await connection_manager.release_connection(connection_id)
        return {"success": True}
    
    @app.post("/api/v1/connections/{connection_id}/close")
    async def close_connection(connection_id: str):
        """Cerrar conexión."""
        await connection_manager.close_connection(connection_id)
        return {"success": True}
    
    @app.get("/api/v1/connections/{connection_id}")
    async def get_connection_info(connection_id: str):
        """Obtener información de conexión."""
        connection = connection_manager.get_connection(connection_id)
        if not connection:
            raise HTTPException(status_code=404, detail="Connection not found")
        return connection
    
    @app.get("/api/v1/connections/type/{connection_type}")
    async def get_connections_by_type(connection_type: str):
        """Obtener conexiones por tipo."""
        connections = connection_manager.get_connections_by_type(connection_type)
        return {"connections": connections, "count": len(connections)}
    
    @app.get("/api/v1/connections/summary")
    async def get_connection_manager_summary():
        """Obtener resumen del gestor."""
        return connection_manager.get_connection_manager_summary()
    
    # Transaction Manager
    @app.post("/api/v1/transactions/begin")
    async def begin_transaction(request: Dict[str, Any]):
        """Iniciar transacción."""
        transaction_id = transaction_manager.begin_transaction(
            transaction_id=request.get("transaction_id"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "transaction_id": transaction_id}
    
    @app.post("/api/v1/transactions/{transaction_id}/add-operation")
    async def add_transaction_operation(transaction_id: str, request: Dict[str, Any]):
        """Agregar operación a transacción."""
        # Nota: En producción, execute y rollback necesitarían ser deserializados
        return {
            "success": True,
            "message": "Operation registration endpoint - implement function registration"
        }
    
    @app.post("/api/v1/transactions/{transaction_id}/commit")
    async def commit_transaction(transaction_id: str):
        """Commit transacción."""
        success = await transaction_manager.commit(transaction_id)
        if not success:
            raise HTTPException(status_code=400, detail="Transaction commit failed")
        return {"success": True}
    
    @app.post("/api/v1/transactions/{transaction_id}/rollback")
    async def rollback_transaction(transaction_id: str):
        """Rollback transacción."""
        success = await transaction_manager.rollback(transaction_id)
        if not success:
            raise HTTPException(status_code=400, detail="Transaction rollback failed")
        return {"success": True}
    
    @app.get("/api/v1/transactions/{transaction_id}")
    async def get_transaction_info(transaction_id: str):
        """Obtener información de transacción."""
        transaction = transaction_manager.get_transaction(transaction_id)
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        return transaction
    
    @app.get("/api/v1/transactions/history")
    async def get_transaction_history(limit: int = 100):
        """Obtener historial de transacciones."""
        history = transaction_manager.get_transaction_history(limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/transactions/summary")
    async def get_transaction_manager_summary():
        """Obtener resumen del gestor."""
        return transaction_manager.get_transaction_manager_summary()
    
    # Saga Orchestrator
    @app.post("/api/v1/sagas/create")
    async def create_saga(request: Dict[str, Any]):
        """Crear saga."""
        saga_id = saga_orchestrator.create_saga(
            saga_id=request.get("saga_id", f"saga_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            metadata=request.get("metadata"),
        )
        return {"success": True, "saga_id": saga_id}
    
    @app.post("/api/v1/sagas/{saga_id}/add-step")
    async def add_saga_step(saga_id: str, request: Dict[str, Any]):
        """Agregar step a saga."""
        # Nota: En producción, execute y compensate necesitarían ser deserializados
        return {
            "success": True,
            "message": "Step registration endpoint - implement function registration"
        }
    
    @app.post("/api/v1/sagas/{saga_id}/execute")
    async def execute_saga(saga_id: str, request: Dict[str, Any]):
        """Ejecutar saga."""
        await saga_orchestrator.execute_saga(
            saga_id=saga_id,
            context=request.get("context", {}),
        )
        return {"success": True, "saga_id": saga_id}
    
    @app.get("/api/v1/sagas/{saga_id}")
    async def get_saga_info(saga_id: str):
        """Obtener información de saga."""
        saga = saga_orchestrator.get_saga(saga_id)
        if not saga:
            raise HTTPException(status_code=404, detail="Saga not found")
        return saga
    
    @app.get("/api/v1/sagas/history")
    async def get_saga_history(limit: int = 100):
        """Obtener historial de sagas."""
        history = saga_orchestrator.get_saga_history(limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/sagas/summary")
    async def get_saga_orchestrator_summary():
        """Obtener resumen del orquestador."""
        return saga_orchestrator.get_saga_orchestrator_summary()
    
    # Distributed Coordinator
    @app.post("/api/v1/coordination/register-node")
    async def register_coordination_node(request: Dict[str, Any]):
        """Registrar nodo en coordinación."""
        node_id = distributed_coordinator.register_node(
            node_id=request.get("node_id", f"node_{datetime.now().timestamp()}"),
            address=request.get("address", ""),
            metadata=request.get("metadata"),
        )
        return {"success": True, "node_id": node_id}
    
    @app.post("/api/v1/coordination/propose")
    async def propose_value(request: Dict[str, Any]):
        """Proponer valor para consenso."""
        proposal_id = await distributed_coordinator.propose_value(
            value=request.get("value"),
            proposal_id=request.get("proposal_id"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "proposal_id": proposal_id}
    
    @app.get("/api/v1/coordination/leader")
    async def get_coordination_leader():
        """Obtener información del líder."""
        leader = distributed_coordinator.get_leader()
        if not leader:
            raise HTTPException(status_code=404, detail="No leader elected")
        return leader
    
    @app.get("/api/v1/coordination/status")
    async def get_coordination_status():
        """Obtener estado de coordinación."""
        return distributed_coordinator.get_coordination_status()
    
    @app.get("/api/v1/coordination/summary")
    async def get_distributed_coordinator_summary():
        """Obtener resumen del coordinador."""
        return distributed_coordinator.get_distributed_coordinator_summary()
    
    # Service Mesh
    @app.post("/api/v1/mesh/register-service")
    async def register_mesh_service(request: Dict[str, Any]):
        """Registrar servicio en malla."""
        strategy_str = request.get("load_balancing_strategy", "round_robin")
        strategy = LoadBalancingStrategy(strategy_str)
        
        service_name = service_mesh.register_service(
            service_name=request.get("service_name", ""),
            load_balancing_strategy=strategy,
            health_check_interval=request.get("health_check_interval", 30.0),
            metadata=request.get("metadata"),
        )
        return {"success": True, "service_name": service_name}
    
    @app.post("/api/v1/mesh/register-instance")
    async def register_mesh_instance(request: Dict[str, Any]):
        """Registrar instancia de servicio."""
        instance_id = service_mesh.register_instance(
            instance_id=request.get("instance_id", f"inst_{datetime.now().timestamp()}"),
            service_name=request.get("service_name", ""),
            address=request.get("address", ""),
            port=request.get("port", 0),
            weight=request.get("weight", 1),
            metadata=request.get("metadata"),
        )
        return {"success": True, "instance_id": instance_id}
    
    @app.get("/api/v1/mesh/service/{service_name}/instance")
    async def get_service_instance(service_name: str, client_id: Optional[str] = None):
        """Obtener instancia de servicio."""
        instance = service_mesh.get_instance(service_name, client_id)
        if not instance:
            raise HTTPException(status_code=404, detail="No instance available")
        
        return {
            "instance_id": instance.instance_id,
            "address": instance.address,
            "port": instance.port,
            "status": instance.status.value,
            "weight": instance.weight,
        }
    
    @app.post("/api/v1/mesh/instance/{instance_id}/status")
    async def update_instance_status(instance_id: str, request: Dict[str, Any]):
        """Actualizar estado de instancia."""
        status_str = request.get("status", "unknown")
        status = ServiceStatus(status_str)
        
        service_mesh.update_instance_status(instance_id, status)
        return {"success": True}
    
    @app.get("/api/v1/mesh/service/{service_name}/instances")
    async def get_service_instances(service_name: str):
        """Obtener instancias de servicio."""
        instances = service_mesh.get_service_instances(service_name)
        return {"instances": instances, "count": len(instances)}
    
    @app.get("/api/v1/mesh/summary")
    async def get_service_mesh_summary():
        """Obtener resumen de la malla."""
        return service_mesh.get_service_mesh_summary()
    
    # Adaptive Throttler
    @app.post("/api/v1/throttler/add-rule")
    async def add_throttle_rule(request: Dict[str, Any]):
        """Agregar regla de throttling."""
        strategy_str = request.get("strategy", "adaptive")
        strategy = ThrottleStrategy(strategy_str)
        
        rule_id = adaptive_throttler.add_rule(
            rule_id=request.get("rule_id", f"rule_{datetime.now().timestamp()}"),
            identifier=request.get("identifier", ""),
            base_limit=request.get("base_limit", 100),
            strategy=strategy,
            min_limit=request.get("min_limit", 1),
            max_limit=request.get("max_limit", 1000),
            adjustment_factor=request.get("adjustment_factor", 0.1),
            metadata=request.get("metadata"),
        )
        return {"success": True, "rule_id": rule_id}
    
    @app.post("/api/v1/throttler/record-metric")
    async def record_throttle_metric(request: Dict[str, Any]):
        """Registrar métrica para throttling."""
        adaptive_throttler.record_metric(
            rule_id=request.get("rule_id", ""),
            success_rate=request.get("success_rate"),
            error_rate=request.get("error_rate"),
            response_time=request.get("response_time"),
            queue_size=request.get("queue_size"),
            system_load=request.get("system_load"),
        )
        return {"success": True}
    
    @app.post("/api/v1/throttler/check")
    async def check_throttle(request: Dict[str, Any]):
        """Verificar throttling."""
        allowed, info = await adaptive_throttler.check_throttle(
            rule_id=request.get("rule_id", ""),
            identifier=request.get("identifier"),
        )
        return info
    
    @app.get("/api/v1/throttler/{rule_id}/status")
    async def get_throttle_status(rule_id: str):
        """Obtener estado de throttling."""
        status = adaptive_throttler.get_throttle_status(rule_id)
        return status
    
    @app.get("/api/v1/throttler/summary")
    async def get_adaptive_throttler_summary():
        """Obtener resumen del limitador."""
        return adaptive_throttler.get_adaptive_throttler_summary()
    
    # Backpressure Manager
    @app.post("/api/v1/backpressure/add-rule")
    async def add_backpressure_rule(request: Dict[str, Any]):
        """Agregar regla de backpressure."""
        rule_id = backpressure_manager.add_rule(
            rule_id=request.get("rule_id", f"rule_{datetime.now().timestamp()}"),
            component_id=request.get("component_id", ""),
            queue_threshold=request.get("queue_threshold", 100),
            error_rate_threshold=request.get("error_rate_threshold", 0.1),
            latency_threshold=request.get("latency_threshold", 2.0),
            metadata=request.get("metadata"),
        )
        return {"success": True, "rule_id": rule_id}
    
    @app.post("/api/v1/backpressure/record-metric")
    async def record_backpressure_metric(request: Dict[str, Any]):
        """Registrar métrica de backpressure."""
        backpressure_manager.record_metric(
            component_id=request.get("component_id", ""),
            queue_size=request.get("queue_size"),
            processing_rate=request.get("processing_rate"),
            arrival_rate=request.get("arrival_rate"),
            error_rate=request.get("error_rate"),
            latency=request.get("latency"),
            system_load=request.get("system_load"),
        )
        return {"success": True}
    
    @app.get("/api/v1/backpressure/{component_id}/level")
    async def get_backpressure_level(component_id: str):
        """Obtener nivel de backpressure."""
        level = backpressure_manager.get_backpressure_level(component_id)
        return {"component_id": component_id, "level": level.value}
    
    @app.post("/api/v1/backpressure/{component_id}/check")
    async def check_should_accept_backpressure(component_id: str):
        """Verificar si se debe aceptar petición."""
        allowed = backpressure_manager.should_accept_request(component_id)
        return {"allowed": allowed, "component_id": component_id}
    
    @app.get("/api/v1/backpressure/status")
    async def get_backpressure_status(component_id: Optional[str] = None):
        """Obtener estado de backpressure."""
        status = backpressure_manager.get_backpressure_status(component_id)
        return status
    
    @app.get("/api/v1/backpressure/history")
    async def get_backpressure_history(component_id: Optional[str] = None, limit: int = 100):
        """Obtener historial de backpressure."""
        history = backpressure_manager.get_backpressure_history(component_id, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/backpressure/summary")
    async def get_backpressure_manager_summary():
        """Obtener resumen del gestor."""
        return backpressure_manager.get_backpressure_manager_summary()
    
    @app.post("/api/v1/chat/sessions", response_model=ChatSessionResponse)
    async def create_session(request: ChatCreateRequest):
        """
        Crear una nueva sesión de chat.
        
        La sesión se iniciará automáticamente si se proporciona initial_message.
        """
        try:
            session = await chat_engine.create_session(
                user_id=request.user_id,
                initial_message=request.initial_message,
                auto_continue=request.auto_continue,
            )
            
            # Si hay mensaje inicial, iniciar chat continuo
            if request.initial_message:
                await chat_engine.start_continuous_chat(
                    session.session_id,
                    request.initial_message,
                )
            
            return ChatSessionResponse(
                session_id=session.session_id,
                state=session.state.value,
                is_paused=session.is_paused,
                message_count=len(session.messages),
                auto_continue=session.auto_continue,
            )
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/chat/sessions/{session_id}", response_model=ChatSessionResponse)
    async def get_session(session_id: str):
        """Obtener información de una sesión."""
        session = chat_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return ChatSessionResponse(
            session_id=session.session_id,
            state=session.state.value,
            is_paused=session.is_paused,
            message_count=len(session.messages),
            auto_continue=session.auto_continue,
        )
    
    @app.get("/api/v1/chat/sessions/{session_id}/messages")
    async def get_messages(session_id: str, limit: int = 50):
        """Obtener mensajes de una sesión."""
        session = chat_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = session.messages[-limit:] if limit else session.messages
        return {
            "session_id": session_id,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in messages
            ],
            "total": len(session.messages),
        }
    
    @app.post("/api/v1/chat/sessions/{session_id}/messages")
    async def send_message(session_id: str, request: ChatMessageRequest):
        """
        Enviar un mensaje a la sesión.
        
        Si la sesión está pausada y auto_continue está activado, se reanudará automáticamente.
        """
        try:
            await chat_engine.add_user_message(session_id, request.message)
            
            # Si la sesión no está activa, iniciar chat continuo
            session = chat_engine.get_session(session_id)
            if session and session.state == ChatState.IDLE:
                await chat_engine.start_continuous_chat(session_id)
            
            return ChatResponse(
                success=True,
                message="Message sent successfully",
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/chat/sessions/{session_id}/start")
    async def start_chat(session_id: str, initial_prompt: Optional[str] = None):
        """
        Iniciar chat continuo para una sesión.
        
        El chat comenzará a generar respuestas automáticamente.
        """
        try:
            await chat_engine.start_continuous_chat(session_id, initial_prompt)
            return ChatResponse(
                success=True,
                message="Continuous chat started",
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error starting chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/chat/sessions/{session_id}/pause")
    async def pause_chat(session_id: str, reason: Optional[str] = None):
        """
        Pausar el chat continuo.
        
        El chat dejará de generar respuestas hasta que se reanude.
        """
        try:
            await chat_engine.pause_session(session_id, reason)
            return ChatResponse(
                success=True,
                message="Chat paused",
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error pausing chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/chat/sessions/{session_id}/resume")
    async def resume_chat(session_id: str):
        """
        Reanudar el chat continuo.
        
        El chat comenzará a generar respuestas nuevamente.
        """
        try:
            await chat_engine.resume_session(session_id)
            return ChatResponse(
                success=True,
                message="Chat resumed",
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error resuming chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/chat/sessions/{session_id}/stop")
    async def stop_chat(session_id: str):
        """
        Detener completamente el chat.
        
        La sesión se detendrá y no podrá reanudarse.
        """
        try:
            await chat_engine.stop_session(session_id)
            return ChatResponse(
                success=True,
                message="Chat stopped",
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error stopping chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/chat/sessions/{session_id}/stream")
    async def stream_chat(session_id: str, message: Optional[str] = None):
        """
        Stream de respuestas en tiempo real.
        
        Retorna Server-Sent Events (SSE) con los tokens generados.
        """
        session = chat_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        async def generate_stream():
            """Generar stream de respuestas."""
            try:
                async for token in chat_engine.stream_response(session_id, message):
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in stream: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    @app.delete("/api/v1/chat/sessions/{session_id}")
    async def delete_session(session_id: str):
        """Eliminar una sesión."""
        try:
            await chat_engine.cleanup_session(session_id)
            return ChatResponse(
                success=True,
                message="Session deleted",
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/chat/sessions")
    async def list_sessions(user_id: Optional[str] = None):
        """Listar todas las sesiones activas."""
        sessions = list(chat_engine.sessions.values())
        
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        
        return {
            "sessions": [s.to_dict() for s in sessions],
            "total": len(sessions),
        }
    
    @app.get("/api/v1/chat/sessions/{session_id}/metrics")
    async def get_session_metrics(session_id: str):
        """Obtener métricas de una sesión."""
        if not chat_engine.metrics:
            raise HTTPException(status_code=503, detail="Metrics not enabled")
        
        metrics = chat_engine.metrics.get_session_metrics(session_id)
        if not metrics:
            raise HTTPException(status_code=404, detail="Session metrics not found")
        
        return metrics
    
    @app.get("/api/v1/chat/metrics")
    async def get_global_metrics():
        """Obtener métricas globales del sistema."""
        if not chat_engine.metrics:
            raise HTTPException(status_code=503, detail="Metrics not enabled")
        
        return chat_engine.metrics.get_global_metrics()
    
    @app.get("/api/v1/chat/rate-limit/{identifier}")
    async def get_rate_limit_stats(identifier: str):
        """Obtener estadísticas de rate limiting."""
        stats = rate_limiter.get_stats(identifier)
        return stats
    
    @app.get("/api/v1/chat/cache/stats")
    async def get_cache_stats():
        """Obtener estadísticas del cache."""
        if not chat_engine.cache:
            raise HTTPException(status_code=503, detail="Cache not enabled")
        return chat_engine.cache.get_stats()
    
    @app.post("/api/v1/chat/cache/clear")
    async def clear_cache():
        """Limpiar el cache."""
        if not chat_engine.cache:
            raise HTTPException(status_code=503, detail="Cache not enabled")
        await chat_engine.cache.clear()
        return {"success": True, "message": "Cache cleared"}
    
    # Endpoints de análisis
    @app.get("/api/v1/chat/sessions/{session_id}/analyze")
    async def analyze_session(session_id: str):
        """Analizar una sesión."""
        session = chat_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        insights = await analyzer.analyze(session)
        return {
            "session_id": session_id,
            "insights": {
                "total_messages": insights.total_messages,
                "user_messages": insights.user_messages,
                "assistant_messages": insights.assistant_messages,
                "average_message_length": insights.average_message_length,
                "topics": insights.topics,
                "key_phrases": insights.key_phrases,
                "conversation_duration": insights.conversation_duration,
                "sentiment_trend": insights.sentiment_trend,
                "response_time_stats": insights.response_time_stats,
            }
        }
    
    @app.get("/api/v1/chat/sessions/{session_id}/summary")
    async def get_session_summary(session_id: str):
        """Obtener resumen de una sesión."""
        session = chat_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        summary = await analyzer.generate_summary(session)
        return {"session_id": session_id, "summary": summary}
    
    # Endpoints de exportación
    @app.get("/api/v1/chat/sessions/{session_id}/export/{format}")
    async def export_session(session_id: str, format: str):
        """Exportar sesión en diferentes formatos."""
        session = chat_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        format = format.lower()
        
        if format == "json":
            content = await exporter.export_json(session)
            return JSONResponse(content=json.loads(content))
        elif format == "markdown" or format == "md":
            content = await exporter.export_markdown(session)
            return JSONResponse(content={"content": content, "format": "markdown"})
        elif format == "csv":
            content = await exporter.export_csv(session)
            return JSONResponse(content={"content": content, "format": "csv"})
        elif format == "txt" or format == "text":
            content = await exporter.export_txt(session)
            return JSONResponse(content={"content": content, "format": "text"})
        elif format == "html":
            content = await exporter.export_html(session)
            return JSONResponse(content={"content": content, "format": "html"})
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
    
    # Endpoints de templates
    @app.get("/api/v1/chat/templates")
    async def list_templates(category: Optional[str] = None):
        """Listar plantillas."""
        templates = template_manager.list_templates(category)
        return {
            "templates": [
                {
                    "id": t.id,
                    "name": t.name,
                    "category": t.category,
                    "variables": t.variables,
                }
                for t in templates
            ]
        }
    
    @app.post("/api/v1/chat/templates/{template_id}/render")
    async def render_template(template_id: str, variables: Optional[Dict[str, str]] = None):
        """Renderizar una plantilla."""
        content = template_manager.render(template_id, variables)
        if content is None:
            raise HTTPException(status_code=404, detail="Template not found")
        return {"template_id": template_id, "content": content}
    
    # Endpoints de webhooks
    @app.post("/api/v1/chat/webhooks")
    async def register_webhook(webhook: Dict[str, Any]):
        """Registrar un webhook."""
        from ..core.webhooks import Webhook
        
        webhook_obj = Webhook(
            url=webhook["url"],
            events=[WebhookEvent(e) for e in webhook.get("events", [])],
            secret=webhook.get("secret"),
            enabled=webhook.get("enabled", True),
        )
        
        await webhook_manager.register(webhook_obj)
        return {"success": True, "message": "Webhook registered"}
    
    @app.get("/api/v1/chat/webhooks")
    async def list_webhooks():
        """Listar webhooks registrados."""
        webhooks = webhook_manager.get_webhooks()
        return {
            "webhooks": [
                {
                    "url": w.url,
                    "events": [e.value for e in w.events],
                    "enabled": w.enabled,
                }
                for w in webhooks
            ]
        }
    
    # Endpoints de autenticación
    @app.post("/api/v1/auth/register")
    async def register_user(request: Dict[str, Any]):
        """Registrar nuevo usuario."""
        user = auth_manager.create_user(
            username=request["username"],
            password=request["password"],
            email=request.get("email"),
        )
        token = auth_manager.create_access_token(user)
        return {
            "user_id": user.user_id,
            "username": user.username,
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expires_in": token.expires_in,
        }
    
    @app.post("/api/v1/auth/login")
    async def login(request: Dict[str, Any]):
        """Iniciar sesión."""
        user = auth_manager.authenticate(
            request["username"],
            request["password"],
        )
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        token = auth_manager.create_access_token(user)
        return {
            "user_id": user.user_id,
            "username": user.username,
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expires_in": token.expires_in,
        }
    
    # Endpoints de backup
    @app.post("/api/v1/chat/backup/create")
    async def create_backup():
        """Crear backup manual."""
        backup_path = await backup_manager.create_backup()
        if not backup_path:
            raise HTTPException(status_code=500, detail="Failed to create backup")
        return {"success": True, "backup_path": backup_path}
    
    @app.get("/api/v1/chat/backup/list")
    async def list_backups():
        """Listar backups disponibles."""
        backups = backup_manager.list_backups()
        return {"backups": backups}
    
    @app.get("/api/v1/chat/backup/history")
    async def get_backup_history():
        """Obtener historial de backups."""
        history = backup_manager.get_backup_history()
        return {"history": history}
    
    # Endpoints de Bulk Operations
    from ..core.bulk_operations import (
        BulkSessionOperations,
        BulkMessageOperations,
        BulkExporter,
        BulkAnalytics,
        BulkCleanup,
        BulkProcessor,
        BulkImporter,
        BulkNotifications,
        BulkSearch,
        BulkTesting,
        BulkBackupRestore,
        BulkMigration,
        BulkMetrics,
        BulkRealTimeMetrics,
        BulkAdvancedCache,
        BulkPriorityQueue,
        BulkEnhancedValidator,
        BulkDashboard,
        BulkScheduler,
        BulkRateLimiter,
        BulkAutoCreator,
        BulkAutoExpander,
        BulkAutoProcessor,
        BulkAutoMaintainer,
        BulkInfiniteGenerator,
        BulkSelfSustaining
    )
    
    # Inicializar bulk operations
    bulk_sessions = BulkSessionOperations(
        chat_engine=chat_engine,
        storage=storage,
        max_workers=10
    )
    bulk_messages = BulkMessageOperations(
        chat_engine=chat_engine,
        max_workers=10
    )
    bulk_exporter = BulkExporter(
        storage=storage,
        exporter=exporter,
        max_workers=10
    )
    bulk_analytics = BulkAnalytics(
        analyzer=analyzer,
        storage=storage,
        max_workers=10
    )
    bulk_cleanup = BulkCleanup(
        storage=storage,
        chat_engine=chat_engine,
        max_workers=10
    )
    bulk_processor = BulkProcessor(max_workers=10)
    bulk_importer = BulkImporter(
        chat_engine=chat_engine,
        storage=storage,
        max_workers=10
    )
    bulk_notifications = BulkNotifications(
        notification_manager=notification_manager,
        max_workers=10
    )
    bulk_search = BulkSearch(
        search_engine=search_engine,
        storage=storage,
        max_workers=10
    )
    bulk_testing = BulkTesting(
        chat_engine=chat_engine,
        max_workers=50
    )
    bulk_backup = BulkBackupRestore(
        storage=storage,
        backup_manager=backup_manager,
        max_workers=10
    )
    bulk_migration = BulkMigration(
        chat_engine=chat_engine,
        storage=storage,
        max_workers=10
    )
    bulk_metrics = BulkMetrics()
    bulk_scheduler = BulkScheduler()
    bulk_rate_limiter = BulkRateLimiter(
        max_operations_per_minute=100,
        max_operations_per_hour=1000
    )
    bulk_auto_creator = BulkAutoCreator(
        chat_engine=chat_engine,
        storage=storage,
        config={"max_workers": 10, "batch_size": 10}
    )
    bulk_auto_expander = BulkAutoExpander(
        chat_engine=chat_engine,
        storage=storage
    )
    bulk_auto_processor = BulkAutoProcessor(
        chat_engine=chat_engine,
        storage=storage
    )
    bulk_auto_maintainer = BulkAutoMaintainer(
        chat_engine=chat_engine,
        storage=storage
    )
    bulk_infinite_generator = BulkInfiniteGenerator(
        chat_engine=chat_engine,
        storage=storage
    )
    bulk_self_sustaining = BulkSelfSustaining(
        chat_engine=chat_engine,
        storage=storage,
        config={"max_workers": 10, "batch_size": 10, "max_capacity": 100000}
    )
    
    # Nuevas clases avanzadas (ya importadas arriba)
    
    bulk_realtime_metrics = BulkRealTimeMetrics(window_size_seconds=60)
    bulk_advanced_cache = BulkAdvancedCache(max_size=1000, default_ttl=3600)
    bulk_priority_queue = BulkPriorityQueue()
    bulk_enhanced_validator = BulkEnhancedValidator()
    bulk_dashboard = BulkDashboard(
        metrics=bulk_realtime_metrics,
        cache=bulk_advanced_cache,
        priority_queue=bulk_priority_queue
    )
    
    # Bulk Session Operations
    @app.post("/api/v1/bulk/sessions/create")
    async def bulk_create_sessions(request: Dict[str, Any]):
        """Crear múltiples sesiones en lote."""
        count = request.get("count", 1)
        initial_messages = request.get("initial_messages")
        auto_continue = request.get("auto_continue", True)
        parallel = request.get("parallel", True)
        user_id = request.get("user_id")
        
        session_ids = await bulk_sessions.create_sessions(
            count=count,
            initial_messages=initial_messages,
            auto_continue=auto_continue,
            parallel=parallel,
            user_id=user_id
        )
        return {
            "success": True,
            "created": len(session_ids),
            "session_ids": session_ids
        }
    
    @app.post("/api/v1/bulk/sessions/delete")
    async def bulk_delete_sessions(request: Dict[str, Any]):
        """Eliminar múltiples sesiones."""
        session_ids = request.get("session_ids", [])
        parallel = request.get("parallel", True)
        
        result = await bulk_sessions.delete_sessions(
            session_ids=session_ids,
            parallel=parallel
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "total": result.total,
            "duration": result.duration,
            "errors": result.errors[:10]  # Limitar errores
        }
    
    @app.post("/api/v1/bulk/sessions/pause")
    async def bulk_pause_sessions(request: Dict[str, Any]):
        """Pausar múltiples sesiones."""
        session_ids = request.get("session_ids", [])
        reason = request.get("reason")
        parallel = request.get("parallel", True)
        
        result = await bulk_sessions.pause_sessions(
            session_ids=session_ids,
            reason=reason,
            parallel=parallel
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "duration": result.duration
        }
    
    @app.post("/api/v1/bulk/sessions/resume")
    async def bulk_resume_sessions(request: Dict[str, Any]):
        """Reanudar múltiples sesiones."""
        session_ids = request.get("session_ids", [])
        parallel = request.get("parallel", True)
        
        result = await bulk_sessions.resume_sessions(
            session_ids=session_ids,
            parallel=parallel
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "duration": result.duration
        }
    
    @app.post("/api/v1/bulk/sessions/stop")
    async def bulk_stop_sessions(request: Dict[str, Any]):
        """Detener múltiples sesiones."""
        session_ids = request.get("session_ids", [])
        parallel = request.get("parallel", True)
        
        result = await bulk_sessions.stop_sessions(
            session_ids=session_ids,
            parallel=parallel
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "duration": result.duration
        }
    
    @app.post("/api/v1/bulk/sessions/export")
    async def bulk_export_sessions_direct(request: Dict[str, Any]):
        """Exportar múltiples sesiones (método directo)."""
        session_ids = request.get("session_ids", [])
        format = request.get("format", "json")
        parallel = request.get("parallel", True)
        
        result = await bulk_sessions.export_sessions(
            session_ids=session_ids,
            format=format,
            parallel=parallel
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "total": result.total,
            "duration": result.duration,
            "exports": result.data.get("exports", []) if result.data else []
        }
    
    # Bulk Message Operations
    @app.post("/api/v1/bulk/messages/send")
    async def bulk_send_messages(request: Dict[str, Any]):
        """Enviar mensaje a múltiples sesiones."""
        session_ids = request.get("session_ids", [])
        message = request.get("message", "")
        parallel = request.get("parallel", True)
        
        result = await bulk_messages.send_to_sessions(
            session_ids=session_ids,
            message=message,
            parallel=parallel
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "duration": result.duration
        }
    
    # Bulk Export
    @app.post("/api/v1/bulk/export/sessions")
    async def bulk_export_sessions(request: Dict[str, Any]):
        """Exportar múltiples sesiones."""
        session_ids = request.get("session_ids", [])
        format = request.get("format", "json")
        compress = request.get("compress", False)
        parallel = request.get("parallel", True)
        
        job_id = await bulk_exporter.export_sessions(
            session_ids=session_ids,
            format=format,
            compress=compress,
            parallel=parallel
        )
        return {
            "success": True,
            "job_id": job_id,
            "message": "Export job started"
        }
    
    @app.get("/api/v1/bulk/export/status/{job_id}")
    async def get_export_status(job_id: str):
        """Obtener estado de exportación."""
        status = await bulk_exporter.get_export_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        return status
    
    # Bulk Analytics
    @app.post("/api/v1/bulk/analytics/sessions")
    async def bulk_analyze_sessions(request: Dict[str, Any]):
        """Analizar múltiples sesiones."""
        session_ids = request.get("session_ids", [])
        parallel = request.get("parallel", True)
        
        results = await bulk_analytics.analyze_sessions_bulk(
            session_ids=session_ids,
            parallel=parallel
        )
        return {
            "success": True,
            "analyzed": len(results),
            "results": results
        }
    
    # Bulk Cleanup
    @app.post("/api/v1/bulk/cleanup/sessions")
    async def bulk_cleanup_sessions(request: Dict[str, Any]):
        """Limpiar sesiones antiguas."""
        days_old = request.get("days_old", 30)
        dry_run = request.get("dry_run", False)
        
        result = await bulk_cleanup.cleanup_old_sessions(
            days_old=days_old,
            dry_run=dry_run
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "total": result.total,
            "duration": result.duration,
            "dry_run": dry_run
        }
    
    # Bulk Import
    @app.post("/api/v1/bulk/import/sessions")
    async def bulk_import_sessions(request: Dict[str, Any]):
        """Importar múltiples sesiones."""
        sessions_data = request.get("sessions", [])
        validate = request.get("validate", True)
        parallel = request.get("parallel", True)
        
        job_id = await bulk_importer.import_sessions(
            sessions_data=sessions_data,
            validate=validate,
            parallel=parallel
        )
        return {
            "success": True,
            "job_id": job_id,
            "message": "Import job started"
        }
    
    @app.get("/api/v1/bulk/import/status/{job_id}")
    async def get_import_status(job_id: str):
        """Obtener estado de importación."""
        status = await bulk_importer.get_import_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        return status
    
    # Bulk Notifications
    @app.post("/api/v1/bulk/notifications/send")
    async def bulk_send_notifications(request: Dict[str, Any]):
        """Enviar notificaciones masivas."""
        user_ids = request.get("user_ids", [])
        template = request.get("template", "")
        data = request.get("data", {})
        channels = request.get("channels", ["email"])
        parallel = request.get("parallel", True)
        
        result = await bulk_notifications.send_bulk(
            user_ids=user_ids,
            template=template,
            data=data,
            channels=channels,
            parallel=parallel
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "duration": result.duration
        }
    
    # Bulk Search
    @app.post("/api/v1/bulk/search/execute")
    async def bulk_search_execute(request: Dict[str, Any]):
        """Ejecutar búsqueda masiva."""
        queries = request.get("queries", [])
        filters = request.get("filters", {})
        parallel = request.get("parallel", True)
        
        results = await bulk_search.search_bulk(
            queries=queries,
            filters=filters,
            parallel=parallel
        )
        return {
            "success": True,
            "queries_count": len(queries),
            "results": results
        }
    
    # Bulk Processor
    @app.get("/api/v1/bulk/process/status/{job_id}")
    async def get_process_status(job_id: str):
        """Obtener estado de procesamiento."""
        status = await bulk_processor.get_progress(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        return status
    
    @app.post("/api/v1/bulk/process/cancel/{job_id}")
    async def cancel_process_job(job_id: str):
        """Cancelar job de procesamiento."""
        cancelled = await bulk_processor.cancel_job(job_id)
        if not cancelled:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled")
        return {"success": True, "message": "Job cancelled"}
    
    # Bulk Backup/Restore
    @app.post("/api/v1/bulk/backup/sessions")
    async def bulk_backup_sessions(request: Dict[str, Any]):
        """Crear backup masivo de sesiones."""
        session_ids = request.get("session_ids", [])
        compress = request.get("compress", True)
        encrypt = request.get("encrypt", False)
        
        job_id = await bulk_backup.backup_sessions(
            session_ids=session_ids,
            compress=compress,
            encrypt=encrypt
        )
        return {
            "success": True,
            "job_id": job_id,
            "message": "Backup job started"
        }
    
    @app.get("/api/v1/bulk/backup/status/{job_id}")
    async def get_backup_status_bulk(job_id: str):
        """Obtener estado de backup masivo."""
        status = await bulk_backup.get_backup_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        return status
    
    # Bulk Testing
    @app.post("/api/v1/bulk/testing/load-test")
    async def bulk_load_test(request: Dict[str, Any]):
        """Ejecutar test de carga masivo."""
        concurrent_sessions = request.get("concurrent_sessions", 100)
        duration = request.get("duration", 60)
        operations_per_session = request.get("operations_per_session", 10)
        
        results = await bulk_testing.load_test(
            concurrent_sessions=concurrent_sessions,
            duration=duration,
            operations_per_session=operations_per_session
        )
        return {
            "success": True,
            "results": results
        }
    
    @app.post("/api/v1/bulk/testing/stress-test")
    async def bulk_stress_test(request: Dict[str, Any]):
        """Ejecutar test de estrés."""
        max_sessions = request.get("max_sessions", 1000)
        ramp_up_seconds = request.get("ramp_up_seconds", 60)
        
        results = await bulk_testing.stress_test(
            max_sessions=max_sessions,
            ramp_up_seconds=ramp_up_seconds
        )
        return {
            "success": True,
            "results": results
        }
    
    # Bulk Migration
    @app.post("/api/v1/bulk/migration/start")
    async def bulk_migration_start(request: Dict[str, Any]):
        """Iniciar migración masiva."""
        session_ids = request.get("session_ids", [])
        source_format = request.get("source_format", "v1")
        target_format = request.get("target_format", "v2")
        batch_size = request.get("batch_size", 100)
        
        job_id = await bulk_migration.migrate_sessions(
            session_ids=session_ids,
            source_format=source_format,
            target_format=target_format,
            batch_size=batch_size
        )
        return {
            "success": True,
            "job_id": job_id,
            "message": "Migration job started"
        }
    
    @app.get("/api/v1/bulk/migration/status/{job_id}")
    async def get_migration_status(job_id: str):
        """Obtener estado de migración."""
        status = await bulk_migration.get_migration_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        return status
    
    # Bulk Metrics
    @app.get("/api/v1/bulk/metrics/stats")
    async def get_bulk_metrics_stats(operation: Optional[str] = None):
        """Obtener estadísticas de operaciones bulk."""
        stats = bulk_metrics.get_stats(operation)
        return {"stats": stats}
    
    @app.get("/api/v1/bulk/metrics/history")
    async def get_bulk_metrics_history(limit: int = 100):
        """Obtener historial de operaciones bulk."""
        history = bulk_metrics.get_history(limit)
        return {"history": history}
    
    @app.get("/api/v1/bulk/metrics/summary")
    async def get_bulk_metrics_summary():
        """Obtener resumen de operaciones bulk."""
        summary = bulk_metrics.get_summary()
        return summary
    
    # Bulk Scheduler
    @app.post("/api/v1/bulk/scheduler/schedule")
    async def bulk_scheduler_schedule(request: Dict[str, Any]):
        """Programar operación bulk recurrente."""
        # Nota: Esto requiere una función callable, por lo que es simplificado
        job_id = request.get("job_id")
        schedule = request.get("schedule", "0 2 * * *")
        enabled = request.get("enabled", True)
        config = request.get("config", {})
        
        # En producción, esto necesitaría un registry de operaciones
        return {
            "success": False,
            "message": "Scheduler requires callable operation. Use programmatic API."
        }
    
    @app.get("/api/v1/bulk/scheduler/jobs")
    async def list_scheduled_jobs():
        """Listar jobs programados."""
        jobs = bulk_scheduler.list_jobs()
        return {"jobs": jobs}
    
    @app.post("/api/v1/bulk/scheduler/{job_id}/enable")
    async def enable_scheduled_job(job_id: str):
        """Habilitar job programado."""
        await bulk_scheduler.enable_job(job_id)
        return {"success": True, "message": "Job enabled"}
    
    @app.post("/api/v1/bulk/scheduler/{job_id}/disable")
    async def disable_scheduled_job(job_id: str):
        """Deshabilitar job programado."""
        await bulk_scheduler.disable_job(job_id)
        return {"success": True, "message": "Job disabled"}
    
    # Bulk Rate Limiter
    @app.get("/api/v1/bulk/rate-limit/stats")
    async def get_bulk_rate_limit_stats(operation: Optional[str] = None):
        """Obtener estadísticas de rate limiting."""
        stats = bulk_rate_limiter.get_stats(operation)
        return {"stats": stats}
    
    @app.post("/api/v1/bulk/rate-limit/check")
    async def check_bulk_rate_limit(request: Dict[str, Any]):
        """Verificar rate limit."""
        operation = request.get("operation", "")
        user_id = request.get("user_id")
        
        allowed, error = await bulk_rate_limiter.check_rate_limit(
            operation=operation,
            user_id=user_id
        )
        
        return {
            "allowed": allowed,
            "error": error
        }
    
    # Bulk Auto-Creation (Sistema que nunca se detiene)
    @app.post("/api/v1/bulk/auto-creation/start")
    async def start_auto_creation(request: Dict[str, Any]):
        """Iniciar auto-creación continua que nunca se detiene."""
        creation_rate = request.get("creation_rate", 1.0)
        batch_size = request.get("batch_size", 10)
        initial_messages = request.get("initial_messages")
        auto_continue = request.get("auto_continue", True)
        
        await bulk_auto_creator.start_continuous_creation(
            creation_rate=creation_rate,
            batch_size=batch_size,
            initial_messages=initial_messages,
            auto_continue=auto_continue
        )
        
        return {
            "success": True,
            "message": "Auto-creation started - will continue indefinitely",
            "creation_rate": creation_rate,
            "batch_size": batch_size
        }
    
    @app.post("/api/v1/bulk/auto-creation/stop")
    async def stop_auto_creation():
        """Detener auto-creación (opcional)."""
        await bulk_auto_creator.stop_continuous_creation()
        return {"success": True, "message": "Auto-creation stopped"}
    
    @app.get("/api/v1/bulk/auto-creation/stats")
    async def get_auto_creation_stats():
        """Obtener estadísticas de auto-creación."""
        stats = bulk_auto_creator.get_stats()
        return stats
    
    # Bulk Auto-Expansion
    @app.post("/api/v1/bulk/auto-expansion/start")
    async def start_auto_expansion(request: Dict[str, Any]):
        """Iniciar auto-expansión continua."""
        check_interval = request.get("check_interval", 60.0)
        expansion_rate = request.get("expansion_rate", 1.1)
        max_capacity = request.get("max_capacity", 100000)
        
        await bulk_auto_expander.start_auto_expansion(
            check_interval=check_interval,
            expansion_rate=expansion_rate,
            max_capacity=max_capacity
        )
        
        return {
            "success": True,
            "message": "Auto-expansion started",
            "target_capacity": max_capacity
        }
    
    # Bulk Self-Sustaining System
    @app.post("/api/v1/bulk/self-sustaining/start")
    async def start_self_sustaining(request: Dict[str, Any]):
        """Iniciar sistema auto-sostenible completo."""
        creation_rate = request.get("creation_rate", 1.0)
        expansion_enabled = request.get("expansion_enabled", True)
        processing_enabled = request.get("processing_enabled", True)
        maintenance_enabled = request.get("maintenance_enabled", True)
        
        await bulk_self_sustaining.start_self_sustaining_system(
            creation_rate=creation_rate,
            expansion_enabled=expansion_enabled,
            processing_enabled=processing_enabled,
            maintenance_enabled=maintenance_enabled
        )
        
        return {
            "success": True,
            "message": "Self-sustaining system started - will run indefinitely",
            "components": {
                "auto_creation": True,
                "auto_expansion": expansion_enabled,
                "auto_processing": processing_enabled,
                "auto_maintenance": maintenance_enabled
            }
        }
    
    @app.get("/api/v1/bulk/self-sustaining/stats")
    async def get_self_sustaining_stats():
        """Obtener estadísticas del sistema auto-sostenible."""
        stats = bulk_self_sustaining.get_system_stats()
        return stats
    
    @app.post("/api/v1/bulk/self-sustaining/ensure-continuity")
    async def ensure_continuity():
        """Asegurar que el sistema continúe operando."""
        await bulk_self_sustaining.ensure_continuous_operation()
        return {
            "success": True,
            "message": "System continuity verified and maintained"
        }
    
    # Bulk Infinite Generator
    @app.post("/api/v1/bulk/infinite-generator/create")
    async def create_infinite_generator(request: Dict[str, Any]):
        """Crear generador infinito de sesiones."""
        generator_id = request.get("generator_id", f"gen_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        creation_rate = request.get("creation_rate", 0.5)
        initial_messages = request.get("initial_messages")
        
        gen = await bulk_infinite_generator.create_infinite_session_generator(
            generator_id=generator_id,
            creation_rate=creation_rate,
            initial_messages=initial_messages
        )
        
        # Consumir en background
        async def consume():
            async for session_id in gen:
                logger.info(f"Infinite generator created session: {session_id}")
        
        asyncio.create_task(consume())
        
        return {
            "success": True,
            "generator_id": generator_id,
            "message": "Infinite generator started - will create sessions indefinitely"
        }
    
    # Nuevos endpoints avanzados
    @app.get("/api/v1/bulk/metrics/realtime")
    async def get_realtime_metrics(operation_type: Optional[str] = None):
        """Obtener métricas en tiempo real."""
        metrics = bulk_realtime_metrics.get_metrics(operation_type)
        return metrics
    
    @app.get("/api/v1/bulk/metrics/health")
    async def get_metrics_health():
        """Obtener estado de salud basado en métricas."""
        health = bulk_realtime_metrics.get_health_status()
        return health
    
    @app.post("/api/v1/bulk/metrics/record")
    async def record_metric(request: Dict[str, Any]):
        """Registrar una métrica."""
        bulk_realtime_metrics.record_operation(
            operation_type=request.get("operation_type", "unknown"),
            duration=request.get("duration", 0.0),
            success=request.get("success", True),
            items_processed=request.get("items_processed", 1),
            metadata=request.get("metadata")
        )
        return {"success": True, "message": "Metric recorded"}
    
    @app.get("/api/v1/bulk/cache/stats")
    async def get_cache_stats():
        """Obtener estadísticas del caché."""
        stats = bulk_advanced_cache.get_stats()
        return stats
    
    @app.post("/api/v1/bulk/cache/get")
    async def cache_get(request: Dict[str, Any]):
        """Obtener valor del caché."""
        key = request.get("key")
        if not key:
            raise HTTPException(status_code=400, detail="Key is required")
        value = bulk_advanced_cache.get(key)
        return {"found": value is not None, "value": value}
    
    @app.post("/api/v1/bulk/cache/set")
    async def cache_set(request: Dict[str, Any]):
        """Guardar valor en caché."""
        key = request.get("key")
        value = request.get("value")
        ttl = request.get("ttl")
        if not key:
            raise HTTPException(status_code=400, detail="Key is required")
        bulk_advanced_cache.set(key, value, ttl)
        return {"success": True, "message": "Value cached"}
    
    @app.post("/api/v1/bulk/cache/invalidate")
    async def cache_invalidate(request: Dict[str, Any]):
        """Invalidar entrada del caché."""
        key = request.get("key")
        if not key:
            raise HTTPException(status_code=400, detail="Key is required")
        bulk_advanced_cache.invalidate(key)
        return {"success": True, "message": "Cache entry invalidated"}
    
    @app.get("/api/v1/bulk/queue/stats")
    async def get_queue_stats():
        """Obtener estadísticas de la cola de prioridades."""
        stats = bulk_priority_queue.get_stats()
        return stats
    
    @app.post("/api/v1/bulk/queue/enqueue")
    async def queue_enqueue(request: Dict[str, Any]):
        """Agregar operación a la cola."""
        operation = request.get("operation", {})
        priority = request.get("priority", "medium")
        bulk_priority_queue.enqueue(operation, priority)
        return {"success": True, "message": "Operation enqueued", "queue_size": bulk_priority_queue.size()}
    
    @app.post("/api/v1/bulk/queue/dequeue")
    async def queue_dequeue():
        """Obtener siguiente operación de la cola."""
        operation = bulk_priority_queue.dequeue()
        if not operation:
            raise HTTPException(status_code=404, detail="Queue is empty")
        return {"success": True, "operation": operation}
    
    @app.get("/api/v1/bulk/queue/peek")
    async def queue_peek():
        """Ver siguiente operación sin removerla."""
        operation = bulk_priority_queue.peek()
        if not operation:
            raise HTTPException(status_code=404, detail="Queue is empty")
        return {"operation": operation}
    
    @app.post("/api/v1/bulk/validator/validate")
    async def validate_bulk(request: Dict[str, Any]):
        """Validar datos."""
        operation_type = request.get("operation_type", "default")
        data = request.get("data", {})
        use_cache = request.get("use_cache", True)
        
        is_valid, error = bulk_enhanced_validator.validate(operation_type, data, use_cache)
        return {
            "valid": is_valid,
            "error": error
        }
    
    @app.post("/api/v1/bulk/validator/validate-batch")
    async def validate_batch(request: Dict[str, Any]):
        """Validar lote de items."""
        operation_type = request.get("operation_type", "default")
        items = request.get("items", [])
        
        results = bulk_enhanced_validator.validate_batch(operation_type, items)
        return results
    
    @app.post("/api/v1/bulk/validator/add-rule")
    async def add_validator_rule(request: Dict[str, Any]):
        """Añadir regla de validación."""
        operation_type = request.get("operation_type")
        # Nota: En producción, necesitarías deserializar la función desde string o usar un registro
        # Por ahora, retornamos un mensaje
        return {
            "success": True,
            "message": "Rule registration endpoint - implement validator function registration"
        }
    
    @app.get("/api/v1/bulk/dashboard")
    async def get_dashboard():
        """Obtener datos completos del dashboard."""
        dashboard_data = bulk_dashboard.get_dashboard_data()
        return dashboard_data
    
    @app.get("/api/v1/bulk/dashboard/summary")
    async def get_dashboard_summary():
        """Obtener resumen ejecutivo del dashboard."""
        summary = bulk_dashboard._generate_summary()
        return summary
    
    @app.post("/api/v1/bulk/dashboard/alert")
    async def add_dashboard_alert(request: Dict[str, Any]):
        """Añadir alerta al dashboard."""
        level = request.get("level", "info")
        message = request.get("message", "")
        details = request.get("details", {})
        bulk_dashboard.add_alert(level, message, details)
        return {"success": True, "message": "Alert added"}
    
    @app.get("/api/v1/bulk/dashboard/alerts")
    async def get_dashboard_alerts(level: Optional[str] = None):
        """Obtener alertas del dashboard."""
        alerts = bulk_dashboard.get_alerts(level)
        return {"alerts": alerts, "count": len(alerts)}
    
    # Nuevas funcionalidades avanzadas
    from ..core.bulk_operations import (
        BulkBenchmark,
        BulkAutoTuner
    )
    
    bulk_benchmark = BulkBenchmark()
    bulk_auto_tuner = BulkAutoTuner(
        metrics=bulk_realtime_metrics,
        benchmark=bulk_benchmark
    )
    
    @app.post("/api/v1/bulk/benchmark/run")
    async def run_benchmark(request: Dict[str, Any]):
        """Ejecutar benchmark de una operación."""
        operation_name = request.get("operation_name", "unknown")
        iterations = request.get("iterations", 3)
        warmup = request.get("warmup", 1)
        test_data = request.get("test_data", [])
        
        # Nota: En producción necesitarías deserializar la función
        # Por ahora retornamos un mensaje
        return {
            "success": True,
            "message": "Benchmark endpoint - implement operation execution",
            "operation_name": operation_name,
            "iterations": iterations
        }
    
    @app.get("/api/v1/bulk/benchmark/summary")
    async def get_benchmark_summary():
        """Obtener resumen de benchmarks."""
        summary = bulk_benchmark.get_summary()
        return summary
    
    @app.post("/api/v1/bulk/benchmark/compare")
    async def compare_benchmarks(request: Dict[str, Any]):
        """Comparar dos operaciones."""
        op1 = request.get("operation1")
        op2 = request.get("operation2")
        
        if not op1 or not op2:
            raise HTTPException(status_code=400, detail="Both operations required")
        
        comparison = bulk_benchmark.compare_operations(op1, op2)
        return comparison
    
    @app.post("/api/v1/bulk/autotune/batch-size")
    async def autotune_batch_size(request: Dict[str, Any]):
        """Auto-ajustar tamaño de batch."""
        min_batch = request.get("min_batch", 10)
        max_batch = request.get("max_batch", 1000)
        step = request.get("step", 50)
        iterations = request.get("iterations", 3)
        test_data = request.get("test_data", [])
        
        # Nota: Necesita implementación de ejecución de operación
        return {
            "success": True,
            "message": "Auto-tune endpoint - implement operation execution",
            "range": {"min": min_batch, "max": max_batch, "step": step}
        }
    
    @app.post("/api/v1/bulk/autotune/workers")
    async def autotune_workers(request: Dict[str, Any]):
        """Auto-ajustar número de workers."""
        min_workers = request.get("min_workers", 1)
        max_workers = request.get("max_workers", 50)
        step = request.get("step", 5)
        test_data = request.get("test_data", [])
        
        # Nota: Necesita implementación de ejecución de operación
        return {
            "success": True,
            "message": "Auto-tune workers endpoint - implement operation execution",
            "range": {"min": min_workers, "max": max_workers, "step": step}
        }
    
    @app.get("/api/v1/bulk/autotune/recommendations")
    async def get_tuning_recommendations():
        """Obtener recomendaciones de tuning."""
        recommendations = bulk_auto_tuner.get_tuning_recommendations()
        return recommendations
    
    # Sistemas avanzados de resiliencia y observabilidad
    from ..core.bulk_operations import (
        BulkAdaptiveRateLimiter,
        BulkLoadBalancer,
        BulkLoadPredictor,
        BulkAutoScaler,
        BulkEventSourcing,
        BulkObservability,
        BulkCostOptimizer,
        BulkAnomalyDetector
    )
    
    bulk_adaptive_rate_limiter = BulkAdaptiveRateLimiter(
        initial_rate=100,
        min_rate=10,
        max_rate=1000
    )
    bulk_load_balancer = BulkLoadBalancer(initial_workers=10)
    bulk_load_predictor = BulkLoadPredictor(window_size=60)
    bulk_auto_scaler = BulkAutoScaler(
        load_balancer=bulk_load_balancer,
        load_predictor=bulk_load_predictor,
        metrics=bulk_realtime_metrics,
        min_workers=1,
        max_workers=100
    )
    
    @app.get("/api/v1/bulk/rate-limiter/status")
    async def get_rate_limiter_status():
        """Obtener estado del rate limiter adaptativo."""
        return bulk_adaptive_rate_limiter.get_status()
    
    @app.post("/api/v1/bulk/rate-limiter/check")
    async def check_rate_limit():
        """Verificar si se puede procesar una petición."""
        can_proceed = bulk_adaptive_rate_limiter.can_proceed()
        return {
            "can_proceed": can_proceed,
            "status": bulk_adaptive_rate_limiter.get_status()
        }
    
    @app.post("/api/v1/bulk/rate-limiter/record")
    async def record_rate_limit_result(request: Dict[str, Any]):
        """Registrar resultado de una petición."""
        success = request.get("success", True)
        response_time = request.get("response_time", 0.0)
        
        if success:
            bulk_adaptive_rate_limiter.record_success(response_time)
        else:
            bulk_adaptive_rate_limiter.record_failure()
        
        return {"success": True, "status": bulk_adaptive_rate_limiter.get_status()}
    
    @app.get("/api/v1/bulk/load-balancer/stats")
    async def get_load_balancer_stats():
        """Obtener estadísticas del load balancer."""
        return bulk_load_balancer.get_stats()
    
    @app.post("/api/v1/bulk/load-balancer/select-worker")
    async def select_worker():
        """Seleccionar worker para nueva tarea."""
        worker_id = bulk_load_balancer.select_worker()
        bulk_load_balancer.assign_task(worker_id)
        return {
            "worker_id": worker_id,
            "stats": bulk_load_balancer.get_stats()
        }
    
    @app.post("/api/v1/bulk/load-balancer/complete-task")
    async def complete_load_balancer_task(request: Dict[str, Any]):
        """Marcar tarea como completada."""
        worker_id = request.get("worker_id")
        success = request.get("success", True)
        response_time = request.get("response_time", 0.0)
        
        if worker_id is None:
            raise HTTPException(status_code=400, detail="worker_id required")
        
        bulk_load_balancer.complete_task(worker_id, success, response_time)
        return {"success": True, "stats": bulk_load_balancer.get_stats()}
    
    @app.post("/api/v1/bulk/load-balancer/add-worker")
    async def add_load_balancer_worker():
        """Añadir nuevo worker al load balancer."""
        worker_id = bulk_load_balancer.add_worker()
        return {
            "success": True,
            "worker_id": worker_id,
            "stats": bulk_load_balancer.get_stats()
        }
    
    @app.get("/api/v1/bulk/load-predictor/predict")
    async def predict_load(minutes_ahead: int = 5):
        """Predecir carga futura."""
        prediction = bulk_load_predictor.predict_load(minutes_ahead)
        return prediction
    
    @app.get("/api/v1/bulk/load-predictor/pattern")
    async def get_load_pattern():
        """Obtener patrón de carga por hora."""
        pattern = bulk_load_predictor.get_load_pattern()
        return pattern
    
    @app.post("/api/v1/bulk/load-predictor/record")
    async def record_load(request: Dict[str, Any]):
        """Registrar carga actual."""
        load = request.get("load", 0.0)
        bulk_load_predictor.record_load(load)
        return {"success": True}
    
    @app.get("/api/v1/bulk/autoscaler/evaluate")
    async def evaluate_scaling():
        """Evaluar si es necesario hacer scaling."""
        evaluation = bulk_auto_scaler.evaluate_scaling()
        return evaluation
    
    @app.post("/api/v1/bulk/autoscaler/execute")
    async def execute_scaling():
        """Ejecutar scaling si es necesario."""
        result = await bulk_auto_scaler.execute_scaling()
        return result
    
    @app.get("/api/v1/bulk/autoscaler/history")
    async def get_scaling_history():
        """Obtener historial de scaling."""
        history = bulk_auto_scaler.get_scaling_history()
        return {"history": history, "count": len(history)}
    
    # Instancias de sistemas de observabilidad (ya importadas arriba)
    bulk_event_sourcing = BulkEventSourcing()
    bulk_observability = BulkObservability()
    bulk_cost_optimizer = BulkCostOptimizer()
    bulk_anomaly_detector = BulkAnomalyDetector(threshold_std=2.0)
    
    @app.post("/api/v1/bulk/events/record")
    async def record_event(request: Dict[str, Any]):
        """Registrar evento en event sourcing."""
        event_type = request.get("event_type")
        aggregate_id = request.get("aggregate_id")
        payload = request.get("payload", {})
        metadata = request.get("metadata", {})
        
        if not event_type or not aggregate_id:
            raise HTTPException(status_code=400, detail="event_type and aggregate_id required")
        
        bulk_event_sourcing.record_event(event_type, aggregate_id, payload, metadata)
        return {"success": True, "message": "Event recorded"}
    
    @app.get("/api/v1/bulk/events")
    async def get_events(
        aggregate_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ):
        """Obtener eventos."""
        events = bulk_event_sourcing.get_events(aggregate_id, event_type, limit)
        return {"events": events, "count": len(events)}
    
    @app.post("/api/v1/bulk/traces/start")
    async def start_trace(request: Dict[str, Any]):
        """Iniciar trace distribuido."""
        trace_id = request.get("trace_id")
        operation_name = request.get("operation_name", "unknown")
        metadata = request.get("metadata", {})
        
        if not trace_id:
            raise HTTPException(status_code=400, detail="trace_id required")
        
        bulk_observability.start_trace(trace_id, operation_name, metadata)
        return {"success": True, "trace_id": trace_id}
    
    @app.post("/api/v1/bulk/traces/add-span")
    async def add_trace_span(request: Dict[str, Any]):
        """Añadir span a trace."""
        trace_id = request.get("trace_id")
        span_name = request.get("span_name")
        duration = request.get("duration", 0.0)
        metadata = request.get("metadata", {})
        
        if not trace_id or not span_name:
            raise HTTPException(status_code=400, detail="trace_id and span_name required")
        
        bulk_observability.add_span(trace_id, span_name, duration, metadata)
        return {"success": True}
    
    @app.post("/api/v1/bulk/traces/complete")
    async def complete_trace(request: Dict[str, Any]):
        """Completar trace."""
        trace_id = request.get("trace_id")
        status = request.get("status", "success")
        error = request.get("error")
        
        if not trace_id:
            raise HTTPException(status_code=400, detail="trace_id required")
        
        bulk_observability.complete_trace(trace_id, status, error)
        return {"success": True}
    
    @app.get("/api/v1/bulk/traces/{trace_id}")
    async def get_trace(trace_id: str):
        """Obtener trace completo."""
        trace = bulk_observability.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace
    
    @app.get("/api/v1/bulk/observability/summary")
    async def get_observability_summary():
        """Obtener resumen de observabilidad."""
        return bulk_observability.get_observability_summary()
    
    @app.post("/api/v1/bulk/observability/log")
    async def log_event(request: Dict[str, Any]):
        """Registrar evento de log."""
        level = request.get("level", "info")
        message = request.get("message", "")
        context = request.get("context", {})
        trace_id = request.get("trace_id")
        
        bulk_observability.log_event(level, message, context, trace_id)
        return {"success": True}
    
    @app.get("/api/v1/bulk/observability/logs")
    async def get_observability_logs(
        level: Optional[str] = None,
        trace_id: Optional[str] = None,
        limit: int = 100
    ):
        """Obtener logs."""
        logs = bulk_observability.get_logs(level, trace_id, limit)
        return {"logs": logs, "count": len(logs)}
    
    @app.post("/api/v1/bulk/costs/record")
    async def record_cost(request: Dict[str, Any]):
        """Registrar costo de operación."""
        operation_type = request.get("operation_type")
        cost = request.get("cost", 0.0)
        items_processed = request.get("items_processed", 1)
        duration = request.get("duration", 0.0)
        metadata = request.get("metadata", {})
        
        if not operation_type:
            raise HTTPException(status_code=400, detail="operation_type required")
        
        bulk_cost_optimizer.record_operation_cost(
            operation_type, cost, items_processed, duration, metadata
        )
        return {"success": True}
    
    @app.get("/api/v1/bulk/costs/summary")
    async def get_cost_summary():
        """Obtener resumen de costos."""
        return bulk_cost_optimizer.get_cost_summary()
    
    @app.get("/api/v1/bulk/costs/optimizations")
    async def get_cost_optimizations(operation_type: Optional[str] = None):
        """Obtener sugerencias de optimización."""
        suggestions = bulk_cost_optimizer.suggest_optimizations(operation_type)
        return {"suggestions": suggestions, "count": len(suggestions)}
    
    @app.post("/api/v1/bulk/anomalies/record")
    async def record_anomaly_metric(request: Dict[str, Any]):
        """Registrar métrica para detección de anomalías."""
        metric_name = request.get("metric_name")
        value = request.get("value")
        metadata = request.get("metadata", {})
        
        if metric_name is None or value is None:
            raise HTTPException(status_code=400, detail="metric_name and value required")
        
        bulk_anomaly_detector.record_metric(metric_name, value, metadata)
        return {"success": True}
    
    @app.get("/api/v1/bulk/anomalies")
    async def get_anomalies(
        metric_name: Optional[str] = None,
        limit: int = 100
    ):
        """Obtener anomalías detectadas."""
        anomalies = bulk_anomaly_detector.get_anomalies(metric_name, limit)
        return {"anomalies": anomalies, "count": len(anomalies)}
    
    @app.get("/api/v1/bulk/anomalies/summary")
    async def get_anomaly_summary():
        """Obtener resumen de anomalías."""
        return bulk_anomaly_detector.get_anomaly_summary()
    
    # Sistemas enterprise
    from ..core.bulk_operations import (
        BulkWorkflowEngine,
        BulkMultiTenancy,
        BulkDisasterRecovery,
        BulkComplianceAudit,
        BulkMLOptimizer
    )
    
    bulk_workflow_engine = BulkWorkflowEngine()
    bulk_multi_tenancy = BulkMultiTenancy()
    bulk_disaster_recovery = BulkDisasterRecovery(backup_interval_minutes=60)
    bulk_compliance_audit = BulkComplianceAudit()
    bulk_ml_optimizer = BulkMLOptimizer()
    
    @app.post("/api/v1/bulk/workflows/register")
    async def register_workflow(request: Dict[str, Any]):
        """Registrar workflow."""
        workflow_id = request.get("workflow_id")
        steps = request.get("steps", [])
        name = request.get("name", "")
        description = request.get("description", "")
        
        if not workflow_id:
            raise HTTPException(status_code=400, detail="workflow_id required")
        
        bulk_workflow_engine.register_workflow(workflow_id, steps, name, description)
        return {"success": True, "workflow_id": workflow_id}
    
    @app.post("/api/v1/bulk/workflows/execute")
    async def execute_workflow(request: Dict[str, Any]):
        """Ejecutar workflow."""
        workflow_id = request.get("workflow_id")
        initial_data = request.get("initial_data", {})
        execution_id = request.get("execution_id")
        
        if not workflow_id:
            raise HTTPException(status_code=400, detail="workflow_id required")
        
        execution = await bulk_workflow_engine.execute_workflow(
            workflow_id, initial_data, execution_id
        )
        return execution
    
    @app.get("/api/v1/bulk/workflows/{workflow_id}")
    async def get_workflow(workflow_id: str):
        """Obtener workflow."""
        workflow = bulk_workflow_engine.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return workflow
    
    @app.get("/api/v1/bulk/workflows/executions/{execution_id}")
    async def get_workflow_execution(execution_id: str):
        """Obtener ejecución de workflow."""
        execution = bulk_workflow_engine.get_execution(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        return execution
    
    @app.post("/api/v1/bulk/tenants/register")
    async def register_tenant(request: Dict[str, Any]):
        """Registrar tenant."""
        tenant_id = request.get("tenant_id")
        name = request.get("name", "")
        config = request.get("config", {})
        
        if not tenant_id:
            raise HTTPException(status_code=400, detail="tenant_id required")
        
        bulk_multi_tenancy.register_tenant(tenant_id, name, config)
        return {"success": True, "tenant_id": tenant_id}
    
    @app.get("/api/v1/bulk/tenants/{tenant_id}/stats")
    async def get_tenant_stats(tenant_id: str):
        """Obtener estadísticas del tenant."""
        stats = bulk_multi_tenancy.get_tenant_stats(tenant_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Tenant not found")
        return stats
    
    @app.post("/api/v1/bulk/tenants/{tenant_id}/check-quota")
    async def check_tenant_quota(request: Dict[str, Any], tenant_id: str):
        """Verificar quota del tenant."""
        resource_type = request.get("resource_type")
        amount = request.get("amount", 1)
        
        if not resource_type:
            raise HTTPException(status_code=400, detail="resource_type required")
        
        can_proceed, error = bulk_multi_tenancy.check_quota(tenant_id, resource_type, amount)
        return {"can_proceed": can_proceed, "error": error}
    
    @app.post("/api/v1/bulk/recovery/checkpoint")
    async def create_recovery_checkpoint(request: Dict[str, Any]):
        """Crear checkpoint de recovery."""
        checkpoint_id = request.get("checkpoint_id")
        state = request.get("state", {})
        metadata = request.get("metadata", {})
        
        if not checkpoint_id:
            raise HTTPException(status_code=400, detail="checkpoint_id required")
        
        bulk_disaster_recovery.create_checkpoint(checkpoint_id, state, metadata)
        return {"success": True, "checkpoint_id": checkpoint_id}
    
    @app.get("/api/v1/bulk/recovery/status")
    async def get_recovery_status():
        """Obtener estado de disaster recovery."""
        return bulk_disaster_recovery.get_recovery_status()
    
    @app.get("/api/v1/bulk/recovery/checkpoints/{checkpoint_id}")
    async def get_checkpoint(checkpoint_id: str):
        """Obtener checkpoint."""
        checkpoint = bulk_disaster_recovery.get_checkpoint(checkpoint_id)
        if not checkpoint:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        return checkpoint
    
    @app.post("/api/v1/bulk/compliance/add-rule")
    async def add_compliance_rule(request: Dict[str, Any]):
        """Añadir regla de compliance."""
        rule_id = request.get("rule_id")
        rule_name = request.get("rule_name", "")
        severity = request.get("severity", "medium")
        # Nota: validator function necesitaría ser deserializado en producción
        
        if not rule_id:
            raise HTTPException(status_code=400, detail="rule_id required")
        
        return {
            "success": True,
            "message": "Rule registration endpoint - implement validator function registration"
        }
    
    @app.post("/api/v1/bulk/compliance/audit")
    async def audit_operation(request: Dict[str, Any]):
        """Registrar operación en auditoría."""
        operation_type = request.get("operation_type")
        user_id = request.get("user_id")
        details = request.get("details", {})
        result = request.get("result")
        
        if not operation_type or not user_id:
            raise HTTPException(status_code=400, detail="operation_type and user_id required")
        
        bulk_compliance_audit.audit_operation(operation_type, user_id, details, result)
        return {"success": True}
    
    @app.get("/api/v1/bulk/compliance/logs")
    async def get_compliance_logs(
        user_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        limit: int = 100
    ):
        """Obtener logs de compliance."""
        logs = bulk_compliance_audit.get_audit_logs(user_id, operation_type, limit)
        return {"logs": logs, "count": len(logs)}
    
    @app.get("/api/v1/bulk/compliance/report")
    async def get_compliance_report():
        """Obtener reporte de compliance."""
        return bulk_compliance_audit.get_compliance_report()
    
    @app.post("/api/v1/bulk/ml/record-training")
    async def record_training_data(request: Dict[str, Any]):
        """Registrar datos de entrenamiento."""
        model_name = request.get("model_name")
        features = request.get("features", {})
        target = request.get("target")
        metadata = request.get("metadata", {})
        
        if not model_name or target is None:
            raise HTTPException(status_code=400, detail="model_name and target required")
        
        bulk_ml_optimizer.record_training_data(model_name, features, target, metadata)
        return {"success": True}
    
    @app.post("/api/v1/bulk/ml/train")
    async def train_ml_model(request: Dict[str, Any]):
        """Entrenar modelo ML."""
        model_name = request.get("model_name")
        model_type = request.get("model_type", "linear_regression")
        
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name required")
        
        result = bulk_ml_optimizer.train_model(model_name, model_type)
        return result
    
    @app.post("/api/v1/bulk/ml/predict")
    async def ml_predict(request: Dict[str, Any]):
        """Hacer predicción con modelo ML."""
        model_name = request.get("model_name")
        features = request.get("features", {})
        
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name required")
        
        prediction = bulk_ml_optimizer.predict(model_name, features)
        return prediction
    
    @app.get("/api/v1/bulk/ml/models/{model_name}/stats")
    async def get_ml_model_stats(model_name: str):
        """Obtener estadísticas del modelo ML."""
        stats = bulk_ml_optimizer.get_model_stats(model_name)
        if not stats:
            raise HTTPException(status_code=404, detail="Model not found")
        return stats
    
    # Guardar referencias para acceso externo
    app.state.chat_engine = chat_engine
    app.state.analyzer = analyzer
    app.state.exporter = exporter
    app.state.webhook_manager = webhook_manager
    app.state.template_manager = template_manager
    app.state.auth_manager = auth_manager
    app.state.backup_manager = backup_manager
    app.state.performance_optimizer = performance_optimizer
    app.state.health_monitor = health_monitor
    app.state.task_queue = task_queue
    app.state.alert_manager = alert_manager
    app.state.cluster_manager = cluster_manager
    app.state.feature_flags = feature_flags
    app.state.api_versioning = api_versioning
    app.state.advanced_analytics = advanced_analytics
    app.state.recommendation_engine = recommendation_engine
    app.state.ab_testing = ab_testing
    app.state.event_bus = event_bus
    app.state.security_manager = security_manager
    app.state.i18n_manager = i18n_manager
    app.state.workflow_engine = workflow_engine
    app.state.notification_manager = notification_manager
    app.state.integration_manager = integration_manager
    app.state.benchmark_runner = benchmark_runner
    app.state.api_docs_generator = api_docs_generator
    app.state.advanced_monitoring = advanced_monitoring
    app.state.secrets_manager = secrets_manager
    app.state.ml_optimizer = ml_optimizer
    app.state.deployment_manager = deployment_manager
    app.state.report_generator = report_generator
    app.state.user_manager = user_manager
    app.state.search_engine = search_engine
    app.state.message_queue = message_queue
    app.state.validation_engine = validation_engine
    app.state.throttler = throttler
    app.state.circuit_breaker = circuit_breaker
    app.state.intelligent_optimizer = intelligent_optimizer
    app.state.adaptive_learning = adaptive_learning
    app.state.demand_predictor = demand_predictor
    app.state.intelligent_health = intelligent_health
    app.state.predictive_scaler = predictive_scaler
    app.state.cost_optimizer = cost_optimizer
    app.state.intelligent_alerts = intelligent_alerts
    app.state.advanced_observability = advanced_observability
    app.state.load_balancer = load_balancer
    app.state.resource_manager = resource_manager
    app.state.disaster_recovery = disaster_recovery
    app.state.advanced_security = advanced_security
    app.state.auto_optimizer = auto_optimizer
    app.state.predictive_analytics = predictive_analytics
    app.state.policy_engine = policy_engine
    app.state.audit_system = audit_system
    app.state.task_orchestrator = task_orchestrator
    app.state.resource_allocator = resource_allocator
    app.state.service_orchestrator = service_orchestrator
    app.state.performance_profiler = performance_profiler
    app.state.adaptive_rate_controller = adaptive_rate_controller
    app.state.smart_retry_manager = smart_retry_manager
    app.state.distributed_lock_manager = distributed_lock_manager
    app.state.data_pipeline_manager = data_pipeline_manager
    app.state.event_scheduler = event_scheduler
    app.state.graceful_degradation_manager = graceful_degradation_manager
    app.state.cache_warmer = cache_warmer
    app.state.load_shedder = load_shedder
    app.state.conflict_resolver = conflict_resolver
    app.state.state_machine_manager = state_machine_manager
    app.state.workflow_engine_v2 = workflow_engine_v2
    app.state.event_bus = event_bus
    app.state.feature_toggle_manager = feature_toggle_manager
    app.state.rate_limiter_v2 = rate_limiter_v2
    app.state.circuit_breaker_v2 = circuit_breaker_v2
    app.state.adaptive_optimizer = adaptive_optimizer
    app.state.health_checker_v2 = health_checker_v2
    app.state.auto_scaler = auto_scaler
    app.state.batch_processor = batch_processor
    app.state.performance_monitor = performance_monitor
    app.state.queue_manager = queue_manager
    app.state.connection_manager = connection_manager
    app.state.transaction_manager = transaction_manager
    app.state.saga_orchestrator = saga_orchestrator
    app.state.distributed_coordinator = distributed_coordinator
    app.state.service_mesh = service_mesh
    app.state.adaptive_throttler = adaptive_throttler
    app.state.backpressure_manager = backpressure_manager
    app.state.federated_learning = federated_learning
    app.state.knowledge_manager = knowledge_manager
    app.state.auto_generator = auto_generator
    app.state.architecture_recommender = architecture_recommender
    app.state.mlops_manager = mlops_manager
    app.state.dependency_manager = dependency_manager
    app.state.cicd_manager = cicd_manager
    app.state.code_quality = code_quality
    app.state.business_metrics = business_metrics
    app.state.version_control = version_control
    app.state.log_analyzer = log_analyzer
    app.state.api_performance = api_performance
    app.state.advanced_secrets = advanced_secrets
    app.state.intelligent_cache = intelligent_cache
    app.state.sentiment_analyzer = sentiment_analyzer
    app.state.task_manager = task_manager
    app.state.resource_monitor = resource_monitor
    app.state.push_notifications = push_notifications
    app.state.distributed_sync = distributed_sync
    app.state.query_analyzer = query_analyzer
    app.state.file_manager = file_manager
    app.state.data_compression = data_compression
    app.state.incremental_backup = incremental_backup
    app.state.network_analyzer = network_analyzer
    app.state.config_manager = config_manager
    app.state.mfa_authentication = mfa_authentication
    app.state.advanced_rate_limiter = advanced_rate_limiter
    app.state.user_behavior_analyzer = user_behavior_analyzer
    app.state.event_stream = event_stream
    app.state.security_analyzer = security_analyzer
    app.state.session_manager = session_manager
    app.state.realtime_metrics = realtime_metrics
    app.state.auto_optimizer = auto_optimizer
    app.state.bulk_sessions = bulk_sessions
    app.state.bulk_messages = bulk_messages
    app.state.bulk_exporter = bulk_exporter
    app.state.bulk_analytics = bulk_analytics
    app.state.bulk_cleanup = bulk_cleanup
    app.state.bulk_processor = bulk_processor
    app.state.bulk_importer = bulk_importer
    app.state.bulk_notifications = bulk_notifications
    app.state.bulk_search = bulk_search
    app.state.bulk_backup = bulk_backup
    app.state.bulk_migration = bulk_migration
    app.state.bulk_metrics = bulk_metrics
    app.state.bulk_scheduler = bulk_scheduler
    app.state.bulk_rate_limiter = bulk_rate_limiter
    app.state.bulk_auto_creator = bulk_auto_creator
    app.state.bulk_auto_expander = bulk_auto_expander
    app.state.bulk_auto_processor = bulk_auto_processor
    app.state.bulk_auto_maintainer = bulk_auto_maintainer
    app.state.bulk_infinite_generator = bulk_infinite_generator
    app.state.bulk_self_sustaining = bulk_self_sustaining
    app.state.bulk_realtime_metrics = bulk_realtime_metrics
    app.state.bulk_advanced_cache = bulk_advanced_cache
    app.state.bulk_priority_queue = bulk_priority_queue
    app.state.bulk_enhanced_validator = bulk_enhanced_validator
    app.state.bulk_dashboard = bulk_dashboard
    app.state.bulk_benchmark = bulk_benchmark
    app.state.bulk_auto_tuner = bulk_auto_tuner
    app.state.bulk_adaptive_rate_limiter = bulk_adaptive_rate_limiter
    app.state.bulk_load_balancer = bulk_load_balancer
    app.state.bulk_load_predictor = bulk_load_predictor
    app.state.bulk_auto_scaler = bulk_auto_scaler
    app.state.bulk_event_sourcing = bulk_event_sourcing
    app.state.bulk_observability = bulk_observability
    app.state.bulk_cost_optimizer = bulk_cost_optimizer
    app.state.bulk_anomaly_detector = bulk_anomaly_detector
    
    # Cleanup al cerrar
    @app.on_event("shutdown")
    async def shutdown_event():
        await integration_manager.close()
    
    return app


class ChatAPI:
    """Wrapper para la API de chat."""
    
    def __init__(self, config: Optional[ChatConfig] = None):
        self.config = config or ChatConfig()
        self.app = create_chat_app(self.config)
        self.chat_engine = self.app.state.chat_engine
    
    def run(self, host: str = "0.0.0.0", port: int = 8006, **kwargs):
        """Ejecutar el servidor API."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level=self.config.log_level.lower(),
            **kwargs
        )


=================================================

API REST para el sistema de chat continuo proactivo.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..core.chat_engine import ContinuousChatEngine
from ..core.chat_session import ChatSession, ChatState
from ..core.session_storage import SessionStorage, JSONSessionStorage, RedisSessionStorage
from ..core.rate_limiter import RateLimiter, RateLimitConfig
from ..core.plugins import PluginManager
from ..core.conversation_analyzer import ConversationAnalyzer
from ..core.exporters import ConversationExporter
from ..core.webhooks import WebhookManager, WebhookEvent
from ..core.templates import TemplateManager
from ..core.auth import AuthManager, Role
from ..core.backup_manager import BackupManager, BackupConfig
from ..core.performance_optimizer import PerformanceOptimizer
from ..core.health_monitor import HealthMonitor
from ..core.task_queue import TaskQueue
from ..core.alerting import AlertManager, AlertType, AlertLevel
from ..core.clustering import ClusterManager
from ..core.feature_flags import FeatureFlagManager
from ..core.api_versioning import APIVersionManager
from ..core.advanced_analytics import AdvancedAnalytics
from ..core.recommendations import RecommendationEngine
from ..core.ab_testing import ABTestingFramework, Variant
from ..core.event_system import EventBus, EventType
from ..core.security import SecurityManager
from ..core.i18n import I18nManager, Language
from ..core.workflow import WorkflowEngine, Workflow, WorkflowStep
from ..core.notifications import NotificationManager, NotificationType, NotificationPriority
from ..core.integrations import IntegrationManager, Integration, IntegrationType
from ..core.benchmarking import BenchmarkRunner
from ..core.api_docs import APIDocumentationGenerator, APIDocumentation
from ..core.monitoring import AdvancedMonitoring
from ..core.secrets_manager import SecretsManager
from ..core.ml_optimizer import MLOptimizer
from ..core.deployment import DeploymentManager, DeploymentStatus
from ..core.reports import ReportGenerator, ReportType
from ..core.user_management import UserManager, UserRole, UserStatus
from ..core.search_engine import SearchEngine
from ..core.message_queue import MessageQueue, MessagePriority
from ..core.validation import ValidationEngine, ValidationRule
from ..core.throttling import Throttler, ThrottleConfig
from ..core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from ..core.intelligent_optimizer import IntelligentOptimizer, OptimizationSuggestion
from ..core.adaptive_learning import AdaptiveLearningSystem, LearningPattern
from ..core.demand_predictor import DemandPredictor, DemandForecast
from ..core.intelligent_health import IntelligentHealthChecker, HealthStatus
from ..core.predictive_scaling import PredictiveScaler, ScalingAction
from ..core.cost_optimizer import CostOptimizer, CostOptimizationSuggestion
from ..core.intelligent_alerts import IntelligentAlertSystem, AlertSeverity
from ..core.advanced_observability import AdvancedObservability
from ..core.intelligent_load_balancer import IntelligentLoadBalancer, LoadBalancingAlgorithm
from ..core.resource_manager import ResourceManager, ResourceType
from ..core.disaster_recovery import DisasterRecovery, RecoveryStatus
from ..core.advanced_security import AdvancedSecurity, ThreatLevel, SecurityEventType
from ..core.auto_optimizer import AutoOptimizer, OptimizationResult
from ..core.federated_learning import FederatedLearning, LearningRoundStatus
from ..core.knowledge_manager import KnowledgeManager, KnowledgeType
from ..core.auto_generator import AutoGenerator, GenerationType
from ..core.architecture_recommender import ArchitectureRecommender, ArchitecturePattern
from ..core.mlops_manager import MLOpsManager, ExperimentStatus, ModelStatus
from ..core.dependency_manager import DependencyManager, DependencyStatus, VulnerabilitySeverity
from ..core.cicd_manager import CICDManager, PipelineStatus, StageStatus
from ..core.code_quality import CodeQuality, QualityLevel, CodeSmellType
from ..core.business_metrics import BusinessMetrics, MetricCategory
from ..core.version_control import VersionControl, ChangeType
from ..core.log_analyzer import LogAnalyzer, LogLevel, LogPattern
from ..core.api_performance import APIPerformance, PerformanceMetric
from ..core.advanced_secrets import AdvancedSecrets, SecretType, SecretStatus
from ..core.intelligent_cache import IntelligentCache, CacheStrategy
from ..core.sentiment_analyzer import SentimentAnalyzer, Sentiment, Emotion
from ..core.task_manager import TaskManager, TaskStatus, TaskPriority
from ..core.resource_monitor import ResourceMonitor, ResourceType, AlertLevel
from ..core.push_notifications import PushNotifications, NotificationChannel, NotificationPriority
from ..core.distributed_sync import DistributedSync, SyncStatus, ConflictResolution
from ..core.query_analyzer import QueryAnalyzer, QueryType
from ..core.file_manager import FileManager, FileType, FileStatus
from ..core.data_compression import DataCompression, CompressionAlgorithm
from ..core.incremental_backup import IncrementalBackup, BackupType
from ..core.network_analyzer import NetworkAnalyzer, NetworkEventType
from ..core.config_manager import ConfigManager, ConfigFormat, ConfigStatus
from ..core.mfa_authentication import MFAAuthentication, MFAMethod, MFAStatus
from ..core.advanced_rate_limiter import AdvancedRateLimiter, RateLimitStrategy
from ..core.user_behavior_analyzer import UserBehaviorAnalyzer, BehaviorType
from ..core.event_stream import EventStream, EventType
from ..core.security_analyzer import SecurityAnalyzer, ThreatLevel, ThreatType
from ..core.session_manager import SessionManager, SessionStatus
from ..core.realtime_metrics import RealTimeMetrics, MetricType
from ..core.auto_optimizer import AutoOptimizer, OptimizationTarget
from ..core.predictive_analytics import PredictiveAnalytics, PredictionType
from ..core.resource_allocator import ResourceAllocator, ResourceType as AllocatorResourceType, AllocationStatus
from ..core.service_orchestrator import ServiceOrchestrator, ServiceStatus
from ..core.performance_profiler import PerformanceProfiler, ProfilerScope
from ..api.websocket_api import create_websocket_router
from ..api.graphql_api import create_graphql_router
from ..config.chat_config import ChatConfig

logger = logging.getLogger(__name__)


# Modelos Pydantic para requests/responses
class ChatCreateRequest(BaseModel):
    """Request para crear una nueva sesión de chat."""
    user_id: Optional[str] = None
    initial_message: Optional[str] = None
    auto_continue: bool = True
    response_interval: float = 2.0


class ChatMessageRequest(BaseModel):
    """Request para enviar un mensaje."""
    message: str = Field(..., description="Mensaje del usuario")


class ChatControlRequest(BaseModel):
    """Request para controlar el chat (pausar/reanudar)."""
    action: str = Field(..., description="Acción: 'pause' o 'resume'")
    reason: Optional[str] = None


class ChatResponse(BaseModel):
    """Response de la API."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class ChatSessionResponse(BaseModel):
    """Response con información de la sesión."""
    session_id: str
    state: str
    is_paused: bool
    message_count: int
    auto_continue: bool


def create_chat_app(config: Optional[ChatConfig] = None) -> FastAPI:
    """
    Crear aplicación FastAPI para el chat continuo.
    
    Args:
        config: Configuración del chat (opcional)
    
    Returns:
        FastAPI app configurada
    """
    app_config = config or ChatConfig()
    
    # Crear sistema de almacenamiento
    storage = None
    if app_config.storage_type == "redis" and app_config.redis_url:
        try:
            storage = RedisSessionStorage(
                redis_url=app_config.redis_url,
                ttl=app_config.session_ttl,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Redis storage: {e}, falling back to JSON")
            storage = JSONSessionStorage(storage_path=app_config.storage_path)
    elif app_config.storage_type == "json":
        storage = JSONSessionStorage(storage_path=app_config.storage_path)
    
    # Crear rate limiter
    rate_limiter = RateLimiter(
        RateLimitConfig(
            max_requests=app_config.rate_limit_max_requests,
            time_window=app_config.rate_limit_window,
            max_concurrent=app_config.rate_limit_max_concurrent,
        )
    )
    
    # Crear app FastAPI
    app = FastAPI(
        title="Bulk Chat API",
        description="API para chat continuo proactivo",
        version="2.0.0",
    )
    
    # Crear plugin manager
    plugin_manager = PluginManager()
    # Registrar plugins por defecto
    if app_config.enable_plugins:
        from ..core.plugins import (
            SentimentAnalyzerPlugin,
            ProfanityFilterPlugin,
            ResponseEnhancerPlugin,
        )
        from ..core.plugins import PluginType
        
        plugin_manager.register(SentimentAnalyzerPlugin(), PluginType.ANALYZER)
        plugin_manager.register(ProfanityFilterPlugin(), PluginType.PRE_PROCESSOR)
        plugin_manager.register(ResponseEnhancerPlugin(), PluginType.POST_PROCESSOR)
    
    # Crear servicios adicionales
    analyzer = ConversationAnalyzer()
    exporter = ConversationExporter()
    webhook_manager = WebhookManager()
    template_manager = TemplateManager()
    auth_manager = AuthManager()
    performance_optimizer = PerformanceOptimizer()
    health_monitor = HealthMonitor()
    task_queue = TaskQueue(max_workers=5)
    alert_manager = AlertManager()
    cluster_manager = ClusterManager(
        node_id=f"node_{app_config.api_port}",
        host=app_config.api_host,
        port=app_config.api_port,
    )
    feature_flags = FeatureFlagManager()
    api_versioning = APIVersionManager()
    advanced_analytics = AdvancedAnalytics()
    recommendation_engine = RecommendationEngine()
    ab_testing = ABTestingFramework()
    event_bus = EventBus()
    security_manager = SecurityManager()
    i18n_manager = I18nManager()
    workflow_engine = WorkflowEngine()
    notification_manager = NotificationManager()
    integration_manager = IntegrationManager()
    benchmark_runner = BenchmarkRunner()
    api_docs_generator = APIDocumentationGenerator()
    advanced_monitoring = AdvancedMonitoring()
    secrets_manager = SecretsManager()
    ml_optimizer = MLOptimizer()
    deployment_manager = DeploymentManager()
    report_generator = ReportGenerator()
    user_manager = UserManager()
    search_engine = SearchEngine()
    message_queue = MessageQueue()
    validation_engine = ValidationEngine()
    throttler = Throttler()
    circuit_breaker = CircuitBreaker()
    intelligent_optimizer = IntelligentOptimizer()
    adaptive_learning = AdaptiveLearningSystem()
    demand_predictor = DemandPredictor()
    intelligent_health = IntelligentHealthChecker()
    predictive_scaler = PredictiveScaler()
    cost_optimizer = CostOptimizer()
    intelligent_alerts = IntelligentAlertSystem()
    advanced_observability = AdvancedObservability()
    load_balancer = IntelligentLoadBalancer()
    resource_manager = ResourceManager()
    disaster_recovery = DisasterRecovery()
    advanced_security = AdvancedSecurity()
    auto_optimizer = AutoOptimizer()
    federated_learning = FederatedLearning()
    knowledge_manager = KnowledgeManager()
    auto_generator = AutoGenerator()
    architecture_recommender = ArchitectureRecommender()
    mlops_manager = MLOpsManager()
    dependency_manager = DependencyManager()
    cicd_manager = CICDManager()
    code_quality = CodeQuality()
    business_metrics = BusinessMetrics()
    version_control = VersionControl()
    log_analyzer = LogAnalyzer()
    api_performance = APIPerformance()
    advanced_secrets = AdvancedSecrets()
    intelligent_cache = IntelligentCache(max_size=5000, strategy=CacheStrategy.ADAPTIVE)
    sentiment_analyzer = SentimentAnalyzer()
    task_manager = TaskManager()
    resource_monitor = ResourceMonitor()
    push_notifications = PushNotifications()
    distributed_sync = DistributedSync(node_id="node_1")
    query_analyzer = QueryAnalyzer()
    file_manager = FileManager()
    data_compression = DataCompression()
    incremental_backup = IncrementalBackup()
    network_analyzer = NetworkAnalyzer()
    config_manager = ConfigManager()
    mfa_authentication = MFAAuthentication()
    advanced_rate_limiter = AdvancedRateLimiter()
    user_behavior_analyzer = UserBehaviorAnalyzer()
    event_stream = EventStream()
    security_analyzer = SecurityAnalyzer()
    session_manager = SessionManager()
    realtime_metrics = RealTimeMetrics()
    auto_optimizer = AutoOptimizer()
    predictive_analytics = PredictiveAnalytics()
    policy_engine = PolicyEngine()
    audit_system = AuditSystem()
    task_orchestrator = TaskOrchestrator()
    resource_allocator = ResourceAllocator()
    service_orchestrator = ServiceOrchestrator()
    performance_profiler = PerformanceProfiler()
    adaptive_rate_controller = AdaptiveRateController()
    smart_retry_manager = SmartRetryManager()
    distributed_lock_manager = DistributedLockManager()
    data_pipeline_manager = DataPipelineManager()
    event_scheduler = EventScheduler()
    graceful_degradation_manager = GracefulDegradationManager()
    cache_warmer = CacheWarmer(cache_manager=intelligent_cache)
    load_shedder = LoadShedder()
    conflict_resolver = ConflictResolver()
    state_machine_manager = StateMachineManager()
    workflow_engine_v2 = WorkflowEngineV2()
    event_bus = EventBus()
    feature_toggle_manager = FeatureToggleManager()
    rate_limiter_v2 = RateLimiterV2()
    circuit_breaker_v2 = CircuitBreakerV2()
    adaptive_optimizer = AdaptiveOptimizer()
    health_checker_v2 = HealthCheckerV2()
    auto_scaler = AutoScaler()
    batch_processor = BatchProcessor()
    performance_monitor = PerformanceMonitor()
    queue_manager = QueueManager()
    connection_manager = ConnectionManager()
    transaction_manager = TransactionManager()
    saga_orchestrator = SagaOrchestrator()
    distributed_coordinator = DistributedCoordinator(node_id=f"node_{datetime.now().timestamp()}")
    service_mesh = ServiceMesh()
    adaptive_throttler = AdaptiveThrottler()
    backpressure_manager = BackpressureManager()
    
    # Iniciar scheduler de eventos
    event_scheduler.start_scheduler()
    
    # Iniciar cache warmer
    cache_warmer.start_warming()
    
    # Iniciar event bus
    event_bus.start_processing()
    
    # Iniciar health checker
    health_checker_v2.start_checking()
    
    # Iniciar auto scaler
    auto_scaler.start_scaling()
    
    # Iniciar coordinación distribuida
    distributed_coordinator.start_coordination()
    
    # Iniciar monitoreo de recursos
    asyncio.create_task(resource_monitor.start_monitoring())
    
    # Iniciar task queue
    asyncio.create_task(task_queue.start())
    
    # Iniciar cluster manager
    asyncio.create_task(cluster_manager.start_heartbeat_monitor())
    
    # Configurar alertas automáticas
    async def check_performance_alerts():
        slow_ops = await performance_optimizer.optimize_slow_operations(threshold=2.0)
        if slow_ops:
            await alert_manager.create_alert(
                AlertType.PERFORMANCE,
                AlertLevel.WARNING,
                "Operaciones Lentas Detectadas",
                f"Se detectaron {len(slow_ops)} operaciones lentas",
                {"slow_operations": slow_ops}
            )
    
    # Verificar alertas periódicamente
    async def alert_loop():
        while True:
            try:
                await check_performance_alerts()
                await asyncio.sleep(60)  # Cada minuto
            except Exception as e:
                logger.error(f"Error in alert loop: {e}")
                await asyncio.sleep(60)
    
    asyncio.create_task(alert_loop())
    
    # Crear backup manager
    backup_config = BackupConfig(
        enabled=app_config.enable_backups,
        interval_hours=app_config.backup_interval_hours,
        backup_directory=app_config.backup_directory,
    )
    backup_manager = BackupManager(
        config=backup_config,
        storage_path=app_config.storage_path,
    )
    
    # Iniciar backups en background task
    backup_task = None
    if backup_config.enabled:
        async def start_backups():
            await backup_manager.start()
        backup_task = asyncio.create_task(start_backups())
    
    # Crear motor de chat
    chat_engine = ContinuousChatEngine(
        llm_provider=app_config.get_llm_provider(),
        auto_continue=app_config.auto_continue,
        response_interval=app_config.response_interval,
        max_consecutive_responses=app_config.max_consecutive_responses,
        storage=storage,
        enable_metrics=app_config.enable_metrics,
        auto_save=app_config.auto_save,
        save_interval=app_config.save_interval,
        enable_cache=app_config.enable_cache,
        cache_size=app_config.cache_size,
        cache_ttl=app_config.cache_ttl,
        plugin_manager=plugin_manager,
        webhook_manager=webhook_manager,
    )
    
    # Agregar router de WebSocket
    ws_router = create_websocket_router(chat_engine)
    app.include_router(ws_router)
    
    # Agregar router de GraphQL (setup después de crear app)
    try:
        graphql_router = create_graphql_router(chat_engine)
        app.include_router(graphql_router, prefix="/graphql")
    except Exception as e:
        logger.warning(f"Could not setup GraphQL: {e}")
    
    # Servir dashboard
    try:
        dashboard_path = Path(__file__).parent.parent / "dashboard"
        if dashboard_path.exists():
            app.mount("/dashboard", StaticFiles(directory=str(dashboard_path), html=True), name="dashboard")
    except Exception as e:
        logger.warning(f"Could not mount dashboard: {e}")
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Endpoints
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint básico."""
        return {
            "status": "healthy",
            "service": "bulk_chat",
            "active_sessions": len(chat_engine.sessions),
        }
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Health check detallado."""
        system_health = await health_monitor.get_system_health()
        return system_health
    
    @app.get("/api/v1/performance/metrics")
    async def get_performance_metrics(operation: Optional[str] = None):
        """Obtener métricas de rendimiento."""
        metrics = performance_optimizer.get_metrics(operation)
        return {"metrics": metrics}
    
    @app.get("/api/v1/performance/slow-operations")
    async def get_slow_operations(threshold: float = 1.0):
        """Obtener operaciones lentas."""
        slow_ops = await performance_optimizer.optimize_slow_operations(threshold)
        return {"slow_operations": slow_ops}
    
    @app.get("/api/v1/tasks/queue")
    async def get_task_queue_status():
        """Obtener estado de la cola de tareas."""
        return {
            "queue_size": task_queue.get_queue_size(),
            "active_tasks": len(task_queue.get_active_tasks()),
            "pending_tasks": len(task_queue.get_pending_tasks()),
            "max_workers": task_queue.max_workers,
        }
    
    # Endpoints de alertas
    @app.get("/api/v1/alerts")
    async def get_alerts(
        alert_type: Optional[str] = None,
        level: Optional[str] = None,
        resolved: Optional[bool] = None,
    ):
        """Obtener alertas."""
        from ..core.alerting import AlertType, AlertLevel
        
        alert_type_enum = AlertType(alert_type) if alert_type else None
        level_enum = AlertLevel(level) if level else None
        
        alerts = alert_manager.get_alerts(
            alert_type=alert_type_enum,
            level=level_enum,
            resolved=resolved,
        )
        
        return {
            "alerts": [
                {
                    "id": a.id,
                    "type": a.type.value,
                    "level": a.level.value,
                    "title": a.title,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                    "resolved": a.resolved,
                }
                for a in alerts
            ],
            "total": len(alerts),
        }
    
    @app.post("/api/v1/alerts/{alert_id}/resolve")
    async def resolve_alert(alert_id: str):
        """Resolver alerta."""
        await alert_manager.resolve_alert(alert_id)
        return {"success": True, "message": "Alert resolved"}
    
    # Endpoints de feature flags
    @app.get("/api/v1/feature-flags")
    async def list_feature_flags():
        """Listar feature flags."""
        flags = feature_flags.list_flags()
        return {"feature_flags": flags}
    
    @app.get("/api/v1/feature-flags/{flag_name}")
    async def get_feature_flag(flag_name: str, user_id: Optional[str] = None):
        """Obtener estado de feature flag."""
        enabled = await feature_flags.is_enabled(flag_name, user_id)
        flag = feature_flags.get_flag(flag_name)
        
        if not flag:
            raise HTTPException(status_code=404, detail="Feature flag not found")
        
        return {
            "name": flag_name,
            "enabled": enabled,
            "status": flag.status.value,
            "description": flag.description,
        }
    
    @app.post("/api/v1/feature-flags/{flag_name}/enable")
    async def enable_feature_flag(flag_name: str):
        """Habilitar feature flag."""
        await feature_flags.enable(flag_name)
        return {"success": True, "message": f"Feature flag {flag_name} enabled"}
    
    @app.post("/api/v1/feature-flags/{flag_name}/disable")
    async def disable_feature_flag(flag_name: str):
        """Deshabilitar feature flag."""
        await feature_flags.disable(flag_name)
        return {"success": True, "message": f"Feature flag {flag_name} disabled"}
    
    # Endpoints de versionado
    @app.get("/api/versions")
    async def get_api_versions():
        """Obtener versiones de API disponibles."""
        versions = api_versioning.get_all_versions()
        return {
            "current_version": api_versioning.current_version,
            "supported_versions": api_versioning.get_supported_versions(),
            "versions": versions,
        }
    
    # Endpoints de clustering
    @app.get("/api/v1/cluster/info")
    async def get_cluster_info():
        """Obtener información del cluster."""
        info = cluster_manager.get_cluster_info()
        return info
    
    # Endpoints de analytics avanzado
    @app.get("/api/v1/analytics/patterns")
    async def get_analytics_patterns():
        """Obtener patrones detectados."""
        sessions = list(chat_engine.sessions.values())
        patterns = await advanced_analytics.detect_patterns(sessions)
        
        return {
            "patterns": [
                {
                    "type": p.pattern_type,
                    "frequency": p.frequency,
                    "confidence": p.confidence,
                    "examples": p.examples,
                }
                for p in patterns
            ]
        }
    
    @app.get("/api/v1/analytics/user/{user_id}/behavior")
    async def get_user_behavior(user_id: str):
        """Obtener comportamiento de usuario."""
        sessions = [
            s for s in chat_engine.sessions.values()
            if s.user_id == user_id
        ]
        
        behavior = await advanced_analytics.analyze_user_behavior(user_id, sessions)
        
        return {
            "user_id": user_id,
            "total_sessions": behavior.total_sessions,
            "average_session_duration": behavior.average_session_duration,
            "favorite_topics": behavior.favorite_topics,
            "preferred_time": behavior.preferred_time,
            "engagement_score": behavior.engagement_score,
        }
    
    @app.get("/api/v1/analytics/insights")
    async def get_advanced_insights():
        """Obtener insights avanzados."""
        sessions = list(chat_engine.sessions.values())
        insights = await advanced_analytics.generate_insights(sessions)
        return insights
    
    # Endpoints de recomendaciones
    @app.get("/api/v1/recommendations/{user_id}")
    async def get_recommendations(
        user_id: str,
        item_type: Optional[str] = None,
        limit: int = 10,
    ):
        """Obtener recomendaciones para usuario."""
        recommendations = await recommendation_engine.recommend_items(
            user_id,
            item_type,
            limit,
        )
        return {
            "user_id": user_id,
            "recommendations": [
                {
                    "item_id": r.item_id,
                    "item_type": r.item_type,
                    "score": r.score,
                    "reason": r.reason,
                }
                for r in recommendations
            ],
        }
    
    @app.post("/api/v1/recommendations/interaction")
    async def record_interaction(
        user_id: str,
        item_id: str,
        item_type: str,
        rating: float = 1.0,
    ):
        """Registrar interacción de usuario."""
        await recommendation_engine.record_interaction(
            user_id,
            item_id,
            item_type,
            rating,
        )
        return {"success": True, "message": "Interaction recorded"}
    
    # Endpoints de A/B Testing
    @app.post("/api/v1/ab-testing/experiments")
    async def create_experiment(
        experiment_id: str,
        name: str,
        description: str,
        variants: List[str],
        traffic_split: Optional[Dict[str, float]] = None,
    ):
        """Crear experimento A/B."""
        variant_enums = [Variant(v) for v in variants]
        split_dict = None
        if traffic_split:
            split_dict = {Variant(k): v for k, v in traffic_split.items()}
        
        experiment = ab_testing.create_experiment(
            experiment_id,
            name,
            description,
            variant_enums,
            split_dict,
        )
        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "is_active": experiment.is_active,
        }
    
    @app.get("/api/v1/ab-testing/experiments/{experiment_id}/variant")
    async def get_variant(
        experiment_id: str,
        user_id: str,
    ):
        """Obtener variante asignada a usuario."""
        variant = ab_testing.get_variant(experiment_id, user_id)
        if not variant:
            raise HTTPException(status_code=404, detail="Experiment not found or inactive")
        return {"experiment_id": experiment_id, "user_id": user_id, "variant": variant.value}
    
    @app.get("/api/v1/ab-testing/experiments/{experiment_id}/stats")
    async def get_experiment_stats(experiment_id: str):
        """Obtener estadísticas de experimento."""
        stats = await ab_testing.get_experiment_stats(experiment_id)
        return stats
    
    # Endpoints de eventos
    @app.get("/api/v1/events/history")
    async def get_event_history(
        event_type: Optional[str] = None,
        limit: int = 100,
    ):
        """Obtener historial de eventos."""
        event_type_enum = None
        if event_type:
            event_type_enum = EventType(event_type)
        
        events = event_bus.get_event_history(event_type_enum, limit)
        return {
            "events": [e.to_dict() for e in events],
            "total": len(events),
        }
    
    @app.get("/api/v1/events/subscribers")
    async def get_subscribers():
        """Obtener conteo de suscriptores."""
        return event_bus.get_subscribers_count()
    
    # Endpoints de seguridad
    @app.get("/api/v1/security/audit-logs")
    async def get_audit_logs(
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ):
        """Obtener logs de auditoría."""
        logs = security_manager.get_audit_logs(user_id, action, None, limit)
        return {
            "logs": [
                {
                    "log_id": log.log_id,
                    "user_id": log.user_id,
                    "action": log.action,
                    "resource": log.resource,
                    "timestamp": log.timestamp.isoformat(),
                    "success": log.success,
                }
                for log in logs
            ],
            "total": len(logs),
        }
    
    @app.get("/api/v1/security/stats")
    async def get_security_stats():
        """Obtener estadísticas de seguridad."""
        return security_manager.get_security_stats()
    
    # Endpoints de i18n
    @app.get("/api/v1/i18n/translate")
    async def translate(
        key: str,
        language: Optional[str] = None,
    ):
        """Traducir clave."""
        lang = None
        if language:
            lang = Language(language)
        
        translation = i18n_manager.translate(key, lang)
        return {"key": key, "language": language or i18n_manager.default_language.value, "translation": translation}
    
    @app.get("/api/v1/i18n/languages")
    async def get_supported_languages():
        """Obtener idiomas soportados."""
        return {"languages": i18n_manager.get_supported_languages()}
    
    # Endpoints de workflows
    @app.post("/api/v1/workflows/execute")
    async def execute_workflow(
        workflow_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
    ):
        """Ejecutar workflow."""
        result = await workflow_engine.execute_workflow(workflow_id, initial_context)
        return result
    
    @app.get("/api/v1/workflows")
    async def list_workflows():
        """Listar workflows."""
        workflows = workflow_engine.list_workflows()
        return {"workflows": workflows}
    
    # Endpoints de notificaciones
    @app.post("/api/v1/notifications/send")
    async def send_notification(
        user_id: str,
        title: str,
        message: str,
        notification_type: str = "info",
        priority: str = "medium",
    ):
        """Enviar notificación."""
        notif_type = NotificationType(notification_type)
        notif_priority = NotificationPriority(priority)
        
        notification = await notification_manager.send_notification(
            user_id,
            title,
            message,
            notif_type,
            notif_priority,
        )
        return {
            "notification_id": notification.notification_id,
            "success": True,
        }
    
    @app.get("/api/v1/notifications/{user_id}")
    async def get_notifications(
        user_id: str,
        unread_only: bool = False,
        limit: int = 50,
    ):
        """Obtener notificaciones de usuario."""
        notifications = await notification_manager.get_notifications(
            user_id,
            unread_only,
            limit,
        )
        return {
            "notifications": [
                {
                    "notification_id": n.notification_id,
                    "title": n.title,
                    "message": n.message,
                    "type": n.notification_type.value,
                    "priority": n.priority.value,
                    "created_at": n.created_at.isoformat(),
                    "read_at": n.read_at.isoformat() if n.read_at else None,
                }
                for n in notifications
            ],
            "unread_count": notification_manager.get_unread_count(user_id),
        }
    
    @app.post("/api/v1/notifications/{user_id}/read/{notification_id}")
    async def mark_notification_read(user_id: str, notification_id: str):
        """Marcar notificación como leída."""
        await notification_manager.mark_as_read(user_id, notification_id)
        return {"success": True}
    
    # Endpoints de integraciones
    @app.post("/api/v1/integrations/call")
    async def call_integration(
        integration_id: str,
        data: Dict[str, Any],
    ):
        """Llamar integración."""
        result = await integration_manager.call_integration(integration_id, data)
        return result
    
    @app.get("/api/v1/integrations")
    async def list_integrations():
        """Listar integraciones."""
        integrations = integration_manager.list_integrations()
        return {"integrations": integrations}
    
    # Endpoints de benchmarking
    @app.post("/api/v1/benchmark/run")
    async def run_benchmark(
        benchmark_id: str,
        name: str,
        iterations: int = 10,
        warmup_runs: int = 2,
    ):
        """Ejecutar benchmark (requiere función específica)."""
        # Nota: Este endpoint es un ejemplo. En producción, necesitarías
        # pasar la función a ejecutar de forma segura.
        return {
            "message": "Benchmark endpoint requires function definition",
            "benchmark_id": benchmark_id,
        }
    
    @app.get("/api/v1/benchmark/results")
    async def get_benchmark_results():
        """Obtener resultados de benchmarks."""
        results = benchmark_runner.list_results()
        return {"results": results}
    
    # Endpoints de documentación
    @app.get("/api/v1/docs/openapi")
    async def get_openapi_spec():
        """Obtener especificación OpenAPI."""
        spec = api_docs_generator.generate_openapi_spec()
        return spec
    
    @app.get("/api/v1/docs/markdown")
    async def get_markdown_docs():
        """Obtener documentación en Markdown."""
        docs = api_docs_generator.generate_markdown_docs()
        return {"content": docs, "format": "markdown"}
    
    @app.get("/api/v1/docs/endpoints")
    async def list_endpoints():
        """Listar todos los endpoints documentados."""
        endpoints = api_docs_generator.list_endpoints()
        return {"endpoints": endpoints}
    
    # Endpoints de monitoring
    @app.post("/api/v1/monitoring/metrics")
    async def record_metric(
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Registrar métrica."""
        await advanced_monitoring.record_metric(name, value, tags)
        return {"success": True, "message": f"Metric {name} recorded"}
    
    @app.get("/api/v1/monitoring/metrics/{metric_name}/stats")
    async def get_metric_stats(metric_name: str, window_minutes: int = 60):
        """Obtener estadísticas de métrica."""
        stats = await advanced_monitoring.get_metric_stats(metric_name, window_minutes)
        return stats
    
    @app.get("/api/v1/monitoring/summary")
    async def get_monitoring_summary():
        """Obtener resumen de monitoreo."""
        summary = advanced_monitoring.get_metrics_summary()
        return summary
    
    @app.get("/api/v1/monitoring/alerts")
    async def get_monitoring_alerts(
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
    ):
        """Obtener alertas de monitoreo."""
        alerts = await advanced_monitoring.get_alerts(severity, resolved)
        return {
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "name": a.name,
                    "severity": a.severity,
                    "message": a.message,
                    "metric_name": a.metric_name,
                    "threshold": a.threshold,
                    "current_value": a.current_value,
                    "timestamp": a.timestamp.isoformat(),
                    "resolved": a.resolved,
                }
                for a in alerts
            ]
        }
    
    # Endpoints de secretos
    @app.post("/api/v1/secrets/store")
    async def store_secret(
        secret_id: str,
        name: str,
        value: str,
        secret_type: str = "api_key",
        encrypted: bool = False,
    ):
        """Almacenar secreto."""
        secret = await secrets_manager.store_secret(
            secret_id,
            name,
            value,
            secret_type,
            encrypted,
        )
        return {
            "secret_id": secret.secret_id,
            "name": secret.name,
            "success": True,
        }
    
    @app.get("/api/v1/secrets/{secret_id}")
    async def get_secret(secret_id: str, decrypt: bool = True):
        """Obtener secreto."""
        value = await secrets_manager.get_secret(secret_id, decrypt)
        if value is None:
            raise HTTPException(status_code=404, detail="Secret not found")
        return {"secret_id": secret_id, "value": value}
    
    @app.get("/api/v1/secrets")
    async def list_secrets():
        """Listar secretos."""
        secrets = await secrets_manager.list_secrets()
        return {"secrets": secrets}
    
    # Endpoints de ML Optimizer
    @app.post("/api/v1/ml-optimizer/record")
    async def record_performance(
        parameters: Dict[str, float],
        performance_metric: float,
    ):
        """Registrar datos de rendimiento."""
        await ml_optimizer.record_performance(parameters, performance_metric)
        return {"success": True, "message": "Performance recorded"}
    
    @app.post("/api/v1/ml-optimizer/optimize")
    async def optimize_parameter(
        parameter_name: str,
        min_value: float = 0.0,
        max_value: float = 1.0,
        step: float = 0.1,
    ):
        """Optimizar parámetro."""
        result = await ml_optimizer.optimize_parameter(
            parameter_name,
            min_value,
            max_value,
            step,
        )
        if not result:
            raise HTTPException(status_code=404, detail="Parameter not found or insufficient data")
        
        return {
            "parameter_name": result.parameter_name,
            "optimal_value": result.optimal_value,
            "confidence": result.confidence,
            "improvement": result.improvement,
            "metadata": result.metadata,
        }
    
    @app.post("/api/v1/ml-optimizer/predict")
    async def predict_performance(
        parameter_name: str,
        value: float,
    ):
        """Predecir rendimiento."""
        predicted = await ml_optimizer.predict_performance(parameter_name, value)
        if predicted is None:
            raise HTTPException(status_code=404, detail="Parameter not found or insufficient data")
        
        return {
            "parameter_name": parameter_name,
            "value": value,
            "predicted_performance": predicted,
        }
    
    # Endpoints de deployment
    @app.post("/api/v1/deployment/deploy")
    async def deploy(
        deployment_id: str,
        version: str,
        environment: str,
        rollback_version: Optional[str] = None,
    ):
        """Ejecutar deployment."""
        deployment = await deployment_manager.deploy(
            deployment_id,
            version,
            environment,
            rollback_version,
        )
        return {
            "deployment_id": deployment.deployment_id,
            "version": deployment.version,
            "environment": deployment.environment,
            "status": deployment.status.value,
            "logs": deployment.logs,
        }
    
    @app.post("/api/v1/deployment/{deployment_id}/rollback")
    async def rollback_deployment(
        deployment_id: str,
        version: Optional[str] = None,
    ):
        """Hacer rollback de deployment."""
        deployment = await deployment_manager.rollback(deployment_id, version)
        return {
            "deployment_id": deployment.deployment_id,
            "status": deployment.status.value,
            "logs": deployment.logs,
        }
    
    @app.get("/api/v1/deployment/current")
    async def get_current_version(environment: str):
        """Obtener versión actual en ambiente."""
        version = deployment_manager.get_current_version(environment)
        if not version:
            raise HTTPException(status_code=404, detail="No version found for environment")
        return {"environment": environment, "version": version}
    
    @app.get("/api/v1/deployment")
    async def list_deployments(
        environment: Optional[str] = None,
        limit: int = 50,
    ):
        """Listar deployments."""
        deployments = deployment_manager.list_deployments(environment, limit)
        return {"deployments": deployments}
    
    # Endpoints de reportes
    @app.post("/api/v1/reports/generate")
    async def generate_report(
        report_id: str,
        report_type: str,
        title: str,
        period_start: str,
        period_end: str,
        format: str = "json",
    ):
        """Generar reporte."""
        # Nota: En producción, necesitarías pasar la función data_source
        return {
            "message": "Report generation requires data_source function",
            "report_id": report_id,
        }
    
    @app.get("/api/v1/reports")
    async def list_reports(
        report_type: Optional[str] = None,
        limit: int = 50,
    ):
        """Listar reportes."""
        rtype = ReportType(report_type) if report_type else None
        reports = report_generator.list_reports(rtype, limit)
        return {"reports": reports}
    
    @app.get("/api/v1/reports/{report_id}")
    async def get_report(report_id: str):
        """Obtener reporte."""
        report = report_generator.get_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return {
            "report_id": report.report_id,
            "title": report.title,
            "report_type": report.report_type.value,
            "generated_at": report.generated_at.isoformat(),
            "period_start": report.period_start.isoformat(),
            "period_end": report.period_end.isoformat(),
            "data": report.data,
            "format": report.format,
        }
    
    # Endpoints de gestión de usuarios
    @app.post("/api/v1/users/register")
    async def register_user(
        username: str,
        email: str,
        password: str,
        role: str = "user",
    ):
        """Registrar nuevo usuario."""
        user_role = UserRole(role)
        user = await user_manager.create_user(username, email, password, user_role)
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "status": user.status.value,
        }
    
    @app.post("/api/v1/users/login")
    async def login_user(username: str, password: str):
        """Iniciar sesión."""
        user = await user_manager.authenticate(username, password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        return {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "status": user.status.value,
        }
    
    @app.get("/api/v1/users")
    async def list_users(
        role: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ):
        """Listar usuarios."""
        user_role = UserRole(role) if role else None
        user_status = UserStatus(status) if status else None
        
        users = user_manager.list_users(user_role, user_status, limit)
        return {"users": users}
    
    @app.get("/api/v1/users/{user_id}")
    async def get_user(user_id: str):
        """Obtener usuario."""
        user = user_manager.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "status": user.status.value,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None,
        }
    
    # Endpoints de búsqueda
    @app.get("/api/v1/search")
    async def search(
        q: str,
        item_type: Optional[str] = None,
        limit: int = 20,
    ):
        """Buscar items."""
        results = await search_engine.search(q, item_type, limit)
        return {
            "query": q,
            "results": [
                {
                    "item_id": r.item_id,
                    "item_type": r.item_type,
                    "score": r.score,
                    "title": r.title,
                    "content": r.content,
                    "highlights": r.highlights,
                }
                for r in results
            ],
            "total": len(results),
        }
    
    @app.post("/api/v1/search/index")
    async def index_item(
        item_id: str,
        item_type: str,
        title: str,
        content: str,
    ):
        """Indexar item."""
        await search_engine.index_item(item_id, item_type, title, content)
        return {"success": True, "message": f"Item {item_id} indexed"}
    
    @app.get("/api/v1/search/stats")
    async def get_search_stats():
        """Obtener estadísticas de búsqueda."""
        stats = search_engine.get_index_stats()
        return stats
    
    # Endpoints de cola de mensajes
    @app.post("/api/v1/queue/enqueue")
    async def enqueue_message(
        queue_name: str,
        payload: Dict[str, Any],
        priority: str = "medium",
        max_attempts: int = 3,
    ):
        """Agregar mensaje a la cola."""
        msg_priority = MessagePriority(priority)
        message_id = await message_queue.enqueue(
            queue_name,
            payload,
            msg_priority,
            max_attempts,
        )
        return {
            "message_id": message_id,
            "queue_name": queue_name,
            "success": True,
        }
    
    @app.get("/api/v1/queue/stats")
    async def get_queue_stats():
        """Obtener estadísticas de colas."""
        stats = message_queue.get_queue_stats()
        return stats
    
    @app.get("/api/v1/queue/{queue_name}/size")
    async def get_queue_size(queue_name: str):
        """Obtener tamaño de cola."""
        size = message_queue.get_queue_size(queue_name)
        return {"queue_name": queue_name, "size": size}
    
    # Endpoints de validación
    @app.post("/api/v1/validation/validate")
    async def validate_data(
        context: str,
        data: Dict[str, Any],
    ):
        """Validar datos."""
        errors = await validation_engine.validate(context, data)
        
        if errors:
            return {
                "valid": False,
                "errors": errors,
            }
        
        return {"valid": True, "errors": {}}
    
    @app.post("/api/v1/validation/rules")
    async def register_validation_rule(
        context: str,
        rule_id: str,
        field_name: str,
        validator_type: str,
        error_message: str,
        required: bool = True,
    ):
        """Registrar regla de validación."""
        # Crear validador según tipo
        validator_func = validation_engine.validators.get(validator_type)
        if not validator_func:
            raise HTTPException(status_code=400, detail=f"Unknown validator type: {validator_type}")
        
        rule = ValidationRule(
            rule_id=rule_id,
            field_name=field_name,
            validator=validator_func,
            error_message=error_message,
            required=required,
        )
        
        validation_engine.register_rule(context, rule)
        return {"success": True, "message": f"Rule {rule_id} registered"}
    
    # Endpoints de throttling
    @app.post("/api/v1/throttle/configure")
    async def configure_throttle(
        identifier: str,
        max_requests: int,
        window_seconds: float,
    ):
        """Configurar throttling."""
        throttler.configure(identifier, max_requests, window_seconds)
        return {"success": True, "message": f"Throttling configured for {identifier}"}
    
    @app.get("/api/v1/throttle/status/{identifier}")
    async def get_throttle_status(identifier: str):
        """Obtener estado de throttling."""
        status = await throttler.get_throttle_status(identifier)
        return status
    
    # Endpoints de circuit breaker
    @app.get("/api/v1/circuit-breaker/{identifier}/state")
    async def get_circuit_breaker_state(identifier: str):
        """Obtener estado del circuit breaker."""
        state = await circuit_breaker.get_state(identifier)
        return state
    
    @app.post("/api/v1/circuit-breaker/{identifier}/reset")
    async def reset_circuit_breaker(identifier: str):
        """Resetear circuit breaker."""
        await circuit_breaker.reset(identifier)
        return {"success": True, "message": f"Circuit breaker {identifier} reset"}
    
    # Intelligent Optimizer
    @app.post("/api/v1/optimizer/record-performance")
    async def record_performance(request: Dict[str, Any]):
        """Registrar rendimiento para optimización."""
        await intelligent_optimizer.record_performance(
            operation=request.get("operation", "unknown"),
            parameters=request.get("parameters", {}),
            metrics=request.get("metrics", {}),
        )
        return {"success": True, "message": "Performance recorded"}
    
    @app.post("/api/v1/optimizer/analyze")
    async def analyze_optimization(request: Dict[str, Any]):
        """Analizar y obtener sugerencias de optimización."""
        suggestions = await intelligent_optimizer.analyze_and_suggest(
            operation=request.get("operation", "unknown"),
            current_parameters=request.get("parameters", {}),
        )
        return {
            "suggestions": [
                {
                    "suggestion_id": s.suggestion_id,
                    "parameter": s.parameter,
                    "current_value": s.current_value,
                    "suggested_value": s.suggested_value,
                    "expected_improvement": s.expected_improvement,
                    "confidence": s.confidence,
                    "reason": s.reason,
                }
                for s in suggestions
            ]
        }
    
    @app.post("/api/v1/optimizer/apply/{suggestion_id}")
    async def apply_optimization(suggestion_id: str):
        """Aplicar optimización sugerida."""
        applied = await intelligent_optimizer.apply_optimization(suggestion_id)
        if not applied:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        return {"success": True, "message": "Optimization applied"}
    
    @app.get("/api/v1/optimizer/applied")
    async def get_applied_optimizations():
        """Obtener optimizaciones aplicadas."""
        return intelligent_optimizer.get_applied_optimizations()
    
    @app.get("/api/v1/optimizer/history")
    async def get_optimization_history():
        """Obtener historial de optimizaciones."""
        return {"history": intelligent_optimizer.get_optimization_history()}
    
    # Adaptive Learning
    @app.post("/api/v1/learning/observe")
    async def observe_learning(request: Dict[str, Any]):
        """Observar resultado para aprendizaje."""
        await adaptive_learning.observe(
            context=request.get("context", {}),
            action=request.get("action", ""),
            outcome=request.get("outcome", {}),
        )
        return {"success": True, "message": "Observation recorded"}
    
    @app.post("/api/v1/learning/predict")
    async def predict_action(request: Dict[str, Any]):
        """Predecir mejor acción basado en aprendizaje."""
        prediction = await adaptive_learning.predict_best_action(
            context=request.get("context", {})
        )
        if not prediction:
            return {"prediction": None, "message": "No patterns found for context"}
        return {"prediction": prediction}
    
    @app.get("/api/v1/learning/patterns")
    async def get_learned_patterns(
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.5,
    ):
        """Obtener patrones aprendidos."""
        patterns = adaptive_learning.get_learned_patterns(pattern_type, min_confidence)
        return {"patterns": patterns, "count": len(patterns)}
    
    # Demand Predictor
    @app.post("/api/v1/demand/record")
    async def record_demand(request: Dict[str, Any]):
        """Registrar demanda actual."""
        await demand_predictor.record_demand(
            resource_type=request.get("resource_type", "unknown"),
            value=request.get("value", 0.0),
        )
        return {"success": True, "message": "Demand recorded"}
    
    @app.post("/api/v1/demand/predict")
    async def predict_demand(request: Dict[str, Any]):
        """Predecir demanda futura."""
        forecast = await demand_predictor.predict_demand(
            resource_type=request.get("resource_type", "unknown"),
            time_horizon_minutes=request.get("time_horizon_minutes", 5),
        )
        
        if not forecast:
            return {"forecast": None, "message": "Insufficient data for prediction"}
        
        return {
            "forecast": {
                "forecast_id": forecast.forecast_id,
                "resource_type": forecast.resource_type,
                "predicted_value": forecast.predicted_value,
                "confidence": forecast.confidence,
                "time_horizon_minutes": forecast.time_horizon_minutes,
                "metadata": forecast.metadata,
            }
        }
    
    @app.post("/api/v1/demand/predict-multiple")
    async def predict_multiple_demands(request: Dict[str, Any]):
        """Predecir demanda para múltiples recursos."""
        predictions = await demand_predictor.predict_multiple_resources(
            resource_types=request.get("resource_types", []),
            time_horizon_minutes=request.get("time_horizon_minutes", 5),
        )
        
        return {
            "predictions": {
                resource_type: {
                    "predicted_value": forecast.predicted_value,
                    "confidence": forecast.confidence,
                    "metadata": forecast.metadata,
                }
                for resource_type, forecast in predictions.items()
            }
        }
    
    @app.get("/api/v1/demand/history/{resource_type}")
    async def get_demand_history(resource_type: str, limit: int = 100):
        """Obtener historial de demanda."""
        history = demand_predictor.get_demand_history(resource_type, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/demand/forecasts")
    async def get_forecast_history(resource_type: Optional[str] = None, limit: int = 100):
        """Obtener historial de pronósticos."""
        forecasts = demand_predictor.get_forecast_history(resource_type, limit)
        return {"forecasts": forecasts, "count": len(forecasts)}
    
    # Intelligent Health Checker
    @app.post("/api/v1/health/register-check")
    async def register_health_check(request: Dict[str, Any]):
        """Registrar check de salud."""
        check_id = request.get("check_id")
        # Nota: En producción necesitarías deserializar la función
        return {
            "success": True,
            "message": "Check registration endpoint - implement check function registration",
            "check_id": check_id,
        }
    
    @app.post("/api/v1/health/register-metric")
    async def register_health_metric(request: Dict[str, Any]):
        """Registrar métrica de salud."""
        intelligent_health.register_metric(
            metric_name=request.get("metric_name"),
            threshold_warning=request.get("threshold_warning", 80.0),
            threshold_critical=request.get("threshold_critical", 95.0),
            unit=request.get("unit", ""),
        )
        return {"success": True, "message": "Metric registered"}
    
    @app.post("/api/v1/health/update-metric")
    async def update_health_metric(request: Dict[str, Any]):
        """Actualizar valor de métrica."""
        await intelligent_health.update_metric(
            metric_name=request.get("metric_name"),
            value=request.get("value", 0.0),
        )
        return {"success": True, "message": "Metric updated"}
    
    @app.get("/api/v1/health/check")
    async def run_health_checks():
        """Ejecutar todos los health checks."""
        results = await intelligent_health.run_all_checks()
        return results
    
    @app.get("/api/v1/health/summary")
    async def get_health_summary():
        """Obtener resumen de salud."""
        summary = intelligent_health.get_health_summary()
        return summary
    
    # Intelligent Cache
    @app.post("/api/v1/cache/get")
    async def cache_get(request: Dict[str, Any]):
        """Obtener valor del caché."""
        key = request.get("key")
        default = request.get("default")
        value = await intelligent_cache.get(key, default)
        return {"found": value is not None, "value": value}
    
    @app.post("/api/v1/cache/set")
    async def cache_set(request: Dict[str, Any]):
        """Guardar valor en caché."""
        await intelligent_cache.set(
            key=request.get("key", ""),
            value=request.get("value"),
            ttl=request.get("ttl"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "message": "Value cached"}
    
    @app.post("/api/v1/cache/invalidate")
    async def cache_invalidate(request: Dict[str, Any]):
        """Invalidar entrada."""
        key = request.get("key")
        if key:
            await intelligent_cache.invalidate(key)
        return {"success": True}
    
    @app.post("/api/v1/cache/invalidate-pattern")
    async def cache_invalidate_pattern(request: Dict[str, Any]):
        """Invalidar por patrón."""
        pattern = request.get("pattern", "")
        count = await intelligent_cache.invalidate_pattern(pattern)
        return {"success": True, "invalidated_count": count}
    
    @app.post("/api/v1/cache/prefetch")
    async def cache_prefetch(request: Dict[str, Any]):
        """Pre-cargar entrada."""
        # Nota: En producción, fetch_func necesitaría ser deserializado
        return {
            "success": True,
            "message": "Prefetch endpoint - implement fetch function registration"
        }
    
    @app.get("/api/v1/cache/stats")
    async def get_cache_stats():
        """Obtener estadísticas de caché."""
        stats = intelligent_cache.get_cache_stats()
        return stats
    
    @app.get("/api/v1/cache/patterns")
    async def get_cache_patterns():
        """Analizar patrones de acceso."""
        patterns = intelligent_cache.analyze_access_patterns()
        return {"patterns": patterns}
    
    @app.post("/api/v1/cache/clear")
    async def cache_clear():
        """Limpiar caché."""
        await intelligent_cache.clear()
        return {"success": True, "message": "Cache cleared"}
    
    # Sentiment Analyzer
    @app.post("/api/v1/sentiment/analyze")
    async def analyze_sentiment(request: Dict[str, Any]):
        """Analizar sentimiento."""
        text = request.get("text", "")
        result = sentiment_analyzer.analyze(text)
        
        return {
            "sentiment": result.sentiment.value,
            "polarity": result.polarity,
            "confidence": result.confidence,
            "emotions": result.emotions,
            "keywords": result.keywords,
        }
    
    @app.post("/api/v1/sentiment/analyze-batch")
    async def analyze_sentiment_batch(request: Dict[str, Any]):
        """Analizar lote de textos."""
        texts = request.get("texts", [])
        results = await sentiment_analyzer.analyze_batch(texts)
        
        return {
            "results": [
                {
                    "sentiment": r.sentiment.value,
                    "polarity": r.polarity,
                    "confidence": r.confidence,
                    "emotions": r.emotions,
                    "keywords": r.keywords,
                }
                for r in results
            ],
            "count": len(results),
        }
    
    @app.post("/api/v1/sentiment/summary")
    async def get_sentiment_summary(request: Dict[str, Any]):
        """Obtener resumen de sentimientos."""
        # Asumir que results viene en el request
        results_data = request.get("results", [])
        # Convertir de vuelta a SentimentResult
        from ..core.sentiment_analyzer import SentimentResult as SR
        
        results = [
            SR(
                sentiment=Sentiment(r["sentiment"]),
                polarity=r["polarity"],
                confidence=r["confidence"],
                emotions=r.get("emotions", {}),
                keywords=r.get("keywords", []),
            )
            for r in results_data
        ]
        
        summary = sentiment_analyzer.get_sentiment_summary(results)
        return summary
    
    # Task Manager
    @app.post("/api/v1/tasks/create")
    async def create_task(request: Dict[str, Any]):
        """Crear tarea."""
        priority_str = request.get("priority", "medium")
        priority = TaskPriority(priority_str)
        
        due_date_str = request.get("due_date")
        due_date = datetime.fromisoformat(due_date_str) if due_date_str else None
        
        task_id = task_manager.create_task(
            task_id=request.get("task_id", ""),
            title=request.get("title", ""),
            description=request.get("description", ""),
            priority=priority,
            assignee=request.get("assignee"),
            due_date=due_date,
            dependencies=request.get("dependencies"),
            metadata=request.get("metadata"),
        )
        
        return {"success": True, "task_id": task_id}
    
    @app.post("/api/v1/tasks/{task_id}/update-status")
    async def update_task_status(task_id: str, request: Dict[str, Any]):
        """Actualizar estado de tarea."""
        status_str = request.get("status")
        status = TaskStatus(status_str)
        
        success = task_manager.update_task_status(task_id, status)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {"success": True, "status": status.value}
    
    @app.post("/api/v1/tasks/{task_id}/update-progress")
    async def update_task_progress(task_id: str, request: Dict[str, Any]):
        """Actualizar progreso de tarea."""
        progress = request.get("progress", 0.0)
        
        success = task_manager.update_task_progress(task_id, progress)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {"success": True, "progress": progress}
    
    @app.get("/api/v1/tasks/{task_id}")
    async def get_task(task_id: str):
        """Obtener tarea."""
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task
    
    @app.get("/api/v1/tasks/status/{status}")
    async def get_tasks_by_status(status: str):
        """Obtener tareas por estado."""
        status_enum = TaskStatus(status)
        tasks = task_manager.get_tasks_by_status(status_enum)
        return {"tasks": tasks, "count": len(tasks)}
    
    @app.get("/api/v1/tasks/assignee/{assignee}")
    async def get_tasks_by_assignee(assignee: str):
        """Obtener tareas por asignado."""
        tasks = task_manager.get_tasks_by_assignee(assignee)
        return {"tasks": tasks, "count": len(tasks)}
    
    @app.get("/api/v1/tasks/overdue")
    async def get_overdue_tasks():
        """Obtener tareas vencidas."""
        tasks = task_manager.get_overdue_tasks()
        return {"tasks": tasks, "count": len(tasks)}
    
    @app.post("/api/v1/task-lists/create")
    async def create_task_list(request: Dict[str, Any]):
        """Crear lista de tareas."""
        list_id = task_manager.create_task_list(
            list_id=request.get("list_id", ""),
            name=request.get("name", ""),
            description=request.get("description", ""),
            metadata=request.get("metadata"),
        )
        return {"success": True, "list_id": list_id}
    
    @app.post("/api/v1/task-lists/{list_id}/add-task")
    async def add_task_to_list(list_id: str, request: Dict[str, Any]):
        """Agregar tarea a lista."""
        success = task_manager.add_task_to_list(list_id, request.get("task_id"))
        if not success:
            raise HTTPException(status_code=404, detail="Task list not found")
        return {"success": True}
    
    @app.get("/api/v1/task-lists/{list_id}")
    async def get_task_list(list_id: str):
        """Obtener lista de tareas."""
        task_list = task_manager.get_task_list(list_id)
        if not task_list:
            raise HTTPException(status_code=404, detail="Task list not found")
        return task_list
    
    @app.get("/api/v1/tasks/summary")
    async def get_task_manager_summary():
        """Obtener resumen del gestor de tareas."""
        summary = task_manager.get_task_manager_summary()
        return summary
    
    # Resource Monitor
    @app.get("/api/v1/resources/current")
    async def get_current_resources():
        """Obtener métricas actuales."""
        metrics = resource_monitor.get_current_metrics()
        return metrics
    
    @app.get("/api/v1/resources/history/{resource_type}")
    async def get_resource_history(resource_type: str, limit: int = 100):
        """Obtener historial de recurso."""
        rt = ResourceType(resource_type)
        history = resource_monitor.get_metric_history(rt, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/resources/statistics/{resource_type}")
    async def get_resource_statistics(resource_type: str, period_minutes: int = 60):
        """Obtener estadísticas de recurso."""
        rt = ResourceType(resource_type)
        stats = resource_monitor.get_metric_statistics(rt, period_minutes)
        return stats
    
    @app.get("/api/v1/resources/alerts")
    async def get_resource_alerts(level: Optional[str] = None):
        """Obtener alertas de recursos."""
        alert_level = AlertLevel(level) if level else None
        alerts = resource_monitor.get_active_alerts(alert_level)
        return {"alerts": alerts, "count": len(alerts)}
    
    @app.post("/api/v1/resources/alerts/{alert_id}/resolve")
    async def resolve_resource_alert(alert_id: str):
        """Resolver alerta."""
        success = resource_monitor.resolve_alert(alert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"success": True, "message": "Alert resolved"}
    
    @app.get("/api/v1/resources/summary")
    async def get_resource_monitor_summary():
        """Obtener resumen del monitor."""
        summary = resource_monitor.get_resource_monitor_summary()
        return summary
    
    # Push Notifications
    @app.post("/api/v1/notifications/send")
    async def send_notification(request: Dict[str, Any]):
        """Enviar notificación."""
        channels_str = request.get("channels", [])
        channels = [NotificationChannel(c) for c in channels_str]
        
        priority_str = request.get("priority", "normal")
        priority = NotificationPriority(priority_str)
        
        notification_id = await push_notifications.send_notification(
            user_id=request.get("user_id", ""),
            title=request.get("title", ""),
            message=request.get("message", ""),
            channels=channels,
            priority=priority,
            metadata=request.get("metadata"),
            action_url=request.get("action_url"),
        )
        
        return {"success": True, "notification_id": notification_id}
    
    @app.post("/api/v1/notifications/subscribe")
    async def subscribe_notifications(request: Dict[str, Any]):
        """Suscribir usuario a notificaciones."""
        channels_str = request.get("channels", [])
        channels = [NotificationChannel(c) for c in channels_str]
        
        push_notifications.subscribe_user(
            user_id=request.get("user_id", ""),
            channels=channels,
            preferences=request.get("preferences"),
        )
        
        return {"success": True, "message": "User subscribed"}
    
    @app.post("/api/v1/notifications/unsubscribe")
    async def unsubscribe_notifications(request: Dict[str, Any]):
        """Desuscribir usuario."""
        push_notifications.unsubscribe_user(request.get("user_id", ""))
        return {"success": True, "message": "User unsubscribed"}
    
    @app.get("/api/v1/notifications/user/{user_id}")
    async def get_user_notifications(user_id: str, unread_only: bool = False, limit: int = 50):
        """Obtener notificaciones de usuario."""
        notifications = push_notifications.get_user_notifications(user_id, unread_only, limit)
        return {"notifications": notifications, "count": len(notifications)}
    
    @app.post("/api/v1/notifications/{notification_id}/read")
    async def mark_notification_read(notification_id: str):
        """Marcar notificación como leída."""
        success = push_notifications.mark_as_read(notification_id)
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")
        return {"success": True, "message": "Notification marked as read"}
    
    @app.get("/api/v1/notifications/stats")
    async def get_notification_stats(user_id: Optional[str] = None):
        """Obtener estadísticas de notificaciones."""
        stats = push_notifications.get_notification_stats(user_id)
        return stats
    
    # Distributed Sync
    @app.post("/api/v1/sync/create-resource")
    async def create_sync_resource(request: Dict[str, Any]):
        """Crear recurso sincronizado."""
        operation_id = distributed_sync.create_resource(
            resource_id=request.get("resource_id", ""),
            resource_type=request.get("resource_type", ""),
            data=request.get("data", {}),
        )
        return {"success": True, "operation_id": operation_id}
    
    @app.post("/api/v1/sync/update-resource")
    async def update_sync_resource(request: Dict[str, Any]):
        """Actualizar recurso sincronizado."""
        operation_id = distributed_sync.update_resource(
            resource_id=request.get("resource_id", ""),
            data=request.get("data", {}),
            expected_version=request.get("expected_version"),
        )
        return {"success": True, "operation_id": operation_id}
    
    @app.post("/api/v1/sync/sync-from-remote")
    async def sync_from_remote(request: Dict[str, Any]):
        """Sincronizar desde operaciones remotas."""
        result = await distributed_sync.sync_from_remote(
            remote_operations=request.get("operations", [])
        )
        return result
    
    @app.post("/api/v1/sync/resolve-conflict")
    async def resolve_sync_conflict(request: Dict[str, Any]):
        """Resolver conflicto de sincronización."""
        resolution_str = request.get("resolution", "last_write_wins")
        resolution = ConflictResolution(resolution_str)
        
        success = distributed_sync.resolve_conflict(
            conflict_id=request.get("conflict_id", ""),
            resolution=resolution,
            resolved_data=request.get("resolved_data"),
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Conflict not found")
        
        return {"success": True, "message": "Conflict resolved"}
    
    @app.get("/api/v1/sync/resource/{resource_id}")
    async def get_sync_resource(resource_id: str):
        """Obtener recurso sincronizado."""
        resource = distributed_sync.get_resource(resource_id)
        if not resource:
            raise HTTPException(status_code=404, detail="Resource not found")
        return resource
    
    @app.get("/api/v1/sync/conflicts")
    async def get_sync_conflicts(resolved: Optional[bool] = None):
        """Obtener conflictos de sincronización."""
        conflicts = distributed_sync.get_conflicts(resolved)
        return {"conflicts": conflicts, "count": len(conflicts)}
    
    @app.get("/api/v1/sync/summary")
    async def get_sync_summary():
        """Obtener resumen de sincronización."""
        summary = distributed_sync.get_sync_summary()
        return summary
    
    # Query Analyzer
    @app.post("/api/v1/queries/record")
    async def record_query(request: Dict[str, Any]):
        """Registrar ejecución de query."""
        query_id = query_analyzer.record_query(
            query_text=request.get("query_text", ""),
            execution_time=request.get("execution_time", 0.0),
            rows_affected=request.get("rows_affected", 0),
            error=request.get("error"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "query_id": query_id}
    
    @app.get("/api/v1/queries/slow")
    async def get_slow_queries(threshold: Optional[float] = None, limit: int = 50):
        """Obtener queries lentas."""
        slow = query_analyzer.get_slow_queries(threshold, limit)
        return {"slow_queries": slow, "count": len(slow)}
    
    @app.get("/api/v1/queries/patterns")
    async def get_query_patterns(query_type: Optional[str] = None, limit: int = 50):
        """Obtener patrones de queries."""
        qtype = QueryType(query_type) if query_type else None
        patterns = query_analyzer.get_query_patterns(qtype, limit)
        return {"patterns": patterns, "count": len(patterns)}
    
    @app.get("/api/v1/queries/statistics")
    async def get_query_statistics(
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ):
        """Obtener estadísticas de queries."""
        start = datetime.fromisoformat(start_time) if start_time else None
        end = datetime.fromisoformat(end_time) if end_time else None
        
        stats = query_analyzer.get_query_statistics(start, end)
        return stats
    
    @app.get("/api/v1/queries/summary")
    async def get_query_analyzer_summary():
        """Obtener resumen del analizador."""
        summary = query_analyzer.get_query_analyzer_summary()
        return summary
    
    # File Manager
    @app.post("/api/v1/files/upload")
    async def upload_file(request: Dict[str, Any]):
        """Subir archivo."""
        file_id = request.get("file_id", f"file_{datetime.now().timestamp()}")
        filename = request.get("filename", "")
        data_bytes = request.get("data")
        
        if not filename or not data_bytes:
            raise HTTPException(status_code=400, detail="filename and data required")
        
        # Convertir data a bytes si viene como string (base64)
        if isinstance(data_bytes, str):
            import base64
            data_bytes = base64.b64decode(data_bytes)
        
        file_id = await file_manager.upload_file(
            file_id=file_id,
            filename=filename,
            data=data_bytes,
            mime_type=request.get("mime_type", ""),
            tags=request.get("tags"),
            metadata=request.get("metadata"),
        )
        
        return {"success": True, "file_id": file_id}
    
    @app.get("/api/v1/files/{file_id}")
    async def download_file(file_id: str):
        """Descargar archivo."""
        data = await file_manager.download_file(file_id)
        if data is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        metadata = file_manager.get_file_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        from fastapi.responses import Response
        
        return Response(
            content=data,
            media_type=metadata.get("mime_type", "application/octet-stream"),
            headers={
                "Content-Disposition": f'attachment; filename="{metadata["filename"]}"',
            },
        )
    
    @app.get("/api/v1/files/{file_id}/metadata")
    async def get_file_metadata(file_id: str):
        """Obtener metadatos de archivo."""
        metadata = file_manager.get_file_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        return metadata
    
    @app.get("/api/v1/files/{file_id}/versions")
    async def get_file_versions(file_id: str):
        """Obtener versiones de archivo."""
        versions = file_manager.get_file_versions(file_id)
        return {"versions": versions, "count": len(versions)}
    
    @app.get("/api/v1/files/search")
    async def search_files(
        tags: Optional[str] = None,
        file_type: Optional[str] = None,
        limit: int = 100,
    ):
        """Buscar archivos."""
        tags_list = tags.split(",") if tags else None
        file_type_enum = FileType(file_type) if file_type else None
        
        files = file_manager.search_files(tags_list, file_type_enum, limit)
        return {"files": files, "count": len(files)}
    
    @app.delete("/api/v1/files/{file_id}")
    async def delete_file(file_id: str, permanent: bool = False):
        """Eliminar archivo."""
        success = await file_manager.delete_file(file_id, permanent)
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        return {"success": True, "message": "File deleted"}
    
    @app.post("/api/v1/files/{file_id}/restore")
    async def restore_file(file_id: str):
        """Restaurar archivo eliminado."""
        success = await file_manager.restore_file(file_id)
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        return {"success": True, "message": "File restored"}
    
    @app.get("/api/v1/files/summary")
    async def get_file_manager_summary():
        """Obtener resumen del gestor."""
        summary = file_manager.get_file_manager_summary()
        return summary
    
    # Data Compression
    @app.post("/api/v1/compression/compress")
    async def compress_data(request: Dict[str, Any]):
        """Comprimir datos."""
        data = request.get("data")
        algorithm_str = request.get("algorithm", "gzip")
        algorithm = CompressionAlgorithm(algorithm_str)
        
        if not data:
            raise HTTPException(status_code=400, detail="data required")
        
        # Convertir a bytes si viene como string (base64)
        if isinstance(data, str):
            import base64
            data = base64.b64decode(data)
        
        compressed = data_compression.compress(data, algorithm)
        
        # Convertir a base64 para respuesta
        import base64
        compressed_b64 = base64.b64encode(compressed).decode()
        
        return {
            "success": True,
            "compressed_data": compressed_b64,
            "original_size": len(data),
            "compressed_size": len(compressed),
            "compression_ratio": len(compressed) / len(data) if len(data) > 0 else 1.0,
        }
    
    @app.post("/api/v1/compression/decompress")
    async def decompress_data(request: Dict[str, Any]):
        """Descomprimir datos."""
        compressed_data = request.get("compressed_data")
        algorithm_str = request.get("algorithm", "gzip")
        algorithm = CompressionAlgorithm(algorithm_str)
        
        if not compressed_data:
            raise HTTPException(status_code=400, detail="compressed_data required")
        
        # Convertir de base64
        import base64
        compressed_bytes = base64.b64decode(compressed_data)
        
        data = data_compression.decompress(compressed_bytes, algorithm)
        data_b64 = base64.b64encode(data).decode()
        
        return {
            "success": True,
            "data": data_b64,
            "decompressed_size": len(data),
        }
    
    @app.post("/api/v1/compression/find-best")
    async def find_best_compression(request: Dict[str, Any]):
        """Encontrar mejor algoritmo de compresión."""
        data = request.get("data")
        
        if not data:
            raise HTTPException(status_code=400, detail="data required")
        
        if isinstance(data, str):
            import base64
            data = base64.b64decode(data)
        
        best_algorithm = data_compression.find_best_algorithm(data)
        
        return {
            "success": True,
            "best_algorithm": best_algorithm.value,
        }
    
    @app.get("/api/v1/compression/stats")
    async def get_compression_stats():
        """Obtener estadísticas de compresión."""
        stats = data_compression.get_compression_stats()
        return stats
    
    # Incremental Backup
    @app.post("/api/v1/backup/create")
    async def create_backup(request: Dict[str, Any]):
        """Crear backup."""
        backup_id = request.get("backup_id", f"backup_{datetime.now().timestamp()}")
        data = request.get("data")
        backup_type_str = request.get("backup_type", "incremental")
        backup_type = BackupType(backup_type_str)
        
        if not data:
            raise HTTPException(status_code=400, detail="data required")
        
        if isinstance(data, str):
            import base64
            data = base64.b64decode(data)
        elif isinstance(data, dict):
            data = json.dumps(data).encode()
        
        backup_id = await incremental_backup.create_backup(
            backup_id=backup_id,
            data=data,
            backup_type=backup_type,
            parent_backup_id=request.get("parent_backup_id"),
            metadata=request.get("metadata"),
        )
        
        return {"success": True, "backup_id": backup_id}
    
    @app.post("/api/v1/backup/restore")
    async def restore_backup(request: Dict[str, Any]):
        """Restaurar backup."""
        backup_id = request.get("backup_id")
        
        if not backup_id:
            raise HTTPException(status_code=400, detail="backup_id required")
        
        data = await incremental_backup.restore_backup(backup_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Backup not found")
        
        import base64
        data_b64 = base64.b64encode(data).decode()
        
        return {"success": True, "data": data_b64, "size": len(data)}
    
    @app.get("/api/v1/backup/{backup_id}")
    async def get_backup_info(backup_id: str):
        """Obtener información de backup."""
        backup = incremental_backup.get_backup(backup_id)
        if not backup:
            raise HTTPException(status_code=404, detail="Backup not found")
        return backup
    
    @app.get("/api/v1/backup/list")
    async def list_backups(backup_type: Optional[str] = None, limit: int = 100):
        """Listar backups."""
        btype = BackupType(backup_type) if backup_type else None
        backups = incremental_backup.list_backups(btype, limit)
        return {"backups": backups, "count": len(backups)}
    
    @app.post("/api/v1/backup/create-set")
    async def create_backup_set(request: Dict[str, Any]):
        """Crear conjunto de backups."""
        set_id = await incremental_backup.create_backup_set(
            set_id=request.get("set_id", f"set_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            backup_ids=request.get("backup_ids", []),
            metadata=request.get("metadata"),
        )
        return {"success": True, "set_id": set_id}
    
    @app.get("/api/v1/backup/summary")
    async def get_backup_summary():
        """Obtener resumen de backups."""
        summary = incremental_backup.get_backup_summary()
        return summary
    
    # Network Analyzer
    @app.post("/api/v1/network/record")
    async def record_network_metric(request: Dict[str, Any]):
        """Registrar métrica de red."""
        metric_id = network_analyzer.record_metric(
            endpoint=request.get("endpoint", ""),
            latency=request.get("latency", 0.0),
            bytes_sent=request.get("bytes_sent", 0),
            bytes_received=request.get("bytes_received", 0),
            status_code=request.get("status_code"),
            error=request.get("error"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "metric_id": metric_id}
    
    @app.get("/api/v1/network/endpoint/{endpoint}")
    async def get_network_endpoint_stats(endpoint: str):
        """Obtener estadísticas de endpoint."""
        stats = network_analyzer.get_endpoint_stats(endpoint)
        if not stats:
            raise HTTPException(status_code=404, detail="Endpoint not found")
        return stats
    
    @app.get("/api/v1/network/slow-endpoints")
    async def get_slow_network_endpoints(threshold: float = 1.0, limit: int = 10):
        """Obtener endpoints lentos."""
        slow = network_analyzer.get_slow_endpoints(threshold, limit)
        return {"slow_endpoints": slow, "count": len(slow)}
    
    @app.get("/api/v1/network/events")
    async def get_network_events(
        event_type: Optional[str] = None,
        endpoint: Optional[str] = None,
        limit: int = 100,
    ):
        """Obtener eventos de red."""
        etype = NetworkEventType(event_type) if event_type else None
        events = network_analyzer.get_network_events(etype, endpoint, limit)
        return {"events": events, "count": len(events)}
    
    @app.get("/api/v1/network/summary")
    async def get_network_summary():
        """Obtener resumen de red."""
        summary = network_analyzer.get_network_summary()
        return summary
    
    # Config Manager
    @app.post("/api/v1/config/register")
    async def register_config(request: Dict[str, Any]):
        """Registrar configuración."""
        format_str = request.get("format", "json")
        config_format = ConfigFormat(format_str)
        
        config_id = config_manager.register_config(
            config_id=request.get("config_id", ""),
            name=request.get("name", ""),
            data=request.get("data", {}),
            format=config_format,
            metadata=request.get("metadata"),
        )
        
        return {"success": True, "config_id": config_id}
    
    @app.get("/api/v1/config/{config_id}")
    async def get_config(config_id: str, version: Optional[int] = None):
        """Obtener configuración."""
        config = config_manager.get_config(config_id, version)
        if not config:
            raise HTTPException(status_code=404, detail="Config not found")
        return config
    
    @app.get("/api/v1/config/{config_id}/history")
    async def get_config_history(config_id: str, limit: int = 50):
        """Obtener historial de configuración."""
        history = config_manager.get_config_history(config_id, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/config/changes")
    async def get_config_changes(config_id: Optional[str] = None, limit: int = 100):
        """Obtener cambios de configuración."""
        changes = config_manager.get_config_changes(config_id, limit)
        return {"changes": changes, "count": len(changes)}
    
    @app.post("/api/v1/config/{config_id}/rollback")
    async def rollback_config(config_id: str, request: Dict[str, Any]):
        """Revertir configuración."""
        target_version = request.get("target_version")
        
        if target_version is None:
            raise HTTPException(status_code=400, detail="target_version required")
        
        success = config_manager.rollback_config(config_id, target_version)
        if not success:
            raise HTTPException(status_code=404, detail="Config or version not found")
        
        return {"success": True, "message": f"Config rolled back to version {target_version}"}
    
    @app.post("/api/v1/config/{config_id}/subscribe")
    async def subscribe_config_changes(config_id: str):
        """Suscribirse a cambios de configuración."""
        # Nota: En producción, el listener necesitaría ser registrado de otra manera
        return {
            "success": True,
            "message": "Config change subscription endpoint - implement listener registration"
        }
    
    @app.get("/api/v1/config/summary")
    async def get_config_manager_summary():
        """Obtener resumen del gestor."""
        summary = config_manager.get_config_manager_summary()
        return summary
    
    # MFA Authentication
    @app.post("/api/v1/mfa/setup/totp")
    async def setup_totp(request: Dict[str, Any]):
        """Configurar TOTP para usuario."""
        user_id = request.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")
        
        result = mfa_authentication.setup_totp(user_id)
        return {"success": True, "data": result}
    
    @app.post("/api/v1/mfa/setup/sms")
    async def setup_sms(request: Dict[str, Any]):
        """Configurar SMS para usuario."""
        user_id = request.get("user_id")
        phone = request.get("phone")
        
        if not user_id or not phone:
            raise HTTPException(status_code=400, detail="user_id and phone required")
        
        success = mfa_authentication.setup_sms(user_id, phone)
        return {"success": success}
    
    @app.post("/api/v1/mfa/setup/email")
    async def setup_email(request: Dict[str, Any]):
        """Configurar Email para usuario."""
        user_id = request.get("user_id")
        email = request.get("email")
        
        if not user_id or not email:
            raise HTTPException(status_code=400, detail="user_id and email required")
        
        success = mfa_authentication.setup_email(user_id, email)
        return {"success": success}
    
    @app.post("/api/v1/mfa/initiate")
    async def initiate_mfa(request: Dict[str, Any]):
        """Iniciar proceso de MFA."""
        user_id = request.get("user_id")
        method_str = request.get("method", "totp")
        method = MFAMethod(method_str)
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")
        
        session_id = await mfa_authentication.initiate_mfa(user_id, method)
        return {"success": True, "session_id": session_id}
    
    @app.post("/api/v1/mfa/verify")
    async def verify_mfa(request: Dict[str, Any]):
        """Verificar código MFA."""
        session_id = request.get("session_id")
        code = request.get("code")
        
        if not session_id or not code:
            raise HTTPException(status_code=400, detail="session_id and code required")
        
        verified = await mfa_authentication.verify_mfa(session_id, code)
        return {"success": verified, "verified": verified}
    
    @app.get("/api/v1/mfa/status/{user_id}")
    async def get_mfa_status(user_id: str):
        """Obtener estado MFA del usuario."""
        status = mfa_authentication.get_user_mfa_status(user_id)
        return status
    
    # Advanced Rate Limiter
    @app.post("/api/v1/rate-limit/create-rule")
    async def create_rate_limit_rule(request: Dict[str, Any]):
        """Crear regla de rate limiting."""
        strategy_str = request.get("strategy", "fixed_window")
        strategy = RateLimitStrategy(strategy_str)
        
        rule_id = advanced_rate_limiter.create_rule(
            rule_id=request.get("rule_id", f"rule_{datetime.now().timestamp()}"),
            identifier=request.get("identifier", ""),
            strategy=strategy,
            max_requests=request.get("max_requests", 100),
            window_seconds=request.get("window_seconds", 60),
            burst_size=request.get("burst_size"),
            tokens_per_second=request.get("tokens_per_second"),
            metadata=request.get("metadata"),
        )
        
        return {"success": True, "rule_id": rule_id}
    
    @app.post("/api/v1/rate-limit/check")
    async def check_rate_limit(request: Dict[str, Any]):
        """Verificar rate limit."""
        identifier = request.get("identifier")
        rule_id = request.get("rule_id")
        
        if not identifier:
            raise HTTPException(status_code=400, detail="identifier required")
        
        allowed, info = await advanced_rate_limiter.check_rate_limit(identifier, rule_id)
        
        return {
            "allowed": allowed,
            **info,
        }
    
    @app.post("/api/v1/rate-limit/block")
    async def block_identifier(request: Dict[str, Any]):
        """Bloquear identificador."""
        identifier = request.get("identifier")
        duration_seconds = request.get("duration_seconds", 3600)
        
        if not identifier:
            raise HTTPException(status_code=400, detail="identifier required")
        
        await advanced_rate_limiter.block_identifier(identifier, duration_seconds)
        return {"success": True, "message": f"Blocked {identifier} for {duration_seconds} seconds"}
    
    @app.get("/api/v1/rate-limit/violations")
    async def get_rate_limit_violations(identifier: Optional[str] = None, limit: int = 100):
        """Obtener violaciones de rate limit."""
        violations = advanced_rate_limiter.get_violations(identifier, limit)
        return {"violations": violations, "count": len(violations)}
    
    @app.get("/api/v1/rate-limit/summary")
    async def get_rate_limiter_summary():
        """Obtener resumen del rate limiter."""
        summary = advanced_rate_limiter.get_rate_limiter_summary()
        return summary
    
    # User Behavior Analyzer
    @app.post("/api/v1/behavior/record")
    async def record_user_action(request: Dict[str, Any]):
        """Registrar acción de usuario."""
        action_id = user_behavior_analyzer.record_action(
            user_id=request.get("user_id", ""),
            action_type=request.get("action_type", ""),
            ip_address=request.get("ip_address"),
            user_agent=request.get("user_agent"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "action_id": action_id}
    
    @app.get("/api/v1/behavior/profile/{user_id}")
    async def get_user_profile(user_id: str):
        """Obtener perfil de usuario."""
        profile = user_behavior_analyzer.get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        return profile
    
    @app.get("/api/v1/behavior/high-risk")
    async def get_high_risk_users(threshold: float = 0.7, limit: int = 50):
        """Obtener usuarios de alto riesgo."""
        high_risk = user_behavior_analyzer.get_high_risk_users(threshold, limit)
        return {"users": high_risk, "count": len(high_risk)}
    
    @app.get("/api/v1/behavior/anomalies")
    async def get_behavior_anomalies(user_id: Optional[str] = None, limit: int = 100):
        """Obtener anomalías de comportamiento."""
        anomalies = user_behavior_analyzer.get_anomalies(user_id, limit)
        return {"anomalies": anomalies, "count": len(anomalies)}
    
    @app.get("/api/v1/behavior/summary")
    async def get_behavior_analyzer_summary():
        """Obtener resumen del analizador."""
        summary = user_behavior_analyzer.get_behavior_analyzer_summary()
        return summary
    
    # Event Stream
    @app.post("/api/v1/events/publish")
    async def publish_event(request: Dict[str, Any]):
        """Publicar evento."""
        event_type_str = request.get("event_type", "custom")
        event_type = EventType(event_type_str)
        
        event_id = event_stream.publish(
            event_type=event_type,
            source=request.get("source", "system"),
            data=request.get("data", {}),
            metadata=request.get("metadata"),
        )
        
        return {"success": True, "event_id": event_id}
    
    @app.get("/api/v1/events")
    async def get_events(
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ):
        """Obtener eventos."""
        etype = EventType(event_type) if event_type else None
        start = datetime.fromisoformat(start_time) if start_time else None
        end = datetime.fromisoformat(end_time) if end_time else None
        
        events = event_stream.get_events(etype, source, start, end, limit)
        return {"events": events, "count": len(events)}
    
    @app.post("/api/v1/events/subscribe")
    async def subscribe_events(request: Dict[str, Any]):
        """Suscribirse a eventos."""
        # Nota: En producción, el handler necesitaría ser registrado de otra manera
        return {
            "success": True,
            "message": "Event subscription endpoint - implement handler registration"
        }
    
    @app.get("/api/v1/events/summary")
    async def get_event_stream_summary():
        """Obtener resumen del stream."""
        summary = event_stream.get_event_stream_summary()
        return summary
    
    # Security Analyzer
    @app.post("/api/v1/security/analyze")
    async def analyze_security(request: Dict[str, Any]):
        """Analizar entrada en busca de amenazas."""
        input_data = request.get("input_data", "")
        source = request.get("source", "unknown")
        
        if not input_data:
            raise HTTPException(status_code=400, detail="input_data required")
        
        threats = security_analyzer.analyze_input(input_data, source, request.get("context"))
        
        return {
            "threats_detected": len(threats),
            "threats": [
                {
                    "threat_id": t.threat_id,
                    "threat_type": t.threat_type.value,
                    "threat_level": t.threat_level.value,
                    "description": t.description,
                }
                for t in threats
            ],
        }
    
    @app.post("/api/v1/security/block")
    async def block_source(request: Dict[str, Any]):
        """Bloquear fuente."""
        source = request.get("source")
        duration_seconds = request.get("duration_seconds", 3600)
        
        if not source:
            raise HTTPException(status_code=400, detail="source required")
        
        security_analyzer.block_source(source, duration_seconds)
        return {"success": True, "message": f"Blocked {source} for {duration_seconds} seconds"}
    
    @app.get("/api/v1/security/threats")
    async def get_security_threats(
        threat_type: Optional[str] = None,
        threat_level: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 100,
    ):
        """Obtener amenazas."""
        ttype = ThreatType(threat_type) if threat_type else None
        tlevel = ThreatLevel(threat_level) if threat_level else None
        
        threats = security_analyzer.get_threats(ttype, tlevel, resolved, limit)
        return {"threats": threats, "count": len(threats)}
    
    @app.post("/api/v1/security/threats/{threat_id}/resolve")
    async def resolve_threat(threat_id: str):
        """Resolver amenaza."""
        success = security_analyzer.resolve_threat(threat_id)
        if not success:
            raise HTTPException(status_code=404, detail="Threat not found")
        return {"success": True, "message": "Threat resolved"}
    
    @app.get("/api/v1/security/summary")
    async def get_security_analyzer_summary():
        """Obtener resumen del analizador."""
        summary = security_analyzer.get_security_analyzer_summary()
        return summary
    
    # Session Manager
    @app.post("/api/v1/sessions/create")
    async def create_session_manager(request: Dict[str, Any]):
        """Crear sesión en el gestor."""
        session_id = request.get("session_id", f"session_{datetime.now().timestamp()}")
        user_id = request.get("user_id", "")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")
        
        session_id = session_manager.create_session(session_id, user_id, request.get("metadata"))
        return {"success": True, "session_id": session_id}
    
    @app.post("/api/v1/sessions/{session_id}/activity")
    async def update_session_activity(
        session_id: str,
        request: Dict[str, Any],
    ):
        """Actualizar actividad de sesión."""
        session_manager.update_session_activity(
            session_id,
            message_count=request.get("message_count", 0),
            response_time=request.get("response_time"),
        )
        return {"success": True}
    
    @app.post("/api/v1/sessions/{session_id}/status")
    async def update_session_status(
        session_id: str,
        request: Dict[str, Any],
    ):
        """Actualizar estado de sesión."""
        status_str = request.get("status", "active")
        status = SessionStatus(status_str)
        
        session_manager.update_session_status(session_id, status)
        return {"success": True}
    
    @app.get("/api/v1/sessions/{session_id}")
    async def get_session_info(session_id: str):
        """Obtener información de sesión."""
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    
    @app.get("/api/v1/sessions/{session_id}/analytics")
    async def get_session_analytics(session_id: str):
        """Obtener analíticas de sesión."""
        analytics = session_manager.get_session_analytics(session_id)
        if not analytics:
            raise HTTPException(status_code=404, detail="Session analytics not found")
        return analytics
    
    @app.get("/api/v1/sessions/active")
    async def get_active_sessions(limit: int = 100):
        """Obtener sesiones activas."""
        active = session_manager.get_active_sessions(limit)
        return {"sessions": active, "count": len(active)}
    
    @app.post("/api/v1/sessions/cleanup")
    async def cleanup_sessions(request: Dict[str, Any]):
        """Limpiar sesiones expiradas."""
        max_idle_seconds = request.get("max_idle_seconds", 3600)
        expired = session_manager.cleanup_expired_sessions(max_idle_seconds)
        return {"success": True, "expired_count": len(expired), "expired_sessions": expired}
    
    @app.get("/api/v1/sessions/summary")
    async def get_session_manager_summary():
        """Obtener resumen del gestor."""
        summary = session_manager.get_session_manager_summary()
        return summary
    
    # Real-time Metrics
    @app.post("/api/v1/metrics/record")
    async def record_metric(request: Dict[str, Any]):
        """Registrar métrica."""
        metric_type_str = request.get("metric_type", "gauge")
        metric_type = MetricType(metric_type_str)
        
        metric_id = realtime_metrics.record_metric(
            metric_name=request.get("metric_name", ""),
            value=request.get("value", 0.0),
            metric_type=metric_type,
            labels=request.get("labels"),
            metadata=request.get("metadata"),
        )
        
        return {"success": True, "metric_id": metric_id}
    
    @app.get("/api/v1/metrics/aggregates/{metric_name}")
    async def get_metric_aggregates(metric_name: str, labels: Optional[str] = None):
        """Obtener agregados de métrica."""
        import json
        labels_dict = json.loads(labels) if labels else None
        
        aggregates = realtime_metrics.get_metric_aggregates(metric_name, labels_dict)
        if not aggregates:
            raise HTTPException(status_code=404, detail="Metric aggregates not found")
        return aggregates
    
    @app.get("/api/v1/metrics")
    async def get_metrics(
        metric_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 1000,
    ):
        """Obtener métricas."""
        start = datetime.fromisoformat(start_time) if start_time else None
        end = datetime.fromisoformat(end_time) if end_time else None
        
        metrics = realtime_metrics.get_metrics(metric_name, start, end, limit)
        return {"metrics": metrics, "count": len(metrics)}
    
    @app.post("/api/v1/metrics/alerts/create")
    async def create_metric_alert(request: Dict[str, Any]):
        """Crear alerta de métrica."""
        alert_id = realtime_metrics.create_alert(
            alert_id=request.get("alert_id", f"alert_{datetime.now().timestamp()}"),
            metric_name=request.get("metric_name", ""),
            condition=request.get("condition", "gt"),
            threshold=request.get("threshold", 0.0),
        )
        
        return {"success": True, "alert_id": alert_id}
    
    @app.get("/api/v1/metrics/summary")
    async def get_realtime_metrics_summary():
        """Obtener resumen de métricas."""
        summary = realtime_metrics.get_realtime_metrics_summary()
        return summary
    
    # Auto Optimizer
    @app.post("/api/v1/optimizer/register-parameter")
    async def register_parameter(request: Dict[str, Any]):
        """Registrar parámetro para optimización."""
        target_str = request.get("target", "latency")
        target = OptimizationTarget(target_str)
        
        auto_optimizer.register_parameter(
            parameter_name=request.get("parameter_name", ""),
            min_value=request.get("min_value"),
            max_value=request.get("max_value"),
            current_value=request.get("current_value"),
            target=target,
            step=request.get("step"),
        )
        
        return {"success": True}
    
    @app.post("/api/v1/optimizer/record-performance")
    async def record_performance(request: Dict[str, Any]):
        """Registrar métrica de performance."""
        auto_optimizer.record_performance(
            metric_name=request.get("metric_name", ""),
            value=request.get("value", 0.0),
            context=request.get("context"),
        )
        
        return {"success": True}
    
    @app.get("/api/v1/optimizer/parameter/{parameter_name}")
    async def get_parameter_value(parameter_name: str):
        """Obtener valor de parámetro."""
        value = auto_optimizer.get_parameter_value(parameter_name)
        if value is None:
            raise HTTPException(status_code=404, detail="Parameter not found")
        return {"parameter_name": parameter_name, "value": value}
    
    @app.get("/api/v1/optimizer/optimizations")
    async def get_optimizations(
        parameter_name: Optional[str] = None,
        target: Optional[str] = None,
        limit: int = 100,
    ):
        """Obtener optimizaciones."""
        ttarget = OptimizationTarget(target) if target else None
        
        optimizations = auto_optimizer.get_optimizations(parameter_name, ttarget, limit)
        return {"optimizations": optimizations, "count": len(optimizations)}
    
    @app.get("/api/v1/optimizer/summary")
    async def get_auto_optimizer_summary():
        """Obtener resumen del optimizador."""
        summary = auto_optimizer.get_auto_optimizer_summary()
        return summary
    
    # Adaptive Rate Controller
    @app.post("/api/v1/adaptive-rate/register")
    async def register_adaptive_rate(request: Dict[str, Any]):
        """Registrar identificador para control de tasa adaptativo."""
        identifier = adaptive_rate_controller.register_identifier(
            identifier=request.get("identifier", ""),
            base_rate=request.get("base_rate"),
            min_rate=request.get("min_rate"),
            max_rate=request.get("max_rate"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "identifier": identifier}
    
    @app.post("/api/v1/adaptive-rate/record")
    async def record_adaptive_rate_request(request: Dict[str, Any]):
        """Registrar petición para ajuste de tasa."""
        await adaptive_rate_controller.record_request(
            identifier=request.get("identifier", ""),
            success=request.get("success", True),
            response_time=request.get("response_time", 0.0),
            system_load=request.get("system_load"),
        )
        return {"success": True}
    
    @app.post("/api/v1/adaptive-rate/check")
    async def check_adaptive_rate_limit(request: Dict[str, Any]):
        """Verificar rate limit adaptativo."""
        allowed, info = await adaptive_rate_controller.check_rate_limit(
            request.get("identifier", "")
        )
        return {"allowed": allowed, **info}
    
    @app.get("/api/v1/adaptive-rate/{identifier}")
    async def get_adaptive_rate_limit(identifier: str):
        """Obtener límite de tasa adaptativo."""
        rate_limit = adaptive_rate_controller.get_rate_limit(identifier)
        if not rate_limit:
            raise HTTPException(status_code=404, detail="Identifier not found")
        return rate_limit
    
    @app.get("/api/v1/adaptive-rate/{identifier}/history")
    async def get_adaptive_rate_history(identifier: str, limit: int = 100):
        """Obtener historial de ajustes."""
        history = adaptive_rate_controller.get_adjustment_history(identifier, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/adaptive-rate/summary")
    async def get_adaptive_rate_summary():
        """Obtener resumen del controlador."""
        return adaptive_rate_controller.get_adaptive_rate_controller_summary()
    
    # Smart Retry Manager
    @app.post("/api/v1/retry/create")
    async def create_retry_operation(request: Dict[str, Any]):
        """Crear operación con reintentos."""
        from ..core.smart_retry_manager import RetryConfig, RetryStrategy
        
        retry_strategy_str = request.get("strategy", "exponential_backoff")
        retry_strategy = RetryStrategy(retry_strategy_str)
        
        config = RetryConfig(
            max_attempts=request.get("max_attempts", 3),
            initial_delay=request.get("initial_delay", 1.0),
            max_delay=request.get("max_delay", 60.0),
            backoff_multiplier=request.get("backoff_multiplier", 2.0),
            strategy=retry_strategy,
            retryable_errors=request.get("retryable_errors", []),
        )
        
        operation_id = smart_retry_manager.create_retry_operation(
            operation_id=request.get("operation_id", f"op_{datetime.now().timestamp()}"),
            operation_type=request.get("operation_type", "unknown"),
            config=config,
            metadata=request.get("metadata"),
        )
        return {"success": True, "operation_id": operation_id}
    
    @app.get("/api/v1/retry/{operation_id}")
    async def get_retry_operation(operation_id: str):
        """Obtener información de operación."""
        operation = smart_retry_manager.get_operation(operation_id)
        if not operation:
            raise HTTPException(status_code=404, detail="Operation not found")
        return operation
    
    @app.get("/api/v1/retry/patterns/{operation_type}")
    async def get_retry_patterns(operation_type: str):
        """Obtener patrones aprendidos."""
        patterns = smart_retry_manager.get_learning_patterns(operation_type)
        return patterns
    
    @app.get("/api/v1/retry/summary")
    async def get_retry_manager_summary():
        """Obtener resumen del gestor."""
        return smart_retry_manager.get_smart_retry_manager_summary()
    
    # Distributed Lock Manager
    @app.post("/api/v1/locks/acquire")
    async def acquire_distributed_lock(request: Dict[str, Any]):
        """Adquirir lock distribuido."""
        lock_id = await distributed_lock_manager.acquire_lock(
            resource_id=request.get("resource_id", ""),
            owner_id=request.get("owner_id", ""),
            ttl_seconds=request.get("ttl_seconds"),
            wait_timeout=request.get("wait_timeout"),
            metadata=request.get("metadata"),
        )
        if not lock_id:
            raise HTTPException(status_code=409, detail="Lock not available")
        return {"success": True, "lock_id": lock_id}
    
    @app.post("/api/v1/locks/{lock_id}/release")
    async def release_distributed_lock(lock_id: str, request: Dict[str, Any]):
        """Liberar lock distribuido."""
        success = await distributed_lock_manager.release_lock(
            lock_id=lock_id,
            owner_id=request.get("owner_id"),
        )
        if not success:
            raise HTTPException(status_code=404, detail="Lock not found or already released")
        return {"success": True}
    
    @app.post("/api/v1/locks/{lock_id}/renew")
    async def renew_distributed_lock(lock_id: str, request: Dict[str, Any]):
        """Renovar lock distribuido."""
        success = await distributed_lock_manager.renew_lock(
            lock_id=lock_id,
            owner_id=request.get("owner_id"),
            ttl_seconds=request.get("ttl_seconds"),
        )
        if not success:
            raise HTTPException(status_code=404, detail="Lock not found")
        return {"success": True}
    
    @app.get("/api/v1/locks/{lock_id}")
    async def get_distributed_lock(lock_id: str):
        """Obtener información de lock."""
        lock = distributed_lock_manager.get_lock(lock_id)
        if not lock:
            raise HTTPException(status_code=404, detail="Lock not found")
        return lock
    
    @app.get("/api/v1/locks/resource/{resource_id}")
    async def get_resource_lock(resource_id: str):
        """Obtener lock de recurso."""
        lock = distributed_lock_manager.get_resource_lock(resource_id)
        if not lock:
            raise HTTPException(status_code=404, detail="No lock for resource")
        return lock
    
    @app.post("/api/v1/locks/cleanup")
    async def cleanup_expired_locks():
        """Limpiar locks expirados."""
        expired_count = await distributed_lock_manager.cleanup_expired_locks()
        return {"success": True, "expired_count": expired_count}
    
    @app.get("/api/v1/locks/summary")
    async def get_lock_manager_summary():
        """Obtener resumen del gestor."""
        return distributed_lock_manager.get_distributed_lock_manager_summary()
    
    # Data Pipeline Manager
    @app.post("/api/v1/pipelines/create")
    async def create_data_pipeline(request: Dict[str, Any]):
        """Crear pipeline de datos."""
        pipeline_id = data_pipeline_manager.create_pipeline(
            pipeline_id=request.get("pipeline_id", f"pipeline_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            description=request.get("description", ""),
            metadata=request.get("metadata"),
        )
        return {"success": True, "pipeline_id": pipeline_id}
    
    @app.post("/api/v1/pipelines/{pipeline_id}/add-stage")
    async def add_pipeline_stage(pipeline_id: str, request: Dict[str, Any]):
        """Agregar stage a pipeline."""
        # Nota: En producción, processor necesitaría ser deserializado
        return {
            "success": True,
            "message": "Stage registration endpoint - implement processor function registration"
        }
    
    @app.post("/api/v1/pipelines/{pipeline_id}/execute")
    async def execute_data_pipeline(pipeline_id: str, request: Dict[str, Any]):
        """Ejecutar pipeline."""
        execution_id = await data_pipeline_manager.execute_pipeline(
            pipeline_id=pipeline_id,
            input_data=request.get("input_data"),
            execution_id=request.get("execution_id"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "execution_id": execution_id}
    
    @app.get("/api/v1/pipelines/{pipeline_id}")
    async def get_data_pipeline(pipeline_id: str):
        """Obtener información de pipeline."""
        pipeline = data_pipeline_manager.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        return pipeline
    
    @app.get("/api/v1/pipelines/executions/{execution_id}")
    async def get_pipeline_execution(execution_id: str):
        """Obtener información de ejecución."""
        execution = data_pipeline_manager.get_execution(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        return execution
    
    @app.post("/api/v1/pipelines/executions/{execution_id}/cancel")
    async def cancel_pipeline_execution(execution_id: str):
        """Cancelar ejecución."""
        success = await data_pipeline_manager.cancel_execution(execution_id)
        if not success:
            raise HTTPException(status_code=404, detail="Execution not found or cannot be cancelled")
        return {"success": True}
    
    @app.get("/api/v1/pipelines/executions/history")
    async def get_pipeline_execution_history(pipeline_id: Optional[str] = None, limit: int = 100):
        """Obtener historial de ejecuciones."""
        history = data_pipeline_manager.get_pipeline_execution_history(pipeline_id, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/pipelines/summary")
    async def get_pipeline_manager_summary():
        """Obtener resumen del gestor."""
        return data_pipeline_manager.get_data_pipeline_manager_summary()
    
    # Event Scheduler
    @app.post("/api/v1/scheduler/schedule")
    async def schedule_event(request: Dict[str, Any]):
        """Programar evento."""
        schedule_type_str = request.get("schedule_type", "interval")
        schedule_type = ScheduleType(schedule_type_str)
        
        # Nota: En producción, handler necesitaría ser deserializado
        return {
            "success": True,
            "message": "Event scheduling endpoint - implement handler function registration"
        }
    
    @app.post("/api/v1/scheduler/{event_id}/pause")
    async def pause_scheduled_event(event_id: str):
        """Pausar evento."""
        success = await event_scheduler.pause_event(event_id)
        if not success:
            raise HTTPException(status_code=404, detail="Event not found")
        return {"success": True}
    
    @app.post("/api/v1/scheduler/{event_id}/resume")
    async def resume_scheduled_event(event_id: str):
        """Reanudar evento."""
        success = await event_scheduler.resume_event(event_id)
        if not success:
            raise HTTPException(status_code=404, detail="Event not found")
        return {"success": True}
    
    @app.post("/api/v1/scheduler/{event_id}/cancel")
    async def cancel_scheduled_event(event_id: str):
        """Cancelar evento."""
        success = await event_scheduler.cancel_event(event_id)
        if not success:
            raise HTTPException(status_code=404, detail="Event not found")
        return {"success": True}
    
    @app.get("/api/v1/scheduler/{event_id}")
    async def get_scheduled_event(event_id: str):
        """Obtener información de evento."""
        event = event_scheduler.get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        return event
    
    @app.get("/api/v1/scheduler/history")
    async def get_event_run_history(event_id: Optional[str] = None, limit: int = 100):
        """Obtener historial de ejecuciones."""
        history = event_scheduler.get_event_run_history(event_id, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/scheduler/summary")
    async def get_event_scheduler_summary():
        """Obtener resumen del scheduler."""
        return event_scheduler.get_event_scheduler_summary()
    
    # Graceful Degradation Manager
    @app.post("/api/v1/degradation/register-service")
    async def register_service_degradation(request: Dict[str, Any]):
        """Registrar servicio para degradación."""
        service_id = graceful_degradation_manager.register_service(
            service_id=request.get("service_id", ""),
            initial_state=ServiceState(request.get("initial_state", "healthy")),
            metadata=request.get("metadata"),
        )
        return {"success": True, "service_id": service_id}
    
    @app.post("/api/v1/degradation/register-fallback")
    async def register_fallback_strategy(request: Dict[str, Any]):
        """Registrar estrategia de fallback."""
        # Nota: En producción, handler necesitaría ser deserializado
        return {
            "success": True,
            "message": "Fallback registration endpoint - implement handler function registration"
        }
    
    @app.post("/api/v1/degradation/add-rule")
    async def add_degradation_rule(request: Dict[str, Any]):
        """Agregar regla de degradación."""
        degradation_level_str = request.get("degradation_level", "degraded")
        degradation_level = DegradationLevel(degradation_level_str)
        
        rule_id = graceful_degradation_manager.add_degradation_rule(
            rule_id=request.get("rule_id", f"rule_{datetime.now().timestamp()}"),
            service_id=request.get("service_id", ""),
            metric_name=request.get("metric_name", ""),
            threshold=request.get("threshold", 0.0),
            degradation_level=degradation_level,
            metadata=request.get("metadata"),
        )
        return {"success": True, "rule_id": rule_id}
    
    @app.post("/api/v1/degradation/record-metric")
    async def record_degradation_metric(request: Dict[str, Any]):
        """Registrar métrica para degradación."""
        await graceful_degradation_manager.record_metric(
            service_id=request.get("service_id", ""),
            metric_name=request.get("metric_name", ""),
            value=request.get("value", 0.0),
        )
        return {"success": True}
    
    @app.post("/api/v1/degradation/record-call")
    async def record_service_call(request: Dict[str, Any]):
        """Registrar llamada a servicio."""
        await graceful_degradation_manager.record_service_call(
            service_id=request.get("service_id", ""),
            success=request.get("success", True),
            response_time=request.get("response_time", 0.0),
        )
        return {"success": True}
    
    @app.get("/api/v1/degradation/service/{service_id}")
    async def get_service_health_degradation(service_id: str):
        """Obtener salud de servicio."""
        health = graceful_degradation_manager.get_service_health(service_id)
        if not health:
            raise HTTPException(status_code=404, detail="Service not found")
        return health
    
    @app.get("/api/v1/degradation/status")
    async def get_degradation_status():
        """Obtener estado de degradación."""
        return graceful_degradation_manager.get_degradation_status()
    
    @app.get("/api/v1/degradation/history")
    async def get_degradation_history(limit: int = 100):
        """Obtener historial de degradación."""
        history = graceful_degradation_manager.get_degradation_history(limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/degradation/summary")
    async def get_degradation_manager_summary():
        """Obtener resumen del gestor."""
        return graceful_degradation_manager.get_graceful_degradation_manager_summary()
    
    # Cache Warmer
    @app.post("/api/v1/cache-warmer/register-rule")
    async def register_warming_rule(request: Dict[str, Any]):
        """Registrar regla de precalentamiento."""
        # Nota: En producción, fetch_function necesitaría ser deserializado
        return {
            "success": True,
            "message": "Warming rule registration endpoint - implement fetch function registration"
        }
    
    @app.post("/api/v1/cache-warmer/record-access")
    async def record_cache_access(request: Dict[str, Any]):
        """Registrar acceso a cache."""
        cache_warmer.record_access(
            key=request.get("key", ""),
        )
        return {"success": True}
    
    @app.post("/api/v1/cache-warmer/start")
    async def start_cache_warming():
        """Iniciar precalentamiento."""
        cache_warmer.start_warming()
        return {"success": True, "message": "Cache warming started"}
    
    @app.post("/api/v1/cache-warmer/stop")
    async def stop_cache_warming():
        """Detener precalentamiento."""
        cache_warmer.stop_warming()
        return {"success": True, "message": "Cache warming stopped"}
    
    @app.get("/api/v1/cache-warmer/patterns")
    async def get_cache_access_patterns(limit: int = 50):
        """Obtener patrones de acceso."""
        patterns = cache_warmer.get_access_patterns(limit)
        return {"patterns": patterns, "count": len(patterns)}
    
    @app.get("/api/v1/cache-warmer/statistics")
    async def get_warming_statistics():
        """Obtener estadísticas de precalentamiento."""
        return cache_warmer.get_warming_statistics()
    
    @app.get("/api/v1/cache-warmer/summary")
    async def get_cache_warmer_summary():
        """Obtener resumen del warmer."""
        return cache_warmer.get_cache_warmer_summary()
    
    # Load Shedder
    @app.post("/api/v1/load-shedder/record-metric")
    async def record_load_shedder_metric(request: Dict[str, Any]):
        """Registrar métrica de carga."""
        load_shedder.record_load_metric(
            cpu_usage=request.get("cpu_usage"),
            memory_usage=request.get("memory_usage"),
            request_rate=request.get("request_rate"),
            response_time=request.get("response_time"),
            queue_size=request.get("queue_size"),
            error_rate=request.get("error_rate"),
        )
        return {"success": True}
    
    @app.post("/api/v1/load-shedder/add-rule")
    async def add_shedding_rule(request: Dict[str, Any]):
        """Agregar regla de descarga."""
        strategy_str = request.get("strategy", "priority")
        strategy = SheddingStrategy(strategy_str)
        
        rule_id = load_shedder.add_shedding_rule(
            rule_id=request.get("rule_id", f"rule_{datetime.now().timestamp()}"),
            metric_name=request.get("metric_name", ""),
            threshold=request.get("threshold", 0.0),
            shedding_percentage=request.get("shedding_percentage", 0.1),
            strategy=strategy,
            metadata=request.get("metadata"),
        )
        return {"success": True, "rule_id": rule_id}
    
    @app.post("/api/v1/load-shedder/check-request")
    async def check_should_accept_request(request: Dict[str, Any]):
        """Verificar si se debe aceptar petición."""
        priority_str = request.get("priority", "normal")
        priority = RequestPriority(priority_str)
        
        allowed = await load_shedder.should_accept_request(
            request_id=request.get("request_id", ""),
            priority=priority,
        )
        return {"allowed": allowed}
    
    @app.get("/api/v1/load-shedder/statistics")
    async def get_load_shedder_statistics(window_minutes: int = 5):
        """Obtener estadísticas de carga."""
        stats = load_shedder.get_load_statistics(window_minutes)
        return stats
    
    @app.get("/api/v1/load-shedder/history")
    async def get_shedding_history(limit: int = 100):
        """Obtener historial de descarga."""
        history = load_shedder.get_shedding_history(limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/load-shedder/summary")
    async def get_load_shedder_summary():
        """Obtener resumen del shedder."""
        return load_shedder.get_load_shedder_summary()
    
    # Conflict Resolver
    @app.post("/api/v1/conflicts/register")
    async def register_conflict(request: Dict[str, Any]):
        """Registrar conflicto."""
        conflict_type_str = request.get("conflict_type", "data_update")
        conflict_type = ConflictType(conflict_type_str)
        
        conflict_id = conflict_resolver.register_conflict(
            conflict_id=request.get("conflict_id", f"conflict_{datetime.now().timestamp()}"),
            conflict_type=conflict_type,
            resource_id=request.get("resource_id", ""),
            conflicting_values=request.get("conflicting_values", {}),
            metadata=request.get("metadata"),
        )
        return {"success": True, "conflict_id": conflict_id}
    
    @app.post("/api/v1/conflicts/{conflict_id}/resolve")
    async def resolve_conflict(conflict_id: str, request: Dict[str, Any]):
        """Resolver conflicto."""
        strategy_str = request.get("strategy", "last_write_wins")
        strategy = ResolutionStrategy(strategy_str)
        
        success = await conflict_resolver.apply_resolution(
            conflict_id=conflict_id,
            resolved_value=request.get("resolved_value"),
            strategy=strategy,
            resolver_id=request.get("resolver_id", "user"),
        )
        if not success:
            raise HTTPException(status_code=404, detail="Conflict not found or already resolved")
        return {"success": True}
    
    @app.post("/api/v1/conflicts/register-rule")
    async def register_resolution_rule(request: Dict[str, Any]):
        """Registrar regla de resolución."""
        conflict_type_str = request.get("conflict_type", "data_update")
        conflict_type = ConflictType(conflict_type_str)
        
        strategy_str = request.get("strategy", "last_write_wins")
        strategy = ResolutionStrategy(strategy_str)
        
        conflict_resolver.register_resolution_rule(conflict_type, strategy)
        return {"success": True}
    
    @app.get("/api/v1/conflicts/{conflict_id}")
    async def get_conflict_info(conflict_id: str):
        """Obtener información de conflicto."""
        conflict = conflict_resolver.get_conflict(conflict_id)
        if not conflict:
            raise HTTPException(status_code=404, detail="Conflict not found")
        return conflict
    
    @app.get("/api/v1/conflicts/pending")
    async def get_pending_conflicts(limit: int = 100):
        """Obtener conflictos pendientes."""
        conflicts = conflict_resolver.get_pending_conflicts(limit)
        return {"conflicts": conflicts, "count": len(conflicts)}
    
    @app.get("/api/v1/conflicts/history")
    async def get_resolution_history(limit: int = 100):
        """Obtener historial de resoluciones."""
        history = conflict_resolver.get_resolution_history(limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/conflicts/summary")
    async def get_conflict_resolver_summary():
        """Obtener resumen del resolvedor."""
        return conflict_resolver.get_conflict_resolver_summary()
    
    # State Machine Manager
    @app.post("/api/v1/state-machines/create")
    async def create_state_machine(request: Dict[str, Any]):
        """Crear máquina de estados."""
        machine_id = state_machine_manager.create_state_machine(
            machine_id=request.get("machine_id", f"machine_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            initial_state=request.get("initial_state", ""),
            states=request.get("states", []),
            metadata=request.get("metadata"),
        )
        return {"success": True, "machine_id": machine_id}
    
    @app.post("/api/v1/state-machines/{machine_id}/add-transition")
    async def add_state_transition(machine_id: str, request: Dict[str, Any]):
        """Agregar transición."""
        # Nota: En producción, condition y on_transition necesitarían ser deserializados
        return {
            "success": True,
            "message": "Transition registration endpoint - implement function registration"
        }
    
    @app.post("/api/v1/state-machines/{machine_id}/transition")
    async def transition_state_machine(machine_id: str, request: Dict[str, Any]):
        """Realizar transición."""
        success = await state_machine_manager.transition(
            machine_id=machine_id,
            to_state=request.get("to_state", ""),
            transition_id=request.get("transition_id"),
            metadata=request.get("metadata"),
        )
        if not success:
            raise HTTPException(status_code=400, detail="Transition not allowed")
        return {"success": True}
    
    @app.get("/api/v1/state-machines/{machine_id}")
    async def get_state_machine(machine_id: str):
        """Obtener información de máquina de estados."""
        machine = state_machine_manager.get_state_machine(machine_id)
        if not machine:
            raise HTTPException(status_code=404, detail="State machine not found")
        return machine
    
    @app.get("/api/v1/state-machines/history")
    async def get_state_machine_history(machine_id: Optional[str] = None, limit: int = 100):
        """Obtener historial de estados."""
        history = state_machine_manager.get_state_history(machine_id, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/state-machines/summary")
    async def get_state_machine_manager_summary():
        """Obtener resumen del gestor."""
        return state_machine_manager.get_state_machine_manager_summary()
    
    # Workflow Engine V2
    @app.post("/api/v1/workflows-v2/create")
    async def create_workflow_v2(request: Dict[str, Any]):
        """Crear workflow."""
        workflow_id = workflow_engine_v2.create_workflow(
            workflow_id=request.get("workflow_id", f"workflow_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            description=request.get("description", ""),
            metadata=request.get("metadata"),
        )
        return {"success": True, "workflow_id": workflow_id}
    
    @app.post("/api/v1/workflows-v2/{workflow_id}/add-step")
    async def add_workflow_step_v2(workflow_id: str, request: Dict[str, Any]):
        """Agregar step a workflow."""
        # Nota: En producción, handler necesitaría ser deserializado
        return {
            "success": True,
            "message": "Step registration endpoint - implement handler function registration"
        }
    
    @app.post("/api/v1/workflows-v2/{workflow_id}/execute")
    async def execute_workflow_v2(workflow_id: str, request: Dict[str, Any]):
        """Ejecutar workflow."""
        execution_id = await workflow_engine_v2.execute_workflow(
            workflow_id=workflow_id,
            context=request.get("context", {}),
            execution_id=request.get("execution_id"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "execution_id": execution_id}
    
    @app.get("/api/v1/workflows-v2/{workflow_id}")
    async def get_workflow_v2(workflow_id: str):
        """Obtener información de workflow."""
        workflow = workflow_engine_v2.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return workflow
    
    @app.get("/api/v1/workflows-v2/executions/{execution_id}")
    async def get_workflow_execution_v2(execution_id: str):
        """Obtener información de ejecución."""
        execution = workflow_engine_v2.get_execution(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        return execution
    
    @app.post("/api/v1/workflows-v2/executions/{execution_id}/cancel")
    async def cancel_workflow_execution_v2(execution_id: str):
        """Cancelar ejecución."""
        success = await workflow_engine_v2.cancel_execution(execution_id)
        if not success:
            raise HTTPException(status_code=404, detail="Execution not found or cannot be cancelled")
        return {"success": True}
    
    @app.get("/api/v1/workflows-v2/summary")
    async def get_workflow_engine_v2_summary():
        """Obtener resumen del motor."""
        return workflow_engine_v2.get_workflow_engine_v2_summary()
    
    # Event Bus
    @app.post("/api/v1/events/publish")
    async def publish_event(request: Dict[str, Any]):
        """Publicar evento."""
        priority_str = request.get("priority", "normal")
        priority = EventPriority(priority_str)
        
        event_id = await event_bus.publish(
            event_type=request.get("event_type", ""),
            source=request.get("source", "api"),
            payload=request.get("payload", {}),
            priority=priority,
            metadata=request.get("metadata"),
        )
        return {"success": True, "event_id": event_id}
    
    @app.post("/api/v1/events/subscribe")
    async def subscribe_to_events(request: Dict[str, Any]):
        """Suscribirse a eventos."""
        # Nota: En producción, handler y filter_func necesitarían ser deserializados
        return {
            "success": True,
            "message": "Subscription endpoint - implement handler function registration"
        }
    
    @app.post("/api/v1/events/unsubscribe")
    async def unsubscribe_from_events(request: Dict[str, Any]):
        """Desuscribirse de eventos."""
        success = await event_bus.unsubscribe(request.get("subscription_id", ""))
        if not success:
            raise HTTPException(status_code=404, detail="Subscription not found")
        return {"success": True}
    
    @app.get("/api/v1/events/history")
    async def get_event_history(event_type: Optional[str] = None, source: Optional[str] = None, limit: int = 100):
        """Obtener historial de eventos."""
        history = event_bus.get_event_history(event_type, source, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/events/subscriptions")
    async def get_event_subscriptions(event_type: Optional[str] = None):
        """Obtener suscripciones."""
        subscriptions = event_bus.get_subscriptions(event_type)
        return {"subscriptions": subscriptions, "count": len(subscriptions)}
    
    @app.get("/api/v1/events/summary")
    async def get_event_bus_summary():
        """Obtener resumen del bus."""
        return event_bus.get_event_bus_summary()
    
    # Feature Toggle Manager
    @app.post("/api/v1/feature-toggles/create")
    async def create_feature_toggle(request: Dict[str, Any]):
        """Crear feature toggle."""
        toggle_type_str = request.get("toggle_type", "boolean")
        toggle_type = ToggleType(toggle_type_str)
        
        toggle_id = feature_toggle_manager.create_toggle(
            toggle_id=request.get("toggle_id", f"toggle_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            toggle_type=toggle_type,
            enabled=request.get("enabled", False),
            percentage=request.get("percentage", 0.0),
            target_users=request.get("target_users"),
            target_attributes=request.get("target_attributes"),
            variants=request.get("variants"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "toggle_id": toggle_id}
    
    @app.get("/api/v1/feature-toggles/{toggle_id}/check")
    async def check_feature_toggle(toggle_id: str, user_id: Optional[str] = None):
        """Verificar si toggle está habilitado."""
        user_attributes = {}  # En producción, obtener de request
        enabled = feature_toggle_manager.is_enabled(toggle_id, user_id, user_attributes)
        
        # Registrar evaluación
        variant = feature_toggle_manager.get_variant(toggle_id, user_id)
        feature_toggle_manager.record_evaluation(toggle_id, user_id, enabled, variant)
        
        return {"enabled": enabled, "variant": variant}
    
    @app.post("/api/v1/feature-toggles/{toggle_id}/update")
    async def update_feature_toggle(toggle_id: str, request: Dict[str, Any]):
        """Actualizar feature toggle."""
        status = None
        if request.get("status"):
            status = ToggleStatus(request.get("status"))
        
        success = feature_toggle_manager.update_toggle(
            toggle_id=toggle_id,
            enabled=request.get("enabled"),
            percentage=request.get("percentage"),
            target_users=request.get("target_users"),
            target_attributes=request.get("target_attributes"),
            variants=request.get("variants"),
            status=status,
        )
        if not success:
            raise HTTPException(status_code=404, detail="Toggle not found")
        return {"success": True}
    
    @app.get("/api/v1/feature-toggles/{toggle_id}")
    async def get_feature_toggle(toggle_id: str):
        """Obtener información de toggle."""
        toggle = feature_toggle_manager.get_toggle(toggle_id)
        if not toggle:
            raise HTTPException(status_code=404, detail="Toggle not found")
        return toggle
    
    @app.get("/api/v1/feature-toggles/{toggle_id}/statistics")
    async def get_feature_toggle_statistics(toggle_id: str):
        """Obtener estadísticas de evaluación."""
        stats = feature_toggle_manager.get_evaluation_statistics(toggle_id)
        return stats
    
    @app.get("/api/v1/feature-toggles/summary")
    async def get_feature_toggle_manager_summary():
        """Obtener resumen del gestor."""
        return feature_toggle_manager.get_feature_toggle_manager_summary()
    
    # Rate Limiter V2
    @app.post("/api/v1/rate-limiter-v2/add-rule")
    async def add_rate_limit_rule_v2(request: Dict[str, Any]):
        """Agregar regla de rate limiting."""
        algorithm_str = request.get("algorithm", "fixed_window")
        algorithm = RateLimitAlgorithm(algorithm_str)
        
        rule_id = rate_limiter_v2.add_rule(
            rule_id=request.get("rule_id", f"rule_{datetime.now().timestamp()}"),
            identifier=request.get("identifier", ""),
            algorithm=algorithm,
            limit=request.get("limit", 100),
            window_seconds=request.get("window_seconds", 60.0),
            tokens=request.get("tokens"),
            refill_rate=request.get("refill_rate"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "rule_id": rule_id}
    
    @app.post("/api/v1/rate-limiter-v2/check")
    async def check_rate_limit_v2(request: Dict[str, Any]):
        """Verificar rate limit."""
        allowed, info = await rate_limiter_v2.check_rate_limit(
            identifier=request.get("identifier", ""),
            rule_id=request.get("rule_id"),
        )
        return info
    
    @app.get("/api/v1/rate-limiter-v2/status")
    async def get_rate_limit_status_v2(identifier: str, rule_id: Optional[str] = None):
        """Obtener estado de rate limiting."""
        status = rate_limiter_v2.get_rate_limit_status(identifier, rule_id)
        return status
    
    @app.get("/api/v1/rate-limiter-v2/history")
    async def get_rate_limit_block_history(limit: int = 100):
        """Obtener historial de bloqueos."""
        history = rate_limiter_v2.get_block_history(limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/rate-limiter-v2/summary")
    async def get_rate_limiter_v2_summary():
        """Obtener resumen del limitador."""
        return rate_limiter_v2.get_rate_limiter_v2_summary()
    
    # Circuit Breaker V2
    @app.post("/api/v1/circuit-breakers-v2/create")
    async def create_circuit_breaker_v2(request: Dict[str, Any]):
        """Crear circuit breaker."""
        strategy_str = request.get("failure_strategy", "count_based")
        strategy = FailureStrategy(strategy_str)
        
        circuit_id = circuit_breaker_v2.create_circuit(
            circuit_id=request.get("circuit_id", f"circuit_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            failure_threshold=request.get("failure_threshold", 5),
            failure_percentage=request.get("failure_percentage", 0.5),
            timeout_seconds=request.get("timeout_seconds", 60.0),
            half_open_max_calls=request.get("half_open_max_calls", 3),
            failure_strategy=strategy,
            metadata=request.get("metadata"),
        )
        return {"success": True, "circuit_id": circuit_id}
    
    @app.get("/api/v1/circuit-breakers-v2/{circuit_id}")
    async def get_circuit_breaker_v2(circuit_id: str):
        """Obtener información de circuit breaker."""
        circuit = circuit_breaker_v2.get_circuit(circuit_id)
        if not circuit:
            raise HTTPException(status_code=404, detail="Circuit breaker not found")
        return circuit
    
    @app.post("/api/v1/circuit-breakers-v2/{circuit_id}/reset")
    async def reset_circuit_breaker_v2(circuit_id: str):
        """Resetear circuit breaker."""
        success = await circuit_breaker_v2.reset_circuit(circuit_id)
        if not success:
            raise HTTPException(status_code=404, detail="Circuit breaker not found")
        return {"success": True}
    
    @app.get("/api/v1/circuit-breakers-v2/summary")
    async def get_circuit_breaker_v2_summary():
        """Obtener resumen del gestor."""
        return circuit_breaker_v2.get_circuit_breaker_v2_summary()
    
    # Adaptive Optimizer
    @app.post("/api/v1/optimizer/register-parameter")
    async def register_optimization_parameter(request: Dict[str, Any]):
        """Registrar parámetro a optimizar."""
        parameter_id = adaptive_optimizer.register_parameter(
            parameter_id=request.get("parameter_id", f"param_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            initial_value=request.get("initial_value", 1.0),
            min_value=request.get("min_value", 0.0),
            max_value=request.get("max_value", 10.0),
            step=request.get("step", 0.1),
            metadata=request.get("metadata"),
        )
        return {"success": True, "parameter_id": parameter_id}
    
    @app.post("/api/v1/optimizer/add-goal")
    async def add_optimization_goal(request: Dict[str, Any]):
        """Agregar objetivo de optimización."""
        target_str = request.get("target", "latency")
        target = OptimizationTarget(target_str)
        
        goal_id = adaptive_optimizer.add_goal(
            goal_id=request.get("goal_id", f"goal_{datetime.now().timestamp()}"),
            target=target,
            target_value=request.get("target_value"),
            maximize=request.get("maximize", True),
            weight=request.get("weight", 1.0),
            metadata=request.get("metadata"),
        )
        return {"success": True, "goal_id": goal_id}
    
    @app.post("/api/v1/optimizer/record-metric")
    async def record_optimization_metric(request: Dict[str, Any]):
        """Registrar métrica para optimización."""
        adaptive_optimizer.record_metric(
            metric_name=request.get("metric_name", ""),
            value=request.get("value", 0.0),
        )
        return {"success": True}
    
    @app.post("/api/v1/optimizer/start")
    async def start_adaptive_optimization():
        """Iniciar optimización."""
        adaptive_optimizer.start_optimization()
        return {"success": True, "message": "Optimization started"}
    
    @app.post("/api/v1/optimizer/stop")
    async def stop_adaptive_optimization():
        """Detener optimización."""
        adaptive_optimizer.stop_optimization()
        return {"success": True, "message": "Optimization stopped"}
    
    @app.get("/api/v1/optimizer/parameter/{parameter_id}")
    async def get_optimization_parameter(parameter_id: str):
        """Obtener información de parámetro."""
        param = adaptive_optimizer.get_parameter(parameter_id)
        if not param:
            raise HTTPException(status_code=404, detail="Parameter not found")
        return param
    
    @app.get("/api/v1/optimizer/results")
    async def get_optimization_results(limit: int = 100):
        """Obtener resultados de optimización."""
        results = adaptive_optimizer.get_optimization_results(limit)
        return {"results": results, "count": len(results)}
    
    @app.get("/api/v1/optimizer/summary")
    async def get_adaptive_optimizer_summary():
        """Obtener resumen del optimizador."""
        return adaptive_optimizer.get_adaptive_optimizer_summary()
    
    # Health Checker V2
    @app.post("/api/v1/health-v2/register-check")
    async def register_health_check_v2(request: Dict[str, Any]):
        """Registrar health check."""
        # Nota: En producción, handler necesitaría ser deserializado
        return {
            "success": True,
            "message": "Health check registration endpoint - implement handler function registration"
        }
    
    @app.post("/api/v1/health-v2/{check_id}/run")
    async def run_health_check_v2(check_id: str):
        """Ejecutar health check manualmente."""
        result = await health_checker_v2.run_check(check_id)
        if not result:
            raise HTTPException(status_code=404, detail="Health check not found")
        return result
    
    @app.get("/api/v1/health-v2/overall")
    async def get_overall_health_v2():
        """Obtener salud general."""
        return health_checker_v2.get_overall_health()
    
    @app.get("/api/v1/health-v2/{check_id}/history")
    async def get_health_check_history_v2(check_id: str, limit: int = 100):
        """Obtener historial de checks."""
        history = health_checker_v2.get_check_history(check_id, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/health-v2/summary")
    async def get_health_checker_v2_summary():
        """Obtener resumen del verificador."""
        return health_checker_v2.get_health_checker_v2_summary()
    
    # Auto Scaler
    @app.post("/api/v1/auto-scaler/add-rule")
    async def add_scaling_rule(request: Dict[str, Any]):
        """Agregar regla de escalado."""
        rule_id = auto_scaler.add_rule(
            rule_id=request.get("rule_id", f"rule_{datetime.now().timestamp()}"),
            metric_name=request.get("metric_name", ""),
            threshold_up=request.get("threshold_up", 80.0),
            threshold_down=request.get("threshold_down", 20.0),
            min_instances=request.get("min_instances", 1),
            max_instances=request.get("max_instances", 10),
            scale_up_step=request.get("scale_up_step", 1),
            scale_down_step=request.get("scale_down_step", 1),
            cooldown_seconds=request.get("cooldown_seconds", 300.0),
            metadata=request.get("metadata"),
        )
        return {"success": True, "rule_id": rule_id}
    
    @app.post("/api/v1/auto-scaler/record-metric")
    async def record_scaling_metric(request: Dict[str, Any]):
        """Registrar métrica para escalado."""
        auto_scaler.record_metric(
            metric_name=request.get("metric_name", ""),
            value=request.get("value", 0.0),
        )
        return {"success": True}
    
    @app.post("/api/v1/auto-scaler/set-instances")
    async def set_scaler_instances(request: Dict[str, Any]):
        """Establecer número de instancias manualmente."""
        auto_scaler.set_instances(request.get("instances", 1))
        return {"success": True}
    
    @app.post("/api/v1/auto-scaler/start")
    async def start_auto_scaling():
        """Iniciar auto-escalado."""
        auto_scaler.start_scaling()
        return {"success": True, "message": "Auto-scaling started"}
    
    @app.post("/api/v1/auto-scaler/stop")
    async def stop_auto_scaling():
        """Detener auto-escalado."""
        auto_scaler.stop_scaling()
        return {"success": True, "message": "Auto-scaling stopped"}
    
    @app.get("/api/v1/auto-scaler/status")
    async def get_scaling_status():
        """Obtener estado de escalado."""
        return auto_scaler.get_scaling_status()
    
    @app.get("/api/v1/auto-scaler/history")
    async def get_scaling_history(limit: int = 100):
        """Obtener historial de escalado."""
        history = auto_scaler.get_scaling_history(limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/auto-scaler/summary")
    async def get_auto_scaler_summary():
        """Obtener resumen del escalador."""
        return auto_scaler.get_auto_scaler_summary()
    
    # Batch Processor
    @app.post("/api/v1/batch/add-item")
    async def add_batch_item(request: Dict[str, Any]):
        """Agregar item a batch."""
        item_id = batch_processor.add_item(
            queue_id=request.get("queue_id", "default"),
            item_id=request.get("item_id", f"item_{datetime.now().timestamp()}"),
            data=request.get("data"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "item_id": item_id}
    
    @app.post("/api/v1/batch/register-processor")
    async def register_batch_processor(request: Dict[str, Any]):
        """Registrar procesador para cola."""
        # Nota: En producción, processor necesitaría ser deserializado
        return {
            "success": True,
            "message": "Processor registration endpoint - implement processor function registration"
        }
    
    @app.get("/api/v1/batch/queue-status")
    async def get_batch_queue_status(queue_id: Optional[str] = None):
        """Obtener estado de cola(s)."""
        status = batch_processor.get_queue_status(queue_id)
        return status
    
    @app.get("/api/v1/batch/history")
    async def get_batch_history(queue_id: Optional[str] = None, limit: int = 100):
        """Obtener historial de batches."""
        history = batch_processor.get_batch_history(queue_id, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/batch/summary")
    async def get_batch_processor_summary():
        """Obtener resumen del procesador."""
        return batch_processor.get_batch_processor_summary()
    
    # Performance Monitor
    @app.post("/api/v1/performance/record-metric")
    async def record_performance_metric(request: Dict[str, Any]):
        """Registrar métrica de rendimiento."""
        metric_type_str = request.get("metric_type", "gauge")
        metric_type = MetricType(metric_type_str)
        
        performance_monitor.record_metric(
            metric_name=request.get("metric_name", ""),
            value=request.get("value", 0.0),
            metric_type=metric_type,
            labels=request.get("labels"),
        )
        return {"success": True}
    
    @app.post("/api/v1/performance/record-latency")
    async def record_performance_latency(request: Dict[str, Any]):
        """Registrar latencia."""
        performance_monitor.record_latency(
            operation_name=request.get("operation_name", ""),
            latency_seconds=request.get("latency_seconds", 0.0),
        )
        return {"success": True}
    
    @app.post("/api/v1/performance/create-snapshot")
    async def create_performance_snapshot():
        """Crear snapshot de rendimiento."""
        snapshot_id = performance_monitor.create_snapshot()
        return {"success": True, "snapshot_id": snapshot_id}
    
    @app.get("/api/v1/performance/summary")
    async def get_performance_summary(window_minutes: int = 5):
        """Obtener resumen de rendimiento."""
        summary = performance_monitor.get_performance_summary(window_minutes)
        return summary
    
    @app.get("/api/v1/performance/metric/{metric_name}")
    async def get_metric_history(metric_name: str, limit: int = 100):
        """Obtener historial de métrica."""
        history = performance_monitor.get_metric_history(metric_name, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/performance/monitor-summary")
    async def get_performance_monitor_summary():
        """Obtener resumen del monitor."""
        return performance_monitor.get_performance_monitor_summary()
    
    # Queue Manager
    @app.post("/api/v1/queues/create")
    async def create_queue(request: Dict[str, Any]):
        """Crear cola."""
        queue_name = queue_manager.create_queue(
            queue_name=request.get("queue_name", ""),
            max_size=request.get("max_size", 10000),
            visibility_timeout=request.get("visibility_timeout", 30.0),
            message_retention=request.get("message_retention", 86400.0),
            dead_letter_queue=request.get("dead_letter_queue"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "queue_name": queue_name}
    
    @app.post("/api/v1/queues/{queue_name}/enqueue")
    async def enqueue_message(queue_name: str, request: Dict[str, Any]):
        """Encolar mensaje."""
        priority_str = request.get("priority", "normal")
        priority = QueuePriority[priority_str.upper()]
        
        message_id = queue_manager.enqueue(
            queue_name=queue_name,
            payload=request.get("payload"),
            priority=priority,
            message_id=request.get("message_id"),
            max_attempts=request.get("max_attempts", 3),
            metadata=request.get("metadata"),
        )
        return {"success": True, "message_id": message_id}
    
    @app.post("/api/v1/queues/{queue_name}/dequeue")
    async def dequeue_message(queue_name: str, timeout: Optional[float] = None):
        """Desencolar mensaje."""
        message = await queue_manager.dequeue(queue_name, timeout)
        if not message:
            raise HTTPException(status_code=404, detail="No messages available")
        
        return {
            "message_id": message.message_id,
            "payload": message.payload,
            "priority": message.priority.value,
            "attempts": message.attempts,
            "created_at": message.created_at.isoformat(),
        }
    
    @app.post("/api/v1/queues/messages/{message_id}/ack")
    async def acknowledge_message(message_id: str):
        """Confirmar mensaje."""
        await queue_manager.acknowledge(message_id)
        return {"success": True}
    
    @app.post("/api/v1/queues/messages/{message_id}/nack")
    async def nack_message(message_id: str, requeue: bool = True):
        """Negar mensaje."""
        await queue_manager.nack(message_id, requeue)
        return {"success": True}
    
    @app.get("/api/v1/queues/{queue_name}/status")
    async def get_queue_status(queue_name: str):
        """Obtener estado de cola."""
        status = queue_manager.get_queue_status(queue_name)
        if not status:
            raise HTTPException(status_code=404, detail="Queue not found")
        return status
    
    @app.get("/api/v1/queues/summary")
    async def get_queue_manager_summary():
        """Obtener resumen del gestor."""
        return queue_manager.get_queue_manager_summary()
    
    # Connection Manager
    @app.post("/api/v1/connections/register")
    async def register_connection(request: Dict[str, Any]):
        """Registrar conexión."""
        connection_id = connection_manager.register_connection(
            connection_id=request.get("connection_id", f"conn_{datetime.now().timestamp()}"),
            connection_type=request.get("connection_type", ""),
            metadata=request.get("metadata"),
        )
        return {"success": True, "connection_id": connection_id}
    
    @app.post("/api/v1/connections/{connection_type}/acquire")
    async def acquire_connection(connection_type: str, timeout: Optional[float] = None):
        """Adquirir conexión."""
        connection_id = await connection_manager.acquire_connection(connection_type, timeout)
        if not connection_id:
            raise HTTPException(status_code=404, detail="No connection available")
        return {"success": True, "connection_id": connection_id}
    
    @app.post("/api/v1/connections/{connection_id}/release")
    async def release_connection(connection_id: str):
        """Liberar conexión."""
        await connection_manager.release_connection(connection_id)
        return {"success": True}
    
    @app.post("/api/v1/connections/{connection_id}/close")
    async def close_connection(connection_id: str):
        """Cerrar conexión."""
        await connection_manager.close_connection(connection_id)
        return {"success": True}
    
    @app.get("/api/v1/connections/{connection_id}")
    async def get_connection_info(connection_id: str):
        """Obtener información de conexión."""
        connection = connection_manager.get_connection(connection_id)
        if not connection:
            raise HTTPException(status_code=404, detail="Connection not found")
        return connection
    
    @app.get("/api/v1/connections/type/{connection_type}")
    async def get_connections_by_type(connection_type: str):
        """Obtener conexiones por tipo."""
        connections = connection_manager.get_connections_by_type(connection_type)
        return {"connections": connections, "count": len(connections)}
    
    @app.get("/api/v1/connections/summary")
    async def get_connection_manager_summary():
        """Obtener resumen del gestor."""
        return connection_manager.get_connection_manager_summary()
    
    # Transaction Manager
    @app.post("/api/v1/transactions/begin")
    async def begin_transaction(request: Dict[str, Any]):
        """Iniciar transacción."""
        transaction_id = transaction_manager.begin_transaction(
            transaction_id=request.get("transaction_id"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "transaction_id": transaction_id}
    
    @app.post("/api/v1/transactions/{transaction_id}/add-operation")
    async def add_transaction_operation(transaction_id: str, request: Dict[str, Any]):
        """Agregar operación a transacción."""
        # Nota: En producción, execute y rollback necesitarían ser deserializados
        return {
            "success": True,
            "message": "Operation registration endpoint - implement function registration"
        }
    
    @app.post("/api/v1/transactions/{transaction_id}/commit")
    async def commit_transaction(transaction_id: str):
        """Commit transacción."""
        success = await transaction_manager.commit(transaction_id)
        if not success:
            raise HTTPException(status_code=400, detail="Transaction commit failed")
        return {"success": True}
    
    @app.post("/api/v1/transactions/{transaction_id}/rollback")
    async def rollback_transaction(transaction_id: str):
        """Rollback transacción."""
        success = await transaction_manager.rollback(transaction_id)
        if not success:
            raise HTTPException(status_code=400, detail="Transaction rollback failed")
        return {"success": True}
    
    @app.get("/api/v1/transactions/{transaction_id}")
    async def get_transaction_info(transaction_id: str):
        """Obtener información de transacción."""
        transaction = transaction_manager.get_transaction(transaction_id)
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        return transaction
    
    @app.get("/api/v1/transactions/history")
    async def get_transaction_history(limit: int = 100):
        """Obtener historial de transacciones."""
        history = transaction_manager.get_transaction_history(limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/transactions/summary")
    async def get_transaction_manager_summary():
        """Obtener resumen del gestor."""
        return transaction_manager.get_transaction_manager_summary()
    
    # Saga Orchestrator
    @app.post("/api/v1/sagas/create")
    async def create_saga(request: Dict[str, Any]):
        """Crear saga."""
        saga_id = saga_orchestrator.create_saga(
            saga_id=request.get("saga_id", f"saga_{datetime.now().timestamp()}"),
            name=request.get("name", ""),
            metadata=request.get("metadata"),
        )
        return {"success": True, "saga_id": saga_id}
    
    @app.post("/api/v1/sagas/{saga_id}/add-step")
    async def add_saga_step(saga_id: str, request: Dict[str, Any]):
        """Agregar step a saga."""
        # Nota: En producción, execute y compensate necesitarían ser deserializados
        return {
            "success": True,
            "message": "Step registration endpoint - implement function registration"
        }
    
    @app.post("/api/v1/sagas/{saga_id}/execute")
    async def execute_saga(saga_id: str, request: Dict[str, Any]):
        """Ejecutar saga."""
        await saga_orchestrator.execute_saga(
            saga_id=saga_id,
            context=request.get("context", {}),
        )
        return {"success": True, "saga_id": saga_id}
    
    @app.get("/api/v1/sagas/{saga_id}")
    async def get_saga_info(saga_id: str):
        """Obtener información de saga."""
        saga = saga_orchestrator.get_saga(saga_id)
        if not saga:
            raise HTTPException(status_code=404, detail="Saga not found")
        return saga
    
    @app.get("/api/v1/sagas/history")
    async def get_saga_history(limit: int = 100):
        """Obtener historial de sagas."""
        history = saga_orchestrator.get_saga_history(limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/sagas/summary")
    async def get_saga_orchestrator_summary():
        """Obtener resumen del orquestador."""
        return saga_orchestrator.get_saga_orchestrator_summary()
    
    # Distributed Coordinator
    @app.post("/api/v1/coordination/register-node")
    async def register_coordination_node(request: Dict[str, Any]):
        """Registrar nodo en coordinación."""
        node_id = distributed_coordinator.register_node(
            node_id=request.get("node_id", f"node_{datetime.now().timestamp()}"),
            address=request.get("address", ""),
            metadata=request.get("metadata"),
        )
        return {"success": True, "node_id": node_id}
    
    @app.post("/api/v1/coordination/propose")
    async def propose_value(request: Dict[str, Any]):
        """Proponer valor para consenso."""
        proposal_id = await distributed_coordinator.propose_value(
            value=request.get("value"),
            proposal_id=request.get("proposal_id"),
            metadata=request.get("metadata"),
        )
        return {"success": True, "proposal_id": proposal_id}
    
    @app.get("/api/v1/coordination/leader")
    async def get_coordination_leader():
        """Obtener información del líder."""
        leader = distributed_coordinator.get_leader()
        if not leader:
            raise HTTPException(status_code=404, detail="No leader elected")
        return leader
    
    @app.get("/api/v1/coordination/status")
    async def get_coordination_status():
        """Obtener estado de coordinación."""
        return distributed_coordinator.get_coordination_status()
    
    @app.get("/api/v1/coordination/summary")
    async def get_distributed_coordinator_summary():
        """Obtener resumen del coordinador."""
        return distributed_coordinator.get_distributed_coordinator_summary()
    
    # Service Mesh
    @app.post("/api/v1/mesh/register-service")
    async def register_mesh_service(request: Dict[str, Any]):
        """Registrar servicio en malla."""
        strategy_str = request.get("load_balancing_strategy", "round_robin")
        strategy = LoadBalancingStrategy(strategy_str)
        
        service_name = service_mesh.register_service(
            service_name=request.get("service_name", ""),
            load_balancing_strategy=strategy,
            health_check_interval=request.get("health_check_interval", 30.0),
            metadata=request.get("metadata"),
        )
        return {"success": True, "service_name": service_name}
    
    @app.post("/api/v1/mesh/register-instance")
    async def register_mesh_instance(request: Dict[str, Any]):
        """Registrar instancia de servicio."""
        instance_id = service_mesh.register_instance(
            instance_id=request.get("instance_id", f"inst_{datetime.now().timestamp()}"),
            service_name=request.get("service_name", ""),
            address=request.get("address", ""),
            port=request.get("port", 0),
            weight=request.get("weight", 1),
            metadata=request.get("metadata"),
        )
        return {"success": True, "instance_id": instance_id}
    
    @app.get("/api/v1/mesh/service/{service_name}/instance")
    async def get_service_instance(service_name: str, client_id: Optional[str] = None):
        """Obtener instancia de servicio."""
        instance = service_mesh.get_instance(service_name, client_id)
        if not instance:
            raise HTTPException(status_code=404, detail="No instance available")
        
        return {
            "instance_id": instance.instance_id,
            "address": instance.address,
            "port": instance.port,
            "status": instance.status.value,
            "weight": instance.weight,
        }
    
    @app.post("/api/v1/mesh/instance/{instance_id}/status")
    async def update_instance_status(instance_id: str, request: Dict[str, Any]):
        """Actualizar estado de instancia."""
        status_str = request.get("status", "unknown")
        status = ServiceStatus(status_str)
        
        service_mesh.update_instance_status(instance_id, status)
        return {"success": True}
    
    @app.get("/api/v1/mesh/service/{service_name}/instances")
    async def get_service_instances(service_name: str):
        """Obtener instancias de servicio."""
        instances = service_mesh.get_service_instances(service_name)
        return {"instances": instances, "count": len(instances)}
    
    @app.get("/api/v1/mesh/summary")
    async def get_service_mesh_summary():
        """Obtener resumen de la malla."""
        return service_mesh.get_service_mesh_summary()
    
    # Adaptive Throttler
    @app.post("/api/v1/throttler/add-rule")
    async def add_throttle_rule(request: Dict[str, Any]):
        """Agregar regla de throttling."""
        strategy_str = request.get("strategy", "adaptive")
        strategy = ThrottleStrategy(strategy_str)
        
        rule_id = adaptive_throttler.add_rule(
            rule_id=request.get("rule_id", f"rule_{datetime.now().timestamp()}"),
            identifier=request.get("identifier", ""),
            base_limit=request.get("base_limit", 100),
            strategy=strategy,
            min_limit=request.get("min_limit", 1),
            max_limit=request.get("max_limit", 1000),
            adjustment_factor=request.get("adjustment_factor", 0.1),
            metadata=request.get("metadata"),
        )
        return {"success": True, "rule_id": rule_id}
    
    @app.post("/api/v1/throttler/record-metric")
    async def record_throttle_metric(request: Dict[str, Any]):
        """Registrar métrica para throttling."""
        adaptive_throttler.record_metric(
            rule_id=request.get("rule_id", ""),
            success_rate=request.get("success_rate"),
            error_rate=request.get("error_rate"),
            response_time=request.get("response_time"),
            queue_size=request.get("queue_size"),
            system_load=request.get("system_load"),
        )
        return {"success": True}
    
    @app.post("/api/v1/throttler/check")
    async def check_throttle(request: Dict[str, Any]):
        """Verificar throttling."""
        allowed, info = await adaptive_throttler.check_throttle(
            rule_id=request.get("rule_id", ""),
            identifier=request.get("identifier"),
        )
        return info
    
    @app.get("/api/v1/throttler/{rule_id}/status")
    async def get_throttle_status(rule_id: str):
        """Obtener estado de throttling."""
        status = adaptive_throttler.get_throttle_status(rule_id)
        return status
    
    @app.get("/api/v1/throttler/summary")
    async def get_adaptive_throttler_summary():
        """Obtener resumen del limitador."""
        return adaptive_throttler.get_adaptive_throttler_summary()
    
    # Backpressure Manager
    @app.post("/api/v1/backpressure/add-rule")
    async def add_backpressure_rule(request: Dict[str, Any]):
        """Agregar regla de backpressure."""
        rule_id = backpressure_manager.add_rule(
            rule_id=request.get("rule_id", f"rule_{datetime.now().timestamp()}"),
            component_id=request.get("component_id", ""),
            queue_threshold=request.get("queue_threshold", 100),
            error_rate_threshold=request.get("error_rate_threshold", 0.1),
            latency_threshold=request.get("latency_threshold", 2.0),
            metadata=request.get("metadata"),
        )
        return {"success": True, "rule_id": rule_id}
    
    @app.post("/api/v1/backpressure/record-metric")
    async def record_backpressure_metric(request: Dict[str, Any]):
        """Registrar métrica de backpressure."""
        backpressure_manager.record_metric(
            component_id=request.get("component_id", ""),
            queue_size=request.get("queue_size"),
            processing_rate=request.get("processing_rate"),
            arrival_rate=request.get("arrival_rate"),
            error_rate=request.get("error_rate"),
            latency=request.get("latency"),
            system_load=request.get("system_load"),
        )
        return {"success": True}
    
    @app.get("/api/v1/backpressure/{component_id}/level")
    async def get_backpressure_level(component_id: str):
        """Obtener nivel de backpressure."""
        level = backpressure_manager.get_backpressure_level(component_id)
        return {"component_id": component_id, "level": level.value}
    
    @app.post("/api/v1/backpressure/{component_id}/check")
    async def check_should_accept_backpressure(component_id: str):
        """Verificar si se debe aceptar petición."""
        allowed = backpressure_manager.should_accept_request(component_id)
        return {"allowed": allowed, "component_id": component_id}
    
    @app.get("/api/v1/backpressure/status")
    async def get_backpressure_status(component_id: Optional[str] = None):
        """Obtener estado de backpressure."""
        status = backpressure_manager.get_backpressure_status(component_id)
        return status
    
    @app.get("/api/v1/backpressure/history")
    async def get_backpressure_history(component_id: Optional[str] = None, limit: int = 100):
        """Obtener historial de backpressure."""
        history = backpressure_manager.get_backpressure_history(component_id, limit)
        return {"history": history, "count": len(history)}
    
    @app.get("/api/v1/backpressure/summary")
    async def get_backpressure_manager_summary():
        """Obtener resumen del gestor."""
        return backpressure_manager.get_backpressure_manager_summary()
    
    @app.post("/api/v1/chat/sessions", response_model=ChatSessionResponse)
    async def create_session(request: ChatCreateRequest):
        """
        Crear una nueva sesión de chat.
        
        La sesión se iniciará automáticamente si se proporciona initial_message.
        """
        try:
            session = await chat_engine.create_session(
                user_id=request.user_id,
                initial_message=request.initial_message,
                auto_continue=request.auto_continue,
            )
            
            # Si hay mensaje inicial, iniciar chat continuo
            if request.initial_message:
                await chat_engine.start_continuous_chat(
                    session.session_id,
                    request.initial_message,
                )
            
            return ChatSessionResponse(
                session_id=session.session_id,
                state=session.state.value,
                is_paused=session.is_paused,
                message_count=len(session.messages),
                auto_continue=session.auto_continue,
            )
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/chat/sessions/{session_id}", response_model=ChatSessionResponse)
    async def get_session(session_id: str):
        """Obtener información de una sesión."""
        session = chat_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return ChatSessionResponse(
            session_id=session.session_id,
            state=session.state.value,
            is_paused=session.is_paused,
            message_count=len(session.messages),
            auto_continue=session.auto_continue,
        )
    
    @app.get("/api/v1/chat/sessions/{session_id}/messages")
    async def get_messages(session_id: str, limit: int = 50):
        """Obtener mensajes de una sesión."""
        session = chat_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = session.messages[-limit:] if limit else session.messages
        return {
            "session_id": session_id,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in messages
            ],
            "total": len(session.messages),
        }
    
    @app.post("/api/v1/chat/sessions/{session_id}/messages")
    async def send_message(session_id: str, request: ChatMessageRequest):
        """
        Enviar un mensaje a la sesión.
        
        Si la sesión está pausada y auto_continue está activado, se reanudará automáticamente.
        """
        try:
            await chat_engine.add_user_message(session_id, request.message)
            
            # Si la sesión no está activa, iniciar chat continuo
            session = chat_engine.get_session(session_id)
            if session and session.state == ChatState.IDLE:
                await chat_engine.start_continuous_chat(session_id)
            
            return ChatResponse(
                success=True,
                message="Message sent successfully",
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/chat/sessions/{session_id}/start")
    async def start_chat(session_id: str, initial_prompt: Optional[str] = None):
        """
        Iniciar chat continuo para una sesión.
        
        El chat comenzará a generar respuestas automáticamente.
        """
        try:
            await chat_engine.start_continuous_chat(session_id, initial_prompt)
            return ChatResponse(
                success=True,
                message="Continuous chat started",
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error starting chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/chat/sessions/{session_id}/pause")
    async def pause_chat(session_id: str, reason: Optional[str] = None):
        """
        Pausar el chat continuo.
        
        El chat dejará de generar respuestas hasta que se reanude.
        """
        try:
            await chat_engine.pause_session(session_id, reason)
            return ChatResponse(
                success=True,
                message="Chat paused",
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error pausing chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/chat/sessions/{session_id}/resume")
    async def resume_chat(session_id: str):
        """
        Reanudar el chat continuo.
        
        El chat comenzará a generar respuestas nuevamente.
        """
        try:
            await chat_engine.resume_session(session_id)
            return ChatResponse(
                success=True,
                message="Chat resumed",
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error resuming chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/chat/sessions/{session_id}/stop")
    async def stop_chat(session_id: str):
        """
        Detener completamente el chat.
        
        La sesión se detendrá y no podrá reanudarse.
        """
        try:
            await chat_engine.stop_session(session_id)
            return ChatResponse(
                success=True,
                message="Chat stopped",
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error stopping chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/chat/sessions/{session_id}/stream")
    async def stream_chat(session_id: str, message: Optional[str] = None):
        """
        Stream de respuestas en tiempo real.
        
        Retorna Server-Sent Events (SSE) con los tokens generados.
        """
        session = chat_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        async def generate_stream():
            """Generar stream de respuestas."""
            try:
                async for token in chat_engine.stream_response(session_id, message):
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in stream: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    @app.delete("/api/v1/chat/sessions/{session_id}")
    async def delete_session(session_id: str):
        """Eliminar una sesión."""
        try:
            await chat_engine.cleanup_session(session_id)
            return ChatResponse(
                success=True,
                message="Session deleted",
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/chat/sessions")
    async def list_sessions(user_id: Optional[str] = None):
        """Listar todas las sesiones activas."""
        sessions = list(chat_engine.sessions.values())
        
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        
        return {
            "sessions": [s.to_dict() for s in sessions],
            "total": len(sessions),
        }
    
    @app.get("/api/v1/chat/sessions/{session_id}/metrics")
    async def get_session_metrics(session_id: str):
        """Obtener métricas de una sesión."""
        if not chat_engine.metrics:
            raise HTTPException(status_code=503, detail="Metrics not enabled")
        
        metrics = chat_engine.metrics.get_session_metrics(session_id)
        if not metrics:
            raise HTTPException(status_code=404, detail="Session metrics not found")
        
        return metrics
    
    @app.get("/api/v1/chat/metrics")
    async def get_global_metrics():
        """Obtener métricas globales del sistema."""
        if not chat_engine.metrics:
            raise HTTPException(status_code=503, detail="Metrics not enabled")
        
        return chat_engine.metrics.get_global_metrics()
    
    @app.get("/api/v1/chat/rate-limit/{identifier}")
    async def get_rate_limit_stats(identifier: str):
        """Obtener estadísticas de rate limiting."""
        stats = rate_limiter.get_stats(identifier)
        return stats
    
    @app.get("/api/v1/chat/cache/stats")
    async def get_cache_stats():
        """Obtener estadísticas del cache."""
        if not chat_engine.cache:
            raise HTTPException(status_code=503, detail="Cache not enabled")
        return chat_engine.cache.get_stats()
    
    @app.post("/api/v1/chat/cache/clear")
    async def clear_cache():
        """Limpiar el cache."""
        if not chat_engine.cache:
            raise HTTPException(status_code=503, detail="Cache not enabled")
        await chat_engine.cache.clear()
        return {"success": True, "message": "Cache cleared"}
    
    # Endpoints de análisis
    @app.get("/api/v1/chat/sessions/{session_id}/analyze")
    async def analyze_session(session_id: str):
        """Analizar una sesión."""
        session = chat_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        insights = await analyzer.analyze(session)
        return {
            "session_id": session_id,
            "insights": {
                "total_messages": insights.total_messages,
                "user_messages": insights.user_messages,
                "assistant_messages": insights.assistant_messages,
                "average_message_length": insights.average_message_length,
                "topics": insights.topics,
                "key_phrases": insights.key_phrases,
                "conversation_duration": insights.conversation_duration,
                "sentiment_trend": insights.sentiment_trend,
                "response_time_stats": insights.response_time_stats,
            }
        }
    
    @app.get("/api/v1/chat/sessions/{session_id}/summary")
    async def get_session_summary(session_id: str):
        """Obtener resumen de una sesión."""
        session = chat_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        summary = await analyzer.generate_summary(session)
        return {"session_id": session_id, "summary": summary}
    
    # Endpoints de exportación
    @app.get("/api/v1/chat/sessions/{session_id}/export/{format}")
    async def export_session(session_id: str, format: str):
        """Exportar sesión en diferentes formatos."""
        session = chat_engine.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        format = format.lower()
        
        if format == "json":
            content = await exporter.export_json(session)
            return JSONResponse(content=json.loads(content))
        elif format == "markdown" or format == "md":
            content = await exporter.export_markdown(session)
            return JSONResponse(content={"content": content, "format": "markdown"})
        elif format == "csv":
            content = await exporter.export_csv(session)
            return JSONResponse(content={"content": content, "format": "csv"})
        elif format == "txt" or format == "text":
            content = await exporter.export_txt(session)
            return JSONResponse(content={"content": content, "format": "text"})
        elif format == "html":
            content = await exporter.export_html(session)
            return JSONResponse(content={"content": content, "format": "html"})
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
    
    # Endpoints de templates
    @app.get("/api/v1/chat/templates")
    async def list_templates(category: Optional[str] = None):
        """Listar plantillas."""
        templates = template_manager.list_templates(category)
        return {
            "templates": [
                {
                    "id": t.id,
                    "name": t.name,
                    "category": t.category,
                    "variables": t.variables,
                }
                for t in templates
            ]
        }
    
    @app.post("/api/v1/chat/templates/{template_id}/render")
    async def render_template(template_id: str, variables: Optional[Dict[str, str]] = None):
        """Renderizar una plantilla."""
        content = template_manager.render(template_id, variables)
        if content is None:
            raise HTTPException(status_code=404, detail="Template not found")
        return {"template_id": template_id, "content": content}
    
    # Endpoints de webhooks
    @app.post("/api/v1/chat/webhooks")
    async def register_webhook(webhook: Dict[str, Any]):
        """Registrar un webhook."""
        from ..core.webhooks import Webhook
        
        webhook_obj = Webhook(
            url=webhook["url"],
            events=[WebhookEvent(e) for e in webhook.get("events", [])],
            secret=webhook.get("secret"),
            enabled=webhook.get("enabled", True),
        )
        
        await webhook_manager.register(webhook_obj)
        return {"success": True, "message": "Webhook registered"}
    
    @app.get("/api/v1/chat/webhooks")
    async def list_webhooks():
        """Listar webhooks registrados."""
        webhooks = webhook_manager.get_webhooks()
        return {
            "webhooks": [
                {
                    "url": w.url,
                    "events": [e.value for e in w.events],
                    "enabled": w.enabled,
                }
                for w in webhooks
            ]
        }
    
    # Endpoints de autenticación
    @app.post("/api/v1/auth/register")
    async def register_user(request: Dict[str, Any]):
        """Registrar nuevo usuario."""
        user = auth_manager.create_user(
            username=request["username"],
            password=request["password"],
            email=request.get("email"),
        )
        token = auth_manager.create_access_token(user)
        return {
            "user_id": user.user_id,
            "username": user.username,
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expires_in": token.expires_in,
        }
    
    @app.post("/api/v1/auth/login")
    async def login(request: Dict[str, Any]):
        """Iniciar sesión."""
        user = auth_manager.authenticate(
            request["username"],
            request["password"],
        )
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        token = auth_manager.create_access_token(user)
        return {
            "user_id": user.user_id,
            "username": user.username,
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expires_in": token.expires_in,
        }
    
    # Endpoints de backup
    @app.post("/api/v1/chat/backup/create")
    async def create_backup():
        """Crear backup manual."""
        backup_path = await backup_manager.create_backup()
        if not backup_path:
            raise HTTPException(status_code=500, detail="Failed to create backup")
        return {"success": True, "backup_path": backup_path}
    
    @app.get("/api/v1/chat/backup/list")
    async def list_backups():
        """Listar backups disponibles."""
        backups = backup_manager.list_backups()
        return {"backups": backups}
    
    @app.get("/api/v1/chat/backup/history")
    async def get_backup_history():
        """Obtener historial de backups."""
        history = backup_manager.get_backup_history()
        return {"history": history}
    
    # Endpoints de Bulk Operations
    from ..core.bulk_operations import (
        BulkSessionOperations,
        BulkMessageOperations,
        BulkExporter,
        BulkAnalytics,
        BulkCleanup,
        BulkProcessor,
        BulkImporter,
        BulkNotifications,
        BulkSearch,
        BulkTesting,
        BulkBackupRestore,
        BulkMigration,
        BulkMetrics,
        BulkRealTimeMetrics,
        BulkAdvancedCache,
        BulkPriorityQueue,
        BulkEnhancedValidator,
        BulkDashboard,
        BulkScheduler,
        BulkRateLimiter,
        BulkAutoCreator,
        BulkAutoExpander,
        BulkAutoProcessor,
        BulkAutoMaintainer,
        BulkInfiniteGenerator,
        BulkSelfSustaining
    )
    
    # Inicializar bulk operations
    bulk_sessions = BulkSessionOperations(
        chat_engine=chat_engine,
        storage=storage,
        max_workers=10
    )
    bulk_messages = BulkMessageOperations(
        chat_engine=chat_engine,
        max_workers=10
    )
    bulk_exporter = BulkExporter(
        storage=storage,
        exporter=exporter,
        max_workers=10
    )
    bulk_analytics = BulkAnalytics(
        analyzer=analyzer,
        storage=storage,
        max_workers=10
    )
    bulk_cleanup = BulkCleanup(
        storage=storage,
        chat_engine=chat_engine,
        max_workers=10
    )
    bulk_processor = BulkProcessor(max_workers=10)
    bulk_importer = BulkImporter(
        chat_engine=chat_engine,
        storage=storage,
        max_workers=10
    )
    bulk_notifications = BulkNotifications(
        notification_manager=notification_manager,
        max_workers=10
    )
    bulk_search = BulkSearch(
        search_engine=search_engine,
        storage=storage,
        max_workers=10
    )
    bulk_testing = BulkTesting(
        chat_engine=chat_engine,
        max_workers=50
    )
    bulk_backup = BulkBackupRestore(
        storage=storage,
        backup_manager=backup_manager,
        max_workers=10
    )
    bulk_migration = BulkMigration(
        chat_engine=chat_engine,
        storage=storage,
        max_workers=10
    )
    bulk_metrics = BulkMetrics()
    bulk_scheduler = BulkScheduler()
    bulk_rate_limiter = BulkRateLimiter(
        max_operations_per_minute=100,
        max_operations_per_hour=1000
    )
    bulk_auto_creator = BulkAutoCreator(
        chat_engine=chat_engine,
        storage=storage,
        config={"max_workers": 10, "batch_size": 10}
    )
    bulk_auto_expander = BulkAutoExpander(
        chat_engine=chat_engine,
        storage=storage
    )
    bulk_auto_processor = BulkAutoProcessor(
        chat_engine=chat_engine,
        storage=storage
    )
    bulk_auto_maintainer = BulkAutoMaintainer(
        chat_engine=chat_engine,
        storage=storage
    )
    bulk_infinite_generator = BulkInfiniteGenerator(
        chat_engine=chat_engine,
        storage=storage
    )
    bulk_self_sustaining = BulkSelfSustaining(
        chat_engine=chat_engine,
        storage=storage,
        config={"max_workers": 10, "batch_size": 10, "max_capacity": 100000}
    )
    
    # Nuevas clases avanzadas (ya importadas arriba)
    
    bulk_realtime_metrics = BulkRealTimeMetrics(window_size_seconds=60)
    bulk_advanced_cache = BulkAdvancedCache(max_size=1000, default_ttl=3600)
    bulk_priority_queue = BulkPriorityQueue()
    bulk_enhanced_validator = BulkEnhancedValidator()
    bulk_dashboard = BulkDashboard(
        metrics=bulk_realtime_metrics,
        cache=bulk_advanced_cache,
        priority_queue=bulk_priority_queue
    )
    
    # Bulk Session Operations
    @app.post("/api/v1/bulk/sessions/create")
    async def bulk_create_sessions(request: Dict[str, Any]):
        """Crear múltiples sesiones en lote."""
        count = request.get("count", 1)
        initial_messages = request.get("initial_messages")
        auto_continue = request.get("auto_continue", True)
        parallel = request.get("parallel", True)
        user_id = request.get("user_id")
        
        session_ids = await bulk_sessions.create_sessions(
            count=count,
            initial_messages=initial_messages,
            auto_continue=auto_continue,
            parallel=parallel,
            user_id=user_id
        )
        return {
            "success": True,
            "created": len(session_ids),
            "session_ids": session_ids
        }
    
    @app.post("/api/v1/bulk/sessions/delete")
    async def bulk_delete_sessions(request: Dict[str, Any]):
        """Eliminar múltiples sesiones."""
        session_ids = request.get("session_ids", [])
        parallel = request.get("parallel", True)
        
        result = await bulk_sessions.delete_sessions(
            session_ids=session_ids,
            parallel=parallel
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "total": result.total,
            "duration": result.duration,
            "errors": result.errors[:10]  # Limitar errores
        }
    
    @app.post("/api/v1/bulk/sessions/pause")
    async def bulk_pause_sessions(request: Dict[str, Any]):
        """Pausar múltiples sesiones."""
        session_ids = request.get("session_ids", [])
        reason = request.get("reason")
        parallel = request.get("parallel", True)
        
        result = await bulk_sessions.pause_sessions(
            session_ids=session_ids,
            reason=reason,
            parallel=parallel
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "duration": result.duration
        }
    
    @app.post("/api/v1/bulk/sessions/resume")
    async def bulk_resume_sessions(request: Dict[str, Any]):
        """Reanudar múltiples sesiones."""
        session_ids = request.get("session_ids", [])
        parallel = request.get("parallel", True)
        
        result = await bulk_sessions.resume_sessions(
            session_ids=session_ids,
            parallel=parallel
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "duration": result.duration
        }
    
    @app.post("/api/v1/bulk/sessions/stop")
    async def bulk_stop_sessions(request: Dict[str, Any]):
        """Detener múltiples sesiones."""
        session_ids = request.get("session_ids", [])
        parallel = request.get("parallel", True)
        
        result = await bulk_sessions.stop_sessions(
            session_ids=session_ids,
            parallel=parallel
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "duration": result.duration
        }
    
    @app.post("/api/v1/bulk/sessions/export")
    async def bulk_export_sessions_direct(request: Dict[str, Any]):
        """Exportar múltiples sesiones (método directo)."""
        session_ids = request.get("session_ids", [])
        format = request.get("format", "json")
        parallel = request.get("parallel", True)
        
        result = await bulk_sessions.export_sessions(
            session_ids=session_ids,
            format=format,
            parallel=parallel
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "total": result.total,
            "duration": result.duration,
            "exports": result.data.get("exports", []) if result.data else []
        }
    
    # Bulk Message Operations
    @app.post("/api/v1/bulk/messages/send")
    async def bulk_send_messages(request: Dict[str, Any]):
        """Enviar mensaje a múltiples sesiones."""
        session_ids = request.get("session_ids", [])
        message = request.get("message", "")
        parallel = request.get("parallel", True)
        
        result = await bulk_messages.send_to_sessions(
            session_ids=session_ids,
            message=message,
            parallel=parallel
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "duration": result.duration
        }
    
    # Bulk Export
    @app.post("/api/v1/bulk/export/sessions")
    async def bulk_export_sessions(request: Dict[str, Any]):
        """Exportar múltiples sesiones."""
        session_ids = request.get("session_ids", [])
        format = request.get("format", "json")
        compress = request.get("compress", False)
        parallel = request.get("parallel", True)
        
        job_id = await bulk_exporter.export_sessions(
            session_ids=session_ids,
            format=format,
            compress=compress,
            parallel=parallel
        )
        return {
            "success": True,
            "job_id": job_id,
            "message": "Export job started"
        }
    
    @app.get("/api/v1/bulk/export/status/{job_id}")
    async def get_export_status(job_id: str):
        """Obtener estado de exportación."""
        status = await bulk_exporter.get_export_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        return status
    
    # Bulk Analytics
    @app.post("/api/v1/bulk/analytics/sessions")
    async def bulk_analyze_sessions(request: Dict[str, Any]):
        """Analizar múltiples sesiones."""
        session_ids = request.get("session_ids", [])
        parallel = request.get("parallel", True)
        
        results = await bulk_analytics.analyze_sessions_bulk(
            session_ids=session_ids,
            parallel=parallel
        )
        return {
            "success": True,
            "analyzed": len(results),
            "results": results
        }
    
    # Bulk Cleanup
    @app.post("/api/v1/bulk/cleanup/sessions")
    async def bulk_cleanup_sessions(request: Dict[str, Any]):
        """Limpiar sesiones antiguas."""
        days_old = request.get("days_old", 30)
        dry_run = request.get("dry_run", False)
        
        result = await bulk_cleanup.cleanup_old_sessions(
            days_old=days_old,
            dry_run=dry_run
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "total": result.total,
            "duration": result.duration,
            "dry_run": dry_run
        }
    
    # Bulk Import
    @app.post("/api/v1/bulk/import/sessions")
    async def bulk_import_sessions(request: Dict[str, Any]):
        """Importar múltiples sesiones."""
        sessions_data = request.get("sessions", [])
        validate = request.get("validate", True)
        parallel = request.get("parallel", True)
        
        job_id = await bulk_importer.import_sessions(
            sessions_data=sessions_data,
            validate=validate,
            parallel=parallel
        )
        return {
            "success": True,
            "job_id": job_id,
            "message": "Import job started"
        }
    
    @app.get("/api/v1/bulk/import/status/{job_id}")
    async def get_import_status(job_id: str):
        """Obtener estado de importación."""
        status = await bulk_importer.get_import_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        return status
    
    # Bulk Notifications
    @app.post("/api/v1/bulk/notifications/send")
    async def bulk_send_notifications(request: Dict[str, Any]):
        """Enviar notificaciones masivas."""
        user_ids = request.get("user_ids", [])
        template = request.get("template", "")
        data = request.get("data", {})
        channels = request.get("channels", ["email"])
        parallel = request.get("parallel", True)
        
        result = await bulk_notifications.send_bulk(
            user_ids=user_ids,
            template=template,
            data=data,
            channels=channels,
            parallel=parallel
        )
        return {
            "success": result.success,
            "processed": result.processed,
            "failed": result.failed,
            "duration": result.duration
        }
    
    # Bulk Search
    @app.post("/api/v1/bulk/search/execute")
    async def bulk_search_execute(request: Dict[str, Any]):
        """Ejecutar búsqueda masiva."""
        queries = request.get("queries", [])
        filters = request.get("filters", {})
        parallel = request.get("parallel", True)
        
        results = await bulk_search.search_bulk(
            queries=queries,
            filters=filters,
            parallel=parallel
        )
        return {
            "success": True,
            "queries_count": len(queries),
            "results": results
        }
    
    # Bulk Processor
    @app.get("/api/v1/bulk/process/status/{job_id}")
    async def get_process_status(job_id: str):
        """Obtener estado de procesamiento."""
        status = await bulk_processor.get_progress(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        return status
    
    @app.post("/api/v1/bulk/process/cancel/{job_id}")
    async def cancel_process_job(job_id: str):
        """Cancelar job de procesamiento."""
        cancelled = await bulk_processor.cancel_job(job_id)
        if not cancelled:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled")
        return {"success": True, "message": "Job cancelled"}
    
    # Bulk Backup/Restore
    @app.post("/api/v1/bulk/backup/sessions")
    async def bulk_backup_sessions(request: Dict[str, Any]):
        """Crear backup masivo de sesiones."""
        session_ids = request.get("session_ids", [])
        compress = request.get("compress", True)
        encrypt = request.get("encrypt", False)
        
        job_id = await bulk_backup.backup_sessions(
            session_ids=session_ids,
            compress=compress,
            encrypt=encrypt
        )
        return {
            "success": True,
            "job_id": job_id,
            "message": "Backup job started"
        }
    
    @app.get("/api/v1/bulk/backup/status/{job_id}")
    async def get_backup_status_bulk(job_id: str):
        """Obtener estado de backup masivo."""
        status = await bulk_backup.get_backup_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        return status
    
    # Bulk Testing
    @app.post("/api/v1/bulk/testing/load-test")
    async def bulk_load_test(request: Dict[str, Any]):
        """Ejecutar test de carga masivo."""
        concurrent_sessions = request.get("concurrent_sessions", 100)
        duration = request.get("duration", 60)
        operations_per_session = request.get("operations_per_session", 10)
        
        results = await bulk_testing.load_test(
            concurrent_sessions=concurrent_sessions,
            duration=duration,
            operations_per_session=operations_per_session
        )
        return {
            "success": True,
            "results": results
        }
    
    @app.post("/api/v1/bulk/testing/stress-test")
    async def bulk_stress_test(request: Dict[str, Any]):
        """Ejecutar test de estrés."""
        max_sessions = request.get("max_sessions", 1000)
        ramp_up_seconds = request.get("ramp_up_seconds", 60)
        
        results = await bulk_testing.stress_test(
            max_sessions=max_sessions,
            ramp_up_seconds=ramp_up_seconds
        )
        return {
            "success": True,
            "results": results
        }
    
    # Bulk Migration
    @app.post("/api/v1/bulk/migration/start")
    async def bulk_migration_start(request: Dict[str, Any]):
        """Iniciar migración masiva."""
        session_ids = request.get("session_ids", [])
        source_format = request.get("source_format", "v1")
        target_format = request.get("target_format", "v2")
        batch_size = request.get("batch_size", 100)
        
        job_id = await bulk_migration.migrate_sessions(
            session_ids=session_ids,
            source_format=source_format,
            target_format=target_format,
            batch_size=batch_size
        )
        return {
            "success": True,
            "job_id": job_id,
            "message": "Migration job started"
        }
    
    @app.get("/api/v1/bulk/migration/status/{job_id}")
    async def get_migration_status(job_id: str):
        """Obtener estado de migración."""
        status = await bulk_migration.get_migration_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        return status
    
    # Bulk Metrics
    @app.get("/api/v1/bulk/metrics/stats")
    async def get_bulk_metrics_stats(operation: Optional[str] = None):
        """Obtener estadísticas de operaciones bulk."""
        stats = bulk_metrics.get_stats(operation)
        return {"stats": stats}
    
    @app.get("/api/v1/bulk/metrics/history")
    async def get_bulk_metrics_history(limit: int = 100):
        """Obtener historial de operaciones bulk."""
        history = bulk_metrics.get_history(limit)
        return {"history": history}
    
    @app.get("/api/v1/bulk/metrics/summary")
    async def get_bulk_metrics_summary():
        """Obtener resumen de operaciones bulk."""
        summary = bulk_metrics.get_summary()
        return summary
    
    # Bulk Scheduler
    @app.post("/api/v1/bulk/scheduler/schedule")
    async def bulk_scheduler_schedule(request: Dict[str, Any]):
        """Programar operación bulk recurrente."""
        # Nota: Esto requiere una función callable, por lo que es simplificado
        job_id = request.get("job_id")
        schedule = request.get("schedule", "0 2 * * *")
        enabled = request.get("enabled", True)
        config = request.get("config", {})
        
        # En producción, esto necesitaría un registry de operaciones
        return {
            "success": False,
            "message": "Scheduler requires callable operation. Use programmatic API."
        }
    
    @app.get("/api/v1/bulk/scheduler/jobs")
    async def list_scheduled_jobs():
        """Listar jobs programados."""
        jobs = bulk_scheduler.list_jobs()
        return {"jobs": jobs}
    
    @app.post("/api/v1/bulk/scheduler/{job_id}/enable")
    async def enable_scheduled_job(job_id: str):
        """Habilitar job programado."""
        await bulk_scheduler.enable_job(job_id)
        return {"success": True, "message": "Job enabled"}
    
    @app.post("/api/v1/bulk/scheduler/{job_id}/disable")
    async def disable_scheduled_job(job_id: str):
        """Deshabilitar job programado."""
        await bulk_scheduler.disable_job(job_id)
        return {"success": True, "message": "Job disabled"}
    
    # Bulk Rate Limiter
    @app.get("/api/v1/bulk/rate-limit/stats")
    async def get_bulk_rate_limit_stats(operation: Optional[str] = None):
        """Obtener estadísticas de rate limiting."""
        stats = bulk_rate_limiter.get_stats(operation)
        return {"stats": stats}
    
    @app.post("/api/v1/bulk/rate-limit/check")
    async def check_bulk_rate_limit(request: Dict[str, Any]):
        """Verificar rate limit."""
        operation = request.get("operation", "")
        user_id = request.get("user_id")
        
        allowed, error = await bulk_rate_limiter.check_rate_limit(
            operation=operation,
            user_id=user_id
        )
        
        return {
            "allowed": allowed,
            "error": error
        }
    
    # Bulk Auto-Creation (Sistema que nunca se detiene)
    @app.post("/api/v1/bulk/auto-creation/start")
    async def start_auto_creation(request: Dict[str, Any]):
        """Iniciar auto-creación continua que nunca se detiene."""
        creation_rate = request.get("creation_rate", 1.0)
        batch_size = request.get("batch_size", 10)
        initial_messages = request.get("initial_messages")
        auto_continue = request.get("auto_continue", True)
        
        await bulk_auto_creator.start_continuous_creation(
            creation_rate=creation_rate,
            batch_size=batch_size,
            initial_messages=initial_messages,
            auto_continue=auto_continue
        )
        
        return {
            "success": True,
            "message": "Auto-creation started - will continue indefinitely",
            "creation_rate": creation_rate,
            "batch_size": batch_size
        }
    
    @app.post("/api/v1/bulk/auto-creation/stop")
    async def stop_auto_creation():
        """Detener auto-creación (opcional)."""
        await bulk_auto_creator.stop_continuous_creation()
        return {"success": True, "message": "Auto-creation stopped"}
    
    @app.get("/api/v1/bulk/auto-creation/stats")
    async def get_auto_creation_stats():
        """Obtener estadísticas de auto-creación."""
        stats = bulk_auto_creator.get_stats()
        return stats
    
    # Bulk Auto-Expansion
    @app.post("/api/v1/bulk/auto-expansion/start")
    async def start_auto_expansion(request: Dict[str, Any]):
        """Iniciar auto-expansión continua."""
        check_interval = request.get("check_interval", 60.0)
        expansion_rate = request.get("expansion_rate", 1.1)
        max_capacity = request.get("max_capacity", 100000)
        
        await bulk_auto_expander.start_auto_expansion(
            check_interval=check_interval,
            expansion_rate=expansion_rate,
            max_capacity=max_capacity
        )
        
        return {
            "success": True,
            "message": "Auto-expansion started",
            "target_capacity": max_capacity
        }
    
    # Bulk Self-Sustaining System
    @app.post("/api/v1/bulk/self-sustaining/start")
    async def start_self_sustaining(request: Dict[str, Any]):
        """Iniciar sistema auto-sostenible completo."""
        creation_rate = request.get("creation_rate", 1.0)
        expansion_enabled = request.get("expansion_enabled", True)
        processing_enabled = request.get("processing_enabled", True)
        maintenance_enabled = request.get("maintenance_enabled", True)
        
        await bulk_self_sustaining.start_self_sustaining_system(
            creation_rate=creation_rate,
            expansion_enabled=expansion_enabled,
            processing_enabled=processing_enabled,
            maintenance_enabled=maintenance_enabled
        )
        
        return {
            "success": True,
            "message": "Self-sustaining system started - will run indefinitely",
            "components": {
                "auto_creation": True,
                "auto_expansion": expansion_enabled,
                "auto_processing": processing_enabled,
                "auto_maintenance": maintenance_enabled
            }
        }
    
    @app.get("/api/v1/bulk/self-sustaining/stats")
    async def get_self_sustaining_stats():
        """Obtener estadísticas del sistema auto-sostenible."""
        stats = bulk_self_sustaining.get_system_stats()
        return stats
    
    @app.post("/api/v1/bulk/self-sustaining/ensure-continuity")
    async def ensure_continuity():
        """Asegurar que el sistema continúe operando."""
        await bulk_self_sustaining.ensure_continuous_operation()
        return {
            "success": True,
            "message": "System continuity verified and maintained"
        }
    
    # Bulk Infinite Generator
    @app.post("/api/v1/bulk/infinite-generator/create")
    async def create_infinite_generator(request: Dict[str, Any]):
        """Crear generador infinito de sesiones."""
        generator_id = request.get("generator_id", f"gen_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        creation_rate = request.get("creation_rate", 0.5)
        initial_messages = request.get("initial_messages")
        
        gen = await bulk_infinite_generator.create_infinite_session_generator(
            generator_id=generator_id,
            creation_rate=creation_rate,
            initial_messages=initial_messages
        )
        
        # Consumir en background
        async def consume():
            async for session_id in gen:
                logger.info(f"Infinite generator created session: {session_id}")
        
        asyncio.create_task(consume())
        
        return {
            "success": True,
            "generator_id": generator_id,
            "message": "Infinite generator started - will create sessions indefinitely"
        }
    
    # Nuevos endpoints avanzados
    @app.get("/api/v1/bulk/metrics/realtime")
    async def get_realtime_metrics(operation_type: Optional[str] = None):
        """Obtener métricas en tiempo real."""
        metrics = bulk_realtime_metrics.get_metrics(operation_type)
        return metrics
    
    @app.get("/api/v1/bulk/metrics/health")
    async def get_metrics_health():
        """Obtener estado de salud basado en métricas."""
        health = bulk_realtime_metrics.get_health_status()
        return health
    
    @app.post("/api/v1/bulk/metrics/record")
    async def record_metric(request: Dict[str, Any]):
        """Registrar una métrica."""
        bulk_realtime_metrics.record_operation(
            operation_type=request.get("operation_type", "unknown"),
            duration=request.get("duration", 0.0),
            success=request.get("success", True),
            items_processed=request.get("items_processed", 1),
            metadata=request.get("metadata")
        )
        return {"success": True, "message": "Metric recorded"}
    
    @app.get("/api/v1/bulk/cache/stats")
    async def get_cache_stats():
        """Obtener estadísticas del caché."""
        stats = bulk_advanced_cache.get_stats()
        return stats
    
    @app.post("/api/v1/bulk/cache/get")
    async def cache_get(request: Dict[str, Any]):
        """Obtener valor del caché."""
        key = request.get("key")
        if not key:
            raise HTTPException(status_code=400, detail="Key is required")
        value = bulk_advanced_cache.get(key)
        return {"found": value is not None, "value": value}
    
    @app.post("/api/v1/bulk/cache/set")
    async def cache_set(request: Dict[str, Any]):
        """Guardar valor en caché."""
        key = request.get("key")
        value = request.get("value")
        ttl = request.get("ttl")
        if not key:
            raise HTTPException(status_code=400, detail="Key is required")
        bulk_advanced_cache.set(key, value, ttl)
        return {"success": True, "message": "Value cached"}
    
    @app.post("/api/v1/bulk/cache/invalidate")
    async def cache_invalidate(request: Dict[str, Any]):
        """Invalidar entrada del caché."""
        key = request.get("key")
        if not key:
            raise HTTPException(status_code=400, detail="Key is required")
        bulk_advanced_cache.invalidate(key)
        return {"success": True, "message": "Cache entry invalidated"}
    
    @app.get("/api/v1/bulk/queue/stats")
    async def get_queue_stats():
        """Obtener estadísticas de la cola de prioridades."""
        stats = bulk_priority_queue.get_stats()
        return stats
    
    @app.post("/api/v1/bulk/queue/enqueue")
    async def queue_enqueue(request: Dict[str, Any]):
        """Agregar operación a la cola."""
        operation = request.get("operation", {})
        priority = request.get("priority", "medium")
        bulk_priority_queue.enqueue(operation, priority)
        return {"success": True, "message": "Operation enqueued", "queue_size": bulk_priority_queue.size()}
    
    @app.post("/api/v1/bulk/queue/dequeue")
    async def queue_dequeue():
        """Obtener siguiente operación de la cola."""
        operation = bulk_priority_queue.dequeue()
        if not operation:
            raise HTTPException(status_code=404, detail="Queue is empty")
        return {"success": True, "operation": operation}
    
    @app.get("/api/v1/bulk/queue/peek")
    async def queue_peek():
        """Ver siguiente operación sin removerla."""
        operation = bulk_priority_queue.peek()
        if not operation:
            raise HTTPException(status_code=404, detail="Queue is empty")
        return {"operation": operation}
    
    @app.post("/api/v1/bulk/validator/validate")
    async def validate_bulk(request: Dict[str, Any]):
        """Validar datos."""
        operation_type = request.get("operation_type", "default")
        data = request.get("data", {})
        use_cache = request.get("use_cache", True)
        
        is_valid, error = bulk_enhanced_validator.validate(operation_type, data, use_cache)
        return {
            "valid": is_valid,
            "error": error
        }
    
    @app.post("/api/v1/bulk/validator/validate-batch")
    async def validate_batch(request: Dict[str, Any]):
        """Validar lote de items."""
        operation_type = request.get("operation_type", "default")
        items = request.get("items", [])
        
        results = bulk_enhanced_validator.validate_batch(operation_type, items)
        return results
    
    @app.post("/api/v1/bulk/validator/add-rule")
    async def add_validator_rule(request: Dict[str, Any]):
        """Añadir regla de validación."""
        operation_type = request.get("operation_type")
        # Nota: En producción, necesitarías deserializar la función desde string o usar un registro
        # Por ahora, retornamos un mensaje
        return {
            "success": True,
            "message": "Rule registration endpoint - implement validator function registration"
        }
    
    @app.get("/api/v1/bulk/dashboard")
    async def get_dashboard():
        """Obtener datos completos del dashboard."""
        dashboard_data = bulk_dashboard.get_dashboard_data()
        return dashboard_data
    
    @app.get("/api/v1/bulk/dashboard/summary")
    async def get_dashboard_summary():
        """Obtener resumen ejecutivo del dashboard."""
        summary = bulk_dashboard._generate_summary()
        return summary
    
    @app.post("/api/v1/bulk/dashboard/alert")
    async def add_dashboard_alert(request: Dict[str, Any]):
        """Añadir alerta al dashboard."""
        level = request.get("level", "info")
        message = request.get("message", "")
        details = request.get("details", {})
        bulk_dashboard.add_alert(level, message, details)
        return {"success": True, "message": "Alert added"}
    
    @app.get("/api/v1/bulk/dashboard/alerts")
    async def get_dashboard_alerts(level: Optional[str] = None):
        """Obtener alertas del dashboard."""
        alerts = bulk_dashboard.get_alerts(level)
        return {"alerts": alerts, "count": len(alerts)}
    
    # Nuevas funcionalidades avanzadas
    from ..core.bulk_operations import (
        BulkBenchmark,
        BulkAutoTuner
    )
    
    bulk_benchmark = BulkBenchmark()
    bulk_auto_tuner = BulkAutoTuner(
        metrics=bulk_realtime_metrics,
        benchmark=bulk_benchmark
    )
    
    @app.post("/api/v1/bulk/benchmark/run")
    async def run_benchmark(request: Dict[str, Any]):
        """Ejecutar benchmark de una operación."""
        operation_name = request.get("operation_name", "unknown")
        iterations = request.get("iterations", 3)
        warmup = request.get("warmup", 1)
        test_data = request.get("test_data", [])
        
        # Nota: En producción necesitarías deserializar la función
        # Por ahora retornamos un mensaje
        return {
            "success": True,
            "message": "Benchmark endpoint - implement operation execution",
            "operation_name": operation_name,
            "iterations": iterations
        }
    
    @app.get("/api/v1/bulk/benchmark/summary")
    async def get_benchmark_summary():
        """Obtener resumen de benchmarks."""
        summary = bulk_benchmark.get_summary()
        return summary
    
    @app.post("/api/v1/bulk/benchmark/compare")
    async def compare_benchmarks(request: Dict[str, Any]):
        """Comparar dos operaciones."""
        op1 = request.get("operation1")
        op2 = request.get("operation2")
        
        if not op1 or not op2:
            raise HTTPException(status_code=400, detail="Both operations required")
        
        comparison = bulk_benchmark.compare_operations(op1, op2)
        return comparison
    
    @app.post("/api/v1/bulk/autotune/batch-size")
    async def autotune_batch_size(request: Dict[str, Any]):
        """Auto-ajustar tamaño de batch."""
        min_batch = request.get("min_batch", 10)
        max_batch = request.get("max_batch", 1000)
        step = request.get("step", 50)
        iterations = request.get("iterations", 3)
        test_data = request.get("test_data", [])
        
        # Nota: Necesita implementación de ejecución de operación
        return {
            "success": True,
            "message": "Auto-tune endpoint - implement operation execution",
            "range": {"min": min_batch, "max": max_batch, "step": step}
        }
    
    @app.post("/api/v1/bulk/autotune/workers")
    async def autotune_workers(request: Dict[str, Any]):
        """Auto-ajustar número de workers."""
        min_workers = request.get("min_workers", 1)
        max_workers = request.get("max_workers", 50)
        step = request.get("step", 5)
        test_data = request.get("test_data", [])
        
        # Nota: Necesita implementación de ejecución de operación
        return {
            "success": True,
            "message": "Auto-tune workers endpoint - implement operation execution",
            "range": {"min": min_workers, "max": max_workers, "step": step}
        }
    
    @app.get("/api/v1/bulk/autotune/recommendations")
    async def get_tuning_recommendations():
        """Obtener recomendaciones de tuning."""
        recommendations = bulk_auto_tuner.get_tuning_recommendations()
        return recommendations
    
    # Sistemas avanzados de resiliencia y observabilidad
    from ..core.bulk_operations import (
        BulkAdaptiveRateLimiter,
        BulkLoadBalancer,
        BulkLoadPredictor,
        BulkAutoScaler,
        BulkEventSourcing,
        BulkObservability,
        BulkCostOptimizer,
        BulkAnomalyDetector
    )
    
    bulk_adaptive_rate_limiter = BulkAdaptiveRateLimiter(
        initial_rate=100,
        min_rate=10,
        max_rate=1000
    )
    bulk_load_balancer = BulkLoadBalancer(initial_workers=10)
    bulk_load_predictor = BulkLoadPredictor(window_size=60)
    bulk_auto_scaler = BulkAutoScaler(
        load_balancer=bulk_load_balancer,
        load_predictor=bulk_load_predictor,
        metrics=bulk_realtime_metrics,
        min_workers=1,
        max_workers=100
    )
    
    @app.get("/api/v1/bulk/rate-limiter/status")
    async def get_rate_limiter_status():
        """Obtener estado del rate limiter adaptativo."""
        return bulk_adaptive_rate_limiter.get_status()
    
    @app.post("/api/v1/bulk/rate-limiter/check")
    async def check_rate_limit():
        """Verificar si se puede procesar una petición."""
        can_proceed = bulk_adaptive_rate_limiter.can_proceed()
        return {
            "can_proceed": can_proceed,
            "status": bulk_adaptive_rate_limiter.get_status()
        }
    
    @app.post("/api/v1/bulk/rate-limiter/record")
    async def record_rate_limit_result(request: Dict[str, Any]):
        """Registrar resultado de una petición."""
        success = request.get("success", True)
        response_time = request.get("response_time", 0.0)
        
        if success:
            bulk_adaptive_rate_limiter.record_success(response_time)
        else:
            bulk_adaptive_rate_limiter.record_failure()
        
        return {"success": True, "status": bulk_adaptive_rate_limiter.get_status()}
    
    @app.get("/api/v1/bulk/load-balancer/stats")
    async def get_load_balancer_stats():
        """Obtener estadísticas del load balancer."""
        return bulk_load_balancer.get_stats()
    
    @app.post("/api/v1/bulk/load-balancer/select-worker")
    async def select_worker():
        """Seleccionar worker para nueva tarea."""
        worker_id = bulk_load_balancer.select_worker()
        bulk_load_balancer.assign_task(worker_id)
        return {
            "worker_id": worker_id,
            "stats": bulk_load_balancer.get_stats()
        }
    
    @app.post("/api/v1/bulk/load-balancer/complete-task")
    async def complete_load_balancer_task(request: Dict[str, Any]):
        """Marcar tarea como completada."""
        worker_id = request.get("worker_id")
        success = request.get("success", True)
        response_time = request.get("response_time", 0.0)
        
        if worker_id is None:
            raise HTTPException(status_code=400, detail="worker_id required")
        
        bulk_load_balancer.complete_task(worker_id, success, response_time)
        return {"success": True, "stats": bulk_load_balancer.get_stats()}
    
    @app.post("/api/v1/bulk/load-balancer/add-worker")
    async def add_load_balancer_worker():
        """Añadir nuevo worker al load balancer."""
        worker_id = bulk_load_balancer.add_worker()
        return {
            "success": True,
            "worker_id": worker_id,
            "stats": bulk_load_balancer.get_stats()
        }
    
    @app.get("/api/v1/bulk/load-predictor/predict")
    async def predict_load(minutes_ahead: int = 5):
        """Predecir carga futura."""
        prediction = bulk_load_predictor.predict_load(minutes_ahead)
        return prediction
    
    @app.get("/api/v1/bulk/load-predictor/pattern")
    async def get_load_pattern():
        """Obtener patrón de carga por hora."""
        pattern = bulk_load_predictor.get_load_pattern()
        return pattern
    
    @app.post("/api/v1/bulk/load-predictor/record")
    async def record_load(request: Dict[str, Any]):
        """Registrar carga actual."""
        load = request.get("load", 0.0)
        bulk_load_predictor.record_load(load)
        return {"success": True}
    
    @app.get("/api/v1/bulk/autoscaler/evaluate")
    async def evaluate_scaling():
        """Evaluar si es necesario hacer scaling."""
        evaluation = bulk_auto_scaler.evaluate_scaling()
        return evaluation
    
    @app.post("/api/v1/bulk/autoscaler/execute")
    async def execute_scaling():
        """Ejecutar scaling si es necesario."""
        result = await bulk_auto_scaler.execute_scaling()
        return result
    
    @app.get("/api/v1/bulk/autoscaler/history")
    async def get_scaling_history():
        """Obtener historial de scaling."""
        history = bulk_auto_scaler.get_scaling_history()
        return {"history": history, "count": len(history)}
    
    # Instancias de sistemas de observabilidad (ya importadas arriba)
    bulk_event_sourcing = BulkEventSourcing()
    bulk_observability = BulkObservability()
    bulk_cost_optimizer = BulkCostOptimizer()
    bulk_anomaly_detector = BulkAnomalyDetector(threshold_std=2.0)
    
    @app.post("/api/v1/bulk/events/record")
    async def record_event(request: Dict[str, Any]):
        """Registrar evento en event sourcing."""
        event_type = request.get("event_type")
        aggregate_id = request.get("aggregate_id")
        payload = request.get("payload", {})
        metadata = request.get("metadata", {})
        
        if not event_type or not aggregate_id:
            raise HTTPException(status_code=400, detail="event_type and aggregate_id required")
        
        bulk_event_sourcing.record_event(event_type, aggregate_id, payload, metadata)
        return {"success": True, "message": "Event recorded"}
    
    @app.get("/api/v1/bulk/events")
    async def get_events(
        aggregate_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ):
        """Obtener eventos."""
        events = bulk_event_sourcing.get_events(aggregate_id, event_type, limit)
        return {"events": events, "count": len(events)}
    
    @app.post("/api/v1/bulk/traces/start")
    async def start_trace(request: Dict[str, Any]):
        """Iniciar trace distribuido."""
        trace_id = request.get("trace_id")
        operation_name = request.get("operation_name", "unknown")
        metadata = request.get("metadata", {})
        
        if not trace_id:
            raise HTTPException(status_code=400, detail="trace_id required")
        
        bulk_observability.start_trace(trace_id, operation_name, metadata)
        return {"success": True, "trace_id": trace_id}
    
    @app.post("/api/v1/bulk/traces/add-span")
    async def add_trace_span(request: Dict[str, Any]):
        """Añadir span a trace."""
        trace_id = request.get("trace_id")
        span_name = request.get("span_name")
        duration = request.get("duration", 0.0)
        metadata = request.get("metadata", {})
        
        if not trace_id or not span_name:
            raise HTTPException(status_code=400, detail="trace_id and span_name required")
        
        bulk_observability.add_span(trace_id, span_name, duration, metadata)
        return {"success": True}
    
    @app.post("/api/v1/bulk/traces/complete")
    async def complete_trace(request: Dict[str, Any]):
        """Completar trace."""
        trace_id = request.get("trace_id")
        status = request.get("status", "success")
        error = request.get("error")
        
        if not trace_id:
            raise HTTPException(status_code=400, detail="trace_id required")
        
        bulk_observability.complete_trace(trace_id, status, error)
        return {"success": True}
    
    @app.get("/api/v1/bulk/traces/{trace_id}")
    async def get_trace(trace_id: str):
        """Obtener trace completo."""
        trace = bulk_observability.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace
    
    @app.get("/api/v1/bulk/observability/summary")
    async def get_observability_summary():
        """Obtener resumen de observabilidad."""
        return bulk_observability.get_observability_summary()
    
    @app.post("/api/v1/bulk/observability/log")
    async def log_event(request: Dict[str, Any]):
        """Registrar evento de log."""
        level = request.get("level", "info")
        message = request.get("message", "")
        context = request.get("context", {})
        trace_id = request.get("trace_id")
        
        bulk_observability.log_event(level, message, context, trace_id)
        return {"success": True}
    
    @app.get("/api/v1/bulk/observability/logs")
    async def get_observability_logs(
        level: Optional[str] = None,
        trace_id: Optional[str] = None,
        limit: int = 100
    ):
        """Obtener logs."""
        logs = bulk_observability.get_logs(level, trace_id, limit)
        return {"logs": logs, "count": len(logs)}
    
    @app.post("/api/v1/bulk/costs/record")
    async def record_cost(request: Dict[str, Any]):
        """Registrar costo de operación."""
        operation_type = request.get("operation_type")
        cost = request.get("cost", 0.0)
        items_processed = request.get("items_processed", 1)
        duration = request.get("duration", 0.0)
        metadata = request.get("metadata", {})
        
        if not operation_type:
            raise HTTPException(status_code=400, detail="operation_type required")
        
        bulk_cost_optimizer.record_operation_cost(
            operation_type, cost, items_processed, duration, metadata
        )
        return {"success": True}
    
    @app.get("/api/v1/bulk/costs/summary")
    async def get_cost_summary():
        """Obtener resumen de costos."""
        return bulk_cost_optimizer.get_cost_summary()
    
    @app.get("/api/v1/bulk/costs/optimizations")
    async def get_cost_optimizations(operation_type: Optional[str] = None):
        """Obtener sugerencias de optimización."""
        suggestions = bulk_cost_optimizer.suggest_optimizations(operation_type)
        return {"suggestions": suggestions, "count": len(suggestions)}
    
    @app.post("/api/v1/bulk/anomalies/record")
    async def record_anomaly_metric(request: Dict[str, Any]):
        """Registrar métrica para detección de anomalías."""
        metric_name = request.get("metric_name")
        value = request.get("value")
        metadata = request.get("metadata", {})
        
        if metric_name is None or value is None:
            raise HTTPException(status_code=400, detail="metric_name and value required")
        
        bulk_anomaly_detector.record_metric(metric_name, value, metadata)
        return {"success": True}
    
    @app.get("/api/v1/bulk/anomalies")
    async def get_anomalies(
        metric_name: Optional[str] = None,
        limit: int = 100
    ):
        """Obtener anomalías detectadas."""
        anomalies = bulk_anomaly_detector.get_anomalies(metric_name, limit)
        return {"anomalies": anomalies, "count": len(anomalies)}
    
    @app.get("/api/v1/bulk/anomalies/summary")
    async def get_anomaly_summary():
        """Obtener resumen de anomalías."""
        return bulk_anomaly_detector.get_anomaly_summary()
    
    # Sistemas enterprise
    from ..core.bulk_operations import (
        BulkWorkflowEngine,
        BulkMultiTenancy,
        BulkDisasterRecovery,
        BulkComplianceAudit,
        BulkMLOptimizer
    )
    
    bulk_workflow_engine = BulkWorkflowEngine()
    bulk_multi_tenancy = BulkMultiTenancy()
    bulk_disaster_recovery = BulkDisasterRecovery(backup_interval_minutes=60)
    bulk_compliance_audit = BulkComplianceAudit()
    bulk_ml_optimizer = BulkMLOptimizer()
    
    @app.post("/api/v1/bulk/workflows/register")
    async def register_workflow(request: Dict[str, Any]):
        """Registrar workflow."""
        workflow_id = request.get("workflow_id")
        steps = request.get("steps", [])
        name = request.get("name", "")
        description = request.get("description", "")
        
        if not workflow_id:
            raise HTTPException(status_code=400, detail="workflow_id required")
        
        bulk_workflow_engine.register_workflow(workflow_id, steps, name, description)
        return {"success": True, "workflow_id": workflow_id}
    
    @app.post("/api/v1/bulk/workflows/execute")
    async def execute_workflow(request: Dict[str, Any]):
        """Ejecutar workflow."""
        workflow_id = request.get("workflow_id")
        initial_data = request.get("initial_data", {})
        execution_id = request.get("execution_id")
        
        if not workflow_id:
            raise HTTPException(status_code=400, detail="workflow_id required")
        
        execution = await bulk_workflow_engine.execute_workflow(
            workflow_id, initial_data, execution_id
        )
        return execution
    
    @app.get("/api/v1/bulk/workflows/{workflow_id}")
    async def get_workflow(workflow_id: str):
        """Obtener workflow."""
        workflow = bulk_workflow_engine.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return workflow
    
    @app.get("/api/v1/bulk/workflows/executions/{execution_id}")
    async def get_workflow_execution(execution_id: str):
        """Obtener ejecución de workflow."""
        execution = bulk_workflow_engine.get_execution(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        return execution
    
    @app.post("/api/v1/bulk/tenants/register")
    async def register_tenant(request: Dict[str, Any]):
        """Registrar tenant."""
        tenant_id = request.get("tenant_id")
        name = request.get("name", "")
        config = request.get("config", {})
        
        if not tenant_id:
            raise HTTPException(status_code=400, detail="tenant_id required")
        
        bulk_multi_tenancy.register_tenant(tenant_id, name, config)
        return {"success": True, "tenant_id": tenant_id}
    
    @app.get("/api/v1/bulk/tenants/{tenant_id}/stats")
    async def get_tenant_stats(tenant_id: str):
        """Obtener estadísticas del tenant."""
        stats = bulk_multi_tenancy.get_tenant_stats(tenant_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Tenant not found")
        return stats
    
    @app.post("/api/v1/bulk/tenants/{tenant_id}/check-quota")
    async def check_tenant_quota(request: Dict[str, Any], tenant_id: str):
        """Verificar quota del tenant."""
        resource_type = request.get("resource_type")
        amount = request.get("amount", 1)
        
        if not resource_type:
            raise HTTPException(status_code=400, detail="resource_type required")
        
        can_proceed, error = bulk_multi_tenancy.check_quota(tenant_id, resource_type, amount)
        return {"can_proceed": can_proceed, "error": error}
    
    @app.post("/api/v1/bulk/recovery/checkpoint")
    async def create_recovery_checkpoint(request: Dict[str, Any]):
        """Crear checkpoint de recovery."""
        checkpoint_id = request.get("checkpoint_id")
        state = request.get("state", {})
        metadata = request.get("metadata", {})
        
        if not checkpoint_id:
            raise HTTPException(status_code=400, detail="checkpoint_id required")
        
        bulk_disaster_recovery.create_checkpoint(checkpoint_id, state, metadata)
        return {"success": True, "checkpoint_id": checkpoint_id}
    
    @app.get("/api/v1/bulk/recovery/status")
    async def get_recovery_status():
        """Obtener estado de disaster recovery."""
        return bulk_disaster_recovery.get_recovery_status()
    
    @app.get("/api/v1/bulk/recovery/checkpoints/{checkpoint_id}")
    async def get_checkpoint(checkpoint_id: str):
        """Obtener checkpoint."""
        checkpoint = bulk_disaster_recovery.get_checkpoint(checkpoint_id)
        if not checkpoint:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        return checkpoint
    
    @app.post("/api/v1/bulk/compliance/add-rule")
    async def add_compliance_rule(request: Dict[str, Any]):
        """Añadir regla de compliance."""
        rule_id = request.get("rule_id")
        rule_name = request.get("rule_name", "")
        severity = request.get("severity", "medium")
        # Nota: validator function necesitaría ser deserializado en producción
        
        if not rule_id:
            raise HTTPException(status_code=400, detail="rule_id required")
        
        return {
            "success": True,
            "message": "Rule registration endpoint - implement validator function registration"
        }
    
    @app.post("/api/v1/bulk/compliance/audit")
    async def audit_operation(request: Dict[str, Any]):
        """Registrar operación en auditoría."""
        operation_type = request.get("operation_type")
        user_id = request.get("user_id")
        details = request.get("details", {})
        result = request.get("result")
        
        if not operation_type or not user_id:
            raise HTTPException(status_code=400, detail="operation_type and user_id required")
        
        bulk_compliance_audit.audit_operation(operation_type, user_id, details, result)
        return {"success": True}
    
    @app.get("/api/v1/bulk/compliance/logs")
    async def get_compliance_logs(
        user_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        limit: int = 100
    ):
        """Obtener logs de compliance."""
        logs = bulk_compliance_audit.get_audit_logs(user_id, operation_type, limit)
        return {"logs": logs, "count": len(logs)}
    
    @app.get("/api/v1/bulk/compliance/report")
    async def get_compliance_report():
        """Obtener reporte de compliance."""
        return bulk_compliance_audit.get_compliance_report()
    
    @app.post("/api/v1/bulk/ml/record-training")
    async def record_training_data(request: Dict[str, Any]):
        """Registrar datos de entrenamiento."""
        model_name = request.get("model_name")
        features = request.get("features", {})
        target = request.get("target")
        metadata = request.get("metadata", {})
        
        if not model_name or target is None:
            raise HTTPException(status_code=400, detail="model_name and target required")
        
        bulk_ml_optimizer.record_training_data(model_name, features, target, metadata)
        return {"success": True}
    
    @app.post("/api/v1/bulk/ml/train")
    async def train_ml_model(request: Dict[str, Any]):
        """Entrenar modelo ML."""
        model_name = request.get("model_name")
        model_type = request.get("model_type", "linear_regression")
        
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name required")
        
        result = bulk_ml_optimizer.train_model(model_name, model_type)
        return result
    
    @app.post("/api/v1/bulk/ml/predict")
    async def ml_predict(request: Dict[str, Any]):
        """Hacer predicción con modelo ML."""
        model_name = request.get("model_name")
        features = request.get("features", {})
        
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name required")
        
        prediction = bulk_ml_optimizer.predict(model_name, features)
        return prediction
    
    @app.get("/api/v1/bulk/ml/models/{model_name}/stats")
    async def get_ml_model_stats(model_name: str):
        """Obtener estadísticas del modelo ML."""
        stats = bulk_ml_optimizer.get_model_stats(model_name)
        if not stats:
            raise HTTPException(status_code=404, detail="Model not found")
        return stats
    
    # Guardar referencias para acceso externo
    app.state.chat_engine = chat_engine
    app.state.analyzer = analyzer
    app.state.exporter = exporter
    app.state.webhook_manager = webhook_manager
    app.state.template_manager = template_manager
    app.state.auth_manager = auth_manager
    app.state.backup_manager = backup_manager
    app.state.performance_optimizer = performance_optimizer
    app.state.health_monitor = health_monitor
    app.state.task_queue = task_queue
    app.state.alert_manager = alert_manager
    app.state.cluster_manager = cluster_manager
    app.state.feature_flags = feature_flags
    app.state.api_versioning = api_versioning
    app.state.advanced_analytics = advanced_analytics
    app.state.recommendation_engine = recommendation_engine
    app.state.ab_testing = ab_testing
    app.state.event_bus = event_bus
    app.state.security_manager = security_manager
    app.state.i18n_manager = i18n_manager
    app.state.workflow_engine = workflow_engine
    app.state.notification_manager = notification_manager
    app.state.integration_manager = integration_manager
    app.state.benchmark_runner = benchmark_runner
    app.state.api_docs_generator = api_docs_generator
    app.state.advanced_monitoring = advanced_monitoring
    app.state.secrets_manager = secrets_manager
    app.state.ml_optimizer = ml_optimizer
    app.state.deployment_manager = deployment_manager
    app.state.report_generator = report_generator
    app.state.user_manager = user_manager
    app.state.search_engine = search_engine
    app.state.message_queue = message_queue
    app.state.validation_engine = validation_engine
    app.state.throttler = throttler
    app.state.circuit_breaker = circuit_breaker
    app.state.intelligent_optimizer = intelligent_optimizer
    app.state.adaptive_learning = adaptive_learning
    app.state.demand_predictor = demand_predictor
    app.state.intelligent_health = intelligent_health
    app.state.predictive_scaler = predictive_scaler
    app.state.cost_optimizer = cost_optimizer
    app.state.intelligent_alerts = intelligent_alerts
    app.state.advanced_observability = advanced_observability
    app.state.load_balancer = load_balancer
    app.state.resource_manager = resource_manager
    app.state.disaster_recovery = disaster_recovery
    app.state.advanced_security = advanced_security
    app.state.auto_optimizer = auto_optimizer
    app.state.predictive_analytics = predictive_analytics
    app.state.policy_engine = policy_engine
    app.state.audit_system = audit_system
    app.state.task_orchestrator = task_orchestrator
    app.state.resource_allocator = resource_allocator
    app.state.service_orchestrator = service_orchestrator
    app.state.performance_profiler = performance_profiler
    app.state.adaptive_rate_controller = adaptive_rate_controller
    app.state.smart_retry_manager = smart_retry_manager
    app.state.distributed_lock_manager = distributed_lock_manager
    app.state.data_pipeline_manager = data_pipeline_manager
    app.state.event_scheduler = event_scheduler
    app.state.graceful_degradation_manager = graceful_degradation_manager
    app.state.cache_warmer = cache_warmer
    app.state.load_shedder = load_shedder
    app.state.conflict_resolver = conflict_resolver
    app.state.state_machine_manager = state_machine_manager
    app.state.workflow_engine_v2 = workflow_engine_v2
    app.state.event_bus = event_bus
    app.state.feature_toggle_manager = feature_toggle_manager
    app.state.rate_limiter_v2 = rate_limiter_v2
    app.state.circuit_breaker_v2 = circuit_breaker_v2
    app.state.adaptive_optimizer = adaptive_optimizer
    app.state.health_checker_v2 = health_checker_v2
    app.state.auto_scaler = auto_scaler
    app.state.batch_processor = batch_processor
    app.state.performance_monitor = performance_monitor
    app.state.queue_manager = queue_manager
    app.state.connection_manager = connection_manager
    app.state.transaction_manager = transaction_manager
    app.state.saga_orchestrator = saga_orchestrator
    app.state.distributed_coordinator = distributed_coordinator
    app.state.service_mesh = service_mesh
    app.state.adaptive_throttler = adaptive_throttler
    app.state.backpressure_manager = backpressure_manager
    app.state.federated_learning = federated_learning
    app.state.knowledge_manager = knowledge_manager
    app.state.auto_generator = auto_generator
    app.state.architecture_recommender = architecture_recommender
    app.state.mlops_manager = mlops_manager
    app.state.dependency_manager = dependency_manager
    app.state.cicd_manager = cicd_manager
    app.state.code_quality = code_quality
    app.state.business_metrics = business_metrics
    app.state.version_control = version_control
    app.state.log_analyzer = log_analyzer
    app.state.api_performance = api_performance
    app.state.advanced_secrets = advanced_secrets
    app.state.intelligent_cache = intelligent_cache
    app.state.sentiment_analyzer = sentiment_analyzer
    app.state.task_manager = task_manager
    app.state.resource_monitor = resource_monitor
    app.state.push_notifications = push_notifications
    app.state.distributed_sync = distributed_sync
    app.state.query_analyzer = query_analyzer
    app.state.file_manager = file_manager
    app.state.data_compression = data_compression
    app.state.incremental_backup = incremental_backup
    app.state.network_analyzer = network_analyzer
    app.state.config_manager = config_manager
    app.state.mfa_authentication = mfa_authentication
    app.state.advanced_rate_limiter = advanced_rate_limiter
    app.state.user_behavior_analyzer = user_behavior_analyzer
    app.state.event_stream = event_stream
    app.state.security_analyzer = security_analyzer
    app.state.session_manager = session_manager
    app.state.realtime_metrics = realtime_metrics
    app.state.auto_optimizer = auto_optimizer
    app.state.bulk_sessions = bulk_sessions
    app.state.bulk_messages = bulk_messages
    app.state.bulk_exporter = bulk_exporter
    app.state.bulk_analytics = bulk_analytics
    app.state.bulk_cleanup = bulk_cleanup
    app.state.bulk_processor = bulk_processor
    app.state.bulk_importer = bulk_importer
    app.state.bulk_notifications = bulk_notifications
    app.state.bulk_search = bulk_search
    app.state.bulk_backup = bulk_backup
    app.state.bulk_migration = bulk_migration
    app.state.bulk_metrics = bulk_metrics
    app.state.bulk_scheduler = bulk_scheduler
    app.state.bulk_rate_limiter = bulk_rate_limiter
    app.state.bulk_auto_creator = bulk_auto_creator
    app.state.bulk_auto_expander = bulk_auto_expander
    app.state.bulk_auto_processor = bulk_auto_processor
    app.state.bulk_auto_maintainer = bulk_auto_maintainer
    app.state.bulk_infinite_generator = bulk_infinite_generator
    app.state.bulk_self_sustaining = bulk_self_sustaining
    app.state.bulk_realtime_metrics = bulk_realtime_metrics
    app.state.bulk_advanced_cache = bulk_advanced_cache
    app.state.bulk_priority_queue = bulk_priority_queue
    app.state.bulk_enhanced_validator = bulk_enhanced_validator
    app.state.bulk_dashboard = bulk_dashboard
    app.state.bulk_benchmark = bulk_benchmark
    app.state.bulk_auto_tuner = bulk_auto_tuner
    app.state.bulk_adaptive_rate_limiter = bulk_adaptive_rate_limiter
    app.state.bulk_load_balancer = bulk_load_balancer
    app.state.bulk_load_predictor = bulk_load_predictor
    app.state.bulk_auto_scaler = bulk_auto_scaler
    app.state.bulk_event_sourcing = bulk_event_sourcing
    app.state.bulk_observability = bulk_observability
    app.state.bulk_cost_optimizer = bulk_cost_optimizer
    app.state.bulk_anomaly_detector = bulk_anomaly_detector
    
    # Cleanup al cerrar
    @app.on_event("shutdown")
    async def shutdown_event():
        await integration_manager.close()
    
    return app


class ChatAPI:
    """Wrapper para la API de chat."""
    
    def __init__(self, config: Optional[ChatConfig] = None):
        self.config = config or ChatConfig()
        self.app = create_chat_app(self.config)
        self.chat_engine = self.app.state.chat_engine
    
    def run(self, host: str = "0.0.0.0", port: int = 8006, **kwargs):
        """Ejecutar el servidor API."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level=self.config.log_level.lower(),
            **kwargs
        )

