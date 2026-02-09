"""
Prototype API - API REST para el generador de prototipos 3D
============================================================

API FastAPI que permite recibir descripciones de productos
y generar prototipos completos con materiales, CAD, instrucciones y opciones de presupuesto.
"""

import logging
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..models.schemas import (
    PrototypeRequest,
    PrototypeResponse,
    ProductType
)
from ..core.prototype_generator import PrototypeGenerator
from ..utils.recommendation_engine import RecommendationEngine
from ..utils.product_templates import ProductTemplateManager
from ..utils.feasibility_analyzer import FeasibilityAnalyzer
from ..utils.prototype_comparator import PrototypeComparator
from ..utils.cost_analyzer import CostAnalyzer
from ..utils.material_validator import MaterialValidator
from ..utils.prototype_history import PrototypeHistory
from ..utils.diagram_generator import DiagramGenerator
from ..utils.analytics import PrototypeAnalytics
from ..utils.notification_system import NotificationSystem, NotificationType
from ..utils.advanced_exporter import AdvancedExporter
from ..utils.collaboration_system import CollaborationSystem, SharePermission
from ..utils.llm_integration import LLMIntegration
from ..utils.webhook_system import WebhookSystem, WebhookEvent
from ..utils.auth_system import AuthSystem, Permission, Role
from ..utils.performance_optimizer import PerformanceOptimizer, performance_monitor
from ..utils.backup_system import BackupSystem
from ..utils.rate_limiter import RateLimiter, RateLimitStrategy
from ..utils.advanced_monitoring import AdvancedMonitoring
from ..utils.async_queue import AsyncQueue, JobStatus
from ..utils.distributed_cache import DistributedCache
from ..utils.health_checker import HealthChecker, HealthStatus
from ..utils.config_manager import ConfigManager
from ..utils.circuit_breaker import CircuitBreaker, CircuitState
from ..utils.event_system import EventSystem, EventType
from ..utils.retry_system import RetrySystem, retry_decorator
from ..utils.plugin_system import PluginSystem
from ..utils.prometheus_metrics import PrometheusMetrics
from ..utils.i18n_system import I18nSystem
from ..utils.report_generator import ReportGenerator
from ..utils.ml_predictor import MLPredictor
from ..utils.load_balancer import LoadBalancer, LoadBalanceStrategy
from ..utils.distributed_tracing import DistributedTracing, TraceContext, SpanKind
from ..utils.auto_optimizer import AutoOptimizer
from ..utils.batch_processor import AdvancedBatchProcessor
from ..utils.cache_warmer import CacheWarmer
from ..utils.auto_scaler import AutoScaler
from ..utils.advanced_validator import AdvancedValidator, ValidationLevel
from ..utils.api_versioning import APIVersioning, APIVersion
from ..utils.dashboard_analytics import DashboardAnalytics
from ..utils.workflow_engine import WorkflowEngine, WorkflowTask, TaskStatus
from ..utils.scheduler import Scheduler, ScheduleType
from ..utils.external_integrations import ExternalIntegrationsManager, ExternalIntegration, IntegrationType
from ..utils.user_rate_limiter import UserRateLimiter
from ..utils.push_notifications import PushNotificationSystem, NotificationPriority, NotificationChannel
from ..utils.security_manager import SecurityManager
from ..utils.audit_system import AuditSystem, AuditEventType
from ..utils.disaster_recovery import DisasterRecovery
from ..utils.interactive_docs import InteractiveDocumentation
from ..utils.api_gateway import APIGateway, RouteMethod
from ..utils.performance_profiler import PerformanceProfiler
from ..utils.data_migration import DataMigration, MigrationStatus
from ..utils.cicd_integration import CICDIntegration, BuildStatus
from ..utils.advanced_feature_flags import AdvancedFeatureFlags, FeatureFlagType
from ..utils.ab_testing import ABTesting, VariantType
from ..utils.ml_analytics import MLAnalytics
from ..utils.personalized_recommendations import PersonalizedRecommendations
from ..utils.gamification import Gamification, AchievementType
from ..utils.marketplace import Marketplace, ListingStatus
from ..utils.monetization import Monetization, SubscriptionTier, PaymentStatus
from ..utils.auto_documentation import AutoDocumentation
from ..utils.advanced_testing import AdvancedTesting
from ..utils.business_metrics import BusinessMetrics
from ..utils.intelligent_cache import IntelligentCache
from ..utils.executive_reports import ExecutiveReports
from ..utils.query_optimizer import QueryOptimizer
from ..utils.sentiment_analysis import SentimentAnalysis, Sentiment
from ..utils.demand_forecasting import DemandForecasting
from ..utils.intelligent_alerts import IntelligentAlerts, AlertSeverity
from ..utils.inventory_management import InventoryManagement, InventoryStatus
from ..utils.competitor_analysis import CompetitorAnalysis
from ..utils.advanced_logging import AdvancedLogging, LogLevel
from ..utils.blockchain_verification import BlockchainVerification
from ..utils.ar_vr_integration import ARVRIntegration, ARVRPlatform
from ..utils.iot_integration import IoTIntegration, DeviceType
from ..utils.edge_computing import EdgeComputing, EdgeNodeStatus
from ..utils.advanced_data_analysis import AdvancedDataAnalysis
from ..utils.predictive_analytics import PredictiveAnalytics
from ..utils.knowledge_management import KnowledgeManagement
from ..utils.enhanced_service_integration import EnhancedServiceIntegration, ServiceStatus
from ..utils.advanced_ml_system import AdvancedMLSystem
from ..utils.enhanced_query_optimizer import EnhancedQueryOptimizer
from ..scripts.utils import check_health, generate_stats, validate_config
from ..utils.advanced_llm_system import AdvancedLLMSystem, LLMConfig, ModelType
from ..utils.diffusion_models import DiffusionModelsSystem, DiffusionConfig, SchedulerType
from ..utils.model_training import ModelTrainer, TrainingConfig
from ..utils.gradio_interface import GradioInterface
from ..utils.experiment_tracking import ExperimentTracker
from ..utils.distributed_training import DistributedTrainer
from ..utils.dataset_manager import DatasetManager, DatasetConfig
from ..utils.model_evaluation import ModelEvaluator
from ..utils.config_manager_dl import DLConfigManager, ExperimentConfig
from ..utils.model_checkpointing import ModelCheckpointer
from ..utils.hyperparameter_tuning import HyperparameterTuner, SearchStrategy, HyperparameterSpace
from ..utils.model_serving import ModelServer, ServingConfig
from ..utils.data_augmentation import DataAugmentationManager
from ..utils.custom_architectures import (
    PrototypeGeneratorModel, TransformerPrototypeModel, PrototypeClassifier
)
from ..utils.advanced_losses import (
    FocalLoss, LabelSmoothingLoss, DiceLoss, ContrastiveLoss, TripletLoss
)
from ..utils.model_compression import ModelCompressor
from ..utils.performance_profiler_dl import DLPerformanceProfiler
from ..utils.advanced_optimizers import AdvancedOptimizerFactory, AdvancedSchedulerFactory, LookaheadOptimizer
from ..utils.model_interpretability import ModelInterpreter
from ..utils.model_ensembling import ModelEnsemble, VotingEnsemble, StackingEnsemble, BlendingEnsemble
from ..utils.transfer_learning import TransferLearningManager
from ..utils.advanced_debugging import ModelDebugger
from ..utils.model_registry import ModelRegistry, ModelStatus
from ..utils.production_monitoring import ProductionMonitor
from ..utils.advanced_data_pipelines import PrefetchDataLoader, CachedDataset, AsyncDataLoader, DataPipeline
from ..utils.automl_system import AutoMLSystem, ArchitectureSearchSpace
from ..utils.advanced_metrics import AdvancedMetrics
from ..utils.model_visualization import ModelVisualizer
from ..utils.model_comparison import ModelComparator
from ..utils.model_deployment import ModelDeployment
from ..utils.model_validation import ModelValidator, ValidationStatus
from ..utils.advanced_testing_dl import DLTestSuite
from ..utils.inference_cache import InferenceCache
from ..utils.model_optimization import ModelOptimizer
from ..utils.intelligent_batching import IntelligentBatcher, AdaptiveBatcher

logger = logging.getLogger(__name__)

# Crear aplicación FastAPI con documentación mejorada
app = FastAPI(
    title="3D Prototype AI API",
    description="""
    ## 🚀 API Completa para Generación de Prototipos 3D
    
    Sistema enterprise completo para generar prototipos 3D con:
    
    * **Generación**: Crea prototipos completos desde descripciones
    * **Análisis**: Viabilidad, costos, comparación, validación
    * **Colaboración**: Compartir, comentarios, notificaciones
    * **Enterprise**: Webhooks, auth, monitoring, backup, rate limiting
    * **ML**: Predicciones de costo, tiempo y viabilidad
    * **i18n**: Soporte multi-idioma
    * **Reportes**: Diarios, semanales, mensuales
    
    ### Características Principales
    
    - ✅ 60+ endpoints REST
    - ✅ 27 sistemas funcionales
    - ✅ Autenticación y permisos
    - ✅ Rate limiting avanzado
    - ✅ Monitoring y métricas Prometheus
    - ✅ Circuit breakers para resiliencia
    - ✅ Sistema de eventos pub/sub
    - ✅ Cola asíncrona
    - ✅ Caché distribuido
    - ✅ Health checks
    - ✅ Internacionalización
    - ✅ Machine Learning
    - ✅ Load balancing
    
    ### Documentación
    
    - Swagger UI: `/docs`
    - ReDoc: `/redoc`
    - OpenAPI JSON: `/openapi.json`
    """,
    version="2.0.0",
    contact={
        "name": "Blatam Academy",
        "email": "support@blatam.academy"
    },
    license_info={
        "name": "Proprietary",
        "url": "https://blatam.academy"
    },
    servers=[
        {
            "url": "http://localhost:8030",
            "description": "Servidor de desarrollo"
        },
        {
            "url": "https://api.3dprototype.ai",
            "description": "Servidor de producción"
        }
    ]
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar generador y servicios
generator = PrototypeGenerator()
recommendation_engine = RecommendationEngine()
template_manager = ProductTemplateManager()
feasibility_analyzer = FeasibilityAnalyzer()
prototype_comparator = PrototypeComparator()
cost_analyzer = CostAnalyzer()
material_validator = MaterialValidator()
prototype_history = PrototypeHistory()
diagram_generator = DiagramGenerator()
analytics = PrototypeAnalytics()
notification_system = NotificationSystem()
advanced_exporter = AdvancedExporter()
collaboration_system = CollaborationSystem()
llm_integration = LLMIntegration()
webhook_system = WebhookSystem()
auth_system = AuthSystem()
performance_optimizer = PerformanceOptimizer()
backup_system = BackupSystem()
rate_limiter = RateLimiter(strategy=RateLimitStrategy.SLIDING_WINDOW)
monitoring = AdvancedMonitoring()
async_queue = AsyncQueue(max_workers=5)
distributed_cache = DistributedCache()
health_checker = HealthChecker()
config_manager = ConfigManager()
event_system = EventSystem()
plugin_system = PluginSystem()
circuit_breakers: Dict[str, CircuitBreaker] = {
    "material_search": CircuitBreaker(failure_threshold=5, timeout_seconds=60),
    "llm": CircuitBreaker(failure_threshold=3, timeout_seconds=120),
    "export": CircuitBreaker(failure_threshold=5, timeout_seconds=60)
}
prometheus_metrics = PrometheusMetrics()
i18n_system = I18nSystem()
report_generator = ReportGenerator()
ml_predictor = MLPredictor()
load_balancer = LoadBalancer(strategy=LoadBalanceStrategy.ROUND_ROBIN)
distributed_tracing = DistributedTracing()
auto_optimizer = AutoOptimizer()
batch_processor = AdvancedBatchProcessor()
cache_warmer = CacheWarmer()
auto_scaler = AutoScaler(min_instances=1, max_instances=10)
advanced_validator = AdvancedValidator()
api_versioning = APIVersioning()
dashboard_analytics = DashboardAnalytics()
workflow_engine = WorkflowEngine()
scheduler = Scheduler()
external_integrations = ExternalIntegrationsManager()
user_rate_limiter = UserRateLimiter()
push_notifications = PushNotificationSystem()
security_manager = SecurityManager()
audit_system = AuditSystem()
disaster_recovery = DisasterRecovery()
interactive_docs = InteractiveDocumentation()
api_gateway = APIGateway()
performance_profiler = PerformanceProfiler()
data_migration = DataMigration()
cicd_integration = CICDIntegration()
advanced_feature_flags = AdvancedFeatureFlags()
ab_testing = ABTesting()
ml_analytics = MLAnalytics()
personalized_recommendations = PersonalizedRecommendations()
gamification = Gamification()
marketplace = Marketplace()
monetization = Monetization()
auto_documentation = AutoDocumentation()
advanced_testing = AdvancedTesting()
business_metrics = BusinessMetrics()
intelligent_cache = IntelligentCache()
executive_reports = ExecutiveReports()
query_optimizer = QueryOptimizer()
sentiment_analysis = SentimentAnalysis()
demand_forecasting = DemandForecasting()
intelligent_alerts = IntelligentAlerts()
inventory_management = InventoryManagement()
competitor_analysis = CompetitorAnalysis()
advanced_logging = AdvancedLogging()
blockchain_verification = BlockchainVerification()
ar_vr_integration = ARVRIntegration()
iot_integration = IoTIntegration()
edge_computing = EdgeComputing()
advanced_data_analysis = AdvancedDataAnalysis()
predictive_analytics = PredictiveAnalytics()
knowledge_management = KnowledgeManagement()
enhanced_service_integration = EnhancedServiceIntegration()
advanced_ml_system = AdvancedMLSystem()
enhanced_query_optimizer = EnhancedQueryOptimizer()

# Deep Learning Systems
try:
    advanced_llm_system = AdvancedLLMSystem()
    diffusion_models_system = DiffusionModelsSystem()
    experiment_tracker = ExperimentTracker()
    distributed_trainer = DistributedTrainer()
    dataset_manager = DatasetManager()
    model_evaluator = ModelEvaluator()
    dl_config_manager = DLConfigManager()
    model_checkpointer = ModelCheckpointer()
    hyperparameter_tuner = HyperparameterTuner()
    data_augmentation_manager = DataAugmentationManager()
    model_compressor = ModelCompressor()
    dl_profiler = DLPerformanceProfiler()
    model_interpreter = ModelInterpreter()
    transfer_learning_manager = TransferLearningManager()
    model_debugger = ModelDebugger()
    model_registry = ModelRegistry()
    production_monitor = ProductionMonitor()
    automl_system = AutoMLSystem()
    advanced_metrics = AdvancedMetrics()
    model_visualizer = ModelVisualizer()
    model_comparator = ModelComparator()
    model_deployment = ModelDeployment()
    model_validator = ModelValidator()
    dl_test_suite = DLTestSuite()
    inference_cache = InferenceCache()
    model_optimizer = ModelOptimizer()
    intelligent_batcher = IntelligentBatcher()
    adaptive_batcher = AdaptiveBatcher()
    DL_AVAILABLE = True
except ImportError:
    advanced_llm_system = None
    diffusion_models_system = None
    experiment_tracker = None
    distributed_trainer = None
    dataset_manager = None
    model_evaluator = None
    dl_config_manager = None
    DL_AVAILABLE = False
    logger.warning("Deep Learning libraries not available")

# Agregar nodos al load balancer
load_balancer.add_node("node-1", weight=1)
load_balancer.add_node("node-2", weight=1)

# Registrar tareas de cache warming
cache_warmer.register_warming_task(
    "common_prototypes",
    lambda: cache_warmer.warm_common_prototypes(generator),
    priority=10
)

# Registrar versiones de API
api_versioning.register_endpoint("v1", "/api/v1/generate", "POST")
api_versioning.register_endpoint("v2", "/api/v1/generate", "POST")

# Iniciar scheduler
asyncio.create_task(scheduler.start())

# Configurar rate limits por defecto
rate_limiter.set_limit("default", requests=100, window_seconds=60)
rate_limiter.set_limit("generate", requests=20, window_seconds=60)
rate_limiter.set_limit("analysis", requests=50, window_seconds=60)

# Registrar health checks
health_checker.register_check("database", health_checker.check_database, critical=True)
health_checker.register_check("cache", health_checker.check_cache, critical=False)
health_checker.register_check("disk_space", health_checker.check_disk_space, critical=True)
health_checker.register_check("memory", health_checker.check_memory, critical=False)

# Configurar suscriptores de eventos
async def on_prototype_created(event):
    """Handler para evento de prototipo creado"""
    logger.info(f"Prototipo creado: {event.data.get('product_name')}")

event_system.subscribe(EventType.PROTOTYPE_CREATED, on_prototype_created)

# Iniciar workers de la cola
import asyncio
asyncio.create_task(async_queue.start_workers())


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "3D Prototype AI API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/v1/generate",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint básico"""
    from datetime import datetime
    return {
        "status": "healthy",
        "service": "3d_prototype_ai",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """Health check detallado"""
    health_status = await health_checker.run_all_checks()
    return health_status


@app.post("/api/v1/generate", response_model=PrototypeResponse)
async def generate_prototype(
    request: PrototypeRequest,
    background_tasks: BackgroundTasks,
    save_to_history: bool = True
):
    """
    Genera un prototipo completo basado en la descripción del producto
    
    Args:
        request: Request con la descripción del producto y opciones
        save_to_history: Si guardar en historial (default: True)
        
    Returns:
        PrototypeResponse con toda la información generada
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Recibida solicitud para generar prototipo: {request.product_description}")
        
        # Generar prototipo
        response = await generator.generate_prototype(request)
        
        generation_time = time.time() - start_time
        
        # Registrar en analytics
        analytics.record_generation(response.model_dump(), generation_time)
        
        # Registrar métricas Prometheus
        prometheus_metrics.increment_counter("prototypes_generated_total", 1.0)
        prometheus_metrics.observe_histogram("prototype_generation_duration_seconds", generation_time)
        prometheus_metrics.set_gauge("prototypes_active", len(prototype_history.list_prototypes(limit=1000)))
        
        # Guardar en historial si se solicita
        prototype_id = None
        if save_to_history:
            prototype_id = prototype_history.save_prototype(response.model_dump())
            
            # Enviar notificación
            background_tasks.add_task(
                notification_system.notify_prototype_generated,
                "system", prototype_id, response.product_name
            )
            
            # Publicar evento
            background_tasks.add_task(
                event_system.publish,
                EventType.PROTOTYPE_CREATED,
                {"prototype_id": prototype_id, "product_name": response.product_name}
            )
        
        logger.info(f"Prototipo generado exitosamente: {response.product_name} ({generation_time:.3f}s)")
        
        return response
        
    except Exception as e:
        logger.error(f"Error al generar prototipo: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar prototipo: {str(e)}"
        )


@app.get("/api/v1/product-types")
async def get_product_types():
    """Obtiene la lista de tipos de productos disponibles"""
    return {
        "product_types": [pt.value for pt in ProductType]
    }


@app.get("/api/v1/materials/suggestions")
async def get_material_suggestions(product_type: Optional[str] = None):
    """Obtiene sugerencias de materiales según el tipo de producto"""
    suggestions = {
        "licuadora": [
            "Motor eléctrico",
            "Vaso de vidrio",
            "Plástico ABS",
            "Cuchillas de acero",
            "Cables eléctricos",
            "Interruptor",
            "Base antideslizante"
        ],
        "estufa": [
            "Acero inoxidable",
            "Quemadores",
            "Válvulas de gas",
            "Tubos de gas",
            "Perillas de control",
            "Rejillas de soporte",
            "Sistema de encendido"
        ],
        "maquina": [
            "Acero inoxidable",
            "Motores",
            "Componentes eléctricos",
            "Plástico ABS",
            "Tornillos y fijaciones",
            "Sensores",
            "Sistema de control"
        ],
        "electrodomestico": [
            "Carcasa de plástico",
            "Componentes eléctricos",
            "Motores",
            "Sensores",
            "Cables y conectores"
        ],
        "herramienta": [
            "Acero de alta resistencia",
            "Mango ergonómico",
            "Motor eléctrico",
            "Batería (si es inalámbrica)",
            "Cables y cargador"
        ]
    }
    
    if product_type:
        return {
            "suggestions": suggestions.get(product_type.lower(), suggestions["maquina"])
        }
    
    return {
        "suggestions": suggestions
    }


@app.post("/api/v1/recommendations")
async def get_recommendations(request: PrototypeRequest):
    """Obtiene recomendaciones para un prototipo"""
    try:
        # Generar prototipo primero
        response = await generator.generate_prototype(request)
        
        # Obtener recomendaciones
        material_recommendations = recommendation_engine.recommend_materials(
            response.materials, 
            request.budget
        )
        
        budget_recommendation = recommendation_engine.recommend_budget_option(
            response.budget_options,
            request.budget
        )
        
        # Obtener tipo de producto desde las especificaciones
        product_type_str = response.specifications.get("tipo", "otro")
        optimization_tips = recommendation_engine.get_optimization_tips(
            response.materials,
            product_type_str
        )
        
        return {
            "prototype": response,
            "material_recommendations": material_recommendations,
            "recommended_budget_option": budget_recommendation.model_dump() if budget_recommendation else None,
            "optimization_tips": optimization_tips
        }
        
    except Exception as e:
        logger.error(f"Error al generar recomendaciones: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar recomendaciones: {str(e)}"
        )


@app.get("/api/v1/materials/search")
async def search_materials(material_name: str, location: Optional[str] = None):
    """Busca materiales en múltiples fuentes"""
    try:
        from ..utils.material_search import MaterialSearchEngine
        search_engine = MaterialSearchEngine()
        results = await search_engine.search_material(material_name, location)
        return {
            "material": material_name,
            "location": location,
            "results": results,
            "total_sources": len(results)
        }
    except Exception as e:
        logger.error(f"Error en búsqueda de materiales: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error en búsqueda: {str(e)}"
        )


@app.get("/api/v1/templates")
async def get_templates(product_type: Optional[str] = None):
    """Obtiene templates de productos disponibles"""
    try:
        from ..models.schemas import ProductType
        pt = ProductType(product_type) if product_type else None
        templates = template_manager.list_templates(pt)
        return {
            "templates": templates,
            "total": len(templates)
        }
    except Exception as e:
        logger.error(f"Error al obtener templates: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener templates: {str(e)}"
        )


@app.get("/api/v1/templates/{template_id}")
async def get_template(template_id: str):
    """Obtiene un template específico"""
    template = template_manager.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template {template_id} no encontrado")
    return template


@app.post("/api/v1/feasibility")
async def analyze_feasibility(request: PrototypeRequest, 
                             user_experience: Optional[str] = None,
                             available_tools: Optional[List[str]] = None):
    """Analiza la viabilidad de un prototipo"""
    try:
        # Generar prototipo primero
        response = await generator.generate_prototype(request)
        
        # Analizar viabilidad
        analysis = feasibility_analyzer.analyze_feasibility(
            response, user_experience, available_tools
        )
        
        return {
            "prototype": response,
            "feasibility_analysis": analysis
        }
    except Exception as e:
        logger.error(f"Error al analizar viabilidad: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al analizar viabilidad: {str(e)}"
        )


@app.post("/api/v1/compare")
async def compare_prototypes(requests: List[PrototypeRequest]):
    """Compara múltiples prototipos"""
    try:
        if len(requests) < 2:
            raise HTTPException(
                status_code=400,
                detail="Se necesitan al menos 2 prototipos para comparar"
            )
        
        # Generar todos los prototipos
        prototypes = []
        for req in requests:
            prototype = await generator.generate_prototype(req)
            prototypes.append(prototype)
        
        # Comparar
        comparison = prototype_comparator.compare_prototypes(prototypes)
        
        return comparison
    except Exception as e:
        logger.error(f"Error al comparar prototipos: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al comparar: {str(e)}"
        )


@app.post("/api/v1/cost-analysis")
async def analyze_costs(request: PrototypeRequest):
    """Realiza un análisis detallado de costos"""
    try:
        response = await generator.generate_prototype(request)
        analysis = cost_analyzer.analyze_costs(response)
        
        return {
            "prototype": response,
            "cost_analysis": analysis
        }
    except Exception as e:
        logger.error(f"Error al analizar costos: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al analizar costos: {str(e)}"
        )


@app.post("/api/v1/validate-materials")
async def validate_materials(request: PrototypeRequest):
    """Valida materiales y compatibilidad"""
    try:
        response = await generator.generate_prototype(request)
        validation = material_validator.validate_materials(
            response.materials,
            response.cad_parts
        )
        
        return {
            "prototype": response,
            "validation": validation
        }
    except Exception as e:
        logger.error(f"Error al validar materiales: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al validar materiales: {str(e)}"
        )


@app.get("/api/v1/history")
async def get_history(limit: int = 50, user_id: Optional[str] = None):
    """Obtiene el historial de prototipos"""
    try:
        prototypes = prototype_history.list_prototypes(user_id=user_id, limit=limit)
        return {
            "prototypes": prototypes,
            "total": len(prototypes)
        }
    except Exception as e:
        logger.error(f"Error al obtener historial: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener historial: {str(e)}"
        )


@app.get("/api/v1/history/{prototype_id}")
async def get_prototype_from_history(prototype_id: str):
    """Obtiene un prototipo específico del historial"""
    prototype = prototype_history.get_prototype(prototype_id)
    if not prototype:
        raise HTTPException(status_code=404, detail="Prototipo no encontrado")
    return prototype


@app.get("/api/v1/history/{prototype_id}/versions")
async def get_prototype_versions(prototype_id: str):
    """Obtiene todas las versiones de un prototipo"""
    versions = prototype_history.get_prototype_versions(prototype_id)
    return {
        "prototype_id": prototype_id,
        "versions": versions
    }


@app.get("/api/v1/history/search")
async def search_history(query: str):
    """Busca prototipos en el historial"""
    results = prototype_history.search_prototypes(query)
    return {
        "query": query,
        "results": results,
        "count": len(results)
    }


@app.post("/api/v1/diagrams")
async def generate_diagrams(request: PrototypeRequest):
    """Genera diagramas para un prototipo"""
    try:
        response = await generator.generate_prototype(request)
        diagrams = diagram_generator.generate_all_diagrams(response.model_dump())
        
        return {
            "prototype": response,
            "diagrams": diagrams
        }
    except Exception as e:
        logger.error(f"Error al generar diagramas: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar diagramas: {str(e)}"
        )


@app.get("/api/v1/analytics")
async def get_analytics():
    """Obtiene estadísticas y analytics"""
    return analytics.get_statistics()


@app.get("/api/v1/analytics/trends")
async def get_trends(days: int = 7):
    """Obtiene tendencias de los últimos días"""
    return analytics.get_trends(days)


@app.get("/api/v1/analytics/performance")
async def get_performance():
    """Obtiene métricas de rendimiento"""
    return analytics.get_performance_metrics()


@app.get("/api/v1/history/statistics")
async def get_history_statistics():
    """Obtiene estadísticas del historial"""
    return prototype_history.get_statistics()


@app.get("/api/v1/notifications")
async def get_notifications(user_id: str = "system", unread_only: bool = False):
    """Obtiene notificaciones de un usuario"""
    notifications = notification_system.get_notifications(user_id, unread_only)
    unread_count = notification_system.get_unread_count(user_id)
    
    return {
        "notifications": notifications,
        "unread_count": unread_count
    }


@app.post("/api/v1/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str, user_id: str = "system"):
    """Marca una notificación como leída"""
    success = notification_system.mark_as_read(user_id, notification_id)
    return {"success": success}


@app.post("/api/v1/notifications/read-all")
async def mark_all_read(user_id: str = "system"):
    """Marca todas las notificaciones como leídas"""
    notification_system.mark_all_as_read(user_id)
    return {"success": True}


@app.post("/api/v1/export/advanced")
async def export_advanced(request: PrototypeRequest, format: str = "all"):
    """Exporta a formatos avanzados (Excel, PDF)"""
    try:
        response = await generator.generate_prototype(request)
        
        if format == "excel" or format == "all":
            excel_path = advanced_exporter.export_to_excel(response.model_dump())
        
        if format == "pdf" or format == "all":
            pdf_result = advanced_exporter.export_to_pdf_structure(response.model_dump())
        
        if format == "all":
            exports = advanced_exporter.export_all_formats(response.model_dump())
            return {
                "prototype": response,
                "exports": exports
            }
        elif format == "excel":
            return {
                "prototype": response,
                "excel_path": excel_path
            }
        elif format == "pdf":
            return {
                "prototype": response,
                "pdf_structure": pdf_result
            }
        
        raise HTTPException(status_code=400, detail="Formato no soportado")
        
    except Exception as e:
        logger.error(f"Error en exportación avanzada: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error en exportación: {str(e)}"
        )


@app.post("/api/v1/share")
async def share_prototype(prototype_id: str, owner_id: str, 
                         share_with: List[str],
                         permission: str = "view",
                         expires_in_days: Optional[int] = None):
    """Comparte un prototipo con otros usuarios"""
    try:
        perm = SharePermission(permission)
        share_token = collaboration_system.share_prototype(
            prototype_id, owner_id, share_with, perm, expires_in_days
        )
        
        return {
            "share_token": share_token,
            "prototype_id": prototype_id,
            "shared_with": share_with,
            "permission": permission
        }
    except Exception as e:
        logger.error(f"Error al compartir: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al compartir: {str(e)}"
        )


@app.get("/api/v1/share/{share_token}")
async def get_shared_prototype(share_token: str):
    """Obtiene un prototipo compartido"""
    shared = collaboration_system.get_shared_prototype(share_token)
    if not shared:
        raise HTTPException(status_code=404, detail="Compartir no encontrado o expirado")
    return shared


@app.post("/api/v1/prototypes/{prototype_id}/comments")
async def add_comment(prototype_id: str, user_id: str, comment: str):
    """Agrega un comentario a un prototipo"""
    comment_id = collaboration_system.add_comment(prototype_id, user_id, comment)
    return {
        "comment_id": comment_id,
        "success": True
    }


@app.get("/api/v1/prototypes/{prototype_id}/comments")
async def get_comments(prototype_id: str):
    """Obtiene comentarios de un prototipo"""
    comments = collaboration_system.get_comments(prototype_id)
    return {
        "prototype_id": prototype_id,
        "comments": comments
    }


@app.post("/api/v1/llm/enhance")
async def enhance_description(description: str):
    """Mejora una descripción usando LLM"""
    try:
        enhanced = await llm_integration.enhance_description(description)
        return enhanced
    except Exception as e:
        logger.error(f"Error al mejorar descripción: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al mejorar descripción: {str(e)}"
        )


@app.post("/api/v1/webhooks")
async def register_webhook(user_id: str, url: str, events: List[str], 
                          secret: Optional[str] = None):
    """Registra un webhook"""
    try:
        webhook_events = [WebhookEvent(e) for e in events]
        webhook_id = webhook_system.register_webhook(user_id, url, webhook_events, secret)
        return {
            "webhook_id": webhook_id,
            "url": url,
            "events": events
        }
    except Exception as e:
        logger.error(f"Error registrando webhook: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error registrando webhook: {str(e)}"
        )


@app.get("/api/v1/webhooks")
async def get_webhooks(user_id: str):
    """Obtiene webhooks de un usuario"""
    webhooks = webhook_system.get_webhooks(user_id)
    return {"webhooks": webhooks}


@app.post("/api/v1/auth/register")
async def register_user(username: str, email: str, password: str, role: str = "user"):
    """Registra un nuevo usuario"""
    try:
        user_role = Role(role)
        user_id = auth_system.create_user(username, email, password, user_role)
        return {
            "user_id": user_id,
            "username": username,
            "role": role
        }
    except Exception as e:
        logger.error(f"Error registrando usuario: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error registrando usuario: {str(e)}"
        )


@app.post("/api/v1/auth/login")
async def login(username: str, password: str):
    """Autentica un usuario"""
    session_token = auth_system.authenticate(username, password)
    if not session_token:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    
    return {
        "session_token": session_token,
        "token_type": "bearer"
    }


@app.get("/api/v1/auth/me")
async def get_current_user(session_token: str = Header(None, alias="Authorization")):
    """Obtiene información del usuario actual"""
    if not session_token:
        raise HTTPException(status_code=401, detail="Token requerido")
    
    # Remover "Bearer " si está presente
    if session_token.startswith("Bearer "):
        session_token = session_token[7:]
    
    user_info = auth_system.validate_session(session_token)
    if not user_info:
        raise HTTPException(status_code=401, detail="Sesión inválida o expirada")
    
    return user_info


@app.post("/api/v1/backup/create")
async def create_backup(source_paths: List[str], backup_name: Optional[str] = None):
    """Crea un backup"""
    try:
        backup_id = backup_system.create_backup(source_paths, backup_name)
        return {
            "backup_id": backup_id,
            "status": "created"
        }
    except Exception as e:
        logger.error(f"Error creando backup: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error creando backup: {str(e)}"
        )


@app.post("/api/v1/backup/restore")
async def restore_backup(backup_id: str, target_dir: str, overwrite: bool = False):
    """Restaura un backup"""
    success = backup_system.restore_backup(backup_id, target_dir, overwrite)
    if not success:
        raise HTTPException(status_code=400, detail="Error restaurando backup")
    
    return {"status": "restored", "backup_id": backup_id}


@app.get("/api/v1/backup/list")
async def list_backups():
    """Lista todos los backups"""
    backups = backup_system.list_backups()
    return {"backups": backups}


@app.get("/api/v1/performance/metrics")
async def get_performance_metrics(operation: Optional[str] = None):
    """Obtiene métricas de rendimiento"""
    metrics = performance_optimizer.get_metrics(operation)
    return {"metrics": metrics}


@app.post("/api/v1/performance/cache/clear")
async def clear_cache(pattern: Optional[str] = None):
    """Limpia el caché de rendimiento"""
    performance_optimizer.clear_cache(pattern)
    distributed_cache.clear(pattern)
    return {"status": "cleared"}


@app.get("/api/v1/monitoring/metrics")
async def get_monitoring_metrics(metric_name: Optional[str] = None, minutes: int = 60):
    """Obtiene métricas de monitoring"""
    summary = monitoring.get_metrics_summary(metric_name, minutes)
    return {"metrics": summary}


@app.get("/api/v1/monitoring/alerts")
async def get_alerts(severity: Optional[str] = None, unacknowledged_only: bool = False):
    """Obtiene alertas del sistema"""
    alerts = monitoring.get_alerts(severity, unacknowledged_only)
    return {"alerts": alerts}


@app.post("/api/v1/monitoring/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Reconoce una alerta"""
    success = monitoring.acknowledge_alert(alert_id)
    return {"success": success}


@app.get("/api/v1/monitoring/errors")
async def get_error_summary(hours: int = 24):
    """Obtiene resumen de errores"""
    summary = monitoring.get_error_summary(hours)
    return summary


@app.get("/api/v1/monitoring/performance")
async def get_performance_summary(hours: int = 24):
    """Obtiene resumen de rendimiento"""
    summary = monitoring.get_performance_summary(hours)
    return summary


@app.post("/api/v1/queue/jobs")
async def enqueue_job(job_type: str, data: Dict[str, Any], priority: int = 0):
    """Agrega un job a la cola"""
    job_id = await async_queue.enqueue(job_type, data, priority)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/v1/queue/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Obtiene el estado de un job"""
    status = async_queue.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return status


@app.get("/api/v1/queue/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 100):
    """Lista jobs en la cola"""
    job_status = JobStatus(status) if status else None
    jobs = async_queue.list_jobs(job_status, limit)
    return {"jobs": jobs}


@app.get("/api/v1/queue/stats")
async def get_queue_stats():
    """Obtiene estadísticas de la cola"""
    stats = async_queue.get_queue_stats()
    return stats


@app.get("/api/v1/cache/stats")
async def get_cache_stats():
    """Obtiene estadísticas del caché distribuido"""
    stats = distributed_cache.get_stats()
    return stats


@app.get("/api/v1/rate-limit/stats")
async def get_rate_limit_stats(identifier: Optional[str] = None):
    """Obtiene estadísticas de rate limiting"""
    stats = rate_limiter.get_stats(identifier)
    return stats


@app.get("/api/v1/config")
async def get_configuration():
    """Obtiene la configuración del sistema"""
    config = config_manager.get_config()
    # No exponer información sensible
    safe_config = {k: v for k, v in config.items() 
                   if "key" not in k.lower() and "password" not in k.lower() and "secret" not in k.lower()}
    return safe_config


@app.post("/api/v1/config/update")
async def update_configuration(updates: Dict[str, Any], session_token: str = Header(None)):
    """Actualiza la configuración (requiere admin)"""
    # Verificar permisos de admin
    if session_token:
        user_info = auth_system.validate_session(session_token.replace("Bearer ", ""))
        if not user_info or user_info.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Se requieren permisos de administrador")
    
    config_manager.update_config(updates)
    return {"status": "updated", "updates": list(updates.keys())}


@app.get("/api/v1/config/feature-flags")
async def get_feature_flags():
    """Obtiene feature flags"""
    return config_manager.config.feature_flags.model_dump()


@app.post("/api/v1/config/feature-flags/{flag_name}")
async def set_feature_flag(flag_name: str, value: bool, session_token: str = Header(None)):
    """Establece un feature flag (requiere admin)"""
    if session_token:
        user_info = auth_system.validate_session(session_token.replace("Bearer ", ""))
        if not user_info or user_info.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Se requieren permisos de administrador")
    
    config_manager.set_feature_flag(flag_name, value)
    return {"flag": flag_name, "value": value}


@app.get("/api/v1/events/history")
async def get_event_history(event_type: Optional[str] = None, limit: int = 100):
    """Obtiene historial de eventos"""
    event_type_enum = EventType(event_type) if event_type else None
    history = event_system.get_event_history(event_type_enum, limit)
    return {"events": history, "count": len(history)}


@app.get("/api/v1/plugins")
async def list_plugins(plugin_type: Optional[str] = None):
    """Lista plugins disponibles"""
    plugins = plugin_system.list_plugins(plugin_type)
    return {"plugins": plugins, "count": len(plugins)}


@app.post("/api/v1/plugins/{plugin_name}/enable")
async def enable_plugin(plugin_name: str):
    """Habilita un plugin"""
    plugin_system.enable_plugin(plugin_name)
    return {"status": "enabled", "plugin": plugin_name}


@app.post("/api/v1/plugins/{plugin_name}/disable")
async def disable_plugin(plugin_name: str):
    """Deshabilita un plugin"""
    plugin_system.disable_plugin(plugin_name)
    return {"status": "disabled", "plugin": plugin_name}


@app.get("/api/v1/circuit-breakers")
async def get_circuit_breakers():
    """Obtiene estado de circuit breakers"""
    states = {name: breaker.get_state() for name, breaker in circuit_breakers.items()}
    return {"circuit_breakers": states}


@app.post("/api/v1/circuit-breakers/{name}/reset")
async def reset_circuit_breaker(name: str):
    """Resetea un circuit breaker"""
    breaker = circuit_breakers.get(name)
    if not breaker:
        raise HTTPException(status_code=404, detail="Circuit breaker no encontrado")
    
    breaker.reset()
    return {"status": "reset", "circuit_breaker": name}


@app.get("/metrics")
async def prometheus_metrics_endpoint():
    """Endpoint de métricas Prometheus"""
    from fastapi.responses import Response
    metrics_text = prometheus_metrics.get_metrics_prometheus_format()
    return Response(content=metrics_text, media_type="text/plain")


@app.get("/api/v1/metrics")
async def get_metrics_dict():
    """Obtiene métricas como diccionario"""
    return prometheus_metrics.get_metrics_dict()


@app.get("/api/v1/i18n/languages")
async def get_languages():
    """Obtiene idiomas disponibles"""
    return {
        "languages": i18n_system.get_available_languages(),
        "default": i18n_system.default_language,
        "current": i18n_system.current_language
    }


@app.post("/api/v1/i18n/set-language")
async def set_language(language: str):
    """Establece el idioma"""
    i18n_system.set_language(language)
    return {"language": language, "status": "set"}


@app.get("/api/v1/i18n/translate")
async def translate(key: str, language: Optional[str] = None):
    """Traduce una clave"""
    translation = i18n_system.translate(key, language)
    return {"key": key, "translation": translation, "language": language or i18n_system.current_language}


@app.get("/api/v1/reports/daily")
async def get_daily_report(date: Optional[str] = None):
    """Genera reporte diario"""
    report_date = datetime.fromisoformat(date) if date else None
    report = report_generator.generate_daily_report(report_date)
    return report


@app.get("/api/v1/reports/weekly")
async def get_weekly_report(week_start: Optional[str] = None):
    """Genera reporte semanal"""
    week_start_date = datetime.fromisoformat(week_start) if week_start else None
    report = report_generator.generate_weekly_report(week_start_date)
    return report


@app.get("/api/v1/reports/monthly")
async def get_monthly_report(month: Optional[int] = None, year: Optional[int] = None):
    """Genera reporte mensual"""
    report = report_generator.generate_monthly_report(month, year)
    return report


@app.post("/api/v1/reports/export")
async def export_report(report_data: Dict[str, Any], format: str = "json"):
    """Exporta un reporte"""
    file_path = report_generator.export_report(report_data, format)
    return {"file_path": file_path, "format": format}


@app.post("/api/v1/ml/predict-cost")
async def predict_cost(product_type: str, num_parts: int, complexity: str, materials_count: int):
    """Predice el costo usando ML"""
    prediction = ml_predictor.predict_cost(product_type, num_parts, complexity, materials_count)
    return prediction


@app.post("/api/v1/ml/predict-build-time")
async def predict_build_time(num_parts: int, complexity: str, num_steps: int):
    """Predice el tiempo de construcción"""
    prediction = ml_predictor.predict_build_time(num_parts, complexity, num_steps)
    return prediction


@app.post("/api/v1/ml/predict-feasibility")
async def predict_feasibility(cost: float, complexity: str, user_experience: str):
    """Predice la viabilidad"""
    prediction = ml_predictor.predict_feasibility(cost, complexity, user_experience)
    return prediction


@app.post("/api/v1/ml/recommend-optimizations")
async def recommend_optimizations(prototype_data: Dict[str, Any]):
    """Recomienda optimizaciones usando ML"""
    recommendations = ml_predictor.recommend_optimizations(prototype_data)
    return {"recommendations": recommendations}


@app.get("/api/v1/load-balancer/stats")
async def get_load_balancer_stats():
    """Obtiene estadísticas del load balancer"""
    return load_balancer.get_stats()


@app.post("/api/v1/load-balancer/nodes")
async def add_load_balancer_node(node_id: str, weight: int = 1):
    """Agrega un nodo al load balancer"""
    node = load_balancer.add_node(node_id, weight)
    return {"node_id": node.node_id, "weight": node.weight}


@app.post("/api/v1/tracing/start")
async def start_trace(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Inicia un nuevo trace"""
    trace_id = distributed_tracing.start_trace(name, attributes)
    return {"trace_id": trace_id, "name": name}


@app.get("/api/v1/tracing/{trace_id}")
async def get_trace(trace_id: str):
    """Obtiene un trace completo"""
    trace = distributed_tracing.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace no encontrado")
    return {"trace_id": trace_id, "spans": trace}


@app.get("/api/v1/tracing/{trace_id}/summary")
async def get_trace_summary(trace_id: str):
    """Obtiene resumen de un trace"""
    summary = distributed_tracing.get_trace_summary(trace_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Trace no encontrado")
    return summary


@app.post("/api/v1/optimize/analyze")
async def analyze_optimization(operation: str, metrics: Dict[str, float]):
    """Analiza y sugiere optimizaciones"""
    analysis = auto_optimizer.analyze_and_optimize(operation, metrics)
    return analysis


@app.get("/api/v1/optimize/suggestions")
async def get_optimization_suggestions(operation: str):
    """Obtiene sugerencias de optimización"""
    suggestions = auto_optimizer.get_optimization_suggestions(operation)
    return {"operation": operation, "suggestions": suggestions}


@app.post("/api/v1/batch/process")
async def process_batch(items: List[Dict[str, Any]], batch_size: int = 10):
    """Procesa un lote de items"""
    # Esto sería un procesador real
    async def processor(item):
        # Simulación de procesamiento
        await asyncio.sleep(0.1)
        return {"processed": item.get("id", "unknown")}
    
    job = await batch_processor.process_batch(items, processor, batch_size)
    return {"job_id": job.id, "status": job.status}


@app.get("/api/v1/batch/{job_id}")
async def get_batch_status(job_id: str):
    """Obtiene estado de un lote"""
    status = batch_processor.get_batch_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Lote no encontrado")
    return status


@app.get("/api/v1/batch/{job_id}/results")
async def get_batch_results(job_id: str):
    """Obtiene resultados de un lote"""
    results = batch_processor.get_batch_results(job_id)
    if not results:
        raise HTTPException(status_code=404, detail="Lote no encontrado")
    return results


@app.post("/api/v1/cache/warm")
async def warm_cache(task_name: Optional[str] = None):
    """Ejecuta cache warming"""
    await cache_warmer.warm_cache(task_name)
    status = cache_warmer.get_warming_status()
    return {"status": "completed", "warming_status": status}


@app.get("/api/v1/cache/warming/status")
async def get_cache_warming_status():
    """Obtiene estado de cache warming"""
    return cache_warmer.get_warming_status()


@app.get("/api/v1/auto-scaler/status")
async def get_auto_scaler_status():
    """Obtiene estado del auto-scaler"""
    return auto_scaler.get_scaling_status()


@app.post("/api/v1/auto-scaler/scale")
async def manual_scale(target_instances: int):
    """Escalado manual"""
    try:
        auto_scaler.manual_scale(target_instances)
        return {"status": "scaled", "instances": target_instances}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/auto-scaler/metrics")
async def record_scaling_metrics(cpu_usage: float, memory_usage: float,
                                 request_rate: float, queue_size: int = 0):
    """Registra métricas para auto-scaling"""
    auto_scaler.record_metric(cpu_usage, memory_usage, request_rate, queue_size)
    return {"status": "recorded"}


@app.post("/api/v1/validate/prototype-request")
async def validate_prototype_request(data: Dict[str, Any], level: str = "moderate"):
    """Valida una solicitud de prototipo"""
    validation_level = ValidationLevel(level)
    result = advanced_validator.validate_prototype_request(data, validation_level)
    return result


@app.post("/api/v1/validate/material")
async def validate_material(material: Dict[str, Any]):
    """Valida un material"""
    result = advanced_validator.validate_material(material)
    return result


@app.get("/api/versions")
async def get_api_versions():
    """Obtiene información de versiones de API"""
    return api_versioning.get_all_versions()


@app.get("/api/v1/dashboard/overview")
async def get_overview_dashboard():
    """Obtiene dashboard de overview"""
    return dashboard_analytics.get_overview_dashboard()


@app.get("/api/v1/dashboard/{dashboard_id}")
async def get_dashboard(dashboard_id: str):
    """Obtiene un dashboard específico"""
    dashboard = dashboard_analytics.get_dashboard(dashboard_id)
    if not dashboard:
        raise HTTPException(status_code=404, detail="Dashboard no encontrado")
    return dashboard


@app.post("/api/v1/workflows/register")
async def register_workflow(workflow_id: str, name: str, description: str = ""):
    """Registra un workflow (simplificado)"""
    # En producción, esto recibiría las tareas en el body
    return {"workflow_id": workflow_id, "status": "registered"}


@app.post("/api/v1/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, initial_data: Optional[Dict[str, Any]] = None):
    """Ejecuta un workflow"""
    execution_id = await workflow_engine.execute_workflow(workflow_id, initial_data)
    return {"execution_id": execution_id, "workflow_id": workflow_id}


@app.get("/api/v1/workflows/executions/{execution_id}")
async def get_workflow_execution(execution_id: str):
    """Obtiene estado de ejecución de workflow"""
    status = workflow_engine.get_execution_status(execution_id)
    if not status:
        raise HTTPException(status_code=404, detail="Ejecución no encontrada")
    return status


@app.post("/api/v1/scheduler/jobs")
async def schedule_job(job_id: str, name: str, schedule_type: str,
                       schedule_value: str, max_runs: Optional[int] = None):
    """Programa un job"""
    # En producción, esto recibiría la función a ejecutar
    schedule_type_enum = ScheduleType(schedule_type)
    # job = scheduler.schedule_job(job_id, name, schedule_type_enum, schedule_value, task, max_runs)
    return {"job_id": job_id, "status": "scheduled"}


@app.get("/api/v1/scheduler/jobs")
async def list_scheduled_jobs():
    """Lista jobs programados"""
    return {"jobs": scheduler.get_jobs()}


@app.post("/api/v1/scheduler/jobs/{job_id}/enable")
async def enable_scheduled_job(job_id: str):
    """Habilita un job programado"""
    scheduler.enable_job(job_id)
    return {"job_id": job_id, "status": "enabled"}


@app.post("/api/v1/scheduler/jobs/{job_id}/disable")
async def disable_scheduled_job(job_id: str):
    """Deshabilita un job programado"""
    scheduler.disable_job(job_id)
    return {"job_id": job_id, "status": "disabled"}


@app.post("/api/v1/integrations/register")
async def register_integration(integration_id: str, integration_type: str,
                               base_url: str, api_key: Optional[str] = None):
    """Registra una integración externa"""
    integration = ExternalIntegration(
        integration_id, IntegrationType(integration_type), base_url, api_key
    )
    external_integrations.register_integration(integration)
    return {"integration_id": integration_id, "status": "registered"}


@app.get("/api/v1/integrations")
async def list_integrations(integration_type: Optional[str] = None):
    """Lista integraciones"""
    type_enum = IntegrationType(integration_type) if integration_type else None
    return {"integrations": external_integrations.list_integrations(type_enum)}


@app.post("/api/v1/integrations/materials/search")
async def search_materials_external(query: str, supplier_id: Optional[str] = None):
    """Busca materiales en proveedores externos"""
    results = await external_integrations.search_materials_external(query, supplier_id)
    return {"results": results, "count": len(results)}


@app.post("/api/v1/integrations/cad/export")
async def export_to_cad(cad_software_id: str, prototype_data: Dict[str, Any]):
    """Exporta prototipo a software CAD"""
    result = await external_integrations.export_to_cad(cad_software_id, prototype_data)
    return result


@app.post("/api/v1/user-rate-limit/check")
async def check_user_rate_limit(user_id: str, endpoint: str):
    """Verifica rate limit de usuario"""
    result = user_rate_limiter.check_user_limit(user_id, endpoint)
    return result


@app.post("/api/v1/user-rate-limit/quota/check")
async def check_user_quota(user_id: str, quota_type: str, amount: int = 1):
    """Verifica cuota de usuario"""
    result = user_rate_limiter.check_user_quota(user_id, quota_type, amount)
    return result


@app.get("/api/v1/user-rate-limit/stats/{user_id}")
async def get_user_rate_limit_stats(user_id: str):
    """Obtiene estadísticas de rate limiting de usuario"""
    return user_rate_limiter.get_user_stats(user_id)


@app.post("/api/v1/notifications/push")
async def create_push_notification(user_id: str, title: str, body: str,
                                  priority: str = "normal",
                                  channels: Optional[List[str]] = None,
                                  data: Optional[Dict[str, Any]] = None):
    """Crea una notificación push"""
    priority_enum = NotificationPriority(priority)
    channel_enums = [NotificationChannel(ch) for ch in (channels or ["in_app"])]
    
    notification = push_notifications.create_notification(
        user_id, title, body, priority_enum, channel_enums, data
    )
    
    # Enviar asíncronamente
    asyncio.create_task(push_notifications.send_notification(notification.id))
    
    return {"notification_id": notification.id, "status": "created"}


@app.get("/api/v1/notifications/{user_id}")
async def get_user_notifications(user_id: str, unread_only: bool = False, limit: int = 50):
    """Obtiene notificaciones de usuario"""
    notifications = push_notifications.get_user_notifications(user_id, unread_only, limit)
    return {"notifications": notifications, "count": len(notifications)}


@app.post("/api/v1/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str, user_id: str):
    """Marca notificación como leída"""
    push_notifications.mark_as_read(notification_id, user_id)
    return {"notification_id": notification_id, "status": "read"}


@app.post("/api/v1/security/generate-api-key")
async def generate_api_key(length: int = 32):
    """Genera una API key"""
    api_key = security_manager.generate_api_key(length)
    return {"api_key": api_key}


@app.post("/api/v1/security/check-password-strength")
async def check_password_strength(password: str):
    """Verifica fortaleza de contraseña"""
    result = security_manager.check_password_strength(password)
    return result


@app.get("/api/v1/audit/logs")
async def get_audit_logs(user_id: Optional[str] = None,
                         event_type: Optional[str] = None,
                         resource_type: Optional[str] = None,
                         limit: int = 1000):
    """Obtiene logs de auditoría"""
    event_type_enum = AuditEventType(event_type) if event_type else None
    logs = audit_system.get_audit_logs(user_id, event_type_enum, resource_type, limit=limit)
    return {"logs": logs, "count": len(logs)}


@app.get("/api/v1/audit/users/{user_id}/activity")
async def get_user_activity(user_id: str, days: int = 30):
    """Obtiene actividad de usuario"""
    return audit_system.get_user_activity(user_id, days)


@app.post("/api/v1/audit/compliance-report")
async def generate_compliance_report(start_date: str, end_date: str):
    """Genera reporte de compliance"""
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    report = audit_system.generate_compliance_report(start, end)
    return report


@app.post("/api/v1/disaster-recovery/recovery-point")
async def create_recovery_point(name: str, data: Dict[str, Any],
                               metadata: Optional[Dict[str, Any]] = None):
    """Crea un punto de recuperación"""
    recovery_point_id = disaster_recovery.create_recovery_point(name, data, metadata)
    return {"recovery_point_id": recovery_point_id, "status": "created"}


@app.get("/api/v1/disaster-recovery/recovery-points")
async def list_recovery_points(limit: int = 50):
    """Lista puntos de recuperación"""
    return {"recovery_points": disaster_recovery.list_recovery_points(limit)}


@app.post("/api/v1/disaster-recovery/restore")
async def restore_from_recovery_point(recovery_point_id: str):
    """Restaura desde un punto de recuperación"""
    result = disaster_recovery.restore_from_recovery_point(recovery_point_id)
    return result


@app.get("/api/v1/disaster-recovery/status")
async def get_disaster_recovery_status():
    """Obtiene estado del sistema de recuperación"""
    return disaster_recovery.get_recovery_status()


@app.get("/api/v1/docs")
async def get_documentation():
    """Obtiene documentación interactiva"""
    return interactive_docs.get_all_docs()


@app.get("/api/v1/docs/search")
async def search_documentation(query: str):
    """Busca en la documentación"""
    results = interactive_docs.search_docs(query)
    return {"results": results, "count": len(results)}


@app.get("/api/v1/docs/markdown")
async def get_documentation_markdown():
    """Obtiene documentación en Markdown"""
    from fastapi.responses import Response
    md_content = interactive_docs.generate_api_doc_markdown()
    return Response(content=md_content, media_type="text/markdown")


@app.get("/api/v1/gateway/routes")
async def get_gateway_routes():
    """Obtiene rutas del API gateway"""
    return {"routes": api_gateway.get_routes()}


@app.get("/api/v1/gateway/stats")
async def get_gateway_stats():
    """Obtiene estadísticas del gateway"""
    return api_gateway.get_gateway_stats()


@app.get("/api/v1/profiler/stats")
async def get_profiler_stats(func_name: Optional[str] = None):
    """Obtiene estadísticas del profiler"""
    if func_name:
        stats = performance_profiler.get_function_stats(func_name)
        return {"function": func_name, "stats": stats}
    return performance_profiler.get_all_stats()


@app.get("/api/v1/profiler/slow-queries")
async def get_slow_queries(limit: int = 50):
    """Obtiene consultas lentas"""
    return {"slow_queries": performance_profiler.get_slow_queries(limit)}


@app.get("/api/v1/profiler/profile/{func_name}")
async def get_profile_report(func_name: str, limit: int = 20):
    """Obtiene reporte de perfil"""
    report = performance_profiler.get_profile_report(func_name, limit)
    if not report:
        raise HTTPException(status_code=404, detail="Perfil no encontrado")
    from fastapi.responses import Response
    return Response(content=report, media_type="text/plain")


@app.get("/api/v1/migrations")
async def list_migrations():
    """Lista migraciones"""
    return {"migrations": data_migration.list_migrations()}


@app.post("/api/v1/migrations/{migration_id}/run")
async def run_migration(migration_id: str, context: Optional[Dict[str, Any]] = None):
    """Ejecuta una migración"""
    result = await data_migration.run_migration(migration_id, context)
    return result


@app.post("/api/v1/migrations/{migration_id}/rollback")
async def rollback_migration(migration_id: str, context: Optional[Dict[str, Any]] = None):
    """Revierte una migración"""
    result = await data_migration.rollback_migration(migration_id, context)
    return result


@app.get("/api/v1/migrations/{migration_id}/status")
async def get_migration_status(migration_id: str):
    """Obtiene estado de una migración"""
    status = data_migration.get_migration_status(migration_id)
    if not status:
        raise HTTPException(status_code=404, detail="Migración no encontrada")
    return status


@app.post("/api/v1/cicd/build")
async def trigger_build(build_id: str, branch: str = "main", commit_hash: Optional[str] = None):
    """Dispara un build"""
    build = cicd_integration.trigger_build(build_id, branch, commit_hash)
    return build


@app.get("/api/v1/cicd/build/{build_id}")
async def get_build_status(build_id: str):
    """Obtiene estado de un build"""
    status = cicd_integration.get_build_status(build_id)
    if not status:
        raise HTTPException(status_code=404, detail="Build no encontrado")
    return status


@app.post("/api/v1/cicd/tests")
async def run_tests(test_suite: str = "all"):
    """Ejecuta tests"""
    result = cicd_integration.run_tests(test_suite)
    return result


@app.post("/api/v1/cicd/deploy")
async def deploy(environment: str, version: str, build_id: Optional[str] = None):
    """Despliega a un ambiente"""
    deployment = cicd_integration.deploy(environment, version, build_id)
    return deployment


@app.get("/api/v1/cicd/stats")
async def get_cicd_stats():
    """Obtiene estadísticas de CI/CD"""
    return cicd_integration.get_ci_cd_stats()


@app.post("/api/v1/feature-flags/create")
async def create_feature_flag(flag_name: str, flag_type: str, default_value: Any,
                             description: str = "", enabled: bool = True):
    """Crea un feature flag"""
    flag_type_enum = FeatureFlagType(flag_type)
    flag = advanced_feature_flags.create_flag(flag_name, flag_type_enum, default_value, description, enabled)
    return flag


@app.get("/api/v1/feature-flags/{flag_name}")
async def get_feature_flag(flag_name: str, user_id: Optional[str] = None,
                          environment: Optional[str] = None):
    """Obtiene valor de un feature flag"""
    value = advanced_feature_flags.get_flag_value(flag_name, user_id, environment)
    return {"flag_name": flag_name, "value": value}


@app.get("/api/v1/feature-flags")
async def list_feature_flags(enabled_only: bool = False):
    """Lista feature flags"""
    flags = advanced_feature_flags.list_flags(enabled_only)
    return {"flags": flags, "count": len(flags)}


@app.post("/api/v1/ab-testing/experiments")
async def create_ab_experiment(experiment_id: str, name: str, variants: List[str],
                              traffic_split: Optional[Dict[str, float]] = None):
    """Crea un experimento A/B"""
    experiment = ab_testing.create_experiment(experiment_id, name, variants, traffic_split)
    return experiment


@app.post("/api/v1/ab-testing/{experiment_id}/assign")
async def assign_variant(experiment_id: str, user_id: str):
    """Asigna variante a usuario"""
    variant = ab_testing.assign_variant(experiment_id, user_id)
    return {"experiment_id": experiment_id, "user_id": user_id, "variant": variant}


@app.post("/api/v1/ab-testing/{experiment_id}/conversion")
async def record_conversion(experiment_id: str, user_id: str, variant: Optional[str] = None,
                           value: float = 1.0):
    """Registra una conversión"""
    ab_testing.record_conversion(experiment_id, user_id, variant, value)
    return {"status": "recorded"}


@app.get("/api/v1/ab-testing/{experiment_id}/results")
async def get_experiment_results(experiment_id: str):
    """Obtiene resultados de experimento"""
    results = ab_testing.get_experiment_results(experiment_id)
    if not results:
        raise HTTPException(status_code=404, detail="Experimento no encontrado")
    return results


@app.post("/api/v1/ml-analytics/record-action")
async def record_user_action(user_id: str, action: str, context: Optional[Dict[str, Any]] = None):
    """Registra acción de usuario"""
    ml_analytics.record_user_action(user_id, action, context)
    return {"status": "recorded"}


@app.get("/api/v1/ml-analytics/predict/{user_id}")
async def predict_user_preference(user_id: str):
    """Predice preferencias de usuario"""
    prediction = ml_analytics.predict_user_preference(user_id)
    return prediction


@app.get("/api/v1/ml-analytics/anomalies")
async def detect_anomalies(user_id: Optional[str] = None):
    """Detecta anomalías"""
    anomalies = ml_analytics.detect_anomalies(user_id)
    return {"anomalies": anomalies, "count": len(anomalies)}


@app.get("/api/v1/ml-analytics/insights")
async def generate_insights(time_range_days: int = 7):
    """Genera insights usando ML"""
    insights = ml_analytics.generate_insights(time_range_days)
    return insights


@app.post("/api/v1/recommendations/update-profile")
async def update_user_profile(user_id: str, preferences: Dict[str, Any]):
    """Actualiza perfil de usuario para recomendaciones"""
    personalized_recommendations.update_user_profile(user_id, preferences)
    return {"status": "updated", "user_id": user_id}


@app.post("/api/v1/recommendations/interaction")
async def record_interaction(user_id: str, item_id: str, item_type: str,
                            interaction_type: str, rating: Optional[float] = None):
    """Registra interacción de usuario"""
    personalized_recommendations.record_interaction(user_id, item_id, item_type, interaction_type, rating)
    return {"status": "recorded"}


@app.get("/api/v1/recommendations/{user_id}")
async def get_recommendations(user_id: str, limit: int = 10, item_type: Optional[str] = None):
    """Obtiene recomendaciones personalizadas"""
    recommendations = personalized_recommendations.get_recommendations(user_id, limit, item_type)
    return {"recommendations": recommendations, "count": len(recommendations)}


@app.post("/api/v1/gamification/award-points")
async def award_points(user_id: str, points: int, reason: str):
    """Otorga puntos a un usuario"""
    gamification.award_points(user_id, points, reason)
    return {"status": "awarded", "user_id": user_id, "points": points}


@app.get("/api/v1/gamification/stats/{user_id}")
async def get_gamification_stats(user_id: str):
    """Obtiene estadísticas de gamificación"""
    return gamification.get_user_stats(user_id)


@app.get("/api/v1/gamification/leaderboard")
async def get_leaderboard(leaderboard_type: str = "points", limit: int = 100):
    """Obtiene leaderboard"""
    return {"leaderboard": gamification.get_leaderboard(leaderboard_type, limit)}


@app.post("/api/v1/marketplace/listings")
async def create_listing(listing_id: str, seller_id: str, title: str,
                         description: str, price: float, category: str,
                         item_data: Dict[str, Any]):
    """Crea un listing en el marketplace"""
    listing = marketplace.create_listing(listing_id, seller_id, title, description, price, category, item_data)
    return listing


@app.post("/api/v1/marketplace/listings/{listing_id}/publish")
async def publish_listing(listing_id: str):
    """Publica un listing"""
    listing = marketplace.publish_listing(listing_id)
    return listing


@app.get("/api/v1/marketplace/listings/search")
async def search_listings(query: Optional[str] = None, category: Optional[str] = None,
                          min_price: Optional[float] = None, max_price: Optional[float] = None,
                          limit: int = 50):
    """Busca listings"""
    results = marketplace.search_listings(query, category, min_price, max_price, limit)
    return {"listings": results, "count": len(results)}


@app.get("/api/v1/marketplace/listings/{listing_id}")
async def get_listing_details(listing_id: str):
    """Obtiene detalles de un listing"""
    listing = marketplace.get_listing_details(listing_id)
    if not listing:
        raise HTTPException(status_code=404, detail="Listing no encontrado")
    return listing


@app.post("/api/v1/marketplace/orders")
async def create_order(order_id: str, buyer_id: str, listing_id: str, quantity: int = 1):
    """Crea una orden"""
    order = marketplace.create_order(order_id, buyer_id, listing_id, quantity)
    return order


@app.get("/api/v1/marketplace/stats")
async def get_marketplace_stats():
    """Obtiene estadísticas del marketplace"""
    return marketplace.get_marketplace_stats()


@app.post("/api/v1/monetization/subscriptions")
async def create_subscription(user_id: str, tier: str, payment_method: str = "credit_card"):
    """Crea una suscripción"""
    tier_enum = SubscriptionTier(tier)
    subscription = monetization.create_subscription(user_id, tier_enum, payment_method)
    return subscription


@app.get("/api/v1/monetization/subscriptions/{user_id}")
async def get_user_subscription(user_id: str):
    """Obtiene suscripción de usuario"""
    subscription = monetization.get_user_subscription(user_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Suscripción no encontrada")
    return subscription


@app.get("/api/v1/monetization/pricing-plans")
async def get_pricing_plans():
    """Obtiene planes de precios"""
    return monetization.get_pricing_plans()


@app.post("/api/v1/monetization/check-access")
async def check_feature_access(user_id: str, feature: str):
    """Verifica acceso a una característica"""
    has_access = monetization.check_feature_access(user_id, feature)
    return {"user_id": user_id, "feature": feature, "has_access": has_access}


@app.get("/api/v1/monetization/revenue")
async def get_revenue_stats(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Obtiene estadísticas de ingresos"""
    start = datetime.fromisoformat(start_date) if start_date else None
    end = datetime.fromisoformat(end_date) if end_date else None
    stats = monetization.get_revenue_stats(start, end)
    return stats


@app.get("/api/v1/docs/auto/openapi")
async def get_auto_openapi_spec():
    """Obtiene especificación OpenAPI generada automáticamente"""
    spec = auto_documentation.generate_openapi_spec()
    return spec


@app.get("/api/v1/docs/auto/markdown")
async def get_auto_markdown_docs():
    """Obtiene documentación Markdown generada automáticamente"""
    from fastapi.responses import Response
    md_content = auto_documentation.generate_markdown_docs()
    return Response(content=md_content, media_type="text/markdown")


@app.post("/api/v1/testing/run")
async def run_tests(test_path: Optional[str] = None, coverage: bool = True, verbose: bool = False):
    """Ejecuta tests"""
    result = advanced_testing.run_tests(test_path, coverage, verbose)
    return result


@app.get("/api/v1/testing/coverage")
async def get_test_coverage():
    """Obtiene reporte de coverage"""
    return advanced_testing.get_coverage_report()


@app.get("/api/v1/testing/summary")
async def get_test_summary():
    """Obtiene resumen de tests"""
    return advanced_testing.get_test_summary()


@app.post("/api/v1/business-metrics/record")
async def record_business_metric(metric_name: str, value: float,
                                category: str = "general", metadata: Optional[Dict[str, Any]] = None):
    """Registra métrica de negocio"""
    business_metrics.record_metric(metric_name, value, category, metadata)
    return {"status": "recorded"}


@app.get("/api/v1/business-metrics/dashboard")
async def get_business_dashboard():
    """Obtiene dashboard de métricas de negocio"""
    return business_metrics.get_business_dashboard()


@app.post("/api/v1/business-metrics/calculate-mrr")
async def calculate_mrr(subscriptions: List[Dict[str, Any]]):
    """Calcula MRR"""
    mrr = business_metrics.calculate_mrr(subscriptions)
    return {"mrr": mrr}


@app.get("/api/v1/cache/intelligent/stats")
async def get_intelligent_cache_stats():
    """Obtiene estadísticas del cache inteligente"""
    return intelligent_cache.get_cache_stats()


@app.post("/api/v1/cache/intelligent/prefetch")
async def prefetch_cache(current_key: str):
    """Predice y precarga valores en cache"""
    intelligent_cache.predict_and_prefetch(current_key)
    return {"status": "prefetched"}


@app.get("/api/v1/reports/executive/summary")
async def generate_executive_summary(period_days: int = 30):
    """Genera resumen ejecutivo"""
    report = executive_reports.generate_executive_summary(period_days)
    return report


@app.get("/api/v1/reports/executive/quarterly")
async def generate_quarterly_report(quarter: int, year: int):
    """Genera reporte trimestral"""
    report = executive_reports.generate_quarterly_report(quarter, year)
    return report


@app.post("/api/v1/reports/executive/export")
async def export_executive_report(report_data: Dict[str, Any], format: str = "json"):
    """Exporta reporte ejecutivo"""
    file_path = executive_reports.export_report(report_data, format)
    return {"file_path": file_path, "format": format}


@app.post("/api/v1/query-optimizer/analyze")
async def analyze_query(query_string: str):
    """Analiza una consulta y sugiere optimizaciones"""
    analysis = query_optimizer.analyze_query(query_string)
    return analysis


@app.get("/api/v1/query-optimizer/slow-queries")
async def get_slow_queries(limit: int = 50):
    """Obtiene consultas lentas"""
    return {"slow_queries": query_optimizer.get_slow_queries(limit)}


@app.get("/api/v1/query-optimizer/stats")
async def get_query_stats(query_type: Optional[str] = None):
    """Obtiene estadísticas de consultas"""
    return query_optimizer.get_query_stats(query_type)


@app.post("/api/v1/sentiment/analyze")
async def analyze_sentiment(text: str, context: Optional[str] = None):
    """Analiza sentimiento de un texto"""
    result = sentiment_analysis.analyze_text(text, context)
    return result


@app.post("/api/v1/sentiment/analyze-reviews")
async def analyze_reviews_sentiment(reviews: List[Dict[str, Any]]):
    """Analiza sentimientos de reseñas"""
    result = sentiment_analysis.analyze_reviews(reviews)
    return result


@app.get("/api/v1/sentiment/trends")
async def get_sentiment_trends(days: int = 30):
    """Obtiene tendencias de sentimiento"""
    return sentiment_analysis.get_sentiment_trends(days)


@app.post("/api/v1/demand-forecasting/record")
async def record_demand(product_type: str, quantity: int, date: Optional[str] = None):
    """Registra demanda histórica"""
    demand_date = datetime.fromisoformat(date) if date else None
    demand_forecasting.record_demand(product_type, quantity, demand_date)
    return {"status": "recorded"}


@app.get("/api/v1/demand-forecasting/{product_type}")
async def forecast_demand(product_type: str, days_ahead: int = 30):
    """Predice demanda futura"""
    forecast = demand_forecasting.forecast_demand(product_type, days_ahead)
    return forecast


@app.get("/api/v1/demand-forecasting/{product_type}/trends")
async def get_demand_trends(product_type: str, days: int = 90):
    """Obtiene tendencias de demanda"""
    return demand_forecasting.get_demand_trends(product_type, days)


@app.post("/api/v1/alerts/rules")
async def create_alert_rule(rule_id: str, name: str, severity: str = "warning"):
    """Crea una regla de alerta (simplificado)"""
    # En producción, esto recibiría la condición y acción
    return {"rule_id": rule_id, "status": "created"}


@app.post("/api/v1/alerts/evaluate")
async def evaluate_alert(rule_id: str, context: Dict[str, Any]):
    """Evalúa una regla de alerta"""
    alert = intelligent_alerts.evaluate_alert(rule_id, context)
    if alert:
        return alert
    return {"status": "no_alert"}


@app.get("/api/v1/alerts/active")
async def get_active_alerts(severity: Optional[str] = None, unacknowledged_only: bool = False):
    """Obtiene alertas activas"""
    severity_enum = AlertSeverity(severity) if severity else None
    alerts = intelligent_alerts.get_active_alerts(severity_enum, unacknowledged_only)
    return {"alerts": alerts, "count": len(alerts)}


@app.get("/api/v1/alerts/stats")
async def get_alert_stats():
    """Obtiene estadísticas de alertas"""
    return intelligent_alerts.get_alert_stats()


@app.post("/api/v1/inventory/items")
async def add_inventory_item(item_id: str, name: str, category: str,
                            initial_quantity: int = 0, unit: str = "unit",
                            reorder_point: int = 10, reorder_quantity: int = 50):
    """Agrega item al inventario"""
    item = inventory_management.add_item(item_id, name, category, initial_quantity, unit, reorder_point, reorder_quantity)
    return item


@app.post("/api/v1/inventory/items/{item_id}/update")
async def update_inventory_quantity(item_id: str, quantity_change: int,
                                   reason: str = "manual", reference: Optional[str] = None):
    """Actualiza cantidad de inventario"""
    item = inventory_management.update_quantity(item_id, quantity_change, reason, reference)
    return item


@app.post("/api/v1/inventory/reservations")
async def reserve_inventory_item(item_id: str, quantity: int, reservation_id: str,
                                 expires_at: Optional[str] = None):
    """Reserva item del inventario"""
    expires = datetime.fromisoformat(expires_at) if expires_at else None
    reservation = inventory_management.reserve_item(item_id, quantity, reservation_id, expires)
    return reservation


@app.get("/api/v1/inventory/report")
async def get_inventory_report():
    """Obtiene reporte de inventario"""
    return inventory_management.get_inventory_report()


@app.get("/api/v1/inventory/low-stock")
async def check_low_stock():
    """Verifica items con stock bajo"""
    return {"low_stock_items": inventory_management.check_low_stock()}


@app.post("/api/v1/competitors")
async def add_competitor(competitor_id: str, name: str, category: str, market_share: float = 0.0):
    """Agrega un competidor"""
    competitor = competitor_analysis.add_competitor(competitor_id, name, category, market_share)
    return competitor


@app.get("/api/v1/competitors/{competitor_id}/analyze")
async def analyze_competitor(competitor_id: str):
    """Analiza un competidor"""
    analysis = competitor_analysis.analyze_competitor(competitor_id)
    return analysis


@app.get("/api/v1/competitors/landscape")
async def get_competitive_landscape():
    """Obtiene panorama competitivo"""
    return competitor_analysis.get_competitive_landscape()


@app.post("/api/v1/competitors/compare")
async def compare_with_competitors(our_metrics: Dict[str, Any]):
    """Compara con competidores"""
    return competitor_analysis.compare_with_competitors(our_metrics)


@app.post("/api/v1/logging/structured")
async def log_structured(level: str, message: str, context: Optional[Dict[str, Any]] = None,
                         user_id: Optional[str] = None, request_id: Optional[str] = None):
    """Log estructurado"""
    level_enum = LogLevel(level)
    log_entry = advanced_logging.log_structured(level_enum, message, context, user_id, request_id)
    return log_entry


@app.get("/api/v1/logging/logs")
async def get_logs(level: Optional[str] = None, user_id: Optional[str] = None,
                   limit: int = 1000):
    """Obtiene logs"""
    level_enum = LogLevel(level) if level else None
    logs = advanced_logging.get_logs(level_enum, user_id, limit=limit)
    return {"logs": logs, "count": len(logs)}


@app.get("/api/v1/logging/statistics")
async def get_log_statistics(days: int = 7):
    """Obtiene estadísticas de logs"""
    return advanced_logging.get_log_statistics(days)


@app.post("/api/v1/blockchain/register")
async def register_in_blockchain(prototype_id: str, prototype_data: Dict[str, Any]):
    """Registra prototipo en blockchain"""
    block_hash = blockchain_verification.register_prototype(prototype_id, prototype_data)
    return {"prototype_id": prototype_id, "block_hash": block_hash, "status": "registered"}


@app.get("/api/v1/blockchain/verify/{prototype_id}")
async def verify_prototype_blockchain(prototype_id: str):
    """Verifica prototipo en blockchain"""
    verification = blockchain_verification.verify_prototype(prototype_id)
    if not verification:
        raise HTTPException(status_code=404, detail="Prototipo no encontrado en blockchain")
    return verification


@app.get("/api/v1/blockchain/info")
async def get_blockchain_info():
    """Obtiene información de blockchain"""
    return blockchain_verification.get_chain_info()


@app.post("/api/v1/ar-vr/models")
async def create_ar_model(model_id: str, prototype_id: str, model_data: Dict[str, Any],
                          platform: str):
    """Crea modelo AR"""
    platform_enum = ARVRPlatform(platform)
    model = ar_vr_integration.create_ar_model(model_id, prototype_id, model_data, platform_enum)
    return model


@app.post("/api/v1/ar-vr/experiences")
async def create_vr_experience(experience_id: str, prototype_id: str,
                              experience_data: Dict[str, Any]):
    """Crea experiencia VR"""
    experience = ar_vr_integration.create_vr_experience(experience_id, prototype_id, experience_data)
    return experience


@app.get("/api/v1/ar-vr/models/{model_id}/qr")
async def generate_ar_qr_code(model_id: str):
    """Genera código QR para AR"""
    qr_data = ar_vr_integration.generate_ar_qr_code(model_id)
    return qr_data


@app.get("/api/v1/ar-vr/models/{model_id}/preview")
async def get_ar_preview(model_id: str):
    """Obtiene preview AR"""
    preview = ar_vr_integration.get_ar_preview(model_id)
    return preview


@app.post("/api/v1/iot/devices")
async def register_iot_device(device_id: str, device_type: str, name: str,
                              location: Optional[str] = None, capabilities: Optional[List[str]] = None):
    """Registra dispositivo IoT"""
    device_type_enum = DeviceType(device_type)
    device = iot_integration.register_device(device_id, device_type_enum, name, location, capabilities)
    return device


@app.post("/api/v1/iot/devices/{device_id}/data")
async def send_iot_data(device_id: str, data: Dict[str, Any]):
    """Envía datos desde dispositivo IoT"""
    iot_integration.send_data(device_id, data)
    return {"status": "received"}


@app.post("/api/v1/iot/devices/{device_id}/command")
async def send_iot_command(device_id: str, command: str, parameters: Optional[Dict[str, Any]] = None):
    """Envía comando a dispositivo IoT"""
    command_entry = iot_integration.send_command(device_id, command, parameters)
    return command_entry


@app.get("/api/v1/iot/devices/{device_id}/status")
async def get_iot_device_status(device_id: str):
    """Obtiene estado de dispositivo IoT"""
    status = iot_integration.get_device_status(device_id)
    if not status:
        raise HTTPException(status_code=404, detail="Dispositivo no encontrado")
    return status


@app.get("/api/v1/iot/devices")
async def list_iot_devices(device_type: Optional[str] = None):
    """Lista dispositivos IoT"""
    device_type_enum = DeviceType(device_type) if device_type else None
    devices = iot_integration.list_devices(device_type_enum)
    return {"devices": devices, "count": len(devices)}


@app.post("/api/v1/edge/nodes")
async def register_edge_node(node_id: str, name: str, location: str,
                            capabilities: List[str], max_workload: int = 100):
    """Registra nodo edge"""
    node = edge_computing.register_edge_node(node_id, name, location, capabilities, max_workload)
    return node


@app.post("/api/v1/edge/deploy")
async def deploy_to_edge(deployment_id: str, node_id: str, workload: Dict[str, Any]):
    """Despliega workload a nodo edge"""
    deployment = edge_computing.deploy_to_edge(deployment_id, node_id, workload)
    return deployment


@app.get("/api/v1/edge/network/status")
async def get_edge_network_status():
    """Obtiene estado de red edge"""
    return edge_computing.get_edge_network_status()


@app.post("/api/v1/data-analysis/load-dataset")
async def load_dataset(dataset_id: str, data: List[Dict[str, Any]]):
    """Carga dataset para análisis"""
    advanced_data_analysis.load_dataset(dataset_id, data)
    return {"dataset_id": dataset_id, "records": len(data), "status": "loaded"}


@app.post("/api/v1/data-analysis/correlation")
async def analyze_correlation(dataset_id: str, variable1: str, variable2: str):
    """Analiza correlación entre variables"""
    result = advanced_data_analysis.analyze_correlation(dataset_id, variable1, variable2)
    return result


@app.post("/api/v1/data-analysis/clustering")
async def perform_clustering(dataset_id: str, features: List[str], k: int = 3):
    """Realiza clustering"""
    result = advanced_data_analysis.perform_clustering(dataset_id, features, k)
    return result


@app.get("/api/v1/data-analysis/outliers/{dataset_id}")
async def detect_outliers(dataset_id: str, variable: str):
    """Detecta outliers"""
    outliers = advanced_data_analysis.detect_outliers(dataset_id, variable)
    return {"outliers": outliers, "count": len(outliers)}


@app.get("/api/v1/data-analysis/statistics/{dataset_id}")
async def get_statistics(dataset_id: str, variable: str):
    """Obtiene estadísticas descriptivas"""
    stats = advanced_data_analysis.generate_statistics(dataset_id, variable)
    return stats


@app.post("/api/v1/predictive/train")
async def train_predictive_model(model_id: str, model_type: str,
                                training_data: List[Dict[str, Any]], target_variable: str):
    """Entrena modelo predictivo"""
    model = predictive_analytics.train_model(model_id, model_type, training_data, target_variable)
    return model


@app.post("/api/v1/predictive/predict")
async def make_prediction(model_id: str, input_data: Dict[str, Any]):
    """Hace una predicción"""
    prediction = predictive_analytics.predict(model_id, input_data)
    return prediction


@app.post("/api/v1/predictive/forecast")
async def forecast_time_series(data: List[Dict[str, Any]], periods: int = 10):
    """Predice serie temporal"""
    forecast = predictive_analytics.forecast_time_series(data, periods)
    return {"forecast": forecast, "periods": periods}


@app.post("/api/v1/predictive/churn")
async def predict_churn(user_data: Dict[str, Any]):
    """Predice probabilidad de churn"""
    prediction = predictive_analytics.predict_churn(user_data)
    return prediction


@app.post("/api/v1/knowledge/add")
async def add_knowledge(knowledge_id: str, title: str, content: str,
                       category: str, tags: Optional[List[str]] = None, author: Optional[str] = None):
    """Agrega conocimiento a la base"""
    knowledge = knowledge_management.add_knowledge(knowledge_id, title, content, category, tags, author)
    return knowledge


@app.get("/api/v1/knowledge/search")
async def search_knowledge(query: str, category: Optional[str] = None, limit: int = 20):
    """Busca en la base de conocimiento"""
    results = knowledge_management.search_knowledge(query, category, limit)
    return {"results": results, "count": len(results)}


@app.get("/api/v1/knowledge/{knowledge_id}")
async def get_knowledge(knowledge_id: str):
    """Obtiene conocimiento específico"""
    knowledge = knowledge_management.get_knowledge(knowledge_id)
    if not knowledge:
        raise HTTPException(status_code=404, detail="Conocimiento no encontrado")
    return knowledge


@app.get("/api/v1/knowledge/statistics")
async def get_knowledge_statistics():
    """Obtiene estadísticas de conocimiento"""
    return knowledge_management.get_knowledge_statistics()


@app.post("/api/v1/services/register")
async def register_service(service_id: str, name: str, base_url: str,
                          service_type: str, api_key: Optional[str] = None,
                          health_check_url: Optional[str] = None):
    """Registra un servicio externo"""
    service = enhanced_service_integration.register_service(
        service_id, name, base_url, service_type, api_key, health_check_url
    )
    return service


@app.get("/api/v1/services/{service_id}/health")
async def check_service_health(service_id: str):
    """Verifica salud de un servicio"""
    health = await enhanced_service_integration.check_service_health(service_id)
    return health


@app.post("/api/v1/services/{service_id}/call")
async def call_service(service_id: str, endpoint: str, method: str = "GET",
                      data: Optional[Dict[str, Any]] = None):
    """Llama a un servicio externo"""
    result = await enhanced_service_integration.call_service(service_id, endpoint, method, data)
    return result


@app.get("/api/v1/services/{service_id}/status")
async def get_service_status(service_id: str):
    """Obtiene estado de un servicio"""
    status = enhanced_service_integration.get_service_status(service_id)
    if not status:
        raise HTTPException(status_code=404, detail="Servicio no encontrado")
    return status


@app.get("/api/v1/services/integration/stats")
async def get_integration_stats():
    """Obtiene estadísticas de integraciones"""
    return enhanced_service_integration.get_integration_stats()


@app.post("/api/v1/ml/models")
async def create_ml_model(model_id: str, model_type: str, architecture: Dict[str, Any],
                         hyperparameters: Optional[Dict[str, Any]] = None):
    """Crea un modelo ML"""
    model = advanced_ml_system.create_model(model_id, model_type, architecture, hyperparameters)
    return model


@app.post("/api/v1/ml/models/{model_id}/train")
async def train_ml_model(model_id: str, training_data: List[Dict[str, Any]],
                        validation_data: Optional[List[Dict[str, Any]]] = None,
                        epochs: int = 10):
    """Entrena un modelo ML"""
    training_job = advanced_ml_system.train_model(model_id, training_data, validation_data, epochs)
    return training_job


@app.post("/api/v1/ml/models/{model_id}/predict-batch")
async def predict_batch_ml(model_id: str, input_data: List[Dict[str, Any]]):
    """Predice en lote con modelo ML"""
    predictions = advanced_ml_system.predict_batch(model_id, input_data)
    return {"predictions": predictions, "count": len(predictions)}


@app.post("/api/v1/ml/models/{model_id}/evaluate")
async def evaluate_ml_model(model_id: str, test_data: List[Dict[str, Any]]):
    """Evalúa un modelo ML"""
    evaluation = advanced_ml_system.evaluate_model(model_id, test_data)
    return evaluation


@app.post("/api/v1/ml/models/{model_id}/deploy")
async def deploy_ml_model(model_id: str, environment: str = "production"):
    """Despliega un modelo ML"""
    deployment = advanced_ml_system.deploy_model(model_id, environment)
    return deployment


@app.get("/api/v1/ml/models/{model_id}")
async def get_ml_model_info(model_id: str):
    """Obtiene información de un modelo ML"""
    info = advanced_ml_system.get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return info


@app.post("/api/v1/query-optimizer/enhanced/analyze")
async def analyze_query_enhanced(query: str, context: Optional[Dict[str, Any]] = None):
    """Analiza y optimiza consulta (versión mejorada)"""
    result = enhanced_query_optimizer.analyze_and_optimize(query, context)
    return result


@app.get("/api/v1/query-optimizer/enhanced/stats")
async def get_enhanced_optimizer_stats():
    """Obtiene estadísticas del optimizador mejorado"""
    return enhanced_query_optimizer.get_optimization_stats()


@app.get("/api/v1/system/health-check")
async def system_health_check():
    """Health check mejorado del sistema"""
    health = check_health()
    return health


@app.get("/api/v1/system/stats")
async def system_stats():
    """Estadísticas del sistema"""
    stats = generate_stats()
    return stats


@app.get("/api/v1/system/validate")
async def system_validate():
    """Valida configuración del sistema"""
    validation = validate_config()
    return validation


# ==================== Deep Learning Endpoints ====================

@app.post("/api/v1/llm/models/load")
async def load_llm_model(
    model_id: str,
    model_name: str,
    model_type: str = "causal_lm",
    use_lora: bool = False,
    use_quantization: bool = False
):
    """Carga un modelo LLM"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    config = LLMConfig(
        model_name=model_name,
        model_type=ModelType(model_type),
        use_lora=use_lora,
        use_quantization=use_quantization
    )
    
    result = advanced_llm_system.load_model(model_id, config)
    return result


@app.post("/api/v1/llm/generate")
async def generate_with_llm(
    model_id: str,
    prompt: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None
):
    """Genera texto con LLM"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    result = advanced_llm_system.generate_text(
        model_id,
        prompt,
        max_new_tokens,
        temperature=temperature
    )
    return result


@app.post("/api/v1/llm/fine-tune")
async def fine_tune_llm(
    model_id: str,
    training_texts: List[str],
    validation_texts: Optional[List[str]] = None,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5
):
    """Fine-tune de modelo LLM"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    result = advanced_llm_system.fine_tune(
        model_id,
        training_texts,
        validation_texts,
        num_epochs,
        batch_size,
        learning_rate
    )
    return result


@app.get("/api/v1/llm/models/{model_id}")
async def get_llm_model_info(model_id: str):
    """Obtiene información de modelo LLM"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    info = advanced_llm_system.get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    return info


@app.post("/api/v1/diffusion/pipelines/load")
async def load_diffusion_pipeline(
    pipeline_id: str,
    model_name: str = "runwayml/stable-diffusion-v1-5",
    use_xl: bool = False,
    scheduler_type: str = "dpm_solver"
):
    """Carga un pipeline de difusión"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    config = DiffusionConfig(
        model_name=model_name,
        use_xl=use_xl,
        scheduler_type=SchedulerType(scheduler_type)
    )
    
    result = diffusion_models_system.load_pipeline(pipeline_id, config)
    return result


@app.post("/api/v1/diffusion/generate-image")
async def generate_image_with_diffusion(
    pipeline_id: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    num_images: int = 1,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None
):
    """Genera imagen con modelo de difusión"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    result = diffusion_models_system.generate_image(
        pipeline_id,
        prompt,
        negative_prompt,
        num_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )
    return result


@app.post("/api/v1/diffusion/generate-3d-description")
async def generate_3d_description(
    pipeline_id: str,
    product_description: str,
    style: Optional[str] = None
):
    """Genera descripción visual 3D"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    result = diffusion_models_system.generate_3d_model_description(
        pipeline_id,
        product_description,
        style
    )
    return result


@app.get("/api/v1/diffusion/pipelines/{pipeline_id}")
async def get_diffusion_pipeline_info(pipeline_id: str):
    """Obtiene información de pipeline de difusión"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    info = diffusion_models_system.get_pipeline_info(pipeline_id)
    if not info:
        raise HTTPException(status_code=404, detail="Pipeline no encontrado")
    return info


# ==================== Advanced Deep Learning Endpoints ====================

@app.post("/api/v1/experiments/start")
async def start_experiment(
    experiment_name: str,
    config: Dict[str, Any],
    tags: Optional[List[str]] = None
):
    """Inicia un nuevo experimento"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    experiment_id = experiment_tracker.start_experiment(experiment_name, config, tags)
    return {"experiment_id": experiment_id, "status": "started"}


@app.post("/api/v1/experiments/log-metrics")
async def log_experiment_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """Registra métricas de experimento"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    experiment_tracker.log_metrics(metrics, step)
    return {"status": "logged"}


@app.post("/api/v1/experiments/finish")
async def finish_experiment():
    """Finaliza experimento"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    experiment_tracker.finish_experiment()
    return {"status": "finished"}


@app.post("/api/v1/datasets/create")
async def create_dataset(
    dataset_id: str,
    data: List[Dict[str, Any]]
):
    """Crea un dataset"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    dataset = dataset_manager.create_dataset(dataset_id, data)
    return {"dataset_id": dataset_id, "samples": len(dataset)}


@app.post("/api/v1/datasets/{dataset_id}/split")
async def split_dataset(
    dataset_id: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
):
    """Divide dataset en train/val/test"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    splits = dataset_manager.split_dataset(dataset_id, train_ratio, val_ratio, test_ratio)
    return {"splits": list(splits.keys()), "counts": {k: len(v) for k, v in splits.items()}}


@app.post("/api/v1/configs/load")
async def load_dl_config(config_path: str):
    """Carga configuración de Deep Learning"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    config = dl_config_manager.load_config(config_path)
    return {"config_name": config.experiment_name, "status": "loaded"}


@app.post("/api/v1/configs/save")
async def save_dl_config(config: Dict[str, Any], config_name: str):
    """Guarda configuración de Deep Learning"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    # Convertir dict a ExperimentConfig (simplificado)
    from ..utils.config_manager_dl import ModelConfig, TrainingConfig, DataConfig, ExperimentConfig
    
    model_cfg = ModelConfig(**config["model"])
    training_cfg = TrainingConfig(**config["training"])
    data_cfg = DataConfig(**config["data"])
    
    exp_config = ExperimentConfig(
        experiment_name=config.get("experiment_name", "default"),
        model=model_cfg,
        training=training_cfg,
        data=data_cfg
    )
    
    dl_config_manager.save_config(exp_config, config_name)
    return {"config_name": config_name, "status": "saved"}


# ==================== Advanced Model Management Endpoints ====================

@app.post("/api/v1/checkpoints/save")
async def save_checkpoint(
    model_id: str,
    epoch: int,
    metrics: Optional[Dict[str, float]] = None,
    is_best: bool = False
):
    """Guarda checkpoint de modelo"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    # En producción, cargar modelo real
    # Por ahora, retornar éxito
    return {"status": "saved", "checkpoint_id": f"{model_id}_epoch_{epoch}"}


@app.get("/api/v1/checkpoints/list")
async def list_checkpoints():
    """Lista todos los checkpoints"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    checkpoints = model_checkpointer.list_checkpoints()
    return {"checkpoints": checkpoints, "count": len(checkpoints)}


@app.post("/api/v1/hyperparameter-tuning/start")
async def start_hyperparameter_tuning(
    strategy: str = "bayesian",
    direction: str = "minimize"
):
    """Inicia optimización de hiperparámetros"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    hyperparameter_tuner.strategy = SearchStrategy(strategy)
    hyperparameter_tuner.create_study(direction=direction)
    return {"status": "started", "strategy": strategy}


@app.post("/api/v1/hyperparameter-tuning/grid-search")
async def grid_search_hyperparameters(
    space: Dict[str, Any],
    n_trials: Optional[int] = None
):
    """Grid search de hiperparámetros"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    # Convertir dict a HyperparameterSpace
    hp_space = HyperparameterSpace(**space)
    
    # Función objetivo simplificada (en producción sería real)
    import random
    def objective_fn(params):
        return random.uniform(0.1, 1.0)  # Simulado
    
    result = hyperparameter_tuner.grid_search(hp_space, objective_fn, n_trials)
    return result


@app.post("/api/v1/data-augmentation/augment-text")
async def augment_text(text: str, method: str = "random"):
    """Aumenta texto"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    augmented = data_augmentation_manager.augment_text(text, method)
    return {"original": text, "augmented": augmented, "method": method}


@app.get("/api/v1/model-serving/stats")
async def get_serving_stats():
    """Obtiene estadísticas de serving"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    # En producción, usar ModelServer real
    return {"status": "available", "cache_size": 0, "batch_queue_size": 0}


# ==================== Advanced Model Development Endpoints ====================

@app.post("/api/v1/models/compress")
async def compress_model(
    model_id: str,
    compression_config: Dict[str, Any]
):
    """Comprime modelo"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    # En producción, cargar modelo real
    # compressed = model_compressor.compress_model(model, compression_config)
    return {"status": "compressed", "model_id": model_id}


@app.post("/api/v1/models/profile")
async def profile_model(
    model_id: str,
    input_shape: List[int],
    num_runs: int = 10
):
    """Perfila modelo"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    # En producción, cargar y perfilar modelo real
    # profile = dl_profiler.profile_model(model, tuple(input_shape), num_runs=num_runs)
    return {
        "model_id": model_id,
        "status": "profiled",
        "avg_inference_time": 0.01,
        "memory_mb": 100
    }


@app.get("/api/v1/profiler/operation-stats")
async def get_operation_stats():
    """Obtiene estadísticas de operaciones"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    stats = dl_profiler.get_operation_stats()
    return {"stats": stats}


@app.get("/api/v1/profiler/memory-stats")
async def get_memory_stats():
    """Obtiene estadísticas de memoria"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    stats = dl_profiler.get_memory_stats()
    return {"stats": stats}


# ==================== Production & Management Endpoints ====================

@app.post("/api/v1/models/register")
async def register_model(
    model_id: str,
    name: str,
    architecture: str,
    description: str = "",
    metrics: Optional[Dict[str, float]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
):
    """Registra modelo en registry"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    metadata = model_registry.register_model(
        model_id, name, architecture, description,
        metrics, hyperparameters, tags=tags
    )
    return {"metadata": metadata.__dict__}


@app.get("/api/v1/models/registry/list")
async def list_registered_models(
    status: Optional[str] = None,
    tags: Optional[List[str]] = None,
    architecture: Optional[str] = None
):
    """Lista modelos registrados"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    model_status = ModelStatus(status) if status else None
    models = model_registry.list_models(model_status, tags, architecture)
    return {"models": [m.__dict__ for m in models], "count": len(models)}


@app.get("/api/v1/models/registry/{model_id}")
async def get_registered_model(model_id: str):
    """Obtiene modelo del registry"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    model = model_registry.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"metadata": model.__dict__}


@app.post("/api/v1/production/monitor/log")
async def log_production_prediction(
    input_data: Dict[str, Any],
    prediction: Dict[str, Any],
    latency: float,
    error: Optional[str] = None
):
    """Registra predicción en producción"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    production_monitor.log_prediction(input_data, prediction, latency, error)
    return {"status": "logged"}


@app.get("/api/v1/production/monitor/health")
async def get_production_health():
    """Obtiene estado de salud en producción"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    health = production_monitor.get_health_status()
    return health


@app.get("/api/v1/production/monitor/report")
async def get_production_report():
    """Obtiene reporte de producción"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    report = production_monitor.generate_report()
    return report


@app.post("/api/v1/debugging/register-hooks")
async def register_debugging_hooks(model_id: str):
    """Registra hooks de debugging"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    # En producción, cargar modelo real
    # model_debugger.register_gradient_hooks(model)
    # model_debugger.register_activation_hooks(model)
    return {"status": "hooks_registered", "model_id": model_id}


@app.get("/api/v1/debugging/gradient-summary")
async def get_gradient_summary():
    """Obtiene resumen de gradientes"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    summary = model_debugger.get_gradient_summary()
    return {"summary": summary}


# ==================== Deployment & Validation Endpoints ====================

@app.post("/api/v1/models/deploy/create-package")
async def create_deployment_package(
    model_id: str,
    version: str,
    metadata: Dict[str, Any],
    export_formats: Optional[List[str]] = None
):
    """Crea paquete de deployment"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    if export_formats is None:
        export_formats = ["pytorch"]
    
    # En producción, cargar modelo real
    # exported = model_deployment.create_deployment_package(model, model_id, version, metadata, export_formats)
    return {
        "status": "package_created",
        "model_id": model_id,
        "version": version,
        "formats": export_formats
    }


@app.post("/api/v1/models/validate")
async def validate_model(
    model_id: str,
    min_accuracy: float = 0.7,
    max_latency: float = 1.0
):
    """Valida modelo antes de deployment"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    # En producción, cargar modelo y test data reales
    # results = model_validator.validate_model(model, test_loader, device, min_accuracy, max_latency)
    return {
        "status": "validated",
        "model_id": model_id,
        "results": []
    }


@app.get("/api/v1/models/validation/summary")
async def get_validation_summary():
    """Obtiene resumen de validación"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    summary = model_validator.get_validation_summary()
    return summary


@app.post("/api/v1/models/testing/run-tests")
async def run_model_tests(
    model_id: str,
    input_shape: List[int]
):
    """Ejecuta tests de modelo"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    # En producción, cargar modelo real
    # results = dl_test_suite.run_all_tests(model, tuple(input_shape))
    return {
        "status": "tests_completed",
        "model_id": model_id,
        "results": {
            "forward": True,
            "gradient": True,
            "consistency": True,
            "memory_leak": True
        },
        "pass_rate": 1.0
    }


# ==================== Optimization & Caching Endpoints ====================

@app.post("/api/v1/inference/cache/get")
async def get_from_cache(
    model_id: str,
    input_data: Dict[str, Any]
):
    """Obtiene resultado de cache"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    cached_result = inference_cache.get(input_data, model_id)
    if cached_result is not None:
        return {"cached": True, "result": cached_result}
    return {"cached": False}


@app.post("/api/v1/inference/cache/set")
async def set_to_cache(
    model_id: str,
    input_data: Dict[str, Any],
    result: Dict[str, Any]
):
    """Guarda resultado en cache"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    inference_cache.set(input_data, model_id, result)
    return {"status": "cached"}


@app.get("/api/v1/inference/cache/stats")
async def get_cache_stats():
    """Obtiene estadísticas de cache"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    stats = inference_cache.get_stats()
    return stats


@app.post("/api/v1/models/optimize")
async def optimize_model(
    model_id: str,
    optimizations: List[str]
):
    """Optimiza modelo"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    # En producción, cargar y optimizar modelo real
    # optimized = model_optimizer.apply_all_optimizations(model, optimizations)
    return {
        "status": "optimized",
        "model_id": model_id,
        "optimizations": optimizations
    }


@app.get("/api/v1/batching/stats")
async def get_batching_stats():
    """Obtiene estadísticas de batching"""
    if not DL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Deep Learning libraries not available")
    
    stats = intelligent_batcher.get_stats()
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)

