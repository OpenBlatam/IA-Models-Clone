"""
Intelligent Routing System with Deep Learning
==============================================

Sistema de enrutamiento inteligente avanzado usando Deep Learning, Transformers,
Graph Neural Networks y técnicas modernas de ML para optimizar rutas y distribuir carga.

Arquitectura:
- routing_config.py: Configuración centralizada
- routing_models.py: Modelos de deep learning
- routing_utils.py: Utilidades de routing
- routing_strategies.py: Estrategias de enrutamiento
- routing_optimizer.py: Optimizaciones de rendimiento
- deep_learning_routing.py: Router de deep learning
- llm_route_optimizer.py: Optimizador LLM
"""

import logging
import uuid
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

# Initialize logger first
logger = logging.getLogger(__name__)

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Deep learning features will be disabled.")

# Transformers imports
try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        AutoConfig,
        GPT2Model,
        GPT2Config,
        BertModel,
        BertConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Transformer features will be disabled.")

# Importar módulos modulares
try:
    from .routing import (
        NodeFeatureExtractor,
        PathGenerator,
        RouteMetricsCalculator,
        GraphBuilder,
        BaseRoutingStrategy,
        create_strategy
    )
    MODULAR_IMPORTS_AVAILABLE = True
except ImportError as e:
    MODULAR_IMPORTS_AVAILABLE = False
    logger.warning(f"Modular routing imports not available: {e}")

# Importar módulos de routing mejorados
try:
    from .deep_learning_routing import (
        DeepLearningRouter,
        RouteFeatures,
        RoutePredictionModel
    )
    DL_ROUTING_AVAILABLE = True
except ImportError:
    DL_ROUTING_AVAILABLE = False
    logger.warning("Deep learning routing module not available")

try:
    from .transformer_routing import (
        TransformerRouteAnalyzer,
        ContextualRouteInfo
    )
    TRANSFORMER_ROUTING_AVAILABLE = True
except ImportError:
    TRANSFORMER_ROUTING_AVAILABLE = False
    logger.warning("Transformer routing module not available")

try:
    from .llm_route_optimizer import (
        LLMRouteOptimizer,
        RouteExplanation
    )
    LLM_ROUTING_AVAILABLE = True
except ImportError:
    LLM_ROUTING_AVAILABLE = False
    logger.warning("LLM route optimizer module not available")

try:
    from .gnn_routing import (
        GNNRouteOptimizer,
        GraphRouteData
    )
    GNN_ROUTING_AVAILABLE = True
except ImportError:
    GNN_ROUTING_AVAILABLE = False
    logger.warning("GNN routing module not available")

try:
    from .rl_routing import (
        RLRouteOptimizer,
        RouteState,
        RouteAction
    )
    RL_ROUTING_AVAILABLE = True
except ImportError:
    RL_ROUTING_AVAILABLE = False
    logger.warning("RL routing module not available")

try:
    from .experiment_tracker import ExperimentTracker
    EXPERIMENT_TRACKING_AVAILABLE = True
except ImportError:
    EXPERIMENT_TRACKING_AVAILABLE = False
    logger.warning("Experiment tracking module not available")

# Importar optimizadores de rendimiento
try:
    from .routing import (
        FastPathCache,
        BatchRouteProcessor,
        GraphHashCalculator,
        VectorizedPathOperations,
        ModelInferenceOptimizer,
        RoutePrecomputation
    )
    OPTIMIZER_AVAILABLE = True
except ImportError as e:
    OPTIMIZER_AVAILABLE = False
    logger.warning(f"Routing optimizer not available: {e}")

# Importar optimizaciones de rendimiento avanzadas
try:
    from .routing_performance import (
        JITModelOptimizer,
        FastInferenceEngine,
        ParallelRouteProcessor,
        VectorizedOperations,
        ModelQuantization,
        FastCache,
        PerformanceProfiler
    )
    PERFORMANCE_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    PERFORMANCE_OPTIMIZER_AVAILABLE = False
    logger.warning(f"Performance optimizer not available: {e}")

# Importar validación y debugging
try:
    from .routing_validation import (
        DataValidator,
        GradientMonitor
    )
    VALIDATION_AVAILABLE = True
except ImportError as e:
    VALIDATION_AVAILABLE = False
    logger.warning(f"Validation module not available: {e}")

# Importar cross-validation
try:
    from .routing_cross_validation import (
        CrossValidator,
        CrossValidationResult
    )
    CROSS_VALIDATION_AVAILABLE = True
except ImportError as e:
    CROSS_VALIDATION_AVAILABLE = False
    logger.warning(f"Cross-validation module not available: {e}")

# Importar configuración YAML
try:
    from .routing_config_yaml import YAMLConfigLoader
    YAML_CONFIG_AVAILABLE = True
except ImportError as e:
    YAML_CONFIG_AVAILABLE = False
    logger.warning(f"YAML config module not available: {e}")

# Importar optimizaciones ultra-rápidas
try:
    from .routing_ultra_fast import (
        UltraFastRouter,
        GPUAccelerator,
        FastJITCompiler,
        VectorizedRouteCalculator,
        PrecomputationEngine,
        AsyncRouteProcessor,
        MemoryPool
    )
    ULTRA_FAST_AVAILABLE = True
except ImportError as e:
    ULTRA_FAST_AVAILABLE = False
    logger.warning(f"Ultra-fast optimizations not available: {e}")

# Importar optimizaciones extremas de rendimiento
try:
    from .routing_extreme_performance import (
        ExtremePerformanceRouter,
        DynamicBatching,
        ONNXRuntimeOptimizer,
        INT8Quantizer,
        SharedMemoryCache,
        ProcessPoolExecutor,
        TorchScriptOptimizer
    )
    EXTREME_PERFORMANCE_AVAILABLE = True
except ImportError as e:
    EXTREME_PERFORMANCE_AVAILABLE = False
    logger.warning(f"Extreme performance optimizations not available: {e}")

# Importar optimizaciones avanzadas adicionales
try:
    from .routing_advanced_optimizations import (
        AdvancedPerformanceRouter,
        KernelFusionOptimizer,
        GraphOptimizer,
        MemoryOptimizer,
        OperatorFusion,
        CacheOptimizer
    )
    ADVANCED_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    ADVANCED_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Advanced optimizations not available: {e}")

# Importar optimizaciones específicas de ML
try:
    from .routing_ml_optimizations import (
        MLPerformanceRouter,
        ModelPruner,
        KnowledgeDistiller,
        ModelEnsembler,
        AdaptiveLearningRate,
        BatchNormFreezer
    )
    ML_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    ML_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"ML optimizations not available: {e}")

# Importar optimizaciones a nivel de sistema
try:
    from .routing_system_optimizations import (
        SystemOptimizer,
        CPUAffinityOptimizer,
        MemoryOptimizer,
        IOOptimizer,
        ThreadPoolOptimizer,
        SystemPerformanceMonitor
    )
    SYSTEM_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    SYSTEM_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"System optimizations not available: {e}")

# Importar optimizaciones de compilación
try:
    from .routing_compilation_optimizations import (
        CompilationOptimizer,
        NumbaJITOptimizer,
        CompiledRouteCalculator,
        FunctionCache
    )
    COMPILATION_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    COMPILATION_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Compilation optimizations not available: {e}")

# Importar optimizaciones de red
try:
    from .routing_network_optimizations import (
        NetworkOptimizer,
        ConnectionPool,
        RequestBatcher,
        AsyncRequestHandler
    )
    NETWORK_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    NETWORK_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Network optimizations not available: {e}")

# Importar optimizaciones de cache
try:
    from .routing_cache_optimizations import (
        CacheOptimizer,
        DistributedCache,
        MultiLevelCache,
        CacheWarmer
    )
    CACHE_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    CACHE_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Cache optimizations not available: {e}")

# Importar optimizaciones de seguridad
try:
    from .routing_security_optimizations import (
        SecurityOptimizer,
        RateLimiter,
        InputValidator
    )
    SECURITY_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    SECURITY_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Security optimizations not available: {e}")

# Importar optimizaciones de monitoreo
try:
    from .routing_monitoring_optimizations import (
        MonitoringOptimizer,
        MetricsCollector,
        PerformanceAnalytics,
        AlertManager
    )
    MONITORING_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    MONITORING_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Monitoring optimizations not available: {e}")

# Importar optimizaciones de deployment
try:
    from .routing_deployment_optimizations import (
        DeploymentOptimizer,
        HealthChecker,
        GracefulShutdown,
        ResourceManager
    )
    DEPLOYMENT_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    DEPLOYMENT_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Deployment optimizations not available: {e}")

# Importar optimizaciones de logging
try:
    from .routing_logging_optimizations import (
        LoggingOptimizer,
        StructuredLogger,
        PerformanceLogger
    )
    LOGGING_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    LOGGING_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Logging optimizations not available: {e}")

# Importar optimizaciones de backup
try:
    from .routing_backup_optimizations import (
        BackupOptimizer,
        SnapshotManager,
        AutoBackup,
        RecoveryManager
    )
    BACKUP_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    BACKUP_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Backup optimizations not available: {e}")

# Importar optimizaciones de API
try:
    from .routing_api_optimizations import (
        APIOptimizer,
        RequestValidator,
        ResponseCache,
        APIVersionManager
    )
    API_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    API_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"API optimizations not available: {e}")

# Importar optimizaciones de serialización
try:
    from .routing_serialization_optimizations import (
        SerializationOptimizer,
        FastSerializer,
        CompressedSerializer
    )
    SERIALIZATION_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    SERIALIZATION_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Serialization optimizations not available: {e}")

# Importar optimizaciones de testing
try:
    from .routing_testing_optimizations import (
        TestOptimizer,
        TestDataGenerator,
        PerformanceTester
    )
    TESTING_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    TESTING_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Testing optimizations not available: {e}")

# Importar optimizaciones de documentación
try:
    from .routing_documentation_optimizations import (
        DocumentationOptimizer,
        CodeAnalyzer,
        DocumentationGenerator
    )
    DOCUMENTATION_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    DOCUMENTATION_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Documentation optimizations not available: {e}")

# Importar optimizaciones de manejo de errores
try:
    from .routing_error_handling_optimizations import (
        ErrorHandlingOptimizer,
        CircuitBreaker,
        RetryHandler,
        ErrorRecovery
    )
    ERROR_HANDLING_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    ERROR_HANDLING_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Error handling optimizations not available: {e}")

# Importar optimizaciones de configuración
try:
    from .routing_configuration_optimizations import (
        ConfigurationOptimizer,
        ConfigurationValidator,
        HotReloadManager,
        EnvironmentConfigLoader
    )
    CONFIGURATION_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    CONFIGURATION_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Configuration optimizations not available: {e}")

# Importar optimizaciones de escalabilidad
try:
    from .routing_scalability_optimizations import (
        ScalabilityOptimizer,
        LoadBalancer,
        ShardingManager,
        DistributedProcessor
    )
    SCALABILITY_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    SCALABILITY_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Scalability optimizations not available: {e}")

# Importar optimizaciones de costos
try:
    from .routing_cost_optimizations import (
        CostOptimizer,
        ResourceTracker,
        AutoScaler
    )
    COST_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    COST_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Cost optimizations not available: {e}")

# Importar optimizaciones de observabilidad
try:
    from .routing_observability_optimizations import (
        ObservabilityOptimizer,
        DistributedTracer,
        TraceContext,
        LogAggregator
    )
    OBSERVABILITY_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    OBSERVABILITY_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Observability optimizations not available: {e}")

# Importar optimizaciones de compliance
try:
    from .routing_compliance_optimizations import (
        ComplianceOptimizer,
        AuditLogger,
        ComplianceChecker,
        DataGovernance
    )
    COMPLIANCE_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    COMPLIANCE_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Compliance optimizations not available: {e}")

# Importar optimizaciones de edge computing
try:
    from .routing_edge_optimizations import (
        EdgeOptimizer,
        EdgeCache,
        OfflineProcessor,
        LatencyOptimizer
    )
    EDGE_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    EDGE_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Edge optimizations not available: {e}")

# Importar optimizaciones de A/B testing
try:
    from .routing_ab_testing_optimizations import (
        ABTestingOptimizer,
        ExperimentManager,
        FeatureFlagManager
    )
    AB_TESTING_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    AB_TESTING_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"A/B testing optimizations not available: {e}")

# Importar optimizaciones de federated learning
try:
    from .routing_federated_optimizations import (
        FederatedOptimizer,
        FederatedClient,
        FederatedAggregator,
        PrivacyPreserver
    )
    FEDERATED_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    FEDERATED_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Federated learning optimizations not available: {e}")

# Importar optimizaciones de real-time analytics
try:
    from .routing_realtime_analytics_optimizations import (
        RealTimeAnalyticsOptimizer,
        StreamProcessor,
        RealTimeMetrics,
        EventProcessor
    )
    REALTIME_ANALYTICS_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    REALTIME_ANALYTICS_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Real-time analytics optimizations not available: {e}")

# Importar optimizaciones de graph database
try:
    from .routing_graphdb_optimizations import (
        GraphDBOptimizer,
        GraphIndex,
        GraphQueryEngine,
        GraphAlgorithmExecutor
    )
    GRAPHDB_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    GRAPHDB_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Graph database optimizations not available: {e}")

# Importar optimizaciones de multi-cloud
try:
    from .routing_multicloud_optimizations import (
        MultiCloudOptimizer,
        MultiCloudManager,
        CrossCloudLoadBalancer,
        CloudProvider,
        CloudRegion
    )
    MULTICLOUD_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    MULTICLOUD_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Multi-cloud optimizations not available: {e}")

# Importar optimizaciones de disaster recovery
try:
    from .routing_disaster_recovery_optimizations import (
        DisasterRecoveryOptimizer,
        BackupStrategy,
        FailoverManager,
        RecoveryProcedure,
        RecoveryStrategy
    )
    DISASTER_RECOVERY_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    DISASTER_RECOVERY_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Disaster recovery optimizations not available: {e}")

# Importar optimizaciones de computación cuántica
try:
    from .routing_quantum_optimizations import (
        QuantumOptimizer,
        QuantumRouter,
        QuantumAlgorithm,
        QAOAOptimizer,
        QuantumAnnealingOptimizer
    )
    QUANTUM_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    QUANTUM_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Quantum optimizations not available: {e}")

# Importar optimizaciones de blockchain
try:
    from .routing_blockchain_optimizations import (
        BlockchainOptimizer,
        Blockchain,
        SmartContract,
        ConsensusAlgorithm
    )
    BLOCKCHAIN_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    BLOCKCHAIN_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Blockchain optimizations not available: {e}")

# Importar optimizaciones de IoT
try:
    from .routing_iot_optimizations import (
        IoTOptimizer,
        IoTDevice,
        IoTProtocol,
        PowerMode
    )
    IOT_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    IOT_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"IoT optimizations not available: {e}")

# Importar optimizaciones de sistemas autónomos
try:
    from .routing_autonomous_optimizations import (
        AutonomousOptimizer,
        AutonomyLevel,
        AdaptationStrategy
    )
    AUTONOMOUS_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    AUTONOMOUS_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Autonomous optimizations not available: {e}")

# Importar optimizaciones de meta-learning
try:
    from .routing_meta_learning_optimizations import (
        MetaLearningOptimizer,
        MetaLearningStrategy
    )
    META_LEARNING_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    META_LEARNING_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Meta-learning optimizations not available: {e}")

# Importar optimizaciones de computación neuromórfica
try:
    from .routing_neuromorphic_optimizations import (
        NeuromorphicOptimizer,
        NeuromorphicArchitecture
    )
    NEUROMORPHIC_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    NEUROMORPHIC_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Neuromorphic optimizations not available: {e}")

# Importar optimizaciones de inteligencia de enjambre
try:
    from .routing_swarm_optimizations import (
        SwarmOptimizer,
        SwarmAlgorithm
    )
    SWARM_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    SWARM_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Swarm optimizations not available: {e}")

# Importar optimizaciones de algoritmos evolutivos
try:
    from .routing_evolutionary_optimizations import (
        EvolutionaryOptimizer,
        EvolutionaryAlgorithm
    )
    EVOLUTIONARY_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    EVOLUTIONARY_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Evolutionary optimizations not available: {e}")

# Importar optimizaciones de digital twins
try:
    from .routing_digital_twin_optimizations import (
        DigitalTwinOptimizer,
        TwinType
    )
    DIGITAL_TWIN_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    DIGITAL_TWIN_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Digital twin optimizations not available: {e}")

# Importar optimizaciones de XAI
try:
    from .routing_xai_optimizations import (
        XAIOptimizer,
        ExplanationMethod
    )
    XAI_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    XAI_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"XAI optimizations not available: {e}")

# Importar optimizaciones de continual learning
try:
    from .routing_continual_learning_optimizations import (
        ContinualLearningOptimizer,
        ContinualLearningStrategy
    )
    CONTINUAL_LEARNING_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    CONTINUAL_LEARNING_OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"Continual learning optimizations not available: {e}")


class RoutingStrategy(Enum):
    """Estrategia de enrutamiento."""
    SHORTEST_PATH = "shortest_path"
    FASTEST_PATH = "fastest_path"
    LEAST_COST = "least_cost"
    LOAD_BALANCED = "load_balanced"
    ADAPTIVE = "adaptive"
    DEEP_LEARNING = "deep_learning"
    TRANSFORMER = "transformer"
    LLM_OPTIMIZED = "llm_optimized"
    GNN_BASED = "gnn_based"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


@dataclass
class RouteNode:
    """
    Nodo de ruta en el grafo de routing.
    
    Attributes:
        node_id: Identificador único del nodo
        name: Nombre descriptivo del nodo
        position: Posición 3D del nodo (x, y, z)
        capacity: Capacidad máxima del nodo
        current_load: Carga actual del nodo
        cost: Costo asociado al nodo
        metadata: Metadatos adicionales
    """
    node_id: str
    name: str
    position: Dict[str, float]
    capacity: float = 1.0
    current_load: float = 0.0
    cost: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteEdge:
    """
    Arista de ruta conectando dos nodos.
    
    Attributes:
        edge_id: Identificador único de la arista
        from_node: ID del nodo origen
        to_node: ID del nodo destino
        distance: Distancia entre nodos
        time: Tiempo de viaje estimado
        cost: Costo de usar esta arista
        capacity: Capacidad máxima de la arista
        current_load: Carga actual de la arista
        metadata: Metadatos adicionales
    """
    edge_id: str
    from_node: str
    to_node: str
    distance: float
    time: float
    cost: float
    capacity: float = 1.0
    current_load: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Route:
    """
    Ruta completa desde un nodo origen a un nodo destino.
    
    Attributes:
        route_id: Identificador único de la ruta
        path: Lista de IDs de nodos que forman la ruta
        total_distance: Distancia total de la ruta
        total_time: Tiempo total estimado
        total_cost: Costo total de la ruta
        strategy: Estrategia usada para calcular la ruta
        created_at: Timestamp de creación
        metadata: Metadatos adicionales
        predicted_metrics: Métricas predichas por modelos ML
        confidence_score: Puntuación de confianza (0-1)
        explanation: Explicación generada por LLM
        ml_model_used: Modelo ML usado para esta ruta
        prediction_features: Features usadas para predicción
    """
    route_id: str
    path: List[str]
    total_distance: float
    total_time: float
    total_cost: float
    strategy: RoutingStrategy
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    predicted_metrics: Optional[Dict[str, float]] = None
    confidence_score: float = 0.0
    explanation: Optional[str] = None
    ml_model_used: Optional[str] = None
    prediction_features: Optional[Dict[str, Any]] = None


class IntelligentRouter:
    """
    Enrutador inteligente con capacidades de Deep Learning.
    
    Calcula rutas óptimas usando múltiples estrategias y técnicas de ML:
    - Algoritmos clásicos (Dijkstra, A*)
    - Deep Learning (GNN, Transformers)
    - Reinforcement Learning
    - LLM para explicaciones y optimización
    
    Attributes:
        strategy: Estrategia de routing por defecto
        nodes: Diccionario de nodos del grafo
        edges: Diccionario de aristas del grafo
        routes: Historial de rutas calculadas
        config: Configuración del router
    """
    
    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
        enable_deep_learning: bool = True,
        enable_transformer: bool = True,
        enable_llm: bool = True,
        config: Optional[RoutingConfig] = None
    ):
        """
        Inicializar enrutador inteligente.
        
        Args:
            strategy: Estrategia de enrutamiento por defecto
            enable_deep_learning: Habilitar módulos de deep learning
            enable_transformer: Habilitar análisis con transformers
            enable_llm: Habilitar optimización con LLM
            config: Configuración personalizada (usa default si None)
        
        Raises:
            ImportError: Si los módulos requeridos no están disponibles
        """
        # Configuración
        self.config = config if config and CONFIG_AVAILABLE else DEFAULT_ROUTING_CONFIG
        if self.config is None:
            logger.warning("No configuration available, using defaults")
        
        # Estado del router
        self.strategy = strategy
        self.nodes: Dict[str, RouteNode] = {}
        self.edges: Dict[str, RouteEdge] = {}
        self.routes: List[Route] = []
        self.max_routes = self.config.cache_max_size if self.config else 10000
        
        # Inicializar utilidades modulares
        if MODULAR_IMPORTS_AVAILABLE:
            self.feature_extractor = NodeFeatureExtractor()
            self.path_generator = PathGenerator()
            self.metrics_calculator = RouteMetricsCalculator()
            self.graph_builder = GraphBuilder()
        else:
            self.feature_extractor = None
            self.path_generator = None
            self.metrics_calculator = None
            self.graph_builder = None
        
        # Inicializar módulos de deep learning
        self.dl_router: Optional[Any] = None
        if enable_deep_learning and DL_ROUTING_AVAILABLE:
            try:
                self.dl_router = DeepLearningRouter()
                logger.info("Deep learning router inicializado")
            except Exception as e:
                logger.warning(f"Error inicializando deep learning router: {e}")
        
        self.transformer_analyzer: Optional[Any] = None
        if enable_transformer and TRANSFORMER_ROUTING_AVAILABLE:
            try:
                self.transformer_analyzer = TransformerRouteAnalyzer()
                logger.info("Transformer analyzer inicializado")
            except Exception as e:
                logger.warning(f"Error inicializando transformer analyzer: {e}")
        
        self.llm_optimizer: Optional[Any] = None
        if enable_llm and LLM_ROUTING_AVAILABLE:
            try:
                self.llm_optimizer = LLMRouteOptimizer()
                logger.info("LLM optimizer inicializado")
            except Exception as e:
                logger.warning(f"Error inicializando LLM optimizer: {e}")
        
        self.gnn_optimizer: Optional[Any] = None
        if GNN_ROUTING_AVAILABLE:
            logger.info("GNN optimizer disponible (se inicializará cuando se necesite)")
        
        self.rl_optimizer: Optional[Any] = None
        if RL_ROUTING_AVAILABLE:
            logger.info("RL optimizer disponible (se inicializará cuando se necesite)")
        
        self.experiment_tracker: Optional[Any] = None
        if EXPERIMENT_TRACKING_AVAILABLE:
            try:
                self.experiment_tracker = ExperimentTracker(
                    project_name="routing_ai",
                    use_wandb=True,
                    use_tensorboard=True
                )
                logger.info("Experiment tracker inicializado")
            except Exception as e:
                logger.warning(f"Error inicializando experiment tracker: {e}")
        
        # Estrategia de routing modular
        self.routing_strategy: Optional[BaseRoutingStrategy] = None
        if MODULAR_IMPORTS_AVAILABLE:
            try:
                self.routing_strategy = create_strategy(strategy.value)
            except Exception as e:
                logger.warning(f"Error creando estrategia modular: {e}, usando implementación básica")
        
        # Optimizadores de rendimiento
        if OPTIMIZER_AVAILABLE and self.config:
            cache_size = self.config.cache_max_size
            cache_ttl = self.config.cache_ttl
            self.path_cache = FastPathCache(max_size=cache_size, ttl=cache_ttl)
            batch_size = self.config.batch_size if self.config else 64
            self.batch_processor = BatchRouteProcessor(batch_size=batch_size)
            self.graph_hash_calc = GraphHashCalculator()
            self.model_optimizer = ModelInferenceOptimizer()
            self.performance_monitor = (
                PerformanceMonitor() if self.config.enable_performance_monitoring else None
            )
            self.precomputation: Optional[RoutePrecomputation] = None
        else:
            self.path_cache = None
            self.batch_processor = None
            self.graph_hash_calc = None
            self.model_optimizer = None
            self.performance_monitor = None
            self.precomputation = None
        
        # Optimizadores avanzados de rendimiento
        if PERFORMANCE_OPTIMIZER_AVAILABLE:
            batch_size = self.config.batch_size if self.config else 64
            cache_size = self.config.cache_max_size if self.config else 10000
            self.jit_optimizer = JITModelOptimizer()
            self.fast_inference = FastInferenceEngine(batch_size=batch_size)
            self.parallel_processor = ParallelRouteProcessor()
            self.vectorized_ops = VectorizedOperations()
            self.fast_cache = FastCache(max_size=cache_size)
            self.profiler = PerformanceProfiler()
        else:
            self.jit_optimizer = None
            self.fast_inference = None
            self.parallel_processor = None
            self.vectorized_ops = None
            self.fast_cache = None
            self.profiler = None
        
        # Validación y debugging
        if VALIDATION_AVAILABLE:
            self.data_validator = DataValidator()
            self.gradient_monitor = None  # Se inicializa cuando hay un modelo
        else:
            self.data_validator = None
            self.gradient_monitor = None
        
        # Cross-validation
        if CROSS_VALIDATION_AVAILABLE:
            self.cross_validator = None  # Se inicializa cuando se necesita
        else:
            self.cross_validator = None
        
        # YAML config loader
        if YAML_CONFIG_AVAILABLE:
            self.yaml_loader = YAMLConfigLoader()
        else:
            self.yaml_loader = None
        
        # Optimizaciones ultra-rápidas
        if ULTRA_FAST_AVAILABLE:
            use_gpu = self.config.use_gpu if self.config else True
            enable_precomp = self.config.enable_precomputation if self.config else True
            self.ultra_fast_router = UltraFastRouter(
                use_gpu=use_gpu,
                enable_precomputation=enable_precomp
            )
            # Precomputar grafo si está habilitado (se hará después de agregar nodos/edges)
            pass
        else:
            self.ultra_fast_router = None
        
        # Optimizaciones extremas de rendimiento
        if EXTREME_PERFORMANCE_AVAILABLE:
            use_onnx = self.config.use_onnx if self.config else True
            use_quantization = getattr(self.config, 'use_quantization', False) if self.config else False
            use_tensorrt = getattr(self.config, 'use_tensorrt', False) if self.config else False
            max_workers = getattr(self.config, 'max_workers', None) if self.config else None
            
            self.extreme_performance_router = ExtremePerformanceRouter(
                use_onnx=use_onnx,
                use_quantization=use_quantization,
                use_tensorrt=use_tensorrt,
                max_workers=max_workers
            )
            logger.info("Extreme performance router initialized")
        else:
            self.extreme_performance_router = None
        
        # Optimizaciones avanzadas adicionales
        if ADVANCED_OPTIMIZATIONS_AVAILABLE:
            self.advanced_optimizer = AdvancedPerformanceRouter()
            logger.info("Advanced performance optimizations initialized")
        else:
            self.advanced_optimizer = None
        
        # Optimizaciones específicas de ML
        if ML_OPTIMIZATIONS_AVAILABLE:
            self.ml_optimizer = MLPerformanceRouter()
            logger.info("ML-specific optimizations initialized")
        else:
            self.ml_optimizer = None
        
        # Optimizaciones a nivel de sistema
        if SYSTEM_OPTIMIZATIONS_AVAILABLE:
            enable_cpu_affinity = getattr(self.config, 'enable_cpu_affinity', False) if self.config else False
            enable_monitoring = getattr(self.config, 'enable_system_monitoring', True) if self.config else True
            try:
                self.system_optimizer = SystemOptimizer(
                    enable_cpu_affinity=enable_cpu_affinity,
                    enable_monitoring=enable_monitoring
                )
                # Aplicar optimizaciones automáticamente
                self.system_optimizer.optimize_all()
                logger.info("System-level optimizations initialized and applied")
            except Exception as e:
                logger.warning(f"System optimizer initialization failed: {e}")
                self.system_optimizer = None
        else:
            self.system_optimizer = None
        
        # Optimizaciones de compilación
        if COMPILATION_OPTIMIZATIONS_AVAILABLE:
            try:
                self.compilation_optimizer = CompilationOptimizer()
                self.compilation_optimizer.optimize_route_calculation()
                logger.info("Compilation optimizations initialized")
            except Exception as e:
                logger.warning(f"Compilation optimizer initialization failed: {e}")
                self.compilation_optimizer = None
        else:
            self.compilation_optimizer = None
        
        # Optimizaciones de red
        if NETWORK_OPTIMIZATIONS_AVAILABLE:
            try:
                self.network_optimizer = NetworkOptimizer()
                self.network_optimizer.optimize_network_settings()
                logger.info("Network optimizations initialized")
            except Exception as e:
                logger.warning(f"Network optimizer initialization failed: {e}")
                self.network_optimizer = None
        else:
            self.network_optimizer = None
        
        # Optimizaciones de cache avanzadas
        if CACHE_OPTIMIZATIONS_AVAILABLE:
            use_distributed = getattr(self.config, 'use_distributed_cache', False) if self.config else False
            redis_host = getattr(self.config, 'redis_host', 'localhost') if self.config else 'localhost'
            try:
                self.cache_optimizer = CacheOptimizer(
                    use_distributed=use_distributed,
                    redis_host=redis_host
                )
                self.cache_optimizer.optimize_cache()
                logger.info("Advanced cache optimizations initialized")
            except Exception as e:
                logger.warning(f"Cache optimizer initialization failed: {e}")
                self.cache_optimizer = None
        else:
            self.cache_optimizer = None
        
        # Optimizaciones de seguridad
        if SECURITY_OPTIMIZATIONS_AVAILABLE:
            enable_rate_limiting = getattr(self.config, 'enable_rate_limiting', True) if self.config else True
            try:
                self.security_optimizer = SecurityOptimizer(
                    enable_rate_limiting=enable_rate_limiting
                )
                logger.info("Security optimizations initialized")
            except Exception as e:
                logger.warning(f"Security optimizer initialization failed: {e}")
                self.security_optimizer = None
        else:
            self.security_optimizer = None
        
        # Optimizaciones de monitoreo
        if MONITORING_OPTIMIZATIONS_AVAILABLE:
            enable_prometheus = getattr(self.config, 'enable_prometheus', True) if self.config else True
            try:
                self.monitoring_optimizer = MonitoringOptimizer(
                    enable_prometheus=enable_prometheus
                )
                logger.info("Monitoring optimizations initialized")
            except Exception as e:
                logger.warning(f"Monitoring optimizer initialization failed: {e}")
                self.monitoring_optimizer = None
        else:
            self.monitoring_optimizer = None
        
        # Optimizaciones de deployment
        if DEPLOYMENT_OPTIMIZATIONS_AVAILABLE:
            try:
                self.deployment_optimizer = DeploymentOptimizer()
                
                # Agregar health checks básicos
                def check_nodes_health():
                    return len(self.nodes) > 0, f"Nodes: {len(self.nodes)}"
                
                def check_routes_health():
                    return len(self.routes) >= 0, f"Routes: {len(self.routes)}"
                
                self.deployment_optimizer.add_health_check('nodes', check_nodes_health)
                self.deployment_optimizer.add_health_check('routes', check_routes_health)
                
                logger.info("Deployment optimizations initialized")
            except Exception as e:
                logger.warning(f"Deployment optimizer initialization failed: {e}")
                self.deployment_optimizer = None
        else:
            self.deployment_optimizer = None
        
        # Optimizaciones de logging
        if LOGGING_OPTIMIZATIONS_AVAILABLE:
            enable_structured = getattr(self.config, 'enable_structured_logging', True) if self.config else True
            log_file = getattr(self.config, 'log_file', None) if self.config else None
            try:
                self.logging_optimizer = LoggingOptimizer(
                    enable_structured=enable_structured,
                    log_file=log_file
                )
                logger.info("Logging optimizations initialized")
            except Exception as e:
                logger.warning(f"Logging optimizer initialization failed: {e}")
                self.logging_optimizer = None
        else:
            self.logging_optimizer = None
        
        # Optimizaciones de backup
        if BACKUP_OPTIMIZATIONS_AVAILABLE:
            snapshot_dir = getattr(self.config, 'snapshot_dir', 'snapshots') if self.config else 'snapshots'
            auto_backup_interval = getattr(self.config, 'auto_backup_interval', 3600.0) if self.config else 3600.0
            try:
                self.backup_optimizer = BackupOptimizer(
                    snapshot_dir=snapshot_dir,
                    auto_backup_interval=auto_backup_interval
                )
                
                # Configurar función de backup
                def backup_data():
                    return {
                        'nodes': {nid: {
                            'name': node.name,
                            'position': node.position,
                            'capacity': node.capacity,
                            'cost': node.cost
                        } for nid, node in self.nodes.items()},
                        'edges': {eid: {
                            'from_node': edge.from_node,
                            'to_node': edge.to_node,
                            'distance': edge.distance,
                            'time': edge.time,
                            'cost': edge.cost,
                            'capacity': edge.capacity
                        } for eid, edge in self.edges.items()},
                        'timestamp': time.time()
                    }
                
                # Habilitar auto-backup si está configurado
                enable_auto_backup = getattr(self.config, 'enable_auto_backup', False) if self.config else False
                if enable_auto_backup:
                    self.backup_optimizer.setup_auto_backup(backup_data)
                
                logger.info("Backup optimizations initialized")
            except Exception as e:
                logger.warning(f"Backup optimizer initialization failed: {e}")
                self.backup_optimizer = None
        else:
            self.backup_optimizer = None
        
        # Optimizaciones de API
        if API_OPTIMIZATIONS_AVAILABLE:
            enable_response_cache = getattr(self.config, 'enable_response_cache', True) if self.config else True
            try:
                self.api_optimizer = APIOptimizer(
                    enable_response_cache=enable_response_cache
                )
                logger.info("API optimizations initialized")
            except Exception as e:
                logger.warning(f"API optimizer initialization failed: {e}")
                self.api_optimizer = None
        else:
            self.api_optimizer = None
        
        # Optimizaciones de serialización
        if SERIALIZATION_OPTIMIZATIONS_AVAILABLE:
            serialization_format = getattr(self.config, 'serialization_format', 'auto') if self.config else 'auto'
            use_compression = getattr(self.config, 'use_compression', False) if self.config else False
            try:
                self.serialization_optimizer = SerializationOptimizer(
                    format=serialization_format,
                    use_compression=use_compression
                )
                logger.info("Serialization optimizations initialized")
            except Exception as e:
                logger.warning(f"Serialization optimizer initialization failed: {e}")
                self.serialization_optimizer = None
        else:
            self.serialization_optimizer = None
        
        # Optimizaciones de testing
        if TESTING_OPTIMIZATIONS_AVAILABLE:
            try:
                self.test_optimizer = TestOptimizer()
                logger.info("Testing optimizations initialized")
            except Exception as e:
                logger.warning(f"Test optimizer initialization failed: {e}")
                self.test_optimizer = None
        else:
            self.test_optimizer = None
        
        # Optimizaciones de documentación
        if DOCUMENTATION_OPTIMIZATIONS_AVAILABLE:
            try:
                self.documentation_optimizer = DocumentationOptimizer()
                logger.info("Documentation optimizations initialized")
            except Exception as e:
                logger.warning(f"Documentation optimizer initialization failed: {e}")
                self.documentation_optimizer = None
        else:
            self.documentation_optimizer = None
        
        # Optimizaciones de manejo de errores
        if ERROR_HANDLING_OPTIMIZATIONS_AVAILABLE:
            try:
                self.error_handling_optimizer = ErrorHandlingOptimizer()
                logger.info("Error handling optimizations initialized")
            except Exception as e:
                logger.warning(f"Error handling optimizer initialization failed: {e}")
                self.error_handling_optimizer = None
        else:
            self.error_handling_optimizer = None
        
        # Optimizaciones de configuración
        if CONFIGURATION_OPTIMIZATIONS_AVAILABLE:
            config_file = getattr(self.config, 'config_file', None) if self.config else None
            try:
                self.configuration_optimizer = ConfigurationOptimizer(config_file=config_file)
                logger.info("Configuration optimizations initialized")
            except Exception as e:
                logger.warning(f"Configuration optimizer initialization failed: {e}")
                self.configuration_optimizer = None
        else:
            self.configuration_optimizer = None
        
        # Optimizaciones de escalabilidad
        if SCALABILITY_OPTIMIZATIONS_AVAILABLE:
            num_shards = getattr(self.config, 'num_shards', 4) if self.config else 4
            try:
                self.scalability_optimizer = ScalabilityOptimizer(num_shards=num_shards)
                logger.info("Scalability optimizations initialized")
            except Exception as e:
                logger.warning(f"Scalability optimizer initialization failed: {e}")
                self.scalability_optimizer = None
        else:
            self.scalability_optimizer = None
        
        # Optimizaciones de costos
        if COST_OPTIMIZATIONS_AVAILABLE:
            try:
                self.cost_optimizer = CostOptimizer()
                logger.info("Cost optimizations initialized")
            except Exception as e:
                logger.warning(f"Cost optimizer initialization failed: {e}")
                self.cost_optimizer = None
        else:
            self.cost_optimizer = None
        
        # Optimizaciones de observabilidad
        if OBSERVABILITY_OPTIMIZATIONS_AVAILABLE:
            try:
                self.observability_optimizer = ObservabilityOptimizer()
                logger.info("Observability optimizations initialized")
            except Exception as e:
                logger.warning(f"Observability optimizer initialization failed: {e}")
                self.observability_optimizer = None
        else:
            self.observability_optimizer = None
        
        # Optimizaciones de compliance
        if COMPLIANCE_OPTIMIZATIONS_AVAILABLE:
            try:
                self.compliance_optimizer = ComplianceOptimizer()
                logger.info("Compliance optimizations initialized")
            except Exception as e:
                logger.warning(f"Compliance optimizer initialization failed: {e}")
                self.compliance_optimizer = None
        else:
            self.compliance_optimizer = None
        
        # Optimizaciones de edge computing
        if EDGE_OPTIMIZATIONS_AVAILABLE:
            try:
                self.edge_optimizer = EdgeOptimizer()
                self.edge_optimizer.optimize_for_edge()
                logger.info("Edge computing optimizations initialized")
            except Exception as e:
                logger.warning(f"Edge optimizer initialization failed: {e}")
                self.edge_optimizer = None
        else:
            self.edge_optimizer = None
        
        # Optimizaciones de A/B testing
        if AB_TESTING_OPTIMIZATIONS_AVAILABLE:
            try:
                self.ab_testing_optimizer = ABTestingOptimizer()
                logger.info("A/B testing optimizations initialized")
            except Exception as e:
                logger.warning(f"A/B testing optimizer initialization failed: {e}")
                self.ab_testing_optimizer = None
        else:
            self.ab_testing_optimizer = None
        
        # Optimizaciones de federated learning
        if FEDERATED_OPTIMIZATIONS_AVAILABLE:
            try:
                self.federated_optimizer = FederatedOptimizer()
                logger.info("Federated learning optimizations initialized")
            except Exception as e:
                logger.warning(f"Federated optimizer initialization failed: {e}")
                self.federated_optimizer = None
        else:
            self.federated_optimizer = None
        
        # Optimizaciones de real-time analytics
        if REALTIME_ANALYTICS_OPTIMIZATIONS_AVAILABLE:
            try:
                self.realtime_analytics_optimizer = RealTimeAnalyticsOptimizer()
                logger.info("Real-time analytics optimizations initialized")
            except Exception as e:
                logger.warning(f"Real-time analytics optimizer initialization failed: {e}")
                self.realtime_analytics_optimizer = None
        else:
            self.realtime_analytics_optimizer = None
        
        # Optimizaciones de graph database
        if GRAPHDB_OPTIMIZATIONS_AVAILABLE:
            try:
                self.graphdb_optimizer = GraphDBOptimizer()
                logger.info("Graph database optimizations initialized")
            except Exception as e:
                logger.warning(f"Graph database optimizer initialization failed: {e}")
                self.graphdb_optimizer = None
        else:
            self.graphdb_optimizer = None
        
        # Optimizaciones de multi-cloud
        if MULTICLOUD_OPTIMIZATIONS_AVAILABLE:
            try:
                self.multi_cloud_optimizer = MultiCloudOptimizer()
                logger.info("Multi-cloud optimizations initialized")
            except Exception as e:
                logger.warning(f"Multi-cloud optimizer initialization failed: {e}")
                self.multi_cloud_optimizer = None
        else:
            self.multi_cloud_optimizer = None
        
        # Optimizaciones de disaster recovery
        if DISASTER_RECOVERY_OPTIMIZATIONS_AVAILABLE:
            backup_strategy = getattr(self.config, 'backup_strategy', 'incremental') if self.config else 'incremental'
            try:
                self.disaster_recovery_optimizer = DisasterRecoveryOptimizer(
                    backup_strategy=backup_strategy
                )
                logger.info("Disaster recovery optimizations initialized")
            except Exception as e:
                logger.warning(f"Disaster recovery optimizer initialization failed: {e}")
                self.disaster_recovery_optimizer = None
        else:
            self.disaster_recovery_optimizer = None
        
        # Optimizaciones de computación cuántica
        if QUANTUM_OPTIMIZATIONS_AVAILABLE:
            use_quantum = getattr(self.config, 'use_quantum', False) if self.config else False
            quantum_algorithm = getattr(self.config, 'quantum_algorithm', 'qaoa') if self.config else 'qaoa'
            try:
                from .routing_quantum_optimizations import QuantumAlgorithm
                algorithm = QuantumAlgorithm.QAOA if quantum_algorithm == 'qaoa' else QuantumAlgorithm.QUANTUM_ANNEALING
                self.quantum_optimizer = QuantumOptimizer(
                    use_quantum=use_quantum,
                    algorithm=algorithm
                )
                logger.info("Quantum computing optimizations initialized")
            except Exception as e:
                logger.warning(f"Quantum optimizer initialization failed: {e}")
                self.quantum_optimizer = None
        else:
            self.quantum_optimizer = None
        
        # Optimizaciones de blockchain
        if BLOCKCHAIN_OPTIMIZATIONS_AVAILABLE:
            enable_blockchain = getattr(self.config, 'enable_blockchain', False) if self.config else False
            try:
                self.blockchain_optimizer = BlockchainOptimizer(
                    enable_blockchain=enable_blockchain
                )
                logger.info("Blockchain optimizations initialized")
            except Exception as e:
                logger.warning(f"Blockchain optimizer initialization failed: {e}")
                self.blockchain_optimizer = None
        else:
            self.blockchain_optimizer = None
        
        # Optimizaciones de IoT
        if IOT_OPTIMIZATIONS_AVAILABLE:
            enable_iot = getattr(self.config, 'enable_iot', False) if self.config else False
            iot_protocol = getattr(self.config, 'iot_protocol', 'mqtt') if self.config else 'mqtt'
            try:
                protocol = IoTProtocol.MQTT if iot_protocol == 'mqtt' else IoTProtocol.COAP
                self.iot_optimizer = IoTOptimizer(
                    enable_iot=enable_iot,
                    protocol=protocol
                )
                logger.info("IoT optimizations initialized")
            except Exception as e:
                logger.warning(f"IoT optimizer initialization failed: {e}")
                self.iot_optimizer = None
        else:
            self.iot_optimizer = None
        
        # Optimizaciones de sistemas autónomos
        if AUTONOMOUS_OPTIMIZATIONS_AVAILABLE:
            autonomy_level = getattr(self.config, 'autonomy_level', 'fully_autonomous') if self.config else 'fully_autonomous'
            try:
                level = AutonomyLevel.FULLY_AUTONOMOUS if autonomy_level == 'fully_autonomous' else AutonomyLevel.SEMI_AUTONOMOUS
                self.autonomous_optimizer = AutonomousOptimizer(
                    autonomy_level=level
                )
                logger.info("Autonomous systems optimizations initialized")
            except Exception as e:
                logger.warning(f"Autonomous optimizer initialization failed: {e}")
                self.autonomous_optimizer = None
        else:
            self.autonomous_optimizer = None
        
        # Optimizaciones de meta-learning
        if META_LEARNING_OPTIMIZATIONS_AVAILABLE:
            meta_strategy = getattr(self.config, 'meta_learning_strategy', 'maml') if self.config else 'maml'
            try:
                strategy = MetaLearningStrategy.MAML if meta_strategy == 'maml' else MetaLearningStrategy.REPTILE
                self.meta_learning_optimizer = MetaLearningOptimizer(
                    strategy=strategy
                )
                logger.info("Meta-learning optimizations initialized")
            except Exception as e:
                logger.warning(f"Meta-learning optimizer initialization failed: {e}")
                self.meta_learning_optimizer = None
        else:
            self.meta_learning_optimizer = None
        
        # Optimizaciones de computación neuromórfica
        if NEUROMORPHIC_OPTIMIZATIONS_AVAILABLE:
            enable_neuromorphic = getattr(self.config, 'enable_neuromorphic', False) if self.config else False
            neuromorphic_arch = getattr(self.config, 'neuromorphic_architecture', 'snn') if self.config else 'snn'
            try:
                from .routing_neuromorphic_optimizations import NeuromorphicArchitecture
                arch = NeuromorphicArchitecture.SPIKING_NEURAL_NETWORK if neuromorphic_arch == 'snn' else NeuromorphicArchitecture.MEMRISTOR_BASED
                self.neuromorphic_optimizer = NeuromorphicOptimizer(
                    enable_neuromorphic=enable_neuromorphic,
                    architecture=arch
                )
                logger.info("Neuromorphic computing optimizations initialized")
            except Exception as e:
                logger.warning(f"Neuromorphic optimizer initialization failed: {e}")
                self.neuromorphic_optimizer = None
        else:
            self.neuromorphic_optimizer = None
        
        # Optimizaciones de inteligencia de enjambre
        if SWARM_OPTIMIZATIONS_AVAILABLE:
            enable_swarm = getattr(self.config, 'enable_swarm', False) if self.config else False
            swarm_algorithm = getattr(self.config, 'swarm_algorithm', 'pso') if self.config else 'pso'
            try:
                algorithm = SwarmAlgorithm.PARTICLE_SWARM if swarm_algorithm == 'pso' else SwarmAlgorithm.ANT_COLONY
                self.swarm_optimizer = SwarmOptimizer(
                    algorithm=algorithm,
                    enable_swarm=enable_swarm
                )
                logger.info("Swarm intelligence optimizations initialized")
            except Exception as e:
                logger.warning(f"Swarm optimizer initialization failed: {e}")
                self.swarm_optimizer = None
        else:
            self.swarm_optimizer = None
        
        # Optimizaciones de algoritmos evolutivos
        if EVOLUTIONARY_OPTIMIZATIONS_AVAILABLE:
            enable_evolutionary = getattr(self.config, 'enable_evolutionary', False) if self.config else False
            evolutionary_algorithm = getattr(self.config, 'evolutionary_algorithm', 'ga') if self.config else 'ga'
            try:
                algorithm = EvolutionaryAlgorithm.GENETIC_ALGORITHM if evolutionary_algorithm == 'ga' else EvolutionaryAlgorithm.DIFFERENTIAL_EVOLUTION
                self.evolutionary_optimizer = EvolutionaryOptimizer(
                    algorithm=algorithm,
                    enable_evolutionary=enable_evolutionary
                )
                logger.info("Evolutionary algorithms optimizations initialized")
            except Exception as e:
                logger.warning(f"Evolutionary optimizer initialization failed: {e}")
                self.evolutionary_optimizer = None
        else:
            self.evolutionary_optimizer = None
        
        # Optimizaciones de digital twins
        if DIGITAL_TWIN_OPTIMIZATIONS_AVAILABLE:
            enable_digital_twin = getattr(self.config, 'enable_digital_twin', False) if self.config else False
            try:
                self.digital_twin_optimizer = DigitalTwinOptimizer(
                    enable_digital_twin=enable_digital_twin
                )
                logger.info("Digital twin optimizations initialized")
            except Exception as e:
                logger.warning(f"Digital twin optimizer initialization failed: {e}")
                self.digital_twin_optimizer = None
        else:
            self.digital_twin_optimizer = None
        
        # Optimizaciones de XAI
        if XAI_OPTIMIZATIONS_AVAILABLE:
            enable_xai = getattr(self.config, 'enable_xai', True) if self.config else True
            xai_method = getattr(self.config, 'xai_method', 'shap') if self.config else 'shap'
            try:
                method = ExplanationMethod.SHAP if xai_method == 'shap' else ExplanationMethod.LIME
                self.xai_optimizer = XAIOptimizer(
                    method=method,
                    enable_xai=enable_xai
                )
                logger.info("XAI optimizations initialized")
            except Exception as e:
                logger.warning(f"XAI optimizer initialization failed: {e}")
                self.xai_optimizer = None
        else:
            self.xai_optimizer = None
        
        # Optimizaciones de continual learning
        if CONTINUAL_LEARNING_OPTIMIZATIONS_AVAILABLE:
            enable_continual = getattr(self.config, 'enable_continual_learning', False) if self.config else False
            continual_strategy = getattr(self.config, 'continual_learning_strategy', 'ewc') if self.config else 'ewc'
            try:
                strategy = ContinualLearningStrategy.ELASTIC_WEIGHT_CONSOLIDATION if continual_strategy == 'ewc' else ContinualLearningStrategy.REPLAY
                self.continual_learning_optimizer = ContinualLearningOptimizer(
                    strategy=strategy,
                    enable_continual=enable_continual
                )
                logger.info("Continual learning optimizations initialized")
            except Exception as e:
                logger.warning(f"Continual learning optimizer initialization failed: {e}")
                self.continual_learning_optimizer = None
        else:
            self.continual_learning_optimizer = None
    
    def add_node(
        self,
        name: str,
        position: Dict[str, float],
        capacity: float = 1.0,
        cost: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Agregar nodo.
        
        Args:
            name: Nombre del nodo
            position: Posición (x, y, z)
            capacity: Capacidad del nodo
            cost: Costo del nodo
            metadata: Metadata adicional
            
        Returns:
            ID del nodo
        """
        node_id = str(uuid.uuid4())
        
        node = RouteNode(
            node_id=node_id,
            name=name,
            position=position,
            capacity=capacity,
            cost=cost,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        logger.debug(f"Added node: {name} ({node_id})")
        
        # Re-precomputar si está habilitado
        if self.ultra_fast_router and self.ultra_fast_router.precomputation:
            try:
                if len(self.nodes) <= self.ultra_fast_router.precomputation.max_nodes:
                    self.ultra_fast_router.precompute_graph(self.nodes, self.edges)
            except Exception as e:
                logger.debug(f"Error re-precomputing after node addition: {e}")
        
        return node_id
    
    def add_edge(
        self,
        from_node: str,
        to_node: str,
        distance: float,
        time: float,
        cost: float = 1.0,
        capacity: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Agregar arista.
        
        Args:
            from_node: ID del nodo origen
            to_node: ID del nodo destino
            distance: Distancia
            time: Tiempo de viaje
            cost: Costo
            capacity: Capacidad
            metadata: Metadata adicional
            
        Returns:
            ID de la arista
        """
        if from_node not in self.nodes:
            raise ValueError(f"From node not found: {from_node}")
        if to_node not in self.nodes:
            raise ValueError(f"To node not found: {to_node}")
        
        edge_id = str(uuid.uuid4())
        
        edge = RouteEdge(
            edge_id=edge_id,
            from_node=from_node,
            to_node=to_node,
            distance=distance,
            time=time,
            cost=cost,
            capacity=capacity,
            metadata=metadata or {}
        )
        
        self.edges[edge_id] = edge
        logger.debug(f"Added edge: {from_node} -> {to_node}")
        
        # Re-precomputar si está habilitado
        if self.ultra_fast_router and self.ultra_fast_router.precomputation:
            try:
                if len(self.nodes) <= self.ultra_fast_router.precomputation.max_nodes:
                    self.ultra_fast_router.precompute_graph(self.nodes, self.edges)
            except Exception as e:
                logger.debug(f"Error re-precomputing after edge addition: {e}")
        
        return edge_id
    
    def find_route(
        self,
        start_node: str,
        end_node: str,
        strategy: Optional[RoutingStrategy] = None
    ) -> Route:
        """
        Encontrar ruta óptima entre dos nodos.
        
        Implementa caching, precomputation y optimizaciones de rendimiento.
        Usa estrategias modulares cuando están disponibles.
        
        Args:
            start_node: ID del nodo de inicio
            end_node: ID del nodo de destino
            strategy: Estrategia de routing (usa default si None)
            
        Returns:
            Route: Objeto Route con la ruta calculada y métricas
        
        Raises:
            ValueError: Si los nodos no existen en el grafo
            RuntimeError: Si no se puede encontrar una ruta válida
        """
        # Validar inputs
        if self.data_validator:
            is_valid, error_msg = self.data_validator.validate_route_path(
                [start_node, end_node], self.nodes, min_length=2
            )
            if not is_valid:
                raise ValueError(f"Invalid route path: {error_msg}")
        
        if start_node not in self.nodes:
            raise ValueError(f"Start node not found: {start_node}")
        if end_node not in self.nodes:
            raise ValueError(f"End node not found: {end_node}")
        
        strategy = strategy or self.strategy
        start_time = time.time()
        
        # Verificar cache primero
        if self.path_cache and self.graph_hash_calc:
            graph_hash = self.graph_hash_calc.calculate_hash(self.nodes, self.edges)
            cached_result = self.path_cache.get(start_node, end_node, strategy.value, graph_hash)
            if cached_result:
                if self.performance_monitor:
                    self.performance_monitor.record_cache_event(True)
                    self.performance_monitor.record_route_time(time.time() - start_time)
                path, total_distance, total_time, total_cost, confidence = cached_result
                route_id = str(uuid.uuid4())
                route = Route(
                    route_id=route_id,
                    path=path,
                    total_distance=total_distance,
                    total_time=total_time,
                    total_cost=total_cost,
                    strategy=strategy,
                    confidence_score=confidence
                )
                self.routes.append(route)
                if len(self.routes) > self.max_routes:
                    self.routes = self.routes[-self.max_routes:]
                return route
        
        if self.performance_monitor:
            self.performance_monitor.record_cache_event(False)
        
        # Construir grafo usando módulo modular
        if self.graph_builder:
            graph = self.graph_builder.build(
                {nid: {
                    'position': node.position,
                    'capacity': node.capacity,
                    'current_load': node.current_load,
                    'cost': node.cost
                } for nid, node in self.nodes.items()},
                {eid: {
                    'from_node': edge.from_node,
                    'to_node': edge.to_node,
                    'distance': edge.distance,
                    'time': edge.time,
                    'cost': edge.cost,
                    'capacity': edge.capacity,
                    'current_load': edge.current_load
                } for eid, edge in self.edges.items()}
            )
        else:
        graph = self._build_graph()
        
        # Usar precomputación ultra-rápida si está disponible
        if self.ultra_fast_router and self.ultra_fast_router.precomputation:
            precomputed_path = self.ultra_fast_router.precomputation.get_path(start_node, end_node)
            if precomputed_path:
                # Usar ruta precomputada (ultra-rápida)
                precomputed_dist = self.ultra_fast_router.precomputation.get_distance(start_node, end_node)
                if precomputed_dist is not None:
                    # Calcular métricas rápidamente
                    total_time = precomputed_dist / 10.0  # Estimación rápida
                    total_cost = precomputed_dist * 0.5  # Estimación rápida
                    
                    route_id = str(uuid.uuid4())
                    route = Route(
                        route_id=route_id,
                        path=precomputed_path,
                        total_distance=precomputed_dist,
                        total_time=total_time,
                        total_cost=total_cost,
                        strategy=strategy,
                        confidence_score=1.0
                    )
                    self.routes.append(route)
                    if len(self.routes) > self.max_routes:
                        self.routes = self.routes[-self.max_routes:]
                    return route
        
        # Usar precomputación estándar si está disponible
        if self.precomputation:
            precomputed_dist = self.precomputation.get_precomputed_distance(start_node, end_node)
            if precomputed_dist is not None:
                # Usar distancia precomputada para acelerar
                pass
        
        # Usar estrategia modular si está disponible
        if self.routing_strategy and MODULAR_IMPORTS_AVAILABLE:
            try:
                path, total_distance, total_time, total_cost, confidence = self.routing_strategy.find_route(
                    graph, start_node, end_node, self.nodes, self.edges
                )
                if path is None:
                    raise ValueError("No path found")
            except Exception as e:
                logger.warning(f"Estrategia modular falló: {e}, usando fallback")
                path, total_distance, total_time, total_cost = self._fallback_route(graph, start_node, end_node, strategy)
                confidence = 0.5
        else:
            # Fallback a implementación básica
            path, total_distance, total_time, total_cost = self._fallback_route(graph, start_node, end_node, strategy)
            confidence = 1.0
        
        # Guardar en cache
        if self.path_cache and self.graph_hash_calc:
            graph_hash = self.graph_hash_calc.calculate_hash(self.nodes, self.edges)
            self.path_cache.put(start_node, end_node, strategy.value, graph_hash, 
                              (path, total_distance, total_time, total_cost, confidence))
        
        if self.performance_monitor:
            self.performance_monitor.record_route_time(time.time() - start_time)
        
        route_id = str(uuid.uuid4())
        
        route = Route(
            route_id=route_id,
            path=path,
            total_distance=total_distance,
            total_time=total_time,
            total_cost=total_cost,
            strategy=strategy,
            confidence_score=confidence
        )
        
        # Mejorar ruta con deep learning si está disponible (ultra-optimizado)
        if self.dl_router:
            try:
                route_features = self._extract_route_features(route, graph)
                
                # Usar optimizaciones avanzadas + extremas (MÁXIMO RENDIMIENTO)
                if self.advanced_optimizer and self.extreme_performance_router and hasattr(self.dl_router, 'model'):
                    try:
                        features_list = list(route_features.values())[:10]
                        example_input = torch.FloatTensor([features_list])
                        
                        # 0. Optimizaciones ML (pruning, batch norm freezing)
                        model_to_optimize = self.dl_router.model
                        optimized_model_name = "dl_router_advanced_extreme"
                        
                        if self.ml_optimizer:
                            try:
                                model_to_optimize = self.ml_optimizer.optimize_model_for_production(
                                    model_to_optimize,
                                    prune_amount=0.2,
                                    freeze_bn=True
                                )
                                optimized_model_name = "dl_router_ml_advanced_extreme"
                            except Exception as e:
                                logger.debug(f"ML optimization failed: {e}")
                        
                        # 1. Optimizaciones avanzadas (kernel fusion, graph optimization, memory)
                        use_kernel_fusion = self.config.use_kernel_fusion if self.config else True
                        use_graph_opt = self.config.use_graph_optimization if self.config else True
                        use_mem_opt = self.config.use_memory_optimization if self.config else True
                        
                        advanced_optimized = self.advanced_optimizer.optimize_model_completely(
                            model_to_optimize,
                            fuse_kernels=use_kernel_fusion,
                            optimize_graph=use_graph_opt,
                            optimize_memory=use_mem_opt
                        )
                        
                        # 2. Optimizaciones extremas (ONNX, Quantization, etc.)
                        optimized_model = self.extreme_performance_router.optimize_model_for_inference(
                            advanced_optimized,
                            optimized_model_name,
                            example_input
                        )
                        
                        # 3. Inferencia extremadamente rápida con dynamic batching
                        predictions = self.extreme_performance_router.batch_infer(
                            optimized_model,
                            [example_input],
                            use_dynamic_batching=True
                        )
                        if predictions:
                            route.predicted_metrics = {"predicted_value": float(predictions[0][0])}
                    except Exception as e:
                        logger.debug(f"ML + Advanced + Extreme performance inference failed: {e}")
                
                # Fallback a optimizaciones extremas solamente
                if not route.predicted_metrics and self.extreme_performance_router and hasattr(self.dl_router, 'model'):
                    try:
                        features_list = list(route_features.values())[:10]
                        example_input = torch.FloatTensor([features_list])
                        
                        # Optimizar con técnicas extremas (ONNX, Quantization, etc.)
                        optimized_model = self.extreme_performance_router.optimize_model_for_inference(
                            self.dl_router.model,
                            "dl_router_extreme",
                            example_input
                        )
                        
                        # Inferencia extremadamente rápida con dynamic batching
                        predictions = self.extreme_performance_router.batch_infer(
                            optimized_model,
                            [example_input],
                            use_dynamic_batching=True
                        )
                        if predictions:
                            route.predicted_metrics = {"predicted_value": float(predictions[0][0])}
                    except Exception as e:
                        logger.debug(f"Extreme performance inference failed: {e}")
                
                # Fallback a optimizaciones ultra-rápidas
                if not route.predicted_metrics and self.ultra_fast_router and hasattr(self.dl_router, 'model'):
                    try:
                        # Optimizar modelo con todas las técnicas
                        features_list = list(route_features.values())[:10]
                        example_input = torch.FloatTensor([features_list])
                        
                        optimized_model = self.ultra_fast_router.optimize_model(
                            self.dl_router.model,
                            "dl_router_ultra",
                            example_input
                        )
                        
                        # Inferencia ultra-rápida
                        with torch.no_grad():
                            if self.ultra_fast_router.gpu_accelerator:
                                example_input = self.ultra_fast_router.gpu_accelerator.to_device(example_input)
                            predictions = optimized_model(example_input)
                            route.predicted_metrics = {"predicted_value": float(predictions[0][0])}
                    except Exception as e:
                        logger.debug(f"Ultra-fast inference failed: {e}")
                
                # Fallback a fast inference engine estándar
                if not route.predicted_metrics and self.fast_inference and hasattr(self.dl_router, 'model'):
                    try:
                        # Compilar modelo con JIT si no está compilado
                        if self.jit_optimizer:
                            compiled_model = self.jit_optimizer.compile_model(
                                self.dl_router.model,
                                "dl_router",
                                example_input=torch.randn(1, 10).to(self.fast_inference.device)
                            )
                            # Usar fast inference
                            features_tensor = torch.FloatTensor([list(route_features.values())[:10]]).to(self.fast_inference.device)
                            predictions = self.fast_inference.batch_predict(compiled_model, [features_tensor])
                            if predictions:
                                route.predicted_metrics = {"predicted_value": float(predictions[0][0])}
                    except Exception as e:
                        logger.debug(f"Fast inference failed, using standard: {e}")
                
                # Fallback a método estándar
                if not route.predicted_metrics:
                    predicted_metrics = self.dl_router.predict_route_metrics(route_features)
                    route.predicted_metrics = predicted_metrics
                    if predicted_metrics:
                        route.confidence_score = predicted_metrics.get("success_probability", confidence)
            except Exception as e:
                logger.warning(f"Error en predicción deep learning: {e}")
        
        # Generar explicación con LLM si está disponible
        if self.llm_optimizer:
            try:
                route_dict = {
                    "route_id": route.route_id,
                    "total_distance": route.total_distance,
                    "total_time": route.total_time,
                    "total_cost": route.total_cost,
                    "strategy": route.strategy.value,
                    "path": route.path
                }
                explanation_obj = self.llm_optimizer.generate_route_explanation(route_dict)
                route.explanation = explanation_obj.explanation
            except Exception as e:
                logger.warning(f"Error generando explicación LLM: {e}")
        
        self.routes.append(route)
        if len(self.routes) > self.max_routes:
            self.routes = self.routes[-self.max_routes:]
        
        return route
    
    def _build_graph(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Construir grafo de nodos y aristas (fallback si módulo modular no está disponible)."""
        graph = {}
        
        for node_id in self.nodes:
            graph[node_id] = {}
        
        for edge in self.edges.values():
            if edge.from_node not in graph:
                graph[edge.from_node] = {}
            
            graph[edge.from_node][edge.to_node] = {
                "distance": edge.distance,
                "time": edge.time,
                "cost": edge.cost,
                "capacity": edge.capacity,
                "load": edge.current_load
            }
        
        return graph
    
    def _fallback_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str,
        strategy: RoutingStrategy
    ) -> Tuple[List[str], float, float, float]:
        """Fallback route finding usando implementación básica."""
        if strategy == RoutingStrategy.SHORTEST_PATH:
            return self._dijkstra(graph, start, end, weight="distance")
        elif strategy == RoutingStrategy.FASTEST_PATH:
            return self._dijkstra(graph, start, end, weight="time")
        elif strategy == RoutingStrategy.LEAST_COST:
            return self._dijkstra(graph, start, end, weight="cost")
        else:
            return self._dijkstra(graph, start, end, weight="distance")
    
    def _extract_route_features(self, route: Route, graph: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """Extraer características de una ruta para predicción."""
        if self.feature_extractor:
            node_features = []
            for node_id in route.path:
                if node_id in self.nodes:
                    node_data = {
                        'position': self.nodes[node_id].position,
                        'capacity': self.nodes[node_id].capacity,
                        'current_load': self.nodes[node_id].current_load,
                        'cost': self.nodes[node_id].cost
                    }
                    features = self.feature_extractor.extract(
                        node_id, node_data, 
                        {eid: {'from_node': e.from_node, 'to_node': e.to_node, 'cost': e.cost} 
                         for eid, e in self.edges.items()},
                        {nid: {'position': n.position} for nid, n in self.nodes.items()}
                    )
                    node_features.append(features.tolist())
            
            return {
                'distance': route.total_distance,
                'time': route.total_time,
                'cost': route.total_cost,
                'node_features': node_features,
                'path_length': len(route.path)
            }
        else:
            return {
                'distance': route.total_distance,
                'time': route.total_time,
                'cost': route.total_cost,
                'path_length': len(route.path)
            }
    
    def _dijkstra(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str,
        weight: str = "distance"
    ) -> tuple:
        """Algoritmo de Dijkstra."""
        import heapq
        
        distances = {node: float('inf') for node in graph}
        distances[start] = 0.0
        previous = {node: None for node in graph}
        queue = [(0.0, start)]
        
        while queue:
            current_dist, current = heapq.heappop(queue)
            
            if current == end:
                break
            
            if current_dist > distances[current]:
                continue
            
            if current not in graph:
                continue
            
            for neighbor, edge_data in graph[current].items():
                edge_weight = edge_data.get(weight, float('inf'))
                new_dist = current_dist + edge_weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(queue, (new_dist, neighbor))
        
        # Reconstruir ruta
        path = []
        current = end
        while current is not None:
            path.insert(0, current)
            current = previous[current]
        
        if path[0] != start:
            raise ValueError("No path found")
        
        # Calcular totales
        total_distance = 0.0
        total_time = 0.0
        total_cost = 0.0
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            if from_node in graph and to_node in graph[from_node]:
                edge_data = graph[from_node][to_node]
                total_distance += edge_data.get("distance", 0.0)
                total_time += edge_data.get("time", 0.0)
                total_cost += edge_data.get("cost", 0.0)
        
        return path, total_distance, total_time, total_cost
    
    def _load_balanced_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str
    ) -> tuple:
        """Ruta balanceada por carga."""
        # Usar costo ajustado por carga
        adjusted_graph = {}
        for from_node, neighbors in graph.items():
            adjusted_graph[from_node] = {}
            for to_node, edge_data in neighbors.items():
                load_factor = edge_data.get("load", 0.0) / max(edge_data.get("capacity", 1.0), 0.001)
                adjusted_cost = edge_data.get("cost", 1.0) * (1.0 + load_factor)
                adjusted_graph[from_node][to_node] = {
                    **edge_data,
                    "cost": adjusted_cost
                }
        
        return self._dijkstra(adjusted_graph, start, end, weight="cost")
    
    def _adaptive_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str
    ) -> tuple:
        """Ruta adaptativa (combina múltiples factores)."""
        # Combinar distancia, tiempo, costo y carga
        adjusted_graph = {}
        for from_node, neighbors in graph.items():
            adjusted_graph[from_node] = {}
            for to_node, edge_data in neighbors.items():
                load_factor = edge_data.get("load", 0.0) / max(edge_data.get("capacity", 1.0), 0.001)
                # Peso combinado
                combined_weight = (
                    edge_data.get("distance", 0.0) * 0.3 +
                    edge_data.get("time", 0.0) * 0.3 +
                    edge_data.get("cost", 0.0) * 0.2 +
                    load_factor * 100.0 * 0.2
                )
                adjusted_graph[from_node][to_node] = {
                    **edge_data,
                    "combined": combined_weight
                }
        
        return self._dijkstra(adjusted_graph, start, end, weight="combined")
    
    def _deep_learning_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str
    ) -> tuple:
        """Ruta usando deep learning."""
        if not self.dl_router:
            # Fallback a adaptive
            return self._adaptive_route(graph, start, end)
        
        # Generar múltiples rutas candidatas
        candidate_routes = []
        
        # Ruta más corta
        try:
            path1, d1, t1, c1 = self._dijkstra(graph, start, end, weight="distance")
            candidate_routes.append((path1, d1, t1, c1))
        except:
            pass
        
        # Ruta más rápida
        try:
            path2, d2, t2, c2 = self._dijkstra(graph, start, end, weight="time")
            candidate_routes.append((path2, d2, t2, c2))
        except:
            pass
        
        # Evaluar cada ruta con deep learning
        best_route = None
        best_score = -1
        
        for path, dist, time, cost in candidate_routes:
            route_features = RouteFeatures(
                distance=dist,
                time=time,
                cost=cost,
                capacity=1.0,
                current_load=0.0
            )
            
            try:
                metrics = self.dl_router.predict_route_metrics(route_features)
                score = metrics.get("success_probability", 0.5)
                
                if score > best_score:
                    best_score = score
                    best_route = (path, dist, time, cost)
            except:
                continue
        
        if best_route:
            return best_route
        
        # Fallback
        return self._adaptive_route(graph, start, end)
    
    def _transformer_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str
    ) -> tuple:
        """Ruta usando análisis transformer."""
        if not self.transformer_analyzer:
            return self._adaptive_route(graph, start, end)
        
        # Generar rutas candidatas
        candidate_routes = []
        try:
            path1, d1, t1, c1 = self._dijkstra(graph, start, end, weight="distance")
            candidate_routes.append({
                "path": path1,
                "distance": d1,
                "time": t1,
                "cost": c1,
                "description": f"Shortest path: {len(path1)} nodes"
            })
        except:
            pass
        
        # Analizar con transformer
        if candidate_routes:
            context = ContextualRouteInfo(
                route_description=f"Route from {start} to {end}",
                constraints=[],
                preferences=[],
                historical_data={},
                environmental_factors={}
            )
            
            try:
                recommendation = self.transformer_analyzer.generate_route_recommendation(
                    context, candidate_routes
                )
                best_route = recommendation.get("recommended_route")
                if best_route:
                    return (
                        best_route["path"],
                        best_route["distance"],
                        best_route["time"],
                        best_route["cost"]
                    )
            except Exception as e:
                logger.warning(f"Error en análisis transformer: {e}")
        
        # Fallback
        return self._adaptive_route(graph, start, end)
    
    def _llm_optimized_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str
    ) -> tuple:
        """Ruta optimizada con LLM."""
        # Primero obtener ruta base
        base_path, base_dist, base_time, base_cost = self._adaptive_route(graph, start, end)
        
        if not self.llm_optimizer:
            return (base_path, base_dist, base_time, base_cost)
        
        # Optimizar con LLM
        route_dict = {
            "route_id": "temp",
            "total_distance": base_dist,
            "total_time": base_time,
            "total_cost": base_cost,
            "path": base_path
        }
        
        try:
            optimization = self.llm_optimizer.optimize_route_with_llm(
                route_dict,
                constraints=[],
                objectives=["minimize_time", "minimize_cost"]
            )
            
            # Por ahora retornar ruta base (en el futuro se podría implementar
            # modificación de ruta basada en recomendaciones)
            return (base_path, base_dist, base_time, base_cost)
        except Exception as e:
            logger.warning(f"Error en optimización LLM: {e}")
            return (base_path, base_dist, base_time, base_cost)
    
    def _extract_route_features(
        self,
        route: Route,
        graph: Dict[str, Dict[str, Dict[str, float]]]
    ) -> RouteFeatures:
        """Extraer features de ruta para deep learning."""
        node_features = []
        edge_features = []
        
        # Extraer features de nodos
        for node_id in route.path:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node_features.extend([
                    node.capacity,
                    node.current_load,
                    node.cost,
                    *node.position.values()
                ])
        
        # Extraer features de aristas
        for i in range(len(route.path) - 1):
            from_node = route.path[i]
            to_node = route.path[i + 1]
            
            if from_node in graph and to_node in graph[from_node]:
                edge_data = graph[from_node][to_node]
                edge_features.extend([
                    edge_data.get("distance", 0.0),
                    edge_data.get("time", 0.0),
                    edge_data.get("cost", 0.0),
                    edge_data.get("capacity", 0.0),
                    edge_data.get("load", 0.0)
                ])
        
        return RouteFeatures(
            distance=route.total_distance,
            time=route.total_time,
            cost=route.total_cost,
            capacity=1.0,
            current_load=0.0,
            node_features=node_features[:10],  # Limitar
            edge_features=edge_features[:5],    # Limitar
            temporal_features=[]  # Se puede agregar información temporal
        )
    
    def find_routes_batch(
        self,
        route_requests: List[Dict[str, str]],
        strategy: Optional[RoutingStrategy] = None
    ) -> List[Route]:
        """Encontrar múltiples rutas en batch para mejor rendimiento."""
        if not self.batch_processor:
            # Fallback a procesamiento secuencial
            return [self.find_route(req['start'], req['end'], strategy) for req in route_requests]
        
        strategy = strategy or self.strategy
        graph = self._build_graph() if not self.graph_builder else self.graph_builder.build(
            {nid: {'position': n.position, 'capacity': n.capacity, 'current_load': n.current_load, 'cost': n.cost}
             for nid, n in self.nodes.items()},
            {eid: {'from_node': e.from_node, 'to_node': e.to_node, 'distance': e.distance, 
                   'time': e.time, 'cost': e.cost, 'capacity': e.capacity, 'current_load': e.current_load}
             for eid, e in self.edges.items()}
        )
        
        routes_data = [
            {'start': req['start'], 'end': req['end'], 'nodes': self.nodes, 'edges': self.edges}
            for req in route_requests
        ]
        
        if self.routing_strategy:
            strategy_func = lambda g, s, e, n, ed: self.routing_strategy.find_route(g, s, e, n, ed)
        else:
            strategy_func = lambda g, s, e, n, ed: self._fallback_route(g, s, e, strategy) + (1.0,)
        
        results = self.batch_processor.process_batch(routes_data, graph, strategy_func)
        
        routes = []
        for i, (path, dist, t, cost, conf) in enumerate(results):
            if path:
                route = Route(
                    route_id=str(uuid.uuid4()),
                    path=path,
                    total_distance=dist,
                    total_time=t,
                    total_cost=cost,
                    strategy=strategy,
                    confidence_score=conf
                )
                routes.append(route)
        
        self.routes.extend(routes)
        if len(self.routes) > self.max_routes:
            self.routes = self.routes[-self.max_routes:]
        
        return routes
    
    def enable_precomputation(self, max_nodes: int = 100):
        """Habilitar precomputación de rutas para mejor rendimiento."""
        if not OPTIMIZER_AVAILABLE:
            logger.warning("Precomputation no disponible sin routing_optimizer")
            return
        
        graph = self._build_graph()
        self.precomputation = RoutePrecomputation(graph)
        self.precomputation.precompute_all_pairs(max_nodes=max_nodes)
        logger.info(f"Precomputation habilitada para grafo con {len(self.nodes)} nodos")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del enrutador."""
        strategy_counts = {}
        ml_routes = 0
        avg_confidence = 0.0
        
        for route in self.routes:
            strategy_counts[route.strategy.value] = strategy_counts.get(route.strategy.value, 0) + 1
            if route.ml_model_used:
                ml_routes += 1
                avg_confidence += route.confidence_score
        
        avg_confidence = avg_confidence / ml_routes if ml_routes > 0 else 0.0
        
        stats = {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_routes": len(self.routes),
            "strategy_counts": strategy_counts,
            "ml_routes": ml_routes,
            "avg_confidence": avg_confidence
        }
        
        # Agregar estadísticas de rendimiento
        if self.performance_monitor:
            perf_stats = self.performance_monitor.get_stats()
            stats['performance'] = perf_stats
        
        # Agregar estadísticas de cache
        if self.path_cache:
            stats['cache_size'] = len(self.path_cache.cache)
            stats['cache_max_size'] = self.path_cache.max_size
            stats['cache_hit_rate'] = perf_stats.get('cache_hit_rate', 0.0) if self.performance_monitor else 0.0
        
        # Agregar estadísticas de deep learning
        if self.dl_router:
            stats["dl_available"] = True
            if hasattr(self.dl_router, 'training_history'):
                stats["dl_training_history"] = len(self.dl_router.training_history)
        else:
            stats["dl_available"] = False
        
        # Agregar estadísticas de transformer
        stats["transformer_available"] = self.transformer_analyzer is not None
        
        # Agregar estadísticas de LLM
        stats["llm_available"] = self.llm_optimizer is not None
        
        # Agregar estadísticas de precomputation
        stats["precomputation_enabled"] = self.precomputation is not None
        
        # Agregar estadísticas de validación
        if self.data_validator:
            stats["validation_enabled"] = True
            # Validar grafo completo
            is_valid, error_msg = self.data_validator.validate_graph(self.nodes, self.edges)
            stats["graph_valid"] = is_valid
            if not is_valid:
                stats["graph_validation_error"] = error_msg
        
        # Agregar estadísticas de gradient monitoring
        if self.gradient_monitor:
            grad_summary = self.gradient_monitor.get_gradient_summary()
            stats["gradient_monitoring"] = grad_summary
        
        # Agregar estadísticas de optimizaciones avanzadas
        if self.advanced_optimizer:
            stats["advanced_optimizations_enabled"] = True
            cache_stats = self.advanced_optimizer.get_cache_stats()
            stats["advanced_cache"] = cache_stats
            if self.dl_router and hasattr(self.dl_router, 'model'):
                memory_stats = self.advanced_optimizer.get_memory_stats(self.dl_router.model)
                stats["model_memory"] = memory_stats
        else:
            stats["advanced_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones extremas
        if self.extreme_performance_router:
            stats["extreme_performance_enabled"] = True
            stats["extreme_performance_config"] = {
                "use_onnx": self.extreme_performance_router.use_onnx,
                "use_quantization": self.extreme_performance_router.use_quantization,
                "use_tensorrt": self.extreme_performance_router.use_tensorrt
            }
        else:
            stats["extreme_performance_enabled"] = False
        
        # Agregar estadísticas de optimizaciones ML
        if self.ml_optimizer:
            stats["ml_optimizations_enabled"] = True
            if self.dl_router and hasattr(self.dl_router, 'model') and self.ml_optimizer.pruner:
                sparsity = self.ml_optimizer.pruner.get_sparsity(self.dl_router.model)
                stats["model_sparsity"] = sparsity
        else:
            stats["ml_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones del sistema
        if self.system_optimizer:
            stats["system_optimizations_enabled"] = True
            system_info = self.system_optimizer.get_system_info()
            stats["system_info"] = system_info
        else:
            stats["system_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de compilación
        if self.compilation_optimizer:
            stats["compilation_optimizations_enabled"] = True
            compilation_stats = self.compilation_optimizer.get_stats()
            stats["compilation_stats"] = compilation_stats
        else:
            stats["compilation_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de red
        if self.network_optimizer:
            stats["network_optimizations_enabled"] = True
            network_stats = self.network_optimizer.get_stats()
            stats["network_stats"] = network_stats
        else:
            stats["network_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de cache
        if self.cache_optimizer:
            stats["cache_optimizations_enabled"] = True
            cache_stats = self.cache_optimizer.get_stats()
            stats["cache_stats"] = cache_stats
        else:
            stats["cache_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de seguridad
        if self.security_optimizer:
            stats["security_optimizations_enabled"] = True
            security_stats = self.security_optimizer.get_stats()
            stats["security_stats"] = security_stats
        else:
            stats["security_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de monitoreo
        if self.monitoring_optimizer:
            stats["monitoring_optimizations_enabled"] = True
            monitoring_stats = self.monitoring_optimizer.get_stats()
            stats["monitoring_stats"] = monitoring_stats
        else:
            stats["monitoring_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de deployment
        if self.deployment_optimizer:
            stats["deployment_optimizations_enabled"] = True
            deployment_stats = self.deployment_optimizer.get_stats()
            stats["deployment_stats"] = deployment_stats
            stats["health"] = self.deployment_optimizer.get_health()
        else:
            stats["deployment_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de logging
        if self.logging_optimizer:
            stats["logging_optimizations_enabled"] = True
            logging_stats = self.logging_optimizer.get_stats()
            stats["logging_stats"] = logging_stats
        else:
            stats["logging_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de backup
        if self.backup_optimizer:
            stats["backup_optimizations_enabled"] = True
            backup_stats = self.backup_optimizer.get_stats()
            stats["backup_stats"] = backup_stats
        else:
            stats["backup_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de API
        if self.api_optimizer:
            stats["api_optimizations_enabled"] = True
            api_stats = self.api_optimizer.get_stats()
            stats["api_stats"] = api_stats
        else:
            stats["api_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de serialización
        if self.serialization_optimizer:
            stats["serialization_optimizations_enabled"] = True
            serialization_stats = self.serialization_optimizer.get_stats()
            stats["serialization_stats"] = serialization_stats
        else:
            stats["serialization_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de testing
        if self.test_optimizer:
            stats["testing_optimizations_enabled"] = True
            test_stats = self.test_optimizer.get_stats()
            stats["test_stats"] = test_stats
        else:
            stats["testing_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de documentación
        if self.documentation_optimizer:
            stats["documentation_optimizations_enabled"] = True
            doc_stats = self.documentation_optimizer.get_stats()
            stats["documentation_stats"] = doc_stats
        else:
            stats["documentation_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de manejo de errores
        if self.error_handling_optimizer:
            stats["error_handling_optimizations_enabled"] = True
            error_stats = self.error_handling_optimizer.get_stats()
            stats["error_handling_stats"] = error_stats
        else:
            stats["error_handling_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de configuración
        if self.configuration_optimizer:
            stats["configuration_optimizations_enabled"] = True
            config_stats = self.configuration_optimizer.get_stats()
            stats["configuration_stats"] = config_stats
        else:
            stats["configuration_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de escalabilidad
        if self.scalability_optimizer:
            stats["scalability_optimizations_enabled"] = True
            scalability_stats = self.scalability_optimizer.get_stats()
            stats["scalability_stats"] = scalability_stats
        else:
            stats["scalability_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de costos
        if self.cost_optimizer:
            stats["cost_optimizations_enabled"] = True
            cost_stats = self.cost_optimizer.get_stats()
            stats["cost_stats"] = cost_stats
            cost_analysis = self.cost_optimizer.get_cost_analysis()
            stats["cost_analysis"] = cost_analysis
        else:
            stats["cost_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de observabilidad
        if self.observability_optimizer:
            stats["observability_optimizations_enabled"] = True
            observability_stats = self.observability_optimizer.get_stats()
            stats["observability_stats"] = observability_stats
        else:
            stats["observability_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de compliance
        if self.compliance_optimizer:
            stats["compliance_optimizations_enabled"] = True
            compliance_stats = self.compliance_optimizer.get_stats()
            stats["compliance_stats"] = compliance_stats
        else:
            stats["compliance_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de edge computing
        if self.edge_optimizer:
            stats["edge_optimizations_enabled"] = True
            edge_stats = self.edge_optimizer.get_stats()
            stats["edge_stats"] = edge_stats
        else:
            stats["edge_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de A/B testing
        if self.ab_testing_optimizer:
            stats["ab_testing_optimizations_enabled"] = True
            ab_stats = self.ab_testing_optimizer.get_stats()
            stats["ab_testing_stats"] = ab_stats
        else:
            stats["ab_testing_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de federated learning
        if self.federated_optimizer:
            stats["federated_optimizations_enabled"] = True
            federated_stats = self.federated_optimizer.get_stats()
            stats["federated_stats"] = federated_stats
        else:
            stats["federated_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de real-time analytics
        if self.realtime_analytics_optimizer:
            stats["realtime_analytics_optimizations_enabled"] = True
            realtime_stats = self.realtime_analytics_optimizer.get_stats()
            stats["realtime_analytics_stats"] = realtime_stats
        else:
            stats["realtime_analytics_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de graph database
        if self.graphdb_optimizer:
            stats["graphdb_optimizations_enabled"] = True
            graphdb_stats = self.graphdb_optimizer.get_stats()
            stats["graphdb_stats"] = graphdb_stats
        else:
            stats["graphdb_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de multi-cloud
        if self.multi_cloud_optimizer:
            stats["multicloud_optimizations_enabled"] = True
            multicloud_stats = self.multi_cloud_optimizer.get_stats()
            stats["multicloud_stats"] = multicloud_stats
        else:
            stats["multicloud_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de disaster recovery
        if self.disaster_recovery_optimizer:
            stats["disaster_recovery_optimizations_enabled"] = True
            dr_stats = self.disaster_recovery_optimizer.get_stats()
            stats["disaster_recovery_stats"] = dr_stats
        else:
            stats["disaster_recovery_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de computación cuántica
        if self.quantum_optimizer:
            stats["quantum_optimizations_enabled"] = True
            quantum_stats = self.quantum_optimizer.get_stats()
            stats["quantum_stats"] = quantum_stats
        else:
            stats["quantum_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de blockchain
        if self.blockchain_optimizer:
            stats["blockchain_optimizations_enabled"] = True
            blockchain_stats = self.blockchain_optimizer.get_stats()
            stats["blockchain_stats"] = blockchain_stats
        else:
            stats["blockchain_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de IoT
        if self.iot_optimizer:
            stats["iot_optimizations_enabled"] = True
            iot_stats = self.iot_optimizer.get_stats()
            stats["iot_stats"] = iot_stats
        else:
            stats["iot_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de sistemas autónomos
        if self.autonomous_optimizer:
            stats["autonomous_optimizations_enabled"] = True
            autonomous_stats = self.autonomous_optimizer.get_stats()
            stats["autonomous_stats"] = autonomous_stats
        else:
            stats["autonomous_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de meta-learning
        if self.meta_learning_optimizer:
            stats["meta_learning_optimizations_enabled"] = True
            meta_stats = self.meta_learning_optimizer.get_stats()
            stats["meta_learning_stats"] = meta_stats
        else:
            stats["meta_learning_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de computación neuromórfica
        if self.neuromorphic_optimizer:
            stats["neuromorphic_optimizations_enabled"] = True
            neuromorphic_stats = self.neuromorphic_optimizer.get_stats()
            stats["neuromorphic_stats"] = neuromorphic_stats
        else:
            stats["neuromorphic_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de inteligencia de enjambre
        if self.swarm_optimizer:
            stats["swarm_optimizations_enabled"] = True
            swarm_stats = self.swarm_optimizer.get_stats()
            stats["swarm_stats"] = swarm_stats
        else:
            stats["swarm_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de algoritmos evolutivos
        if self.evolutionary_optimizer:
            stats["evolutionary_optimizations_enabled"] = True
            evolutionary_stats = self.evolutionary_optimizer.get_stats()
            stats["evolutionary_stats"] = evolutionary_stats
        else:
            stats["evolutionary_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de digital twins
        if self.digital_twin_optimizer:
            stats["digital_twin_optimizations_enabled"] = True
            digital_twin_stats = self.digital_twin_optimizer.get_stats()
            stats["digital_twin_stats"] = digital_twin_stats
        else:
            stats["digital_twin_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de XAI
        if self.xai_optimizer:
            stats["xai_optimizations_enabled"] = True
            xai_stats = self.xai_optimizer.get_stats()
            stats["xai_stats"] = xai_stats
        else:
            stats["xai_optimizations_enabled"] = False
        
        # Agregar estadísticas de optimizaciones de continual learning
        if self.continual_learning_optimizer:
            stats["continual_learning_optimizations_enabled"] = True
            continual_stats = self.continual_learning_optimizer.get_stats()
            stats["continual_learning_stats"] = continual_stats
        else:
            stats["continual_learning_optimizations_enabled"] = False
        
        return stats
    
    def validate_graph(self) -> Tuple[bool, Optional[str]]:
        """
        Validar integridad del grafo completo.
        
        Returns:
            (is_valid, error_message)
        """
        if not self.data_validator:
            return True, None
        
        return self.data_validator.validate_graph(self.nodes, self.edges)
    
    def enable_gradient_monitoring(self, model: Any, log_interval: int = 100):
        """
        Habilitar monitoreo de gradientes para debugging.
        
        Args:
            model: Modelo PyTorch a monitorear
            log_interval: Intervalo de logging
        """
        if VALIDATION_AVAILABLE:
            self.gradient_monitor = GradientMonitor(model, log_interval)
            logger.info("Gradient monitoring enabled")
        else:
            logger.warning("Validation module not available for gradient monitoring")
    
    def setup_cross_validation(
        self,
        n_splits: int = 5,
        cv_type: str = "kfold",
        shuffle: bool = True
    ):
        """
        Configurar cross-validation para entrenamiento de modelos.
        
        Args:
            n_splits: Número de folds
            cv_type: Tipo de CV ('kfold', 'stratified', 'timeseries')
            shuffle: Mezclar datos
        """
        if CROSS_VALIDATION_AVAILABLE:
            self.cross_validator = CrossValidator(
                n_splits=n_splits,
                cv_type=cv_type,
                shuffle=shuffle
            )
            logger.info(f"Cross-validation configured: {cv_type} with {n_splits} folds")
        else:
            logger.warning("Cross-validation module not available")
    
    def load_config_from_yaml(self, config_path: str) -> bool:
        """
        Cargar configuración desde archivo YAML.
        
        Args:
            config_path: Ruta al archivo YAML
        
        Returns:
            True si se cargó exitosamente
        """
        if not self.yaml_loader:
            logger.warning("YAML config loader not available")
            return False
        
        try:
            config_dict = self.yaml_loader.load_config(config_path)
            # Aquí se podría actualizar self.config con los valores del YAML
            logger.info(f"Configuration loaded from {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading YAML config: {e}")
            return False
    
    def initialize_ultra_fast_precomputation(self):
        """
        Inicializar precomputación ultra-rápida del grafo completo.
        Útil cuando el grafo está completo y se quiere precomputar todas las rutas.
        """
        if self.ultra_fast_router and self.ultra_fast_router.precomputation:
            try:
                if len(self.nodes) > 0 and len(self.nodes) <= self.ultra_fast_router.precomputation.max_nodes:
                    self.ultra_fast_router.precompute_graph(self.nodes, self.edges)
                    logger.info("Ultra-fast precomputation initialized")
                    return True
                else:
                    logger.warning(f"Graph too large ({len(self.nodes)} nodes) for precomputation (max {self.ultra_fast_router.precomputation.max_nodes})")
                    return False
            except Exception as e:
                logger.error(f"Error initializing ultra-fast precomputation: {e}")
                return False
        return False
    
    def _gnn_based_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str
    ) -> tuple:
        """Ruta usando Graph Neural Networks."""
        if not GNN_ROUTING_AVAILABLE:
            return self._adaptive_route(graph, start, end)
        
        # Inicializar GNN optimizer si no está inicializado
        if not self.gnn_optimizer:
            try:
                # Convertir nodos y aristas al formato necesario
                nodes_dict = {
                    node_id: {
                        "capacity": node.capacity,
                        "current_load": node.current_load,
                        "cost": node.cost,
                        "position": node.position
                    }
                    for node_id, node in self.nodes.items()
                }
                
                edges_dict = {
                    edge_id: {
                        "from_node": edge.from_node,
                        "to_node": edge.to_node,
                        "distance": edge.distance,
                        "time": edge.time,
                        "cost": edge.cost,
                        "capacity": edge.capacity,
                        "current_load": edge.current_load
                    }
                    for edge_id, edge in self.edges.items()
                }
                
                self.gnn_optimizer = GNNRouteOptimizer(model_type="GCN")
            except Exception as e:
                logger.warning(f"Error inicializando GNN optimizer: {e}")
                return self._adaptive_route(graph, start, end)
        
        try:
            # Construir datos de grafo
            nodes_dict = {
                node_id: {
                    "capacity": node.capacity,
                    "current_load": node.current_load,
                    "cost": node.cost,
                    "position": node.position
                }
                for node_id, node in self.nodes.items()
            }
            
            edges_dict = {
                edge_id: {
                    "from_node": edge.from_node,
                    "to_node": edge.to_node,
                    "distance": edge.distance,
                    "time": edge.time,
                    "cost": edge.cost,
                    "capacity": edge.capacity,
                    "current_load": edge.current_load
                }
                for edge_id, edge in self.edges.items()
            }
            
            graph_data = self.gnn_optimizer.build_graph_data(nodes_dict, edges_dict)
            
            # Encontrar índices de nodos
            node_ids = list(self.nodes.keys())
            start_idx = node_ids.index(start) if start in node_ids else 0
            end_idx = node_ids.index(end) if end in node_ids else len(node_ids) - 1
            
            # Encontrar camino usando GNN
            path_indices = self.gnn_optimizer.find_optimal_path_gnn(graph_data, start_idx, end_idx)
            path = [node_ids[idx] for idx in path_indices if idx < len(node_ids)]
            
            # Calcular métricas
            total_distance = 0.0
            total_time = 0.0
            total_cost = 0.0
            
            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                if from_node in graph and to_node in graph[from_node]:
                    edge_data = graph[from_node][to_node]
                    total_distance += edge_data.get("distance", 0.0)
                    total_time += edge_data.get("time", 0.0)
                    total_cost += edge_data.get("cost", 0.0)
            
            return path, total_distance, total_time, total_cost
        except Exception as e:
            logger.warning(f"Error en ruta GNN: {e}")
            return self._adaptive_route(graph, start, end)
    
    def _rl_based_route(
        self,
        graph: Dict[str, Dict[str, Dict[str, float]]],
        start: str,
        end: str
    ) -> tuple:
        """Ruta usando Reinforcement Learning."""
        if not RL_ROUTING_AVAILABLE:
            return self._adaptive_route(graph, start, end)
        
        # Inicializar RL optimizer si no está inicializado
        if not self.rl_optimizer:
            try:
                nodes_dict = {
                    node_id: {
                        "capacity": node.capacity,
                        "current_load": node.current_load,
                        "cost": node.cost,
                        "position": node.position
                    }
                    for node_id, node in self.nodes.items()
                }
                
                edges_dict = {
                    edge_id: {
                        "from_node": edge.from_node,
                        "to_node": edge.to_node,
                        "distance": edge.distance,
                        "time": edge.time,
                        "cost": edge.cost,
                        "capacity": edge.capacity,
                        "current_load": edge.current_load
                    }
                    for edge_id, edge in self.edges.items()
                }
                
                self.rl_optimizer = RLRouteOptimizer(nodes_dict, edges_dict)
            except Exception as e:
                logger.warning(f"Error inicializando RL optimizer: {e}")
                return self._adaptive_route(graph, start, end)
        
        try:
            # Encontrar ruta usando RL
            route_info = self.rl_optimizer.find_route(start, end)
            
            path = route_info["path"]
            total_distance = route_info["distance"]
            total_time = route_info["time"]
            total_cost = route_info["cost"]
            
            return path, total_distance, total_time, total_cost
        except Exception as e:
            logger.warning(f"Error en ruta RL: {e}")
            return self._adaptive_route(graph, start, end)


# Instancia global
_intelligent_router: Optional[IntelligentRouter] = None


def get_intelligent_router(
    strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE
) -> IntelligentRouter:
    """Obtener instancia global del enrutador."""
    global _intelligent_router
    if _intelligent_router is None:
        _intelligent_router = IntelligentRouter(strategy=strategy)
    return _intelligent_router


