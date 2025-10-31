"""
Ultimate TruthGPT Configuration
==============================

Configuración avanzada para la aplicación definitiva de TruthGPT con todas
las características ultra avanzadas integradas.

Características:
- Configuración modular por componentes
- Variables de entorno
- Configuración de desarrollo y producción
- Configuración de ML y AI
- Configuración de rendimiento
- Configuración de seguridad
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration."""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    connection_pool_size: int = 10
    connection_timeout: int = 5
    socket_timeout: int = 5
    retry_on_timeout: bool = True

@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    cors_origins: List[str] = None
    cors_credentials: bool = True
    cors_methods: List[str] = None
    cors_headers: List[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]
        if self.cors_methods is None:
            self.cors_methods = ["*"]
        if self.cors_headers is None:
            self.cors_headers = ["*"]

@dataclass
class AIConfig:
    """AI and ML configuration."""
    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    
    # Model configurations
    default_model: str = "gpt-3.5-turbo"
    fallback_model: str = "gpt-3.5-turbo"
    max_tokens: int = 4000
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # LangChain configuration
    langchain_verbose: bool = False
    langchain_cache: bool = True
    
    # ML configuration
    enable_ml_optimization: bool = True
    ml_model_cache_size: int = 100
    ml_prediction_threshold: float = 0.7

@dataclass
class WorkflowConfig:
    """Workflow orchestration configuration."""
    max_concurrent_workflows: int = 10
    workflow_timeout: int = 300
    task_timeout: int = 60
    retry_attempts: int = 3
    retry_delay: int = 5
    
    # AI Agent configuration
    enable_ai_history: bool = True
    enable_prompt_evolution: bool = True
    agent_specialization: bool = True
    agent_learning: bool = True
    
    # Workflow types
    supported_business_areas: List[str] = None
    
    def __post_init__(self):
        if self.supported_business_areas is None:
            self.supported_business_areas = [
                "technology", "business", "healthcare", "finance",
                "education", "marketing", "legal", "general"
            ]

@dataclass
class AnalyticsConfig:
    """Advanced analytics configuration."""
    enable_analytics: bool = True
    enable_predictions: bool = True
    enable_trend_analysis: bool = True
    enable_correlation_analysis: bool = True
    
    # ML models
    quality_model_type: str = "RandomForest"
    performance_model_type: str = "GradientBoosting"
    prediction_horizon: int = 7
    correlation_threshold: float = 0.7
    
    # Data collection
    collect_metrics: bool = True
    metrics_retention_days: int = 30
    batch_size: int = 100
    
    # Performance thresholds
    quality_threshold: float = 0.7
    performance_threshold: float = 0.7
    throughput_threshold: float = 10.0

@dataclass
class ClusteringConfig:
    """Smart clustering configuration."""
    enable_clustering: bool = True
    enable_semantic_clustering: bool = True
    enable_lexical_clustering: bool = True
    enable_structural_clustering: bool = True
    
    # Clustering algorithms
    semantic_algorithm: str = "KMeans"
    lexical_algorithm: str = "DBSCAN"
    structural_algorithm: str = "AgglomerativeClustering"
    
    # Clustering parameters
    max_clusters: int = 20
    min_cluster_size: int = 2
    similarity_threshold: float = 0.8
    distance_metric: str = "cosine"
    
    # Feature extraction
    max_features: int = 1000
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    max_df: float = 0.95

@dataclass
class SentimentConfig:
    """Advanced sentiment analysis configuration."""
    enable_sentiment_analysis: bool = True
    enable_emotion_detection: bool = True
    enable_tone_analysis: bool = True
    enable_attitude_analysis: bool = True
    enable_sarcasm_detection: bool = True
    enable_irony_detection: bool = True
    
    # Sentiment models
    sentiment_model: str = "vader"
    emotion_model: str = "nltk"
    tone_model: str = "textblob"
    
    # Analysis parameters
    confidence_threshold: float = 0.6
    emotion_threshold: float = 0.5
    sarcasm_threshold: float = 0.7
    irony_threshold: float = 0.6
    
    # Supported emotions
    supported_emotions: List[str] = None
    
    def __post_init__(self):
        if self.supported_emotions is None:
            self.supported_emotions = [
                "joy", "sadness", "anger", "fear", "surprise",
                "disgust", "trust", "anticipation"
            ]

@dataclass
class ContentMetricsConfig:
    """Content metrics configuration."""
    enable_content_metrics: bool = True
    enable_quality_analysis: bool = True
    enable_performance_analysis: bool = True
    enable_readability_analysis: bool = True
    enable_engagement_analysis: bool = True
    
    # Metrics categories
    metrics_categories: List[str] = None
    
    # Analysis parameters
    optimization_threshold: float = 0.1
    trend_analysis_window: int = 30
    correlation_threshold: float = 0.7
    prediction_horizon: int = 7
    
    # Quality weights
    quality_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics_categories is None:
            self.metrics_categories = [
                "quality", "performance", "engagement", "clarity", "structure"
            ]
        if self.quality_weights is None:
            self.quality_weights = {
                "readability": 0.15,
                "vocabulary_diversity": 0.10,
                "clarity": 0.20,
                "coherence": 0.20,
                "originality": 0.10,
                "relevance": 0.10,
                "completeness": 0.10,
                "structure": 0.05
            }

@dataclass
class PerformanceConfig:
    """Performance configuration."""
    max_documents_per_request: int = 1000
    max_concurrent_requests: int = 50
    request_timeout: int = 300
    cache_ttl: int = 3600
    cache_max_size: int = 10000
    
    # Batch processing
    batch_size: int = 10
    batch_timeout: int = 30
    enable_batch_processing: bool = True
    
    # Memory management
    max_memory_usage: float = 0.8
    gc_threshold: int = 1000
    enable_memory_monitoring: bool = True
    
    # CPU optimization
    max_cpu_usage: float = 0.8
    enable_cpu_monitoring: bool = True
    worker_processes: int = 1

@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_https: bool = False
    enable_rate_limiting: bool = True
    enable_cors: bool = True
    enable_input_validation: bool = True
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_limit: int = 200
    rate_limit_window: int = 60
    
    # API keys
    api_key_encryption: bool = True
    api_key_rotation_days: int = 30
    api_key_length: int = 32
    
    # CORS
    cors_allow_origins: List[str] = None
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = None
    cors_allow_headers: List[str] = None
    
    def __post_init__(self):
        if self.cors_allow_origins is None:
            self.cors_allow_origins = ["*"]
        if self.cors_allow_methods is None:
            self.cors_allow_methods = ["*"]
        if self.cors_allow_headers is None:
            self.cors_allow_headers = ["*"]

@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    log_max_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    log_rotation: str = "time"  # time or size
    
    # Component logging
    enable_component_logging: bool = True
    enable_performance_logging: bool = True
    enable_error_logging: bool = True
    enable_debug_logging: bool = False

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enable_monitoring: bool = True
    enable_health_checks: bool = True
    enable_metrics_collection: bool = True
    enable_alerting: bool = False
    
    # Health check intervals
    health_check_interval: int = 30
    max_health_check_failures: int = 3
    health_check_timeout: int = 10
    
    # Metrics collection
    metrics_collection_interval: int = 60
    metrics_retention_days: int = 7
    enable_prometheus: bool = False
    
    # Alerting
    alert_email: Optional[str] = None
    alert_webhook: Optional[str] = None
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "cpu_usage": 0.8,
                "memory_usage": 0.8,
                "error_rate": 0.1,
                "response_time": 5.0
            }

@dataclass
class UltimateConfig:
    """Main configuration class for Ultimate TruthGPT."""
    database: DatabaseConfig
    api: APIConfig
    ai: AIConfig
    workflow: WorkflowConfig
    analytics: AnalyticsConfig
    clustering: ClusteringConfig
    sentiment: SentimentConfig
    content_metrics: ContentMetricsConfig
    performance: PerformanceConfig
    security: SecurityConfig
    logging: LoggingConfig
    monitoring: MonitoringConfig
    
    # Global settings
    environment: str = "development"
    debug: bool = False
    version: str = "1.0.0-ultimate"
    
    def __post_init__(self):
        # Set debug based on environment
        if self.environment == "development":
            self.debug = True
            self.logging.log_level = "DEBUG"
            self.api.reload = True
        elif self.environment == "production":
            self.debug = False
            self.logging.log_level = "INFO"
            self.api.reload = False
            self.security.enable_https = True

def load_config_from_env() -> UltimateConfig:
    """Load configuration from environment variables."""
    try:
        # Database configuration
        database = DatabaseConfig(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_ssl=os.getenv("REDIS_SSL", "false").lower() == "true"
        )
        
        # API configuration
        api = APIConfig(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            workers=int(os.getenv("WORKERS", "1")),
            reload=os.getenv("RELOAD", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "info")
        )
        
        # AI configuration
        ai = AIConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            default_model=os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4000"))
        )
        
        # Workflow configuration
        workflow = WorkflowConfig(
            max_concurrent_workflows=int(os.getenv("MAX_CONCURRENT_WORKFLOWS", "10")),
            workflow_timeout=int(os.getenv("WORKFLOW_TIMEOUT", "300")),
            enable_ai_history=os.getenv("ENABLE_AI_HISTORY", "true").lower() == "true",
            enable_prompt_evolution=os.getenv("ENABLE_PROMPT_EVOLUTION", "true").lower() == "true"
        )
        
        # Analytics configuration
        analytics = AnalyticsConfig(
            enable_analytics=os.getenv("ENABLE_ANALYTICS", "true").lower() == "true",
            enable_predictions=os.getenv("ENABLE_PREDICTIONS", "true").lower() == "true",
            prediction_horizon=int(os.getenv("PREDICTION_HORIZON", "7")),
            correlation_threshold=float(os.getenv("CORRELATION_THRESHOLD", "0.7"))
        )
        
        # Clustering configuration
        clustering = ClusteringConfig(
            enable_clustering=os.getenv("ENABLE_CLUSTERING", "true").lower() == "true",
            max_clusters=int(os.getenv("MAX_CLUSTERS", "20")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
        )
        
        # Sentiment configuration
        sentiment = SentimentConfig(
            enable_sentiment_analysis=os.getenv("ENABLE_SENTIMENT_ANALYSIS", "true").lower() == "true",
            enable_emotion_detection=os.getenv("ENABLE_EMOTION_DETECTION", "true").lower() == "true",
            confidence_threshold=float(os.getenv("SENTIMENT_CONFIDENCE_THRESHOLD", "0.6"))
        )
        
        # Content metrics configuration
        content_metrics = ContentMetricsConfig(
            enable_content_metrics=os.getenv("ENABLE_CONTENT_METRICS", "true").lower() == "true",
            optimization_threshold=float(os.getenv("OPTIMIZATION_THRESHOLD", "0.1")),
            trend_analysis_window=int(os.getenv("TREND_ANALYSIS_WINDOW", "30"))
        )
        
        # Performance configuration
        performance = PerformanceConfig(
            max_documents_per_request=int(os.getenv("MAX_DOCUMENTS_PER_REQUEST", "1000")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "50")),
            cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
            batch_size=int(os.getenv("BATCH_SIZE", "10"))
        )
        
        # Security configuration
        security = SecurityConfig(
            enable_https=os.getenv("ENABLE_HTTPS", "false").lower() == "true",
            enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
            rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "100"))
        )
        
        # Logging configuration
        logging_config = LoggingConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE"),
            enable_debug_logging=os.getenv("ENABLE_DEBUG_LOGGING", "false").lower() == "true"
        )
        
        # Monitoring configuration
        monitoring = MonitoringConfig(
            enable_monitoring=os.getenv("ENABLE_MONITORING", "true").lower() == "true",
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "false").lower() == "true"
        )
        
        # Main configuration
        config = UltimateConfig(
            database=database,
            api=api,
            ai=ai,
            workflow=workflow,
            analytics=analytics,
            clustering=clustering,
            sentiment=sentiment,
            content_metrics=content_metrics,
            performance=performance,
            security=security,
            logging=logging_config,
            monitoring=monitoring,
            environment=os.getenv("ENVIRONMENT", "development"),
            version=os.getenv("VERSION", "1.0.0-ultimate")
        )
        
        logger.info("Configuration loaded from environment variables")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration from environment: {e}")
        raise

def load_config_from_file(config_file: str) -> UltimateConfig:
    """Load configuration from JSON file."""
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Convert to configuration objects
        config = UltimateConfig(
            database=DatabaseConfig(**config_data.get("database", {})),
            api=APIConfig(**config_data.get("api", {})),
            ai=AIConfig(**config_data.get("ai", {})),
            workflow=WorkflowConfig(**config_data.get("workflow", {})),
            analytics=AnalyticsConfig(**config_data.get("analytics", {})),
            clustering=ClusteringConfig(**config_data.get("clustering", {})),
            sentiment=SentimentConfig(**config_data.get("sentiment", {})),
            content_metrics=ContentMetricsConfig(**config_data.get("content_metrics", {})),
            performance=PerformanceConfig(**config_data.get("performance", {})),
            security=SecurityConfig(**config_data.get("security", {})),
            logging=LoggingConfig(**config_data.get("logging", {})),
            monitoring=MonitoringConfig(**config_data.get("monitoring", {})),
            environment=config_data.get("environment", "development"),
            debug=config_data.get("debug", False),
            version=config_data.get("version", "1.0.0-ultimate")
        )
        
        logger.info(f"Configuration loaded from file: {config_file}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration from file: {e}")
        raise

def save_config_to_file(config: UltimateConfig, config_file: str) -> None:
    """Save configuration to JSON file."""
    try:
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = asdict(config)
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        logger.info(f"Configuration saved to file: {config_file}")
        
    except Exception as e:
        logger.error(f"Error saving configuration to file: {e}")
        raise

def get_default_config() -> UltimateConfig:
    """Get default configuration."""
    return UltimateConfig(
        database=DatabaseConfig(),
        api=APIConfig(),
        ai=AIConfig(),
        workflow=WorkflowConfig(),
        analytics=AnalyticsConfig(),
        clustering=ClusteringConfig(),
        sentiment=SentimentConfig(),
        content_metrics=ContentMetricsConfig(),
        performance=PerformanceConfig(),
        security=SecurityConfig(),
        logging=LoggingConfig(),
        monitoring=MonitoringConfig()
    )

# Global configuration instance
_global_config: Optional[UltimateConfig] = None

def get_global_config() -> UltimateConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config_from_env()
    return _global_config

def set_global_config(config: UltimateConfig) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config
    logger.info("Global configuration updated")

# Configuration presets
DEVELOPMENT_CONFIG = UltimateConfig(
    database=DatabaseConfig(),
    api=APIConfig(reload=True, log_level="debug"),
    ai=AIConfig(),
    workflow=WorkflowConfig(),
    analytics=AnalyticsConfig(),
    clustering=ClusteringConfig(),
    sentiment=SentimentConfig(),
    content_metrics=ContentMetricsConfig(),
    performance=PerformanceConfig(max_documents_per_request=100),
    security=SecurityConfig(enable_https=False),
    logging=LoggingConfig(log_level="DEBUG", enable_debug_logging=True),
    monitoring=MonitoringConfig(),
    environment="development",
    debug=True
)

PRODUCTION_CONFIG = UltimateConfig(
    database=DatabaseConfig(),
    api=APIConfig(reload=False, log_level="info"),
    ai=AIConfig(),
    workflow=WorkflowConfig(),
    analytics=AnalyticsConfig(),
    clustering=ClusteringConfig(),
    sentiment=SentimentConfig(),
    content_metrics=ContentMetricsConfig(),
    performance=PerformanceConfig(max_documents_per_request=1000),
    security=SecurityConfig(enable_https=True, enable_rate_limiting=True),
    logging=LoggingConfig(log_level="INFO", enable_debug_logging=False),
    monitoring=MonitoringConfig(enable_monitoring=True, enable_alerting=True),
    environment="production",
    debug=False
)

TESTING_CONFIG = UltimateConfig(
    database=DatabaseConfig(redis_db=1),
    api=APIConfig(port=8001, log_level="debug"),
    ai=AIConfig(),
    workflow=WorkflowConfig(max_concurrent_workflows=2),
    analytics=AnalyticsConfig(),
    clustering=ClusteringConfig(),
    sentiment=SentimentConfig(),
    content_metrics=ContentMetricsConfig(),
    performance=PerformanceConfig(max_documents_per_request=10),
    security=SecurityConfig(enable_https=False),
    logging=LoggingConfig(log_level="DEBUG", enable_debug_logging=True),
    monitoring=MonitoringConfig(enable_monitoring=False),
    environment="testing",
    debug=True
)

























