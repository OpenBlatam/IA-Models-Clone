"""
Unified Configuration System

This module provides a centralized configuration system that manages all
aspects of the unified AI History Comparison System including quantum computing,
blockchain, IoT, AR/VR, edge computing, and performance optimizations.
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "ai_history_comparison"
    username: str = "postgres"
    password: str = "password"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True

@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: LogLevel = LogLevel.INFO
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    rate_limit: int = 1000
    rate_limit_window: int = 3600

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    enable_2fa: bool = True
    enable_biometric: bool = False

@dataclass
class QuantumConfig:
    """Quantum computing configuration"""
    enabled: bool = True
    simulator_qubits: int = 32
    max_circuit_depth: int = 1000
    default_shots: int = 1024
    algorithms_enabled: List[str] = field(default_factory=lambda: [
        "grover", "shor", "deutsch_jozsa", "qft", "vqe", "qaoa"
    ])
    ml_enabled: bool = True
    optimization_enabled: bool = True
    cryptography_enabled: bool = True

@dataclass
class BlockchainConfig:
    """Blockchain configuration"""
    enabled: bool = True
    supported_chains: List[str] = field(default_factory=lambda: [
        "ethereum", "bitcoin", "solana", "polygon", "bsc", "avalanche"
    ])
    ethereum_rpc: str = "https://mainnet.infura.io/v3/YOUR_KEY"
    bitcoin_rpc: str = "https://api.blockcypher.com/v1/btc/main"
    solana_rpc: str = "https://api.mainnet-beta.solana.com"
    defi_enabled: bool = True
    nft_enabled: bool = True
    cross_chain_enabled: bool = True
    smart_contracts_enabled: bool = True

@dataclass
class IoTConfig:
    """IoT configuration"""
    enabled: bool = True
    protocols: List[str] = field(default_factory=lambda: [
        "mqtt", "coap", "http", "websocket", "modbus", "opc_ua"
    ])
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    coap_server: str = "localhost"
    coap_port: int = 5683
    http_server: str = "http://localhost:8080"
    device_management_enabled: bool = True
    real_time_processing: bool = True
    edge_computing_enabled: bool = True

@dataclass
class ARVRConfig:
    """AR/VR configuration"""
    enabled: bool = True
    renderers: List[str] = field(default_factory=lambda: ["opengl", "vulkan"])
    tracking_types: List[str] = field(default_factory=lambda: [
        "head", "hand", "eye", "body", "object", "spatial"
    ])
    interaction_types: List[str] = field(default_factory=lambda: [
        "gaze", "gesture", "voice", "touch", "haptic", "controller"
    ])
    frame_rate: int = 90
    resolution: tuple = (1920, 1080)
    field_of_view: float = 110.0
    haptic_enabled: bool = True
    voice_recognition_enabled: bool = True
    gesture_recognition_enabled: bool = True

@dataclass
class EdgeConfig:
    """Edge computing configuration"""
    enabled: bool = True
    node_types: List[str] = field(default_factory=lambda: [
        "gateway", "edge_server", "mobile_edge", "iot_edge", "fog_node"
    ])
    processing_types: List[str] = field(default_factory=lambda: [
        "real_time", "batch", "streaming", "event_driven", "scheduled"
    ])
    load_balancing: str = "least_loaded"
    fault_tolerance_enabled: bool = True
    auto_scaling_enabled: bool = True
    health_check_interval: int = 30
    max_nodes: int = 100

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    optimization_enabled: bool = True
    memory_optimization: bool = True
    cpu_optimization: bool = True
    io_optimization: bool = True
    database_optimization: bool = True
    cache_optimization: bool = True
    network_optimization: bool = True
    profiling_enabled: bool = True
    metrics_enabled: bool = True
    monitoring_enabled: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    jaeger_enabled: bool = True
    elasticsearch_enabled: bool = True
    kibana_enabled: bool = True
    metrics_interval: int = 60
    log_retention_days: int = 30
    alerting_enabled: bool = True
    dashboard_enabled: bool = True

@dataclass
class AIConfig:
    """AI/ML configuration"""
    enabled: bool = True
    models_enabled: List[str] = field(default_factory=lambda: [
        "gpt-4", "claude-3", "gemini-pro", "llama-2", "mistral"
    ])
    providers: List[str] = field(default_factory=lambda: [
        "openai", "anthropic", "google", "meta", "mistral"
    ])
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    retry_attempts: int = 3
    cache_enabled: bool = True
    cache_ttl: int = 3600

@dataclass
class UnifiedConfig:
    """Unified configuration system"""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    version: str = "1.0.0"
    
    # Core configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api: APIConfig = field(default_factory=APIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Advanced feature configurations
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    blockchain: BlockchainConfig = field(default_factory=BlockchainConfig)
    iot: IoTConfig = field(default_factory=IoTConfig)
    ar_vr: ARVRConfig = field(default_factory=ARVRConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    
    # Additional settings
    features: Dict[str, bool] = field(default_factory=lambda: {
        "quantum_computing": True,
        "blockchain_integration": True,
        "iot_integration": True,
        "ar_vr_support": True,
        "edge_computing": True,
        "performance_optimization": True,
        "advanced_security": True,
        "real_time_monitoring": True,
        "ai_ml_enhancement": True,
        "multi_modal_analysis": True
    })
    
    def __post_init__(self):
        """Post-initialization setup"""
        self._load_from_environment()
        self._validate_config()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Environment
        env = os.getenv("ENVIRONMENT", "development")
        self.environment = Environment(env)
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Database
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", self.database.port))
        self.database.database = os.getenv("DB_NAME", self.database.database)
        self.database.username = os.getenv("DB_USER", self.database.username)
        self.database.password = os.getenv("DB_PASSWORD", self.database.password)
        
        # Redis
        self.redis.host = os.getenv("REDIS_HOST", self.redis.host)
        self.redis.port = int(os.getenv("REDIS_PORT", self.redis.port))
        self.redis.password = os.getenv("REDIS_PASSWORD", self.redis.password)
        
        # API
        self.api.host = os.getenv("API_HOST", self.api.host)
        self.api.port = int(os.getenv("API_PORT", self.api.port))
        self.api.workers = int(os.getenv("API_WORKERS", self.api.workers))
        
        # Security
        self.security.secret_key = os.getenv("SECRET_KEY", self.security.secret_key)
        
        # Quantum
        self.quantum.enabled = os.getenv("QUANTUM_ENABLED", "true").lower() == "true"
        
        # Blockchain
        self.blockchain.enabled = os.getenv("BLOCKCHAIN_ENABLED", "true").lower() == "true"
        self.blockchain.ethereum_rpc = os.getenv("ETHEREUM_RPC", self.blockchain.ethereum_rpc)
        
        # IoT
        self.iot.enabled = os.getenv("IOT_ENABLED", "true").lower() == "true"
        self.iot.mqtt_broker = os.getenv("MQTT_BROKER", self.iot.mqtt_broker)
        
        # AR/VR
        self.ar_vr.enabled = os.getenv("ARVR_ENABLED", "true").lower() == "true"
        
        # Edge
        self.edge.enabled = os.getenv("EDGE_ENABLED", "true").lower() == "true"
        
        # Performance
        self.performance.optimization_enabled = os.getenv("PERFORMANCE_OPTIMIZATION", "true").lower() == "true"
        
        # Monitoring
        self.monitoring.enabled = os.getenv("MONITORING_ENABLED", "true").lower() == "true"
        
        # AI
        self.ai.enabled = os.getenv("AI_ENABLED", "true").lower() == "true"
    
    def _validate_config(self):
        """Validate configuration settings"""
        # Validate database connection
        if not self.database.host:
            raise ValueError("Database host is required")
        
        # Validate security settings
        if not self.security.secret_key or self.security.secret_key == "your-secret-key-here":
            logger.warning("Using default secret key. Please set SECRET_KEY environment variable.")
        
        # Validate API settings
        if self.api.port < 1 or self.api.port > 65535:
            raise ValueError("API port must be between 1 and 65535")
        
        # Validate quantum settings
        if self.quantum.enabled and self.quantum.simulator_qubits < 1:
            raise ValueError("Quantum simulator qubits must be at least 1")
        
        # Validate blockchain settings
        if self.blockchain.enabled and not self.blockchain.supported_chains:
            raise ValueError("At least one blockchain must be supported")
        
        # Validate IoT settings
        if self.iot.enabled and not self.iot.protocols:
            raise ValueError("At least one IoT protocol must be enabled")
        
        # Validate AR/VR settings
        if self.ar_vr.enabled and not self.ar_vr.renderers:
            raise ValueError("At least one AR/VR renderer must be enabled")
        
        # Validate edge settings
        if self.edge.enabled and not self.edge.node_types:
            raise ValueError("At least one edge node type must be enabled")
        
        logger.info("Configuration validation completed successfully")
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "UnifiedConfig":
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return cls._from_dict(config_data)
    
    @classmethod
    def _from_dict(cls, config_data: Dict[str, Any]) -> "UnifiedConfig":
        """Create configuration from dictionary"""
        # Create base config
        config = cls()
        
        # Update with provided data
        for key, value in config_data.items():
            if hasattr(config, key):
                if isinstance(value, dict):
                    # Handle nested configurations
                    nested_config = getattr(config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {}
        
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                # Handle nested configurations
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
        
        return config_dict
    
    def to_file(self, config_path: Union[str, Path], format: str = "yaml"):
        """Save configuration to file"""
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            if format.lower() == "yaml":
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {config_path}")
    
    def get_feature_status(self, feature_name: str) -> bool:
        """Get feature status"""
        return self.features.get(feature_name, False)
    
    def enable_feature(self, feature_name: str):
        """Enable a feature"""
        self.features[feature_name] = True
        logger.info(f"Enabled feature: {feature_name}")
    
    def disable_feature(self, feature_name: str):
        """Disable a feature"""
        self.features[feature_name] = False
        logger.info(f"Disabled feature: {feature_name}")
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT
    
    def get_database_url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.database.username}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.database}"
    
    def get_redis_url(self) -> str:
        """Get Redis URL"""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.db}"
        else:
            return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "environment": self.environment.value,
            "version": self.version,
            "debug": self.debug,
            "features_enabled": {
                feature: enabled for feature, enabled in self.features.items() if enabled
            },
            "advanced_features": {
                "quantum_computing": self.quantum.enabled,
                "blockchain_integration": self.blockchain.enabled,
                "iot_integration": self.iot.enabled,
                "ar_vr_support": self.ar_vr.enabled,
                "edge_computing": self.edge.enabled,
                "performance_optimization": self.performance.optimization_enabled,
                "monitoring": self.monitoring.enabled,
                "ai_ml": self.ai.enabled
            },
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "workers": self.api.workers
            }
        }

# Global configuration instance
_global_config: Optional[UnifiedConfig] = None

def get_config() -> UnifiedConfig:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = UnifiedConfig()
    return _global_config

def load_config(config_path: Union[str, Path]) -> UnifiedConfig:
    """Load configuration from file"""
    global _global_config
    _global_config = UnifiedConfig.from_file(config_path)
    return _global_config

def save_config(config_path: Union[str, Path], format: str = "yaml"):
    """Save current configuration to file"""
    config = get_config()
    config.to_file(config_path, format)





















