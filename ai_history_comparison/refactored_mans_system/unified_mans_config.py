"""
Unified MANS Configuration - Centralized Configuration Management

This module provides unified configuration management for all MANS systems:
- Advanced AI configuration
- Space Technology configuration
- Neural Network configuration
- Generative AI configuration
- Computer Vision configuration
- NLP configuration
- Reinforcement Learning configuration
- Transfer Learning configuration
- Federated Learning configuration
- Explainable AI configuration
- AI Ethics configuration
- AI Safety configuration
- Satellite Communication configuration
- Space Weather configuration
- Space Debris configuration
- Interplanetary Networking configuration
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"

class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# Individual system configurations
class AdvancedAIConfig(BaseModel):
    """Advanced AI configuration"""
    enabled: bool = True
    neural_networks: bool = True
    generative_ai: bool = True
    computer_vision: bool = True
    natural_language_processing: bool = True
    reinforcement_learning: bool = True
    transfer_learning: bool = True
    federated_learning: bool = True
    explainable_ai: bool = True
    ai_ethics: bool = True
    ai_safety: bool = True
    max_models: int = 100
    training_timeout_hours: int = 24
    inference_timeout_seconds: int = 30
    model_accuracy_threshold: float = 0.85

class SpaceTechnologyConfig(BaseModel):
    """Space Technology configuration"""
    enabled: bool = True
    satellite_communication: bool = True
    space_based_data_processing: bool = True
    orbital_mechanics: bool = True
    space_weather_monitoring: bool = True
    satellite_constellation_management: bool = True
    deep_space_communication: bool = True
    space_debris_tracking: bool = True
    interplanetary_networking: bool = True
    space_based_ai: bool = True
    satellite_imagery_processing: bool = True
    max_satellites: int = 1000
    communication_latency_ms: float = 100.0
    orbital_accuracy_meters: float = 1.0

class NeuralNetworkConfig(BaseModel):
    """Neural Network configuration"""
    enabled: bool = True
    transformer: bool = True
    cnn: bool = True
    rnn: bool = True
    lstm: bool = True
    gru: bool = True
    gan: bool = True
    vae: bool = True
    bert: bool = True
    gpt: bool = True
    resnet: bool = True
    max_layers: int = 100
    max_parameters: int = 1000000000
    training_batch_size: int = 32
    learning_rate: float = 0.001

class GenerativeAIConfig(BaseModel):
    """Generative AI configuration"""
    enabled: bool = True
    gpt: bool = True
    bert: bool = True
    t5: bool = True
    dalle: bool = True
    stable_diffusion: bool = True
    midjourney: bool = True
    chatgpt: bool = True
    claude: bool = True
    max_context_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1000

class ComputerVisionConfig(BaseModel):
    """Computer Vision configuration"""
    enabled: bool = True
    classification: bool = True
    detection: bool = True
    segmentation: bool = True
    recognition: bool = True
    tracking: bool = True
    reconstruction: bool = True
    enhancement: bool = True
    generation: bool = True
    analysis: bool = True
    understanding: bool = True
    max_image_size: int = 4096
    supported_formats: List[str] = ["jpg", "png", "tiff", "bmp"]
    processing_timeout_seconds: int = 60

class NLPConfig(BaseModel):
    """Natural Language Processing configuration"""
    enabled: bool = True
    sentiment_analysis: bool = True
    named_entity_recognition: bool = True
    summarization: bool = True
    translation: bool = True
    question_answering: bool = True
    text_generation: bool = True
    parsing: bool = True
    classification: bool = True
    max_text_length: int = 10000
    supported_languages: List[str] = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
    processing_timeout_seconds: int = 30

class ReinforcementLearningConfig(BaseModel):
    """Reinforcement Learning configuration"""
    enabled: bool = True
    q_learning: bool = True
    policy_gradient: bool = True
    actor_critic: bool = True
    deep_q_network: bool = True
    proximal_policy_optimization: bool = True
    trust_region_policy_optimization: bool = True
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    learning_rate: float = 0.001
    discount_factor: float = 0.99

class TransferLearningConfig(BaseModel):
    """Transfer Learning configuration"""
    enabled: bool = True
    fine_tuning: bool = True
    feature_extraction: bool = True
    domain_adaptation: bool = True
    multi_task_learning: bool = True
    meta_learning: bool = True
    few_shot_learning: bool = True
    zero_shot_learning: bool = True
    max_source_domains: int = 10
    adaptation_epochs: int = 100

class FederatedLearningConfig(BaseModel):
    """Federated Learning configuration"""
    enabled: bool = True
    horizontal_federation: bool = True
    vertical_federation: bool = True
    federated_averaging: bool = True
    secure_aggregation: bool = True
    differential_privacy: bool = True
    max_clients: int = 1000
    communication_rounds: int = 100
    local_epochs: int = 5
    privacy_budget: float = 1.0

class ExplainableAIConfig(BaseModel):
    """Explainable AI configuration"""
    enabled: bool = True
    lime: bool = True
    shap: bool = True
    integrated_gradients: bool = True
    attention_visualization: bool = True
    feature_importance: bool = True
    counterfactual_explanations: bool = True
    causal_inference: bool = True
    explanation_confidence_threshold: float = 0.8
    max_explanation_features: int = 20

class AIEthicsConfig(BaseModel):
    """AI Ethics configuration"""
    enabled: bool = True
    fairness: bool = True
    transparency: bool = True
    privacy: bool = True
    accountability: bool = True
    non_maleficence: bool = True
    beneficence: bool = True
    autonomy: bool = True
    justice: bool = True
    bias_detection_threshold: float = 0.1
    fairness_metrics: List[str] = ["demographic_parity", "equalized_odds", "calibration"]

class AISafetyConfig(BaseModel):
    """AI Safety configuration"""
    enabled: bool = True
    robustness: bool = True
    interpretability: bool = True
    verifiability: bool = True
    controllability: bool = True
    alignment: bool = True
    adversarial_robustness: bool = True
    out_of_distribution_detection: bool = True
    safety_threshold: float = 0.95
    robustness_epsilon: float = 0.1

class SatelliteCommunicationConfig(BaseModel):
    """Satellite Communication configuration"""
    enabled: bool = True
    leo: bool = True
    meo: bool = True
    geo: bool = True
    heo: bool = True
    cubesat: bool = True
    l_band: bool = True
    s_band: bool = True
    c_band: bool = True
    x_band: bool = True
    ku_band: bool = True
    ka_band: bool = True
    max_data_rate_mbps: float = 1000.0
    signal_strength_threshold: float = 0.8

class SpaceWeatherConfig(BaseModel):
    """Space Weather configuration"""
    enabled: bool = True
    solar_flare: bool = True
    coronal_mass_ejection: bool = True
    solar_wind: bool = True
    geomagnetic_storm: bool = True
    radiation_belt: bool = True
    aurora: bool = True
    ionospheric_disturbance: bool = True
    cosmic_ray: bool = True
    monitoring_frequency_minutes: int = 15
    alert_threshold_kp: float = 5.0

class SpaceDebrisConfig(BaseModel):
    """Space Debris configuration"""
    enabled: bool = True
    collision_detection: bool = True
    orbit_prediction: bool = True
    avoidance_maneuvers: bool = True
    debris_catalog: bool = True
    tracking_accuracy_meters: float = 10.0
    collision_risk_threshold: float = 0.1
    prediction_horizon_days: int = 7

class InterplanetaryNetworkingConfig(BaseModel):
    """Interplanetary Networking configuration"""
    enabled: bool = True
    delay_tolerant_networking: bool = True
    bundle_protocol: bool = True
    interplanetary_internet: bool = True
    deep_space_communication: bool = True
    max_communication_distance_km: float = 1e9
    signal_propagation_delay_seconds: float = 1000.0

class UnifiedMANSConfig(BaseModel):
    """
    Unified configuration for the entire MANS system,
    integrating all advanced technology configurations.
    """
    # System Information
    system_name: str = "AI History Comparison System - MANS Unified"
    environment: Environment = Environment.DEVELOPMENT
    debug_mode: bool = True
    log_level: LogLevel = LogLevel.INFO
    
    # API Configuration
    api_prefix: str = "/api/v1/mans"
    database_url: str = "sqlite:///./mans_unified.db"
    cache_enabled: bool = True
    metrics_enabled: bool = True
    events_enabled: bool = True
    
    # Advanced Technology Configurations
    advanced_ai: AdvancedAIConfig = Field(default_factory=AdvancedAIConfig)
    space_technology: SpaceTechnologyConfig = Field(default_factory=SpaceTechnologyConfig)
    neural_network: NeuralNetworkConfig = Field(default_factory=NeuralNetworkConfig)
    generative_ai: GenerativeAIConfig = Field(default_factory=GenerativeAIConfig)
    computer_vision: ComputerVisionConfig = Field(default_factory=ComputerVisionConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    reinforcement_learning: ReinforcementLearningConfig = Field(default_factory=ReinforcementLearningConfig)
    transfer_learning: TransferLearningConfig = Field(default_factory=TransferLearningConfig)
    federated_learning: FederatedLearningConfig = Field(default_factory=FederatedLearningConfig)
    explainable_ai: ExplainableAIConfig = Field(default_factory=ExplainableAIConfig)
    ai_ethics: AIEthicsConfig = Field(default_factory=AIEthicsConfig)
    ai_safety: AISafetyConfig = Field(default_factory=AISafetyConfig)
    satellite_communication: SatelliteCommunicationConfig = Field(default_factory=SatelliteCommunicationConfig)
    space_weather: SpaceWeatherConfig = Field(default_factory=SpaceWeatherConfig)
    space_debris: SpaceDebrisConfig = Field(default_factory=SpaceDebrisConfig)
    interplanetary_networking: InterplanetaryNetworkingConfig = Field(default_factory=InterplanetaryNetworkingConfig)
    
    # Performance Configuration
    max_concurrent_requests: int = 2000
    request_timeout_seconds: int = 60
    cache_ttl_seconds: int = 7200
    rate_limit_requests_per_minute: int = 2000
    
    # Security Configuration
    enable_authentication: bool = True
    enable_authorization: bool = True
    enable_encryption: bool = True
    enable_rate_limiting: bool = True
    enable_cors: bool = True
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        self._setup_logging()
        logger.info(f"UnifiedMANSConfig loaded for environment: {self.environment}")
        if self.debug_mode:
            logger.warning("MANS system is running in DEBUG mode. Disable for production.")

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.value),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.setLevel(getattr(logging, self.log_level.value))

    def get_feature_status(self, feature_name: str) -> bool:
        """Returns the enabled status of a specific MANS feature"""
        feature_config = getattr(self, feature_name, None)
        if feature_config and isinstance(feature_config, BaseModel) and hasattr(feature_config, 'enabled'):
            return feature_config.enabled
        return False

    def get_all_enabled_features(self) -> List[str]:
        """Returns list of all enabled MANS features"""
        enabled_features = []
        feature_configs = [
            ("advanced_ai", self.advanced_ai),
            ("space_technology", self.space_technology),
            ("neural_network", self.neural_network),
            ("generative_ai", self.generative_ai),
            ("computer_vision", self.computer_vision),
            ("nlp", self.nlp),
            ("reinforcement_learning", self.reinforcement_learning),
            ("transfer_learning", self.transfer_learning),
            ("federated_learning", self.federated_learning),
            ("explainable_ai", self.explainable_ai),
            ("ai_ethics", self.ai_ethics),
            ("ai_safety", self.ai_safety),
            ("satellite_communication", self.satellite_communication),
            ("space_weather", self.space_weather),
            ("space_debris", self.space_debris),
            ("interplanetary_networking", self.interplanetary_networking)
        ]
        
        for feature_name, feature_config in feature_configs:
            if feature_config.enabled:
                enabled_features.append(feature_name)
        
        return enabled_features

    def update_config(self, new_settings: Dict[str, Any]):
        """Updates configuration settings dynamically"""
        for key, value in new_settings.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if isinstance(current_value, BaseModel) and isinstance(value, dict):
                    # Update nested BaseModel configs
                    updated_nested = current_value.copy(update=value)
                    setattr(self, key, updated_nested)
                else:
                    setattr(self, key, value)
                logger.info(f"MANS configuration '{key}' updated to '{getattr(self, key)}'")
        
        self._setup_logging()  # Re-apply logging level if changed

    def get_system_summary(self) -> Dict[str, Any]:
        """Returns a comprehensive system summary"""
        return {
            "system_name": self.system_name,
            "environment": self.environment.value,
            "debug_mode": self.debug_mode,
            "log_level": self.log_level.value,
            "enabled_features": self.get_all_enabled_features(),
            "total_features": 16,
            "enabled_count": len(self.get_all_enabled_features()),
            "api_prefix": self.api_prefix,
            "performance": {
                "max_concurrent_requests": self.max_concurrent_requests,
                "request_timeout_seconds": self.request_timeout_seconds,
                "cache_ttl_seconds": self.cache_ttl_seconds,
                "rate_limit_requests_per_minute": self.rate_limit_requests_per_minute
            },
            "security": {
                "authentication": self.enable_authentication,
                "authorization": self.enable_authorization,
                "encryption": self.enable_encryption,
                "rate_limiting": self.enable_rate_limiting,
                "cors": self.enable_cors
            }
        }





















