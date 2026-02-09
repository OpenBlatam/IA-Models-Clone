"""
Configuration for Robot Maintenance Teaching AI.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OpenRouterConfig:
    """Configuration for Open Router API integration."""
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "openai/gpt-4-turbo"
    timeout: int = 90
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 3000


@dataclass
class MLConfig:
    """Configuration for Machine Learning models."""
    model_type: str = "ensemble"
    prediction_threshold: float = 0.7
    training_data_path: str = "data/training"
    model_save_path: str = "ml_models/saved_models"
    use_pretrained: bool = True
    retrain_interval_days: int = 30
    feature_engineering: bool = True
    cross_validation_folds: int = 5


@dataclass
class NLPConfig:
    """Configuration for Natural Language Processing."""
    language: str = "es"
    model_name: str = "es_core_news_md"
    use_transformer: bool = True
    transformer_model: str = "dccuchile/bert-base-spanish-wwm-uncased"
    enable_ner: bool = True
    enable_sentiment: bool = True
    enable_keyword_extraction: bool = True
    similarity_threshold: float = 0.75


@dataclass
class MaintenanceConfig:
    """Main configuration for the Robot Maintenance Teaching system."""
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    
    robot_types: List[str] = field(default_factory=lambda: [
        "industrial_robot",
        "service_robot",
        "collaborative_robot",
        "mobile_robot",
        "medical_robot",
        "agricultural_robot"
    ])
    
    maintenance_categories: List[str] = field(default_factory=lambda: [
        "preventive",
        "corrective",
        "predictive",
        "emergency",
        "scheduled",
        "condition_based"
    ])
    
    difficulty_levels: List[str] = field(default_factory=lambda: [
        "beginner",
        "intermediate",
        "advanced",
        "expert"
    ])
    
    response_style: str = "technical"
    use_visual_aids: bool = True
    provide_troubleshooting: bool = True
    adaptive_learning: bool = True
    
    conversation_history_path: str = "data/conversations"
    max_history_length: int = 100
    
    cache_enabled: bool = True
    cache_ttl: int = 7200
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    
    # Metrics
    metrics_enabled: bool = True
    
    # Security
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 300  # 5 minutes
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.openrouter.api_key:
            raise ValueError("OPENROUTER_API_KEY must be set")
        
        if self.rate_limit_per_minute <= 0:
            raise ValueError("rate_limit_per_minute must be positive")
        
        if self.rate_limit_per_hour <= 0:
            raise ValueError("rate_limit_per_hour must be positive")
        
        if self.cache_ttl <= 0:
            raise ValueError("cache_ttl must be positive")
        
        return True






