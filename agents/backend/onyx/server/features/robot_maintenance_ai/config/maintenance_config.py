"""
Configuration for Robot Maintenance AI with Open Router, NLP, and ML.
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
class NLPConfig:
    """Configuration for NLP processing."""
    language: str = "es"
    use_spacy: bool = True
    use_transformers: bool = True
    model_name: str = "dccuchile/bert-base-spanish-wwm-uncased"
    enable_ner: bool = True
    enable_sentiment: bool = True
    enable_keyword_extraction: bool = True


@dataclass
class MLConfig:
    """Configuration for Machine Learning models."""
    enable_predictive_maintenance: bool = True
    model_path: str = "ml_models"
    use_tensorflow: bool = True
    use_pytorch: bool = False
    enable_anomaly_detection: bool = True
    training_data_path: str = "data/training"
    retrain_interval_days: int = 30


@dataclass
class MaintenanceConfig:
    """Main configuration for the Robot Maintenance AI system."""
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    
    robot_types: List[str] = field(default_factory=lambda: [
        "robots_industriales", "robots_medicos", "robots_servicio",
        "robots_agricolas", "maquinaria_cnc", "sistemas_automatizados"
    ])
    
    maintenance_categories: List[str] = field(default_factory=lambda: [
        "preventivo", "correctivo", "predictivo", "diagnostico",
        "calibracion", "lubricacion", "reemplazo_piezas", "actualizacion_software"
    ])
    
    difficulty_levels: List[str] = field(default_factory=lambda: [
        "principiante", "intermedio", "avanzado", "experto"
    ])
    
    response_style: str = "tecnico_practico"
    use_visual_aids: bool = True
    provide_step_by_step: bool = True
    include_safety_warnings: bool = True
    
    conversation_history_path: str = "conversations"
    max_history_length: int = 100
    
    cache_enabled: bool = True
    cache_ttl: int = 3600
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.openrouter.api_key:
            raise ValueError("OPENROUTER_API_KEY must be set in environment variables")
        return True





