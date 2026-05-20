"""
Configuración del Autonomous Long-Term Agent
Organizada en grupos lógicos con comentarios claros
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Configuración de la aplicación
    
    Organizada en secciones lógicas:
    - OpenRouter: Configuración de API
    - Agent: Configuración del agente
    - Learning: Configuración de aprendizaje
    - Reasoning: Configuración de razonamiento
    - Self-Reflection: Configuración de self-reflection (EvoAgent)
    - Experience Learning: Configuración de ELL framework
    - World Model: Configuración de world model (EvoAgent)
    - Server: Configuración del servidor
    - Storage: Configuración de almacenamiento
    - Rate Limiting: Configuración de rate limiting
    """
    
    # ============================================
    # OpenRouter Configuration
    # ============================================
    openrouter_api_key: str = ""
    openrouter_http_referer: str = "https://blatam-academy.com"
    openrouter_model: str = "anthropic/claude-3.5-sonnet"
    openrouter_temperature: float = 0.7
    openrouter_max_tokens: int = 4000
    
    # ============================================
    # Agent Configuration
    # ============================================
    agent_poll_interval: float = 1.0  # segundos entre iteraciones
    agent_max_concurrent_tasks: int = 10
    agent_max_parallel_instances: int = 5
    agent_auto_restart: bool = True
    
    # ============================================
    # Learning Configuration
    # ============================================
    learning_enabled: bool = True
    learning_adaptation_rate: float = 0.1
    knowledge_base_retention_days: int = 30
    max_knowledge_entries: int = 10000
    
    # ============================================
    # Long-Horizon Reasoning Configuration
    # ============================================
    reasoning_depth: int = 5
    reasoning_timeout: float = 60.0
    enable_web_research: bool = False
    
    # ============================================
    # Self-Reflection Configuration (EvoAgent paper)
    # ============================================
    enable_self_reflection: bool = True
    self_reflection_interval: float = 300.0  # 5 minutes
    self_reflection_on_performance: bool = True
    self_reflection_on_strategy: bool = True
    self_reflection_on_capabilities: bool = True
    
    # ============================================
    # Experience-Driven Learning Configuration (ELL paper)
    # ============================================
    enable_experience_learning: bool = True
    max_experiences: int = 5000
    skill_abstraction_threshold: int = 3  # Minimum experiences to abstract a skill
    
    # ============================================
    # Continual World Model Configuration (EvoAgent paper)
    # ============================================
    enable_world_model: bool = True
    world_model_max_observations: int = 1000
    world_model_change_threshold: float = 0.3
    
    # ============================================
    # Server Configuration
    # ============================================
    host: str = "0.0.0.0"
    port: int = 8001
    
    # ============================================
    # Storage Configuration
    # ============================================
    storage_path: str = "./data/autonomous_agent"
    enable_persistence: bool = True
    
    # ============================================
    # Logging Configuration
    # ============================================
    log_level: str = "INFO"
    
    # ============================================
    # Rate Limiting Configuration
    # ============================================
    rate_limit_max_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )


settings = Settings()




