"""
Config Manager - Sistema de configuración avanzada
===================================================
"""

import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class FeatureFlags(BaseModel):
    """Feature flags del sistema"""
    enable_llm: bool = False
    enable_redis: bool = False
    enable_webhooks: bool = True
    enable_notifications: bool = True
    enable_analytics: bool = True
    enable_backup: bool = True
    enable_rate_limiting: bool = True
    enable_monitoring: bool = True


class AdvancedConfig(BaseSettings):
    """Configuración avanzada del sistema"""
    
    # General
    app_name: str = "3D Prototype AI"
    app_version: str = "2.0.0"
    debug: bool = False
    environment: str = "production"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8030
    workers: int = 4
    
    # Database (preparado para futura implementación)
    database_url: Optional[str] = None
    
    # Redis
    redis_url: Optional[str] = None
    redis_enabled: bool = False
    
    # LLM
    llm_provider: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_default: int = 100
    rate_limit_window: int = 60
    
    # Cache
    cache_ttl_default: int = 3600
    cache_type: str = "memory"  # memory, redis, distributed
    
    # Monitoring
    monitoring_enabled: bool = True
    log_level: str = "INFO"
    log_retention_days: int = 30
    
    # Backup
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30
    
    # Feature Flags
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class ConfigManager:
    """Gestor de configuración avanzada"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = Path(config_file) if config_file else Path("config.json")
        self.config = AdvancedConfig()
        self._load_config()
    
    def _load_config(self):
        """Carga configuración desde archivo"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
                    # Actualizar configuración
                    for key, value in file_config.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                logger.info(f"Configuración cargada desde {self.config_file}")
            except Exception as e:
                logger.warning(f"Error cargando configuración: {e}")
    
    def save_config(self):
        """Guarda configuración a archivo"""
        config_dict = self.config.model_dump()
        
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Configuración guardada en {self.config_file}")
    
    def get_config(self) -> Dict[str, Any]:
        """Obtiene la configuración completa"""
        return self.config.model_dump()
    
    def update_config(self, updates: Dict[str, Any]):
        """Actualiza configuración"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.save_config()
        logger.info(f"Configuración actualizada: {list(updates.keys())}")
    
    def get_feature_flag(self, flag_name: str) -> bool:
        """Obtiene el valor de un feature flag"""
        flags = self.config.feature_flags
        return getattr(flags, flag_name, False)
    
    def set_feature_flag(self, flag_name: str, value: bool):
        """Establece un feature flag"""
        if hasattr(self.config.feature_flags, flag_name):
            setattr(self.config.feature_flags, flag_name, value)
            self.save_config()
            logger.info(f"Feature flag {flag_name} = {value}")
    
    def validate_config(self) -> List[str]:
        """Valida la configuración y retorna errores"""
        errors = []
        
        if self.config.port < 1 or self.config.port > 65535:
            errors.append("Puerto inválido")
        
        if self.config.workers < 1:
            errors.append("Número de workers inválido")
        
        if self.config.redis_enabled and not self.config.redis_url:
            errors.append("Redis habilitado pero no hay URL configurada")
        
        if self.config.feature_flags.enable_llm and not self.config.llm_api_key:
            errors.append("LLM habilitado pero no hay API key")
        
        return errors




