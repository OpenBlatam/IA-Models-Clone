"""
Advanced Config - Sistema de configuración avanzado
====================================================
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import os
from pydantic import BaseSettings, Field

logger = logging.getLogger(__name__)


class AdvancedConfig:
    """
    Sistema de configuración avanzado con validación y hot-reload.
    """
    
    def __init__(self, config_file: str = "config/advanced_config.json"):
        """
        Inicializar configuración avanzada.
        
        Args:
            config_file: Archivo de configuración
        """
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.config: Dict[str, Any] = self._load_default_config()
        self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Carga configuración por defecto"""
        return {
            "performance": {
                "cache_enabled": True,
                "cache_ttl_seconds": 3600,
                "max_cache_size": 256,
                "enable_profiling": False
            },
            "rag": {
                "enabled": True,
                "top_k_papers": 3,
                "min_similarity_score": 0.7,
                "use_hybrid_search": True
            },
            "improvements": {
                "max_suggestions": 10,
                "min_confidence": 0.5,
                "apply_automatically": False,
                "require_approval": True
            },
            "notifications": {
                "webhooks_enabled": True,
                "email_enabled": False,
                "slack_enabled": False,
                "discord_enabled": False
            },
            "security": {
                "rate_limit_enabled": True,
                "auth_required": False,
                "api_key_required": False,
                "max_file_size_mb": 50
            },
            "features": {
                "test_generation": True,
                "documentation_generation": True,
                "git_integration": True,
                "collaboration": True
            }
        }
    
    def _load_config(self):
        """Carga configuración desde archivo"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
                    self._merge_config(self.config, file_config)
                logger.info(f"Configuración cargada: {self.config_file}")
            except Exception as e:
                logger.error(f"Error cargando configuración: {e}")
        else:
            self._save_config()
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Fusiona configuraciones"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración usando path.
        
        Args:
            key_path: Path de la clave (ej: "performance.cache_enabled")
            default: Valor por defecto
            
        Returns:
            Valor de configuración
        """
        keys = key_path.split(".")
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        Establece un valor de configuración.
        
        Args:
            key_path: Path de la clave
            value: Valor a establecer
        """
        keys = key_path.split(".")
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self._save_config()
        
        logger.info(f"Configuración actualizada: {key_path} = {value}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Obtiene una sección completa de configuración.
        
        Args:
            section: Nombre de la sección
            
        Returns:
            Configuración de la sección
        """
        return self.config.get(section, {})
    
    def update_section(self, section: str, updates: Dict[str, Any]):
        """
        Actualiza una sección completa.
        
        Args:
            section: Nombre de la sección
            updates: Valores a actualizar
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section].update(updates)
        self._save_config()
    
    def validate_config(self) -> List[str]:
        """
        Valida la configuración actual.
        
        Returns:
            Lista de errores de validación (vacía si es válida)
        """
        errors = []
        
        # Validar valores
        cache_ttl = self.get("performance.cache_ttl_seconds")
        if cache_ttl and (cache_ttl < 0 or cache_ttl > 86400):
            errors.append("cache_ttl_seconds debe estar entre 0 y 86400")
        
        max_file_size = self.get("security.max_file_size_mb")
        if max_file_size and (max_file_size < 1 or max_file_size > 500):
            errors.append("max_file_size_mb debe estar entre 1 y 500")
        
        return errors
    
    def _save_config(self):
        """Guarda configuración en disco"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
    
    def reload(self):
        """Recarga configuración desde disco"""
        self._load_config()
        logger.info("Configuración recargada")




