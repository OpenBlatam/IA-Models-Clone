"""
Configuration Validator Module
=============================

Valida y normaliza la configuración del agente.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Valida y normaliza la configuración del agente.
    
    Asegura que todos los valores de configuración sean válidos
    y proporciona valores por defecto cuando sea necesario.
    """
    
    # Rangos válidos para diferentes configuraciones
    VALID_PRIORITY_RANGE = (1, 10)
    VALID_TEMPERATURE_RANGE = (0.0, 2.0)
    VALID_THRESHOLD_RANGE = (0.0, 1.0)
    VALID_SLEEP_RANGE = (0.1, 3600.0)  # 0.1s a 1 hora
    
    @staticmethod
    def validate_and_normalize(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validar y normalizar configuración.
        
        Args:
            config: Configuración a validar
            
        Returns:
            Configuración validada y normalizada
        """
        if config is None:
            config = {}
        
        validated = config.copy()
        
        # Validar flags booleanos
        boolean_keys = [
            "react_enabled", "lats_enabled", "tot_enabled",
            "tom_enabled", "personality_enabled", "toolformer_enabled",
            "learning_enabled"
        ]
        for key in boolean_keys:
            if key in validated:
                validated[key] = bool(validated[key])
        
        # Validar umbrales de estrategias
        threshold_keys = {
            "react_threshold": (1, 10),
            "tot_threshold": (1, 10),
            "lats_threshold": (1, 10)
        }
        for key, (min_val, max_val) in threshold_keys.items():
            if key in validated:
                validated[key] = max(min_val, min(max_val, int(validated[key])))
        
        # Validar reflection_threshold
        if "reflection_threshold" in validated:
            validated["reflection_threshold"] = max(
                1, int(validated["reflection_threshold"])
            )
        
        # Validar planning_horizon
        if "planning_horizon" in validated:
            validated["planning_horizon"] = max(
                1, min(10, int(validated["planning_horizon"]))
            )
        
        # Validar performance_threshold
        if "performance_threshold" in validated:
            threshold = float(validated["performance_threshold"])
            validated["performance_threshold"] = max(
                0.0, min(1.0, threshold)
            )
        
        # Validar error_threshold
        if "error_threshold" in validated:
            validated["error_threshold"] = max(1, int(validated["error_threshold"]))
        
        # Validar tot_strategy
        if "tot_strategy" in validated:
            valid_strategies = ["bfs", "dfs", "beam"]
            if validated["tot_strategy"] not in valid_strategies:
                logger.warning(
                    f"Invalid tot_strategy: {validated['tot_strategy']}, "
                    f"using 'bfs'"
                )
                validated["tot_strategy"] = "bfs"
        
        # Validar personality_profile
        if "personality_profile" in validated:
            profile = validated["personality_profile"]
            if isinstance(profile, dict):
                # Validar que los valores estén en rango [0.0, 1.0]
                for key, value in profile.items():
                    if isinstance(value, (int, float)):
                        profile[key] = max(0.0, min(1.0, float(value)))
        
        return validated
    
    @staticmethod
    def validate_priority(priority: int) -> int:
        """
        Validar prioridad de tarea.
        
        Args:
            priority: Prioridad a validar
            
        Returns:
            Prioridad validada
        """
        min_priority, max_priority = ConfigValidator.VALID_PRIORITY_RANGE
        return max(min_priority, min(max_priority, int(priority)))
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Obtener configuración por defecto.
        
        Returns:
            Dict con valores por defecto
        """
        return {
            # Estrategias
            "react_enabled": True,
            "lats_enabled": True,
            "tot_enabled": True,
            "tom_enabled": True,
            "personality_enabled": False,
            "toolformer_enabled": True,
            "learning_enabled": True,
            
            # Umbrales de estrategias
            "react_threshold": 7,
            "tot_threshold": 8,
            "lats_threshold": 9,
            
            # Reflection y Planning
            "reflection_threshold": 5,
            "planning_horizon": 3,
            
            # Learning
            "performance_threshold": 0.6,
            "error_threshold": 3,
            
            # Tree of Thoughts
            "tot_strategy": "bfs",
            
            # Autonomy
            "autonomy_level": "fully_autonomous"
        }
    
    @staticmethod
    def merge_with_defaults(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combinar configuración con valores por defecto.
        
        Args:
            config: Configuración proporcionada
            
        Returns:
            Configuración combinada y validada
        """
        defaults = ConfigValidator.get_default_config()
        merged = {**defaults, **(config or {})}
        return ConfigValidator.validate_and_normalize(merged)
