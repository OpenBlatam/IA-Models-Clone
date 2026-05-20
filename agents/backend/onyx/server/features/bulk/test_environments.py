"""
Configuración de Entornos
Gestiona diferentes entornos (dev, staging, prod)
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

class EnvironmentConfig:
    """Configuración de entorno."""
    
    ENVIRONMENTS = {
        "dev": {
            "base_url": "http://localhost:8000",
            "timeout": 30,
            "rate_limit": 100,  # Más permisivo en dev
            "debug": True,
            "mock_enabled": True
        },
        "staging": {
            "base_url": "https://staging-api.example.com",
            "timeout": 30,
            "rate_limit": 50,
            "debug": False,
            "mock_enabled": False
        },
        "production": {
            "base_url": "https://api.example.com",
            "timeout": 20,
            "rate_limit": 10,
            "debug": False,
            "mock_enabled": False
        }
    }
    
    def __init__(self, env: str = None):
        self.current_env = env or os.getenv("BUL_ENV", "dev")
        self.config = self.ENVIRONMENTS.get(self.current_env, self.ENVIRONMENTS["dev"])
    
    def get_base_url(self) -> str:
        """Obtiene URL base del entorno."""
        return self.config["base_url"]
    
    def get_timeout(self) -> int:
        """Obtiene timeout del entorno."""
        return self.config["timeout"]
    
    def is_debug(self) -> bool:
        """Verifica si está en modo debug."""
        return self.config.get("debug", False)
    
    def get_config(self) -> Dict[str, Any]:
        """Obtiene configuración completa."""
        return {
            **self.config,
            "environment": self.current_env
        }
    
    def print_config(self):
        """Imprime configuración actual."""
        print(f"\n{'='*70}")
        print(f"  ⚙️ CONFIGURACIÓN DE ENTORNO")
        print(f"{'='*70}\n")
        print(f"Entorno actual: {self.current_env.upper()}")
        print(f"Base URL: {self.config['base_url']}")
        print(f"Timeout: {self.config['timeout']}s")
        print(f"Debug: {self.config.get('debug', False)}")
        print(f"Rate Limit: {self.config.get('rate_limit', 10)} req/min")
        print(f"Mock Enabled: {self.config.get('mock_enabled', False)}")
        print(f"\n{'='*70}\n")

def get_environment_config(env: str = None) -> EnvironmentConfig:
    """Obtiene configuración de entorno."""
    return EnvironmentConfig(env)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuración de Entornos")
    parser.add_argument("--env", choices=["dev", "staging", "production"], 
                       default="dev", help="Entorno a usar")
    
    args = parser.parse_args()
    
    config = get_environment_config(args.env)
    config.print_config()
































