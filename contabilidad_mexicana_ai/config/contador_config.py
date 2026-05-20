"""
Configuration for Contabilidad Mexicana AI with Open Router.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class OpenRouterConfig:
    """Configuration for Open Router API integration."""
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "anthropic/claude-3.5-sonnet"
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.3
    max_tokens: int = 4000


@dataclass
class ContadorConfig:
    """Main configuration for the Contador AI system."""
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    
    # Regímenes fiscales soportados
    regimenes_fiscales: List[str] = field(default_factory=lambda: [
        "RESICO", "PFAE", "Plataformas", "Sueldos y Salarios",
        "Personas Físicas", "Personas Morales"
    ])
    
    # Tipos de impuestos
    tipos_impuestos: List[str] = field(default_factory=lambda: [
        "ISR", "IVA", "IEPS"
    ])
    
    # Servicios disponibles
    servicios: List[str] = field(default_factory=lambda: [
        "calculo_impuestos",
        "asesoria_fiscal",
        "guias_fiscales",
        "tramites_sat",
        "declaraciones",
        "devoluciones",
        "regularizacion"
    ])
    
    # Response settings
    response_style: str = "profesional"
    include_examples: bool = True
    include_formulas: bool = True
    include_references: bool = True
    
    # Storage settings
    conversation_history_path: str = "conversations/contador"
    max_history_length: int = 50
    
    # Performance settings
    cache_enabled: bool = True
    cache_ttl: int = 3600
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.openrouter.api_key:
            raise ValueError("OPENROUTER_API_KEY must be set")
        return True
