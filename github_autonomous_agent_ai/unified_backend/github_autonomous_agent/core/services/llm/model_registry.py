"""
Model Registry - Registro y gestión de modelos LLM.

Sigue principios de configuración centralizada y gestión de modelos.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import json

from config.logging_config import get_logger

logger = get_logger(__name__)


class ModelProvider(str, Enum):
    """Proveedores de modelos."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    META = "meta"
    MISTRAL = "mistralai"
    COHERE = "cohere"
    PERPLEXITY = "perplexity"
    OTHER = "other"


@dataclass
class ModelConfig:
    """Configuración de un modelo LLM."""
    name: str
    provider: ModelProvider
    model_id: str  # ID completo en OpenRouter (ej: "openai/gpt-4o")
    max_tokens: int
    max_output_tokens: int
    supports_streaming: bool = True
    supports_function_calling: bool = False
    default_temperature: float = 0.7
    default_max_tokens: Optional[int] = None
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    context_window: int = 4096
    description: str = ""
    capabilities: List[str] = None
    recommended_use_cases: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.recommended_use_cases is None:
            self.recommended_use_cases = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Crear desde diccionario."""
        if isinstance(data.get("provider"), str):
            data["provider"] = ModelProvider(data["provider"])
        return cls(**data)


class ModelRegistry:
    """
    Registry de modelos LLM con configuraciones.
    
    Sigue principios de gestión centralizada de modelos.
    """
    
    def __init__(self):
        """Inicializar registry."""
        self.models: Dict[str, ModelConfig] = {}
        self._register_default_models()
    
    def register(self, config: ModelConfig) -> None:
        """Registrar un modelo."""
        self.models[config.model_id] = config
        logger.debug(f"Modelo '{config.model_id}' registrado")
    
    def get(self, model_id: str) -> Optional[ModelConfig]:
        """Obtener configuración de un modelo."""
        return self.models.get(model_id)
    
    def list_models(
        self,
        provider: Optional[ModelProvider] = None,
        capability: Optional[str] = None
    ) -> List[ModelConfig]:
        """
        Listar modelos con filtros opcionales.
        
        Args:
            provider: Filtrar por proveedor
            capability: Filtrar por capacidad
            
        Returns:
            Lista de configuraciones de modelos
        """
        models = list(self.models.values())
        
        if provider:
            models = [m for m in models if m.provider == provider]
        
        if capability:
            models = [m for m in models if capability in m.capabilities]
        
        return models
    
    def get_recommended_model(
        self,
        use_case: str,
        budget_conscious: bool = False
    ) -> Optional[ModelConfig]:
        """
        Obtener modelo recomendado para un caso de uso.
        
        Args:
            use_case: Caso de uso (code_analysis, documentation, etc.)
            budget_conscious: Priorizar modelos económicos
            
        Returns:
            ModelConfig recomendado o None
        """
        candidates = [
            m for m in self.models.values()
            if use_case in m.recommended_use_cases
        ]
        
        if not candidates:
            return None
        
        if budget_conscious:
            # Ordenar por costo (input + output)
            candidates.sort(
                key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output
            )
        else:
            # Ordenar por capacidad (más tokens = mejor)
            candidates.sort(key=lambda m: m.max_tokens, reverse=True)
        
        return candidates[0]
    
    def estimate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Estimar costo de un request.
        
        Args:
            model_id: ID del modelo
            input_tokens: Tokens de entrada
            output_tokens: Tokens de salida
            
        Returns:
            Costo estimado en dólares
        """
        config = self.get(model_id)
        if not config:
            return 0.0
        
        input_cost = (input_tokens / 1000) * config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * config.cost_per_1k_output
        
        return input_cost + output_cost
    
    def _register_default_models(self):
        """Registrar modelos por defecto con configuraciones."""
        
        # OpenAI Models
        self.register(ModelConfig(
            name="GPT-4o",
            provider=ModelProvider.OPENAI,
            model_id="openai/gpt-4o",
            max_tokens=128000,
            max_output_tokens=16384,
            supports_function_calling=True,
            default_temperature=0.7,
            cost_per_1k_input=0.0025,
            cost_per_1k_output=0.01,
            context_window=128000,
            description="GPT-4o: Modelo más avanzado de OpenAI",
            capabilities=["code", "analysis", "reasoning", "multimodal"],
            recommended_use_cases=["code_analysis", "complex_reasoning", "documentation"]
        ))
        
        self.register(ModelConfig(
            name="GPT-4o Mini",
            provider=ModelProvider.OPENAI,
            model_id="openai/gpt-4o-mini",
            max_tokens=128000,
            max_output_tokens=16384,
            supports_function_calling=True,
            default_temperature=0.7,
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            context_window=128000,
            description="GPT-4o Mini: Versión económica y rápida",
            capabilities=["code", "analysis", "general"],
            recommended_use_cases=["code_analysis", "documentation", "general"]
        ))
        
        # Anthropic Models
        self.register(ModelConfig(
            name="Claude 3.5 Sonnet",
            provider=ModelProvider.ANTHROPIC,
            model_id="anthropic/claude-3.5-sonnet",
            max_tokens=200000,
            max_output_tokens=8192,
            supports_function_calling=True,
            default_temperature=0.7,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            context_window=200000,
            description="Claude 3.5 Sonnet: Excelente para análisis y razonamiento",
            capabilities=["code", "analysis", "reasoning", "long_context"],
            recommended_use_cases=["code_analysis", "documentation", "refactoring"]
        ))
        
        self.register(ModelConfig(
            name="Claude 3 Opus",
            provider=ModelProvider.ANTHROPIC,
            model_id="anthropic/claude-3-opus",
            max_tokens=200000,
            max_output_tokens=4096,
            supports_function_calling=True,
            default_temperature=0.7,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            context_window=200000,
            description="Claude 3 Opus: Modelo más potente de Anthropic",
            capabilities=["code", "analysis", "reasoning", "complex_tasks"],
            recommended_use_cases=["complex_analysis", "refactoring", "code_generation"]
        ))
        
        # Google Models
        self.register(ModelConfig(
            name="Gemini Pro 1.5",
            provider=ModelProvider.GOOGLE,
            model_id="google/gemini-pro-1.5",
            max_tokens=1000000,
            max_output_tokens=8192,
            supports_function_calling=False,
            default_temperature=0.7,
            cost_per_1k_input=0.000125,
            cost_per_1k_output=0.0005,
            context_window=1000000,
            description="Gemini Pro 1.5: Contexto muy largo, económico",
            capabilities=["code", "long_context", "multimodal"],
            recommended_use_cases=["large_codebase_analysis", "documentation"]
        ))
        
        logger.info(f"Registrados {len(self.models)} modelos por defecto")
    
    def save_to_file(self, filepath: str) -> None:
        """Guardar registry a archivo JSON."""
        data = {
            model_id: config.to_dict()
            for model_id, config in self.models.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Registry guardado en {filepath}")
    
    def load_from_file(self, filepath: str) -> None:
        """Cargar registry desde archivo JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for model_id, config_data in data.items():
            config = ModelConfig.from_dict(config_data)
            self.register(config)
        
        logger.info(f"Registry cargado desde {filepath}: {len(data)} modelos")


# Instancia global
_model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Obtener instancia global del registry."""
    return _model_registry



