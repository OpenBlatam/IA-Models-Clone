"""
Model Validator - Validación y verificación de modelos LLM.

Valida que los modelos solicitados estén disponibles y sean compatibles
con las operaciones requeridas.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from config.logging_config import get_logger

logger = get_logger(__name__)


class ModelCapability(str, Enum):
    """Capacidades de un modelo."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    MULTIMODAL = "multimodal"


@dataclass
class ModelInfo:
    """Información sobre un modelo."""
    model_id: str
    name: str
    provider: str
    capabilities: List[ModelCapability]
    max_tokens: int
    supports_streaming: bool = False
    cost_per_1k_tokens: Optional[float] = None
    is_available: bool = True
    last_checked: Optional[datetime] = None
    
    def has_capability(self, capability: ModelCapability) -> bool:
        """Verificar si el modelo tiene una capacidad."""
        return capability in self.capabilities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "provider": self.provider,
            "capabilities": [c.value for c in self.capabilities],
            "max_tokens": self.max_tokens,
            "supports_streaming": self.supports_streaming,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "is_available": self.is_available,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None
        }


class ModelValidator:
    """
    Validador de modelos LLM.
    
    Características:
    - Validación de disponibilidad de modelos
    - Verificación de capacidades
    - Cache de información de modelos
    - Validación de compatibilidad
    """
    
    def __init__(self, cache_ttl_seconds: int = 3600):
        """
        Inicializar validador.
        
        Args:
            cache_ttl_seconds: TTL del cache de modelos en segundos
        """
        self.cache_ttl = cache_ttl_seconds
        self.models_cache: Dict[str, ModelInfo] = {}
        self.available_models: Set[str] = set()
        self.last_fetch: Optional[datetime] = None
        
        # Modelos conocidos con sus capacidades
        self.known_models: Dict[str, Dict[str, Any]] = {
            "openai/gpt-4": {
                "name": "GPT-4",
                "provider": "openai",
                "capabilities": [
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CODE_ANALYSIS,
                    ModelCapability.STREAMING,
                    ModelCapability.FUNCTION_CALLING
                ],
                "max_tokens": 8192,
                "supports_streaming": True
            },
            "openai/gpt-4-turbo": {
                "name": "GPT-4 Turbo",
                "provider": "openai",
                "capabilities": [
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CODE_ANALYSIS,
                    ModelCapability.STREAMING,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.VISION
                ],
                "max_tokens": 128000,
                "supports_streaming": True
            },
            "anthropic/claude-3-opus": {
                "name": "Claude 3 Opus",
                "provider": "anthropic",
                "capabilities": [
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CODE_ANALYSIS,
                    ModelCapability.STREAMING,
                    ModelCapability.VISION
                ],
                "max_tokens": 200000,
                "supports_streaming": True
            },
            "google/gemini-pro": {
                "name": "Gemini Pro",
                "provider": "google",
                "capabilities": [
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.STREAMING,
                    ModelCapability.MULTIMODAL
                ],
                "max_tokens": 32768,
                "supports_streaming": True
            }
        }
    
    async def fetch_available_models(self, llm_service: Any) -> List[str]:
        """
        Obtener lista de modelos disponibles desde OpenRouter.
        
        Args:
            llm_service: Instancia de LLMService
            
        Returns:
            Lista de IDs de modelos disponibles
        """
        try:
            models = await llm_service.get_available_models()
            if models:
                self.available_models = {m.get("id", "") for m in models if m.get("id")}
                self.last_fetch = datetime.now()
                logger.info(f"Modelos disponibles actualizados: {len(self.available_models)} modelos")
            return list(self.available_models)
        except Exception as e:
            logger.error(f"Error al obtener modelos disponibles: {e}")
            return list(self.available_models)
    
    def validate_model(
        self,
        model_id: str,
        required_capability: Optional[ModelCapability] = None,
        require_streaming: bool = False
    ) -> tuple[bool, Optional[str]]:
        """
        Validar que un modelo esté disponible y tenga las capacidades requeridas.
        
        Args:
            model_id: ID del modelo
            required_capability: Capacidad requerida (opcional)
            require_streaming: Si requiere soporte de streaming
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        # Verificar cache
        if self.last_fetch and (datetime.now() - self.last_fetch).total_seconds() > self.cache_ttl:
            # Cache expirado, necesitamos refrescar
            logger.debug("Cache de modelos expirado, necesita refrescar")
        
        # Verificar si el modelo está en la lista de disponibles
        if self.available_models and model_id not in self.available_models:
            return False, f"Modelo '{model_id}' no está disponible"
        
        # Obtener información del modelo
        model_info = self.get_model_info(model_id)
        if not model_info:
            # Modelo desconocido, asumir que está disponible pero sin info
            logger.warning(f"Modelo desconocido: {model_id}, asumiendo disponible")
            return True, None
        
        # Verificar disponibilidad
        if not model_info.is_available:
            return False, f"Modelo '{model_id}' no está disponible actualmente"
        
        # Verificar capacidad requerida
        if required_capability and not model_info.has_capability(required_capability):
            return False, (
                f"Modelo '{model_id}' no soporta la capacidad requerida: "
                f"{required_capability.value}"
            )
        
        # Verificar streaming
        if require_streaming and not model_info.supports_streaming:
            return False, f"Modelo '{model_id}' no soporta streaming"
        
        return True, None
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Obtener información de un modelo.
        
        Args:
            model_id: ID del modelo
            
        Returns:
            Información del modelo o None
        """
        # Verificar cache
        if model_id in self.models_cache:
            return self.models_cache[model_id]
        
        # Buscar en modelos conocidos
        if model_id in self.known_models:
            model_data = self.known_models[model_id]
            model_info = ModelInfo(
                model_id=model_id,
                name=model_data["name"],
                provider=model_data["provider"],
                capabilities=model_data["capabilities"],
                max_tokens=model_data["max_tokens"],
                supports_streaming=model_data.get("supports_streaming", False),
                is_available=model_id in self.available_models if self.available_models else True,
                last_checked=datetime.now()
            )
            self.models_cache[model_id] = model_info
            return model_info
        
        # Modelo desconocido, crear info básica
        logger.debug(f"Modelo desconocido: {model_id}, creando info básica")
        model_info = ModelInfo(
            model_id=model_id,
            name=model_id.split("/")[-1] if "/" in model_id else model_id,
            provider=model_id.split("/")[0] if "/" in model_id else "unknown",
            capabilities=[ModelCapability.TEXT_GENERATION],
            max_tokens=4096,  # Default conservador
            supports_streaming=False,
            is_available=model_id in self.available_models if self.available_models else True,
            last_checked=datetime.now()
        )
        self.models_cache[model_id] = model_info
        return model_info
    
    def validate_models_for_parallel(
        self,
        model_ids: List[str],
        max_models: int = 10
    ) -> tuple[bool, Optional[str], List[str]]:
        """
        Validar múltiples modelos para generación paralela.
        
        Args:
            model_ids: Lista de IDs de modelos
            max_models: Máximo de modelos permitidos
            
        Returns:
            Tupla (es_válido, mensaje_error, modelos_válidos)
        """
        if len(model_ids) > max_models:
            return False, f"Máximo {max_models} modelos permitidos para generación paralela", []
        
        valid_models = []
        for model_id in model_ids:
            is_valid, error = self.validate_model(model_id)
            if is_valid:
                valid_models.append(model_id)
            else:
                logger.warning(f"Modelo inválido para generación paralela: {model_id} - {error}")
        
        if not valid_models:
            return False, "Ningún modelo válido para generación paralela", []
        
        return True, None, valid_models
    
    def get_recommended_model(
        self,
        capability: ModelCapability,
        prefer_streaming: bool = False,
        max_cost: Optional[float] = None
    ) -> Optional[str]:
        """
        Obtener modelo recomendado para una capacidad.
        
        Args:
            capability: Capacidad requerida
            prefer_streaming: Preferir modelos con streaming
            max_cost: Costo máximo por 1k tokens (opcional)
            
        Returns:
            ID del modelo recomendado o None
        """
        candidates = []
        
        for model_id, model_info in self.models_cache.items():
            if not model_info.is_available:
                continue
            
            if not model_info.has_capability(capability):
                continue
            
            if prefer_streaming and not model_info.supports_streaming:
                continue
            
            if max_cost and model_info.cost_per_1k_tokens:
                if model_info.cost_per_1k_tokens > max_cost:
                    continue
            
            candidates.append((model_id, model_info))
        
        if not candidates:
            return None
        
        # Ordenar por costo (si está disponible) o por nombre
        candidates.sort(key=lambda x: (
            x[1].cost_per_1k_tokens or float('inf'),
            x[0]
        ))
        
        return candidates[0][0]
    
    def clear_cache(self):
        """Limpiar cache de modelos."""
        self.models_cache.clear()
        self.available_models.clear()
        self.last_fetch = None
        logger.info("Cache de modelos limpiado")


def get_model_validator(cache_ttl_seconds: int = 3600) -> ModelValidator:
    """Factory function para obtener instancia singleton del validador."""
    if not hasattr(get_model_validator, "_instance"):
        get_model_validator._instance = ModelValidator(cache_ttl_seconds)
    return get_model_validator._instance



