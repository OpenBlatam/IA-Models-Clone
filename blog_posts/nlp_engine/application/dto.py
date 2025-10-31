from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from ..core.enums import AnalysisType, ProcessingTier
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🎯 APPLICATION DTOs - Data Transfer Objects
==========================================

DTOs para transferencia de datos entre capas de la aplicación.
"""



@dataclass
class AnalysisRequest:
    """DTO para request de análisis."""
    text: str
    analysis_types: List[AnalysisType] = field(default_factory=lambda: [AnalysisType.SENTIMENT, AnalysisType.QUALITY_ASSESSMENT])
    processing_tier: Optional[ProcessingTier] = None
    client_id: str = "default"
    request_id: Optional[str] = None
    use_cache: bool = True
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> Any:
        """Validar request."""
        if not self.text or not isinstance(self.text, str):
            raise ValueError("Text must be a non-empty string")
        
        if len(self.text) > 50000:  # Límite razonable
            raise ValueError("Text too long (max 50000 characters)")
        
        if not self.analysis_types:
            raise ValueError("At least one analysis type must be specified")
        
        # Convertir strings a enums si es necesario
        if isinstance(self.analysis_types[0], str):
            try:
                self.analysis_types = [AnalysisType[at.upper()] for at in self.analysis_types]
            except KeyError as e:
                raise ValueError(f"Invalid analysis type: {e}")
        
        # Convertir tier si es string
        if isinstance(self.processing_tier, str):
            try:
                self.processing_tier = ProcessingTier(self.processing_tier)
            except ValueError:
                raise ValueError(f"Invalid processing tier: {self.processing_tier}")


@dataclass
class AnalysisResponse:
    """DTO para response de análisis."""
    success: bool
    request_id: str
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    @classmethod
    def success_response(
        cls,
        request_id: str,
        analysis_results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> 'AnalysisResponse':
        """Factory para response exitoso."""
        return cls(
            success=True,
            request_id=request_id,
            analysis_results=analysis_results,
            metadata=metadata or {},
            metrics=metrics
        )
    
    @classmethod
    def error_response(
        cls,
        request_id: str,
        errors: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'AnalysisResponse':
        """Factory para response de error."""
        return cls(
            success=False,
            request_id=request_id,
            errors=errors,
            metadata=metadata or {}
        )


@dataclass
class BatchAnalysisRequest:
    """DTO para request de análisis en lote."""
    texts: List[str]
    analysis_types: List[AnalysisType] = field(default_factory=lambda: [AnalysisType.SENTIMENT, AnalysisType.QUALITY_ASSESSMENT])
    processing_tier: Optional[ProcessingTier] = None
    client_id: str = "default"
    request_id: Optional[str] = None
    max_concurrency: int = 50
    use_cache: bool = True
    timeout_seconds: float = 300.0  # Más tiempo para lotes
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> Any:
        """Validar request de lote."""
        if not self.texts or not isinstance(self.texts, list):
            raise ValueError("Texts must be a non-empty list")
        
        if len(self.texts) > 1000:  # Límite de lote
            raise ValueError("Too many texts (max 1000 per batch)")
        
        # Validar cada texto
        for i, text in enumerate(self.texts):
            if not text or not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a non-empty string")
            
            if len(text) > 10000:  # Límite por texto en lote
                raise ValueError(f"Text at index {i} too long (max 10000 characters)")
        
        if self.max_concurrency <= 0:
            raise ValueError("Max concurrency must be positive")


@dataclass
class StreamAnalysisRequest:
    """DTO para request de análisis en streaming."""
    analysis_types: List[AnalysisType] = field(default_factory=lambda: [AnalysisType.SENTIMENT])
    processing_tier: Optional[ProcessingTier] = None
    client_id: str = "default"
    session_id: Optional[str] = None
    buffer_size: int = 100
    flush_interval_seconds: float = 1.0
    use_cache: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheRequest:
    """DTO para operaciones de cache."""
    operation: str  # get, set, delete, clear
    key: Optional[str] = None
    value: Optional[Any] = None
    ttl: Optional[int] = None
    pattern: Optional[str] = None
    
    def __post_init__(self) -> Any:
        """Validar request de cache."""
        valid_operations = ['get', 'set', 'delete', 'clear', 'invalidate_pattern']
        if self.operation not in valid_operations:
            raise ValueError(f"Invalid operation. Must be one of: {valid_operations}")
        
        if self.operation in ['get', 'set', 'delete'] and not self.key:
            raise ValueError(f"Key is required for operation: {self.operation}")
        
        if self.operation == 'set' and self.value is None:
            raise ValueError("Value is required for set operation")
        
        if self.operation == 'invalidate_pattern' and not self.pattern:
            raise ValueError("Pattern is required for invalidate_pattern operation")


@dataclass
class HealthCheckRequest:
    """DTO para request de health check."""
    component: Optional[str] = None  # None significa check completo
    deep_check: bool = False
    timeout_seconds: float = 10.0
    include_metrics: bool = True


@dataclass
class HealthCheckResponse:
    """DTO para response de health check."""
    status: str  # healthy, degraded, unhealthy
    timestamp: float
    uptime_seconds: float
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    version: str = "1.0.0-modular"


@dataclass
class MetricsRequest:
    """DTO para request de métricas."""
    metric_names: Optional[List[str]] = None  # None significa todas
    time_range: Optional[str] = None  # ej: "1h", "24h"
    format_type: str = "json"  # json, prometheus, etc.
    include_history: bool = False
    aggregation: Optional[str] = None  # sum, avg, max, min


@dataclass
class MetricsResponse:
    """DTO para response de métricas."""
    timestamp: float
    metrics: Dict[str, Any]
    format_type: str
    time_range: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigurationRequest:
    """DTO para request de configuración."""
    operation: str  # get, set, reload, validate
    key: Optional[str] = None
    value: Optional[Any] = None
    section: Optional[str] = None
    
    def __post_init__(self) -> Any:
        """Validar request de configuración."""
        valid_operations = ['get', 'set', 'reload', 'validate', 'get_all']
        if self.operation not in valid_operations:
            raise ValueError(f"Invalid operation. Must be one of: {valid_operations}")


@dataclass
class ConfigurationResponse:
    """DTO para response de configuración."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: __import__('time').time())


@dataclass 
class ValidationResult:
    """DTO para resultados de validación."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    field_errors: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_error(self, field: str, message: str):
        """Añadir error de campo específico."""
        if field not in self.field_errors:
            self.field_errors[field] = []
        self.field_errors[field].append(message)
        self.errors.append(f"{field}: {message}")
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Añadir warning."""
        self.warnings.append(message)


# Utility functions para DTOs
async def create_analysis_request(
    text: str,
    analysis_types: Optional[List[str]] = None,
    tier: Optional[str] = None,
    **kwargs
) -> AnalysisRequest:
    """Helper para crear AnalysisRequest desde parámetros simples."""
    
    # Convertir strings a enums
    if analysis_types:
        enum_types = []
        for at in analysis_types:
            try:
                enum_types.append(AnalysisType[at.upper()])
            except KeyError:
                raise ValueError(f"Invalid analysis type: {at}")
    else:
        enum_types = [AnalysisType.SENTIMENT, AnalysisType.QUALITY_ASSESSMENT]
    
    # Convertir tier
    tier_enum = None
    if tier:
        try:
            tier_enum = ProcessingTier(tier)
        except ValueError:
            raise ValueError(f"Invalid processing tier: {tier}")
    
    return AnalysisRequest(
        text=text,
        analysis_types=enum_types,
        processing_tier=tier_enum,
        **kwargs
    )


def serialize_response(response: AnalysisResponse) -> Dict[str, Any]:
    """Serializar response para API."""
    return {
        'success': response.success,
        'request_id': response.request_id,
        'analysis': response.analysis_results,
        'metadata': response.metadata,
        'errors': response.errors,
        'metrics': response.metrics,
        'timestamp': response.timestamp
    } 