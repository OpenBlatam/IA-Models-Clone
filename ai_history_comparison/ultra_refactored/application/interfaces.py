"""
Application Interfaces - Interfaces de Aplicación
===============================================

Interfaces que definen los contratos para los servicios
y repositorios del sistema.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..domain.models import HistoryEntry, ComparisonResult, ModelType
from ..domain.value_objects import ContentMetrics, QualityScore, SimilarityScore


class IHistoryRepository(ABC):
    """
    Interface para repositorio de historial.
    
    Define las operaciones de acceso a datos para entradas de historial.
    """
    
    @abstractmethod
    async def save(self, entry: HistoryEntry) -> HistoryEntry:
        """Guardar una entrada de historial."""
        pass
    
    @abstractmethod
    async def get_by_id(self, entry_id: str) -> Optional[HistoryEntry]:
        """Obtener entrada por ID."""
        pass
    
    @abstractmethod
    async def list(
        self,
        user_id: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[HistoryEntry]:
        """Listar entradas con filtros."""
        pass
    
    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Eliminar entrada por ID."""
        pass
    
    @abstractmethod
    async def count(
        self,
        user_id: Optional[str] = None,
        model_type: Optional[ModelType] = None
    ) -> int:
        """Contar entradas con filtros."""
        pass


class IComparisonRepository(ABC):
    """
    Interface para repositorio de comparaciones.
    
    Define las operaciones de acceso a datos para resultados de comparación.
    """
    
    @abstractmethod
    async def save(self, comparison: ComparisonResult) -> ComparisonResult:
        """Guardar resultado de comparación."""
        pass
    
    @abstractmethod
    async def get_by_id(self, comparison_id: str) -> Optional[ComparisonResult]:
        """Obtener comparación por ID."""
        pass
    
    @abstractmethod
    async def list(
        self,
        entry_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ComparisonResult]:
        """Listar comparaciones con filtros."""
        pass
    
    @abstractmethod
    async def delete(self, comparison_id: str) -> bool:
        """Eliminar comparación por ID."""
        pass
    
    @abstractmethod
    async def count(self, entry_id: Optional[str] = None) -> int:
        """Contar comparaciones con filtros."""
        pass


class IContentAnalyzer(ABC):
    """
    Interface para analizador de contenido.
    
    Define las operaciones de análisis de contenido.
    """
    
    @abstractmethod
    async def analyze_content(self, content: str) -> ContentMetrics:
        """Analizar contenido y extraer métricas."""
        pass
    
    @abstractmethod
    async def extract_keywords(self, content: str) -> List[str]:
        """Extraer palabras clave del contenido."""
        pass
    
    @abstractmethod
    async def detect_language(self, content: str) -> str:
        """Detectar idioma del contenido."""
        pass
    
    @abstractmethod
    async def analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Analizar sentimiento del contenido."""
        pass


class IQualityAssessor(ABC):
    """
    Interface para evaluador de calidad.
    
    Define las operaciones de evaluación de calidad.
    """
    
    @abstractmethod
    async def assess_quality(self, entry: HistoryEntry) -> QualityScore:
        """Evaluar calidad de una entrada."""
        pass
    
    @abstractmethod
    async def assess_readability(self, content: str) -> float:
        """Evaluar legibilidad del contenido."""
        pass
    
    @abstractmethod
    async def assess_coherence(self, content: str) -> float:
        """Evaluar coherencia del contenido."""
        pass
    
    @abstractmethod
    async def assess_relevance(self, content: str, context: Optional[str] = None) -> float:
        """Evaluar relevancia del contenido."""
        pass


class ISimilarityCalculator(ABC):
    """
    Interface para calculador de similitud.
    
    Define las operaciones de cálculo de similitud.
    """
    
    @abstractmethod
    async def calculate_similarity(self, entry1: HistoryEntry, entry2: HistoryEntry) -> SimilarityScore:
        """Calcular similitud entre dos entradas."""
        pass
    
    @abstractmethod
    async def calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calcular similitud de contenido."""
        pass
    
    @abstractmethod
    async def calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """Calcular similitud semántica."""
        pass
    
    @abstractmethod
    async def calculate_structural_similarity(self, content1: str, content2: str) -> float:
        """Calcular similitud estructural."""
        pass


class IEventPublisher(ABC):
    """
    Interface para publicador de eventos.
    
    Define las operaciones de publicación de eventos del dominio.
    """
    
    @abstractmethod
    async def publish(self, event: Any) -> None:
        """Publicar un evento."""
        pass
    
    @abstractmethod
    async def publish_batch(self, events: List[Any]) -> None:
        """Publicar múltiples eventos."""
        pass


class ICacheService(ABC):
    """
    Interface para servicio de caché.
    
    Define las operaciones de caché.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Establecer valor en el caché."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Eliminar valor del caché."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Limpiar todo el caché."""
        pass


class ILogger(ABC):
    """
    Interface para logger.
    
    Define las operaciones de logging.
    """
    
    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log mensaje de información."""
        pass
    
    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log mensaje de advertencia."""
        pass
    
    @abstractmethod
    def error(self, message: str, **kwargs) -> None:
        """Log mensaje de error."""
        pass
    
    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log mensaje de debug."""
        pass


class IConfigurationService(ABC):
    """
    Interface para servicio de configuración.
    
    Define las operaciones de configuración.
    """
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Obtener valor de configuración."""
        pass
    
    @abstractmethod
    def get_int(self, key: str, default: int = 0) -> int:
        """Obtener valor entero de configuración."""
        pass
    
    @abstractmethod
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Obtener valor flotante de configuración."""
        pass
    
    @abstractmethod
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Obtener valor booleano de configuración."""
        pass
    
    @abstractmethod
    def get_list(self, key: str, default: List = None) -> List:
        """Obtener lista de configuración."""
        pass




