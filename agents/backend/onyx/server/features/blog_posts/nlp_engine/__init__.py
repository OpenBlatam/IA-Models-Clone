from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from nlp_engine import NLPEngine
from nlp_engine.core.enums import AnalysisType, ProcessingTier
from .core.entities import (
from .core.enums import (
from .core.domain_services import (
from .application.dto import (
from .application.use_cases import (
from .application.services import (
from .interfaces import (
            from .demo_infrastructure import MockAnalyzerFactory
            from .demo_infrastructure import MockCacheRepository
            from .demo_infrastructure import MockMetricsCollector
            from .demo_infrastructure import MockConfigurationService
            from .demo_infrastructure import MockStructuredLogger
            from .core.entities import TextFingerprint
                from .core.entities import TextFingerprint
                from .core.entities import TextFingerprint
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🚀 NLP ENGINE - Sistema Modular Enterprise
=========================================

Motor de Procesamiento de Lenguaje Natural con arquitectura modular
enterprise-grade implementando Clean Architecture y SOLID principles.

Capas:
- Core: Domain Logic, Entities, Value Objects, Domain Services
- Interfaces: Ports & Contracts (ABCs)
- Application: Use Cases, Services, DTOs
- Infrastructure: Implementaciones concretas (no incluidas en core)

Características:
- ⚡ Ultra-Fast Processing: < 0.1ms latency target
- 🏗️  Clean Architecture: Separación clara de responsabilidades
- 🔧 Dependency Injection: Inversión de dependencias completa
- 📊 Multi-Tier Processing: ULTRA_FAST, BALANCED, HIGH_QUALITY, RESEARCH_GRADE
- 🗄️  Advanced Caching: Multi-level con eviction policies
- 📈 Real-time Metrics: Performance monitoring completo
- 🔍 Health Checks: Monitoreo de salud del sistema
- 📝 Structured Logging: Logging enterprise con contexto
- 🎯 Batch Processing: Análisis en lote con concurrencia controlada
- 🌊 Stream Processing: Análisis en tiempo real
- ⚙️  Configuration Management: Multi-environment config
- 🔒 Type Safety: Typing completo en Python

Ejemplo de uso:

```python

# Crear motor
engine = NLPEngine()
await engine.initialize()

# Análisis simple
result = await engine.analyze(
    text="Este es un texto excelente!",
    analysis_types=[AnalysisType.SENTIMENT, AnalysisType.QUALITY_ASSESSMENT],
    tier=ProcessingTier.BALANCED
)

print(f"Sentimiento: {result.get_sentiment_score()}")
print(f"Calidad: {result.get_quality_score()}")
```
"""

# Version info
__version__ = "1.0.0-modular"
__author__ = "Enterprise NLP Team"
__description__ = "Modular NLP Engine with Clean Architecture"

# Core exports
    AnalysisResult, 
    TextFingerprint, 
    AnalysisScore, 
    ProcessingMetrics,
    AnalysisError
)

    AnalysisType,
    ProcessingTier, 
    CacheStrategy,
    Environment,
    LogLevel,
    AnalysisStatus,
    ErrorType,
    MetricType
)

    AnalysisOrchestrator,
    TextProcessor,
    ScoreValidator
)

# Application exports
    AnalysisRequest,
    AnalysisResponse,
    BatchAnalysisRequest,
    create_analysis_request,
    serialize_response
)

    AnalyzeTextUseCase,
    BatchAnalysisUseCase,
    StreamAnalysisUseCase
)

    AnalysisService,
    CacheService,
    MetricsService,
    ConfigurationService
)

# Interface exports (for implementations)
    # Analyzer interfaces
    IAnalyzer,
    IAnalyzerFactory,
    IAdvancedAnalyzer,
    IConfigurableAnalyzer,
    
    # Cache interfaces
    ICacheRepository,
    IDistributedCache,
    ICacheKeyGenerator,
    ICacheEvictionPolicy,
    ICacheSerializer,
    
    # Metrics interfaces
    IMetricsCollector,
    IPerformanceMonitor,
    IHealthChecker,
    IAlertManager,
    IStructuredLogger,
    IMetricsExporter,
    
    # Config interfaces
    IConfigurationService,
    IEnvironmentConfigLoader,
    IFileConfigLoader,
    ISecretManager,
    IConfigValidator,
    IConfigMerger,
    IConfigTransformer
)


class NLPEngine:
    """
    🚀 MAIN NLP ENGINE - Facade principal del sistema modular.
    
    Proporciona una API simplificada sobre toda la arquitectura modular.
    """
    
    def __init__(
        self,
        analyzer_factory: Optional[IAnalyzerFactory] = None,
        cache_repository: Optional[ICacheRepository] = None,
        metrics_collector: Optional[IMetricsCollector] = None,
        config_service: Optional[IConfigurationService] = None,
        logger: Optional[IStructuredLogger] = None
    ):
        """
        Inicializar motor NLP con dependencias inyectadas.
        
        Args:
            analyzer_factory: Factory para crear analyzers
            cache_repository: Repositorio de cache
            metrics_collector: Collector de métricas
            config_service: Servicio de configuración
            logger: Logger estructurado
        """
        # Usar implementaciones mock por defecto si no se proporcionan
        if not analyzer_factory:
            analyzer_factory = MockAnalyzerFactory()
        
        if not cache_repository:
            cache_repository = MockCacheRepository()
        
        if not metrics_collector:
            metrics_collector = MockMetricsCollector()
        
        if not config_service:
            config_service = MockConfigurationService()
        
        if not logger:
            logger = MockStructuredLogger()
        
        # Infrastructure
        self._analyzer_factory = analyzer_factory
        self._cache_repository = cache_repository
        self._metrics_collector = metrics_collector
        self._config_service = config_service
        self._logger = logger
        
        # Application Layer
        self._analysis_use_case: Optional[AnalyzeTextUseCase] = None
        self._batch_use_case: Optional[BatchAnalysisUseCase] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Inicializar el motor NLP."""
        if self._initialized:
            return
        
        # Crear use cases
        self._analysis_use_case = AnalyzeTextUseCase(
            analyzer_factory=self._analyzer_factory,
            cache_repository=self._cache_repository,
            metrics_collector=self._metrics_collector,
            config_service=self._config_service,
            logger=self._logger
        )
        
        self._batch_use_case = BatchAnalysisUseCase(
            analyze_text_use_case=self._analysis_use_case
        )
        
        self._initialized = True
        self._logger.log_structured('INFO', 'NLP Engine initialized successfully')
    
    async def analyze(
        self,
        text: str,
        analysis_types: Optional[List[AnalysisType]] = None,
        tier: Optional[ProcessingTier] = None,
        use_cache: bool = True,
        client_id: str = "api_client"
    ) -> AnalysisResult:
        """
        Analizar texto individual.
        
        Args:
            text: Texto a analizar
            analysis_types: Tipos de análisis a realizar
            tier: Tier de procesamiento
            use_cache: Si usar cache
            client_id: ID del cliente
            
        Returns:
            AnalysisResult con resultados
        """
        if not self._initialized:
            await self.initialize()
        
        if not analysis_types:
            analysis_types = [AnalysisType.SENTIMENT, AnalysisType.QUALITY_ASSESSMENT]
        
        request = AnalysisRequest(
            text=text,
            analysis_types=analysis_types,
            processing_tier=tier,
            client_id=client_id,
            use_cache=use_cache
        )
        
        response = await self._analysis_use_case.execute(request)
        
        if response.success:
            # Convertir response a AnalysisResult para API simplificada
            fingerprint = TextFingerprint.create(text)
            result = AnalysisResult(fingerprint=fingerprint)
            
            # Llenar scores desde response
            for analysis_type_str, score_data in response.analysis_results.get('scores', {}).items():
                analysis_type = AnalysisType[analysis_type_str]
                result.add_score(
                    analysis_type=analysis_type,
                    value=score_data['value'],
                    confidence=score_data['confidence'],
                    method=score_data['method'],
                    metadata=score_data.get('metadata', {})
                )
            
            return result
        else:
            raise RuntimeError(f"Analysis failed: {', '.join(response.errors)}")
    
    async def analyze_batch(
        self,
        texts: List[str],
        analysis_types: Optional[List[AnalysisType]] = None,
        tier: Optional[ProcessingTier] = None,
        max_concurrency: int = 50,
        client_id: str = "batch_client"
    ) -> List[AnalysisResult]:
        """
        Analizar lote de textos.
        
        Args:
            texts: Lista de textos
            analysis_types: Tipos de análisis
            tier: Tier de procesamiento
            max_concurrency: Máxima concurrencia
            client_id: ID del cliente
            
        Returns:
            Lista de AnalysisResult
        """
        if not self._initialized:
            await self.initialize()
        
        if not analysis_types:
            analysis_types = [AnalysisType.SENTIMENT]
        
        batch_request = BatchAnalysisRequest(
            texts=texts,
            analysis_types=analysis_types,
            processing_tier=tier,
            max_concurrency=max_concurrency,
            client_id=client_id
        )
        
        responses = await self._batch_use_case.execute(batch_request)
        
        results = []
        for i, response in enumerate(responses):
            if response.success:
                fingerprint = TextFingerprint.create(texts[i])
                result = AnalysisResult(fingerprint=fingerprint)
                
                for analysis_type_str, score_data in response.analysis_results.get('scores', {}).items():
                    analysis_type = AnalysisType[analysis_type_str]
                    result.add_score(
                        analysis_type=analysis_type,
                        value=score_data['value'],
                        confidence=score_data['confidence'],
                        method=score_data['method'],
                        metadata=score_data.get('metadata', {})
                    )
                
                results.append(result)
            else:
                # Crear resultado de error
                fingerprint = TextFingerprint.create(texts[i])
                result = AnalysisResult(fingerprint=fingerprint)
                
                for error in response.errors:
                    result.add_error(ErrorType.PROCESSING_ERROR, error)
                
                results.append(result)
        
        return results
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Obtener estado de salud del sistema."""
        if not self._initialized:
            await self.initialize()
        
        # Mock health status
        return {
            'status': 'healthy',
            'version': __version__,
            'initialized': self._initialized,
            'components': {
                'analyzer_factory': 'healthy',
                'cache_repository': 'healthy',
                'metrics_collector': 'healthy'
            },
            'timestamp': time.time()
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del sistema."""
        if not self._initialized:
            await self.initialize()
        
        return await self._metrics_collector.get_metrics_summary()
    
    def get_version(self) -> str:
        """Obtener versión del motor."""
        return __version__
    
    def get_supported_analysis_types(self) -> List[AnalysisType]:
        """Obtener tipos de análisis soportados."""
        return list(AnalysisType)
    
    def get_supported_tiers(self) -> List[ProcessingTier]:
        """Obtener tiers soportados."""
        return list(ProcessingTier)


# Public API
__all__ = [
    # Main Engine
    'NLPEngine',
    
    # Core entities
    'AnalysisResult',
    'TextFingerprint', 
    'AnalysisScore',
    'ProcessingMetrics',
    'AnalysisError',
    
    # Enums
    'AnalysisType',
    'ProcessingTier',
    'CacheStrategy',
    'Environment',
    'LogLevel',
    'AnalysisStatus',
    'ErrorType',
    'MetricType',
    
    # Domain services
    'AnalysisOrchestrator',
    'TextProcessor',
    'ScoreValidator',
    
    # Application DTOs
    'AnalysisRequest',
    'AnalysisResponse',
    'BatchAnalysisRequest',
    'create_analysis_request',
    'serialize_response',
    
    # Use cases
    'AnalyzeTextUseCase',
    'BatchAnalysisUseCase',
    'StreamAnalysisUseCase',
    
    # Services
    'AnalysisService',
    'CacheService',
    'MetricsService',
    'ConfigurationService',
    
    # All interfaces for implementation
    'IAnalyzer',
    'IAnalyzerFactory',
    'IAdvancedAnalyzer',
    'IConfigurableAnalyzer',
    'ICacheRepository',
    'IDistributedCache',
    'ICacheKeyGenerator',
    'ICacheEvictionPolicy',
    'ICacheSerializer',
    'IMetricsCollector',
    'IPerformanceMonitor',
    'IHealthChecker',
    'IAlertManager',
    'IStructuredLogger',
    'IMetricsExporter',
    'IConfigurationService',
    'IEnvironmentConfigLoader',
    'IFileConfigLoader',
    'ISecretManager',
    'IConfigValidator',
    'IConfigMerger',
    'IConfigTransformer',
    
    # Version
    '__version__'
] 