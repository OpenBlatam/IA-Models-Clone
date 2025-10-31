from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, AsyncGenerator
import logging
from nlp_engine.core.entities import AnalysisResult, TextFingerprint, AnalysisScore
from nlp_engine.core.enums import AnalysisType, ProcessingTier, CacheStrategy, Environment
from nlp_engine.core.domain_services import AnalysisOrchestrator, TextProcessor, ScoreValidator
from nlp_engine.application.dto import AnalysisRequest, BatchAnalysisRequest
from nlp_engine.application.use_cases import AnalyzeTextUseCase, BatchAnalysisUseCase
from nlp_engine.application.services import AnalysisService, CacheService, MetricsService
from demo_infrastructure import (
        from nlp_engine.application.dto import HealthCheckRequest
from typing import Any, List, Dict, Optional
"""
🎯 DEMO MODULAR COMPLETE - Demostración de Arquitectura Modular NLP
================================================================

Demostración completa del sistema modular NLP con todas las capas:
- Core (Domain Logic, Entities, Domain Services)
- Interfaces (Ports & Contracts)
- Application (Use Cases, Services, DTOs)
- Infrastructure (Mock implementations)

Este demo muestra un sistema enterprise-grade completamente funcional.
"""


# Core Layer

# Application Layer

# Mock Infrastructure (for demo)
    MockAnalyzerFactory, MockCacheRepository, MockMetricsCollector,
    MockConfigurationService, MockStructuredLogger, MockPerformanceMonitor,
    MockHealthChecker, MockCacheKeyGenerator
)


class ModularNLPEngineDemo:
    """Demo completo del motor NLP modular."""
    
    def __init__(self) -> Any:
        self.logger = self._setup_logging()
        
        # Infrastructure Layer (Mock implementations for demo)
        self._setup_infrastructure()
        
        # Application Services
        self._setup_application_services()
        
        # Use Cases
        self._setup_use_cases()
        
        # Demo data
        self._setup_demo_data()
    
    def _setup_logging(self) -> logging.Logger:
        """Configurar logging para el demo."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(self.__class__.__name__)
    
    def _setup_infrastructure(self) -> None:
        """Configurar capa de infraestructura (mocks para demo)."""
        self.analyzer_factory = MockAnalyzerFactory()
        self.cache_repository = MockCacheRepository()
        self.metrics_collector = MockMetricsCollector()
        self.config_service = MockConfigurationService()
        self.structured_logger = MockStructuredLogger()
        self.performance_monitor = MockPerformanceMonitor()
        self.health_checker = MockHealthChecker()
        self.cache_key_generator = MockCacheKeyGenerator()
    
    def _setup_application_services(self) -> None:
        """Configurar servicios de aplicación."""
        # Analysis Service
        self.analysis_service = AnalysisService(
            config_service=self.config_service,
            metrics_collector=self.metrics_collector,
            cache_repository=self.cache_repository,
            performance_monitor=self.performance_monitor,
            health_checker=self.health_checker
        )
        
        # Cache Service
        self.cache_service = CacheService(
            cache_repository=self.cache_repository,
            cache_key_generator=self.cache_key_generator,
            metrics_collector=self.metrics_collector
        )
        
        # Metrics Service
        self.metrics_service = MetricsService(
            metrics_collector=self.metrics_collector,
            performance_monitor=self.performance_monitor,
            health_checker=self.health_checker
        )
    
    def _setup_use_cases(self) -> None:
        """Configurar casos de uso."""
        # Main Use Case
        self.analyze_text_use_case = AnalyzeTextUseCase(
            analyzer_factory=self.analyzer_factory,
            cache_repository=self.cache_repository,
            metrics_collector=self.metrics_collector,
            config_service=self.config_service,
            logger=self.structured_logger
        )
        
        # Batch Use Case
        self.batch_analysis_use_case = BatchAnalysisUseCase(
            analyze_text_use_case=self.analyze_text_use_case
        )
    
    def _setup_demo_data(self) -> None:
        """Configurar datos de prueba."""
        self.demo_texts = [
            "Este es un texto excelente con muy buena calidad y sentimiento positivo.",
            "El servicio fue terrible, muy mal, no lo recomiendo para nada.",
            "Un texto neutral sin emociones particulares, simplemente informativo.",
            "¡Increíble experiencia! Lo mejor que he probado en mucho tiempo.",
            "No está mal, tampoco es excepcional, algo promedio en general.",
            "Texto muy corto.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur."
        ]
    
    async def run_complete_demo(self) -> Any:
        """Ejecutar demostración completa del sistema modular."""
        print("🎯 DEMO MODULAR NLP ENGINE - Arquitectura Enterprise")
        print("=" * 60)
        
        # Inicializar sistema
        await self._demo_system_initialization()
        
        # Demo 1: Análisis individual
        await self._demo_individual_analysis()
        
        # Demo 2: Análisis en lote
        await self._demo_batch_analysis()
        
        # Demo 3: Performance y optimización
        await self._demo_performance_optimization()
        
        # Demo 4: Cache management
        await self._demo_cache_management()
        
        # Demo 5: Métricas y monitoreo
        await self._demo_metrics_monitoring()
        
        # Demo 6: Health checks
        await self._demo_health_checks()
        
        # Demo 7: Domain logic
        await self._demo_domain_logic()
        
        # Demo 8: Different tiers
        await self._demo_processing_tiers()
        
        # Resumen final
        await self._demo_final_summary()
    
    async def _demo_system_initialization(self) -> Any:
        """Demo de inicialización del sistema."""
        print("\n🚀 1. INICIALIZACIÓN DEL SISTEMA")
        print("-" * 40)
        
        try:
            await self.analysis_service.initialize()
            print("✅ Sistema inicializado correctamente")
            
            # Validar configuración
            config_errors = self.config_service.validate_config()
            if not config_errors:
                print("✅ Configuración validada")
            else:
                print(f"⚠️  Errores de configuración: {config_errors}")
            
            # Mostrar configuración actual
            current_tier = self.analysis_service.get_default_tier()
            cache_strategy = self.analysis_service.get_cache_strategy()
            
            print(f"📊 Tier por defecto: {current_tier.value}")
            print(f"🗄️  Estrategia de cache: {cache_strategy.value}")
            
        except Exception as e:
            print(f"❌ Error en inicialización: {e}")
    
    async def _demo_individual_analysis(self) -> Any:
        """Demo de análisis individual."""
        print("\n📝 2. ANÁLISIS INDIVIDUAL DE TEXTO")
        print("-" * 40)
        
        text = self.demo_texts[0]
        print(f"Texto: '{text[:50]}...'")
        
        # Crear request
        request = AnalysisRequest(
            text=text,
            analysis_types=[AnalysisType.SENTIMENT, AnalysisType.QUALITY_ASSESSMENT],
            processing_tier=ProcessingTier.BALANCED,
            client_id="demo_client",
            use_cache=True
        )
        
        try:
            # Ejecutar análisis
            start_time = time.time()
            response = await self.analyze_text_use_case.execute(request)
            duration = (time.time() - start_time) * 1000
            
            print(f"⏱️  Duración: {duration:.2f}ms")
            print(f"✅ Éxito: {response.success}")
            print(f"🆔 Request ID: {response.request_id}")
            
            if response.success:
                results = response.analysis_results
                print("📊 Resultados:")
                
                for analysis_type, score_data in results.get('scores', {}).items():
                    print(f"  - {analysis_type}: {score_data.get('value', 0):.2f} "
                          f"(confianza: {score_data.get('confidence', 0):.2f})")
                
                metadata = response.metadata
                print(f"🎯 Performance Grade: {metadata.get('performance_grade', 'N/A')}")
                print(f"🗄️  Cache Hit: {metadata.get('cache_hit', False)}")
            else:
                print(f"❌ Errores: {response.errors}")
                
        except Exception as e:
            print(f"❌ Error en análisis: {e}")
    
    async def _demo_batch_analysis(self) -> Any:
        """Demo de análisis en lote."""
        print("\n📋 3. ANÁLISIS EN LOTE")
        print("-" * 40)
        
        batch_texts = self.demo_texts[:5]
        print(f"Analizando {len(batch_texts)} textos en paralelo...")
        
        # Crear request de lote
        batch_request = BatchAnalysisRequest(
            texts=batch_texts,
            analysis_types=[AnalysisType.SENTIMENT],
            processing_tier=ProcessingTier.ULTRA_FAST,
            max_concurrency=3,
            client_id="batch_demo"
        )
        
        try:
            start_time = time.time()
            responses = await self.batch_analysis_use_case.execute(batch_request)
            duration = (time.time() - start_time) * 1000
            
            print(f"⏱️  Duración total: {duration:.2f}ms")
            print(f"📊 Resultados procesados: {len(responses)}")
            
            successful = sum(1 for r in responses if r.success)
            failed = len(responses) - successful
            
            print(f"✅ Exitosos: {successful}")
            print(f"❌ Fallidos: {failed}")
            
            # Mostrar algunos resultados
            print("\n📈 Muestra de resultados:")
            for i, response in enumerate(responses[:3]):
                if response.success:
                    sentiment_score = response.analysis_results.get('scores', {}).get('SENTIMENT', {})
                    print(f"  Texto {i+1}: Sentimiento = {sentiment_score.get('value', 0):.2f}")
                    
        except Exception as e:
            print(f"❌ Error en análisis en lote: {e}")
    
    async def _demo_performance_optimization(self) -> Any:
        """Demo de optimización de performance."""
        print("\n⚡ 4. OPTIMIZACIÓN DE PERFORMANCE")
        print("-" * 40)
        
        print("Ejecutando múltiples análisis para generar métricas...")
        
        # Ejecutar varios análisis rápidos
        tasks = []
        for i in range(10):
            text = self.demo_texts[i % len(self.demo_texts)]
            request = AnalysisRequest(
                text=text,
                analysis_types=[AnalysisType.SENTIMENT],
                processing_tier=ProcessingTier.ULTRA_FAST
            )
            tasks.append(self.analyze_text_use_case.execute(request))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = (time.time() - start_time) * 1000
        
        successful_results = [r for r in results if not isinstance(r, Exception) and r.success]
        
        print(f"📊 Análisis completados: {len(successful_results)}/10")
        print(f"⏱️  Tiempo total: {total_duration:.2f}ms")
        print(f"⚡ Promedio por análisis: {total_duration/10:.2f}ms")
        print(f"🚀 Throughput: {10000/total_duration:.1f} análisis/segundo")
        
        # Obtener métricas de performance
        performance_summary = self.metrics_service.get_performance_summary()
        print(f"💾 Memoria: {performance_summary.get('memory_usage', {}).get('used_mb', 0):.1f}MB")
        print(f"🖥️  CPU: {performance_summary.get('cpu_usage', 0):.1f}%")
    
    async def _demo_cache_management(self) -> Any:
        """Demo de gestión de cache."""
        print("\n🗄️  5. GESTIÓN DE CACHE")
        print("-" * 40)
        
        # Obtener estadísticas de cache
        cache_stats = self.cache_service.get_cache_stats()
        print(f"📈 Estadísticas de Cache:")
        print(f"  - Hits: {cache_stats['hits']}")
        print(f"  - Misses: {cache_stats['misses']}")
        print(f"  - Hit Rate: {cache_stats['hit_rate']:.2%}")
        print(f"  - Total Requests: {cache_stats['total_requests']}")
        
        # Optimizar cache
        print("\n🔧 Optimizando cache...")
        optimization_result = await self.cache_service.optimize_cache()
        
        if optimization_result['success']:
            print("✅ Optimización completada")
            optimizations = optimization_result.get('optimizations', [])
            if optimizations:
                print(f"   Optimizaciones realizadas: {', '.join(optimizations)}")
            else:
                print("   No se requirieron optimizaciones")
        else:
            print(f"❌ Error en optimización: {optimization_result.get('error')}")
    
    async def _demo_metrics_monitoring(self) -> Any:
        """Demo de métricas y monitoreo."""
        print("\n📊 6. MÉTRICAS Y MONITOREO")
        print("-" * 40)
        
        # Generar reporte de performance
        performance_report = await self.metrics_service.generate_performance_report()
        
        if 'error' not in performance_report:
            summary = performance_report.get('summary', {})
            print(f"🎯 Estado General: {summary.get('overall_status', 'unknown')}")
            print(f"📝 Total de Requests: {summary.get('total_requests', 0)}")
            print(f"⏱️  Latencia Promedio: {summary.get('avg_latency', 0):.2f}ms")
            print(f"🗄️  Cache Hit Rate: {summary.get('cache_hit_rate', 0):.2%}")
            
            # Métricas detalladas
            metrics = performance_report.get('metrics', {})
            print("\n📈 Métricas Detalladas:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  - {metric_name}: {value}")
        else:
            print(f"❌ Error obteniendo métricas: {performance_report['error']}")
    
    async def _demo_health_checks(self) -> Any:
        """Demo de health checks."""
        print("\n🏥 7. HEALTH CHECKS")
        print("-" * 40)
        
        
        # Health check completo del sistema
        health_request = HealthCheckRequest(
            deep_check=True,
            include_metrics=True
        )
        
        health_response = await self.metrics_service.handle_health_check_request(health_request)
        
        print(f"🎯 Estado del Sistema: {health_response.status}")
        print(f"⏰ Uptime: {health_response.uptime_seconds:.1f} segundos")
        
        # Componentes
        if health_response.components:
            print("\n🔧 Estado de Componentes:")
            for component, status in health_response.components.items():
                comp_status = status.get('status', 'unknown')
                print(f"  - {component}: {comp_status}")
        
        # Errores si los hay
        if health_response.errors:
            print(f"\n⚠️  Errores: {health_response.errors}")
    
    async def _demo_domain_logic(self) -> Any:
        """Demo de lógica de dominio."""
        print("\n🏗️  8. LÓGICA DE DOMINIO")
        print("-" * 40)
        
        # Text Processor
        text_processor = TextProcessor()
        raw_text = "  ¡¡¡Texto con MUCHOS espacios   y símbolos!!!  "
        sanitized = text_processor.sanitize_text(raw_text)
        print(f"🧹 Texto sanitizado: '{sanitized}'")
        
        # Text Fingerprint
        fingerprint = TextFingerprint.create(sanitized)
        print(f"🔍 Fingerprint: {fingerprint.short_hash}")
        print(f"📏 Longitud: {fingerprint.length}")
        
        # Score Validator
        score_validator = ScoreValidator()
        test_score = 85.5
        is_valid_sentiment = score_validator.validate_sentiment_score(test_score, "transformer")
        print(f"✅ Score {test_score} válido para sentimiento: {is_valid_sentiment}")
        
        # Analysis Orchestrator
        orchestrator = AnalysisOrchestrator()
        optimal_tier = orchestrator.determine_optimal_tier(
            text_length=len(sanitized),
            analysis_types=[AnalysisType.SENTIMENT, AnalysisType.QUALITY_ASSESSMENT]
        )
        print(f"🎯 Tier óptimo determinado: {optimal_tier.value}")
    
    async def _demo_processing_tiers(self) -> Any:
        """Demo de diferentes tiers de procesamiento."""
        print("\n⚙️  9. TIERS DE PROCESAMIENTO")
        print("-" * 40)
        
        text = self.demo_texts[0]
        tiers = [ProcessingTier.ULTRA_FAST, ProcessingTier.BALANCED, ProcessingTier.HIGH_QUALITY]
        
        for tier in tiers:
            print(f"\n🔧 Probando tier: {tier.value}")
            
            request = AnalysisRequest(
                text=text,
                analysis_types=[AnalysisType.SENTIMENT],
                processing_tier=tier,
                use_cache=False  # Sin cache para medir tiempo real
            )
            
            start_time = time.time()
            response = await self.analyze_text_use_case.execute(request)
            duration = (time.time() - start_time) * 1000
            
            if response.success:
                sentiment = response.analysis_results.get('scores', {}).get('SENTIMENT', {})
                score = sentiment.get('value', 0)
                confidence = sentiment.get('confidence', 0)
                
                print(f"  ⏱️  Duración: {duration:.2f}ms")
                print(f"  📊 Score: {score:.2f} (confianza: {confidence:.2f})")
                print(f"  🎯 Grade: {response.metadata.get('performance_grade', 'N/A')}")
            else:
                print(f"  ❌ Error: {response.errors}")
    
    async def _demo_final_summary(self) -> Any:
        """Resumen final del demo."""
        print("\n🎉 10. RESUMEN FINAL")
        print("-" * 40)
        
        # Estadísticas finales
        final_performance = await self.metrics_service.generate_performance_report()
        
        if 'error' not in final_performance:
            summary = final_performance.get('summary', {})
            metrics = final_performance.get('metrics', {})
            
            print("📊 ESTADÍSTICAS FINALES:")
            print(f"  🎯 Estado: {summary.get('overall_status', 'unknown')}")
            print(f"  📝 Total Análisis: {summary.get('total_requests', 0)}")
            print(f"  ⏱️  Latencia Promedio: {summary.get('avg_latency', 0):.2f}ms")
            print(f"  🗄️  Cache Hit Rate: {summary.get('cache_hit_rate', 0):.2%}")
            
            print("\n🏗️  ARQUITECTURA MODULAR:")
            print("  ✅ Core Layer: Entities, Value Objects, Domain Services")
            print("  ✅ Interfaces Layer: Ports & Contracts")
            print("  ✅ Application Layer: Use Cases, Services, DTOs")
            print("  ✅ Infrastructure Layer: Mock implementations")
            
            print("\n🚀 CARACTERÍSTICAS ENTERPRISE:")
            print("  ✅ Clean Architecture")
            print("  ✅ SOLID Principles")
            print("  ✅ Dependency Injection")
            print("  ✅ Async/Await Pattern")
            print("  ✅ Comprehensive Error Handling")
            print("  ✅ Performance Monitoring")
            print("  ✅ Health Checks")
            print("  ✅ Cache Management")
            print("  ✅ Structured Logging")
            print("  ✅ Multiple Processing Tiers")
            
        # Cleanup
        try:
            await self.analysis_service.shutdown()
            print("\n✅ Sistema cerrado correctamente")
        except Exception as e:
            print(f"\n⚠️  Error en cierre: {e}")


async def main():
    """Función principal del demo."""
    demo = ModularNLPEngineDemo()
    await demo.run_complete_demo()


match __name__:
    case "__main__":
    asyncio.run(main()) 