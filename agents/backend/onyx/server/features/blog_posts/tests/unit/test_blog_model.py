from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import pytest
import time
import json
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock
from nlp_engine.optimized.modular_engine import (
from nlp_engine.optimized.ultra_turbo_engine import UltraTurboNLPEngine
from nlp_engine.core.entities import (
from nlp_engine.core.enums import (
        import psutil
        import os
from typing import Any, List, Dict, Optional
import logging
"""
🧪 COMPREHENSIVE BLOG MODEL TESTS
=================================

Test suite completo para el modelo blog y sistema NLP ultra-optimizado.
Incluye tests de performance, funcionalidad, entidades del dominio y análisis de contenido.
"""


# Import sistema NLP optimizado
    ModularNLPEngine, create_modular_engine, quick_sentiment_analysis,
    quick_quality_analysis, ultra_fast_mixed_analysis
)
    AnalysisResult, AnalysisScore, ProcessingMetrics, TextFingerprint
)
    AnalysisType, ProcessingTier, AnalysisStatus, OptimizationTier
)


class TestBlogContentModels:
    """Tests para modelos de contenido de blog."""
    
    def test_text_fingerprint_creation(self) -> Any:
        """Test creación de fingerprint para contenido de blog."""
        blog_content = """
        La inteligencia artificial está revolucionando el marketing digital.
        Las empresas que adoptan estas tecnologías están viendo mejores resultados.
        """
        
        fingerprint = TextFingerprint.create(blog_content, language_hint="es")
        
        assert fingerprint.length == len(blog_content)
        assert fingerprint.language_hint == "es"
        assert len(fingerprint.hash_value) == 32  # BLAKE2b 16 bytes = 32 chars hex
        assert fingerprint.short_hash == fingerprint.hash_value[:8]
    
    def test_analysis_score_validation(self) -> Any:
        """Test validación de scores de análisis."""
        # Valid score
        score = AnalysisScore(value=85.5, confidence=0.95, method="ultra_fast")
        assert score.weighted_value == 85.5 * 0.95
        assert score.is_high_confidence()
        
        # Invalid score values
        with pytest.raises(ValueError):
            AnalysisScore(value=105.0)  # > 100
        
        with pytest.raises(ValueError):
            AnalysisScore(value=50.0, confidence=1.5)  # > 1.0
    
    def test_analysis_result_lifecycle(self) -> Any:
        """Test ciclo de vida completo de análisis de blog."""
        blog_text = "Este es un excelente artículo sobre IA en marketing."
        fingerprint = TextFingerprint.create(blog_text)
        
        result = AnalysisResult(fingerprint=fingerprint)
        
        # Estado inicial
        assert result.status == AnalysisStatus.PENDING
        assert len(result.scores) == 0
        assert result.is_valid() == False
        
        # Añadir scores
        result.add_score(AnalysisType.SENTIMENT, 85.0, confidence=0.95, method="ultra_fast")
        assert result.status == AnalysisStatus.PROCESSING
        
        result.add_score(AnalysisType.QUALITY_ASSESSMENT, 78.5, confidence=0.90, method="modular")
        
        # Completar análisis
        metrics = ProcessingMetrics(
            start_time_ns=1000000,
            end_time_ns=1500000,  # 0.5ms
            cache_hit=True,
            cache_level="L1",
            model_used="ultra_turbo",
            tier=ProcessingTier.ULTRA_FAST
        )
        
        result.complete(metrics)
        assert result.status == AnalysisStatus.COMPLETED
        assert result.is_valid() == True
        assert result.has_high_confidence_scores(threshold=0.8) == True
        assert result.get_performance_grade() == "A+"


class TestModularNLPEngine:
    """Tests para el motor NLP modular ultra-optimizado."""
    
    @pytest.fixture
    async def engine(self) -> Any:
        """Crear motor para tests."""
        engine = create_modular_engine(OptimizationTier.ULTRA)
        await engine.initialize()
        return engine
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine) -> Any:
        """Test inicialización del motor."""
        assert engine.initialized == True
        assert engine.optimization_tier == OptimizationTier.ULTRA
        assert engine.thread_pool._max_workers == 4
    
    @pytest.mark.asyncio
    async def test_single_blog_analysis(self, engine) -> Any:
        """Test análisis individual de contenido de blog."""
        blog_content = """
        La automatización con inteligencia artificial está transformando 
        el panorama del marketing digital. Las empresas más innovadoras 
        están implementando soluciones que permiten personalizar experiencias 
        a escala masiva.
        """
        
        # Test análisis de sentimiento
        sentiment_result = await engine.analyze_single(blog_content, "sentiment")
        
        assert "score" in sentiment_result
        assert 0.0 <= sentiment_result["score"] <= 1.0
        assert sentiment_result["confidence"] == 0.95
        assert sentiment_result["processing_time_ms"] < 1.0  # Ultra-fast
        assert sentiment_result["analysis_type"] == "sentiment"
        
        # Test análisis de calidad
        quality_result = await engine.analyze_single(blog_content, "quality")
        
        assert "score" in quality_result
        assert quality_result["analysis_type"] == "quality"
        assert quality_result["metadata"]["ultra_optimized"] == True
    
    @pytest.mark.asyncio
    async def test_batch_blog_analysis(self, engine) -> Any:
        """Test análisis en lote de múltiples blogs."""
        blog_posts = [
            "Excelente artículo sobre machine learning aplicado al marketing.",
            "Tutorial paso a paso para implementar chatbots con IA.",
            "Análisis profundo de las tendencias en automatización empresarial.",
            "Guía completa para optimizar campañas publicitarias con datos.",
            "Casos de éxito en transformación digital con inteligencia artificial."
        ]
        
        # Test análisis de sentimiento en lote
        sentiment_result = await engine.analyze_sentiment(blog_posts)
        
        assert "scores" in sentiment_result
        assert len(sentiment_result["scores"]) == len(blog_posts)
        assert sentiment_result["total_texts"] == 5
        assert sentiment_result["processing_time_ms"] < 10.0  # Ultra-fast batch
        assert sentiment_result["throughput_ops_per_second"] > 500  # High throughput
        assert sentiment_result["optimization_tier"] == OptimizationTier.ULTRA.value
        
        # Verificar que todos los scores son válidos
        for score in sentiment_result["scores"]:
            assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_mixed_analysis_performance(self, engine) -> Any:
        """Test análisis mixto (sentimiento + calidad) con métricas de performance."""
        blog_content = [
            "La inteligencia artificial está revolucionando el marketing digital moderno.",
            "Tutorial: Cómo implementar chatbots inteligentes en tu empresa.",
            "Análisis de ROI en proyectos de automatización con machine learning."
        ]
        
        start_time = time.perf_counter()
        
        result = await engine.analyze_batch_mixed(
            blog_content, 
            include_sentiment=True, 
            include_quality=True
        )
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Verificar estructura del resultado
        assert "results" in result
        assert "sentiment" in result["results"]
        assert "quality" in result["results"]
        assert result["total_texts"] == 3
        assert result["analyses_performed"] == 2
        
        # Verificar performance ultra-rápida
        assert result["total_processing_time_ms"] < 5.0
        assert result["combined_throughput_ops_per_second"] > 1000
        assert total_time < 10.0  # Test completo en menos de 10ms
        
        # Verificar optimizaciones aplicadas
        assert result["optimization_summary"]["ultra_optimized"] == True
        assert result["optimization_summary"]["parallel_analyses"] == True
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, engine) -> Any:
        """Test performance del sistema de cache."""
        blog_text = "Análisis de performance para contenido de blog con cache optimizado."
        
        # Primera ejecución (cache miss)
        result1 = await engine.analyze_single(blog_text, "sentiment")
        initial_cache_hits = engine.cache_hits
        
        # Segunda ejecución (cache hit)
        result2 = await engine.analyze_single(blog_text, "sentiment")
        final_cache_hits = engine.cache_hits
        
        # Verificar que el cache funcionó
        assert final_cache_hits > initial_cache_hits
        assert result1["score"] == result2["score"]  # Mismos resultados
        assert result2["processing_time_ms"] <= result1["processing_time_ms"]  # Más rápido
    
    @pytest.mark.asyncio
    async def test_engine_stats_tracking(self, engine) -> Any:
        """Test tracking de estadísticas del motor."""
        blog_posts = ["Post 1", "Post 2", "Post 3"]
        
        initial_stats = engine.get_stats()
        initial_requests = initial_stats["total_requests"]
        
        await engine.analyze_sentiment(blog_posts)
        
        final_stats = engine.get_stats()
        
        # Verificar incremento en estadísticas
        assert final_stats["total_requests"] > initial_requests
        assert final_stats["successful_requests"] > initial_stats["successful_requests"]
        assert final_stats["error_rate"] == 0.0  # Sin errores
        assert final_stats["cache_hit_ratio"] >= 0.0
        assert final_stats["ultra_optimizations"]["lru_cache_enabled"] == True
    
    @pytest.mark.asyncio
    async def test_health_check(self, engine) -> Any:
        """Test health check del sistema."""
        health = await engine.health_check()
        
        assert health["status"] == "healthy"
        assert health["response_time_ms"] < 5.0
        assert health["optimization_tier"] == OptimizationTier.ULTRA.value
        assert health["ultra_optimized"] == True
        assert "stats" in health


class TestUltraTurboEngine:
    """Tests para el motor ultra-turbo (máximo rendimiento)."""
    
    @pytest.mark.asyncio
    async def test_ultra_turbo_initialization(self) -> Any:
        """Test inicialización del motor ultra-turbo."""
        engine = UltraTurboNLPEngine()
        success = await engine.initialize()
        
        assert success == True
        assert engine.initialized == True
        assert hasattr(engine, 'sentiment_model')
        assert hasattr(engine, 'quality_analyzer')
    
    @pytest.mark.asyncio
    async def test_ultra_turbo_performance(self) -> Any:
        """Test performance extrema del motor ultra-turbo."""
        engine = UltraTurboNLPEngine()
        await engine.initialize()
        
        # Test con lote grande para medir throughput máximo
        large_batch = [
            f"Blog post número {i} sobre inteligencia artificial y marketing digital."
            for i in range(100)
        ]
        
        start_time = time.perf_counter()
        
        result = await engine.turbo_sentiment_analysis(large_batch, use_ultra_turbo=True)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Verificar performance extrema
        assert len(result["scores"]) == 100
        assert processing_time < 20.0  # < 20ms para 100 textos
        assert result["throughput_ops_per_second"] > 5000  # > 5K ops/s
        assert result["average_latency_ms"] < 0.2  # < 0.2ms por texto


class TestConvenienceFunctions:
    """Tests para funciones de conveniencia."""
    
    @pytest.mark.asyncio
    async def test_quick_sentiment_analysis(self) -> Any:
        """Test función rápida de análisis de sentimiento."""
        blog_texts = [
            "Excelente artículo sobre IA generativa.",
            "Tutorial mediocre sin ejemplos prácticos.",
            "Análisis extraordinario de tendencias tecnológicas."
        ]
        
        scores = await quick_sentiment_analysis(blog_texts, OptimizationTier.ULTRA)
        
        assert len(scores) == 3
        assert all(0.0 <= score <= 1.0 for score in scores)
        # El primer y tercer texto deberían ser más positivos
        assert scores[0] > 0.6  # "Excelente"
        assert scores[2] > 0.6  # "Extraordinario"
    
    @pytest.mark.asyncio
    async def test_quick_quality_analysis(self) -> Any:
        """Test función rápida de análisis de calidad."""
        blog_texts = [
            """Este es un artículo muy corto.""",  # Baja calidad por longitud
            """Este es un artículo de longitud media que explica conceptos de inteligencia 
            artificial de manera clara y estructurada. Incluye ejemplos prácticos y 
            está bien organizado en párrafos coherentes.""",  # Alta calidad
            """Un texto extremadamente largo que repite muchas ideas sin aportar valor real 
            al lector, con frases muy largas que no tienen estructura clara y que podrían 
            ser simplificadas para mejorar la comprensión del contenido presentado."""  # Calidad media
        ]
        
        scores = await quick_quality_analysis(blog_texts, OptimizationTier.ULTRA)
        
        assert len(scores) == 3
        assert all(0.0 <= score <= 1.0 for score in scores)
        # El segundo texto debería tener mejor calidad
        assert scores[1] > scores[0]  # Artículo medio > artículo corto
    
    @pytest.mark.asyncio
    async def test_ultra_fast_mixed_analysis(self) -> Any:
        """Test análisis mixto ultra-rápido."""
        blog_posts = [
            "Increíble tutorial sobre machine learning aplicado al marketing digital.",
            "Guía práctica para implementar automatización en procesos empresariales."
        ]
        
        start_time = time.perf_counter()
        
        result = await ultra_fast_mixed_analysis(blog_posts, OptimizationTier.EXTREME)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Verificar performance extrema
        assert processing_time < 3.0  # < 3ms total
        assert "results" in result
        assert "sentiment" in result["results"]
        assert "quality" in result["results"]
        assert result["combined_throughput_ops_per_second"] > 1000


class TestBlogContentScenarios:
    """Tests para escenarios específicos de contenido de blog."""
    
    @pytest.mark.asyncio
    async def test_technical_blog_analysis(self) -> Any:
        """Test análisis de blog técnico."""
        technical_blog = """
        # Implementación de Algoritmos de Machine Learning en Marketing

        La implementación de algoritmos de machine learning en marketing digital 
        requiere una comprensión profunda de los datos del cliente y los objetivos 
        comerciales. En este artículo, exploraremos las mejores prácticas para 
        desarrollar modelos predictivos eficaces.

        ## Preparación de Datos

        El primer paso crítico es la preparación de datos. Los datos deben ser:
        - Limpios y consistentes
        - Representativos del problema empresarial
        - Suficientes para el entrenamiento del modelo

        ## Selección de Algoritmos

        Para marketing digital, recomendamos:
        1. Regresión logística para clasificación binaria
        2. Random Forest para problemas complejos
        3. Redes neuronales para datasets grandes

        La elección del algoritmo depende del caso de uso específico y la cantidad 
        de datos disponibles.
        """
        
        engine = create_modular_engine(OptimizationTier.ULTRA)
        await engine.initialize()
        
        result = await engine.analyze_batch_mixed(
            [technical_blog],
            include_sentiment=True,
            include_quality=True
        )
        
        # Blog técnico debería tener alta calidad y sentimiento neutral-positivo
        quality_score = result["results"]["quality"]["scores"][0]
        sentiment_score = result["results"]["sentiment"]["scores"][0]
        
        assert quality_score > 0.7  # Alta calidad por estructura y contenido
        assert 0.4 <= sentiment_score <= 0.8  # Neutral a positivo (contenido técnico)
    
    @pytest.mark.asyncio
    async def test_promotional_blog_analysis(self) -> Any:
        """Test análisis de blog promocional."""
        promotional_blog = """
        ¡Descubre la Revolución del Marketing con IA!

        ¿Estás listo para transformar tu negocio? Nuestra plataforma de marketing 
        con inteligencia artificial es simplemente INCREÍBLE. Miles de empresas 
        ya están viendo resultados EXTRAORDINARIOS.

        ¡NO PIERDAS ESTA OPORTUNIDAD ÚNICA! Regístrate ahora y obtén:
        ✅ Análisis predictivo avanzado
        ✅ Automatización completa de campañas  
        ✅ ROI garantizado del 300%

        ¡Es PERFECTO para tu empresa! ¡ACTÚA YA!
        """
        
        engine = create_modular_engine(OptimizationTier.ULTRA)
        await engine.initialize()
        
        result = await engine.analyze_batch_mixed([promotional_blog])
        
        quality_score = result["results"]["quality"]["scores"][0]
        sentiment_score = result["results"]["sentiment"]["scores"][0]
        
        # Blog promocional: sentimiento muy positivo, calidad variable
        assert sentiment_score > 0.8  # Muy positivo por palabras como "INCREÍBLE", "EXTRAORDINARIOS"
        # La calidad puede ser menor por el estilo promocional
    
    @pytest.mark.asyncio
    async def test_educational_blog_analysis(self) -> Any:
        """Test análisis de blog educativo."""
        educational_blog = """
        Conceptos Fundamentales de Inteligencia Artificial para Principiantes

        La inteligencia artificial puede parecer compleja, pero entender sus 
        conceptos básicos es más sencillo de lo que imaginas. En esta guía, 
        aprenderás los fundamentos de manera clara y estructurada.

        ¿Qué es la Inteligencia Artificial?

        La IA es una rama de la informática que busca crear sistemas capaces 
        de realizar tareas que normalmente requieren inteligencia humana. 
        Estos sistemas pueden aprender, razonar y tomar decisiones.

        Tipos de Inteligencia Artificial:

        1. IA Débil: Sistemas especializados en tareas específicas
        2. IA Fuerte: Sistemas con capacidades cognitivas generales
        3. Superinteligencia: Hipotética IA superior a la humana

        Aplicaciones Prácticas:

        - Asistentes virtuales (Siri, Alexa)
        - Recomendaciones en plataformas (Netflix, Amazon)
        - Vehículos autónomos
        - Diagnóstico médico

        La IA está transformando múltiples industrias y su comprensión es 
        fundamental para el futuro profesional.
        """
        
        engine = create_modular_engine(OptimizationTier.ULTRA)
        await engine.initialize()
        
        result = await engine.analyze_batch_mixed([educational_blog])
        
        quality_score = result["results"]["quality"]["scores"][0]
        sentiment_score = result["results"]["sentiment"]["scores"][0]
        
        # Blog educativo: alta calidad por estructura y claridad, sentimiento neutro-positivo
        assert quality_score > 0.8  # Alta calidad por estructura clara y contenido organizado
        assert 0.5 <= sentiment_score <= 0.7  # Neutral a ligeramente positivo


class TestPerformanceBenchmarks:
    """Tests de performance y benchmarks del sistema."""
    
    @pytest.mark.asyncio
    async def test_scalability_benchmark(self) -> Any:
        """Test de escalabilidad con diferentes tamaños de lote."""
        engine = create_modular_engine(OptimizationTier.EXTREME)
        await engine.initialize()
        
        # Test con diferentes tamaños
        batch_sizes = [1, 10, 50, 100, 500]
        results = {}
        
        for size in batch_sizes:
            blog_batch = [
                f"Blog post {i} sobre tecnología y marketing digital avanzado."
                for i in range(size)
            ]
            
            start_time = time.perf_counter()
            result = await engine.analyze_sentiment(blog_batch)
            processing_time = (time.perf_counter() - start_time) * 1000
            
            throughput = size / (processing_time / 1000) if processing_time > 0 else 0
            
            results[size] = {
                "processing_time_ms": processing_time,
                "throughput_ops_per_second": throughput,
                "avg_latency_ms": processing_time / size
            }
        
        # Verificar escalabilidad sub-lineal
        assert results[1]["avg_latency_ms"] > results[100]["avg_latency_ms"]  # Mejora con lotes grandes
        assert results[500]["throughput_ops_per_second"] > results[1]["throughput_ops_per_second"]
        
        # Verificar targets de performance
        assert results[100]["throughput_ops_per_second"] > 10000  # > 10K ops/s para lotes de 100
        assert results[500]["avg_latency_ms"] < 0.1  # < 0.1ms promedio para lotes grandes
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self) -> Any:
        """Test eficiencia de memoria con lotes grandes."""
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        engine = create_modular_engine(OptimizationTier.EXTREME)
        await engine.initialize()
        
        # Procesar lote muy grande
        large_batch = [
            f"Análisis de contenido extenso número {i} para testing de memoria y performance."
            for i in range(1000)
        ]
        
        result = await engine.analyze_batch_mixed(large_batch)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Verificar eficiencia de memoria
        assert memory_used < 50  # < 50MB para procesar 1000 textos
        assert result["total_texts"] == 1000
        assert result["total_processing_time_ms"] < 100  # < 100ms total
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self) -> Any:
        """Test procesamiento concurrente de múltiples requests."""
        engine = create_modular_engine(OptimizationTier.EXTREME)
        await engine.initialize()
        
        # Crear múltiples tareas concurrentes
        tasks = []
        for i in range(10):
            blog_batch = [
                f"Contenido concurrente {i}-{j} para test de paralelismo."
                for j in range(20)
            ]
            task = engine.analyze_sentiment(blog_batch)
            tasks.append(task)
        
        start_time = time.perf_counter()
        
        # Ejecutar todas las tareas en paralelo
        results = await asyncio.gather(*tasks)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Verificar que todas las tareas se completaron
        assert len(results) == 10
        assert all(len(result["scores"]) == 20 for result in results)
        
        # Verificar eficiencia del paralelismo
        total_texts = 10 * 20  # 200 textos
        combined_throughput = total_texts / (total_time / 1000)
        
        assert combined_throughput > 5000  # > 5K ops/s en paralelo
        assert total_time < 50  # < 50ms total para procesar 200 textos en paralelo


# Funciones de utilidad para testing
def create_sample_blog_content(blog_type: str = "technical") -> str:
    """Crear contenido de blog de muestra para testing."""
    samples = {
        "technical": """
        Implementación de Arquitecturas de Microservicios con Kubernetes
        
        Los microservicios representan un paradigma arquitectónico que permite 
        desarrollar aplicaciones como un conjunto de servicios pequeños e independientes.
        
        Ventajas principales:
        - Escalabilidad independiente
        - Tecnologías heterogéneas
        - Despliegue continuo
        - Tolerancia a fallos
        """,
        "tutorial": """
        Tutorial: Cómo Configurar un Pipeline de CI/CD con GitHub Actions
        
        Paso 1: Crear el archivo de workflow
        Paso 2: Configurar las acciones de build
        Paso 3: Implementar tests automatizados
        Paso 4: Configurar deployment automático
        
        Siguiendo estos pasos tendrás un pipeline completo funcionando.
        """,
        "promotional": """
        ¡Descubre la Mejor Plataforma de Marketing Digital!
        
        ¿Buscas resultados INCREÍBLES? Nuestra solución es PERFECTA para ti.
        ¡Obtén un ROI del 300% garantizado! ¡ACTÚA AHORA!
        
        ✅ Automatización completa
        ✅ Analytics avanzado  
        ✅ Soporte 24/7
        """
    }
    
    return samples.get(blog_type, samples["technical"])


async def run_performance_benchmark() -> Dict[str, Any]:
    """Ejecutar benchmark completo de performance."""
    engine = create_modular_engine(OptimizationTier.EXTREME)
    await engine.initialize()
    
    # Test casos diversos
    test_cases = [
        ("single_analysis", [create_sample_blog_content()]),
        ("small_batch", [create_sample_blog_content() for _ in range(10)]),
        ("medium_batch", [create_sample_blog_content() for _ in range(50)]),
        ("large_batch", [create_sample_blog_content() for _ in range(200)])
    ]
    
    results = {}
    
    for test_name, blog_batch in test_cases:
        start_time = time.perf_counter()
        
        result = await engine.analyze_batch_mixed(blog_batch)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        results[test_name] = {
            "batch_size": len(blog_batch),
            "processing_time_ms": processing_time,
            "throughput_ops_per_second": len(blog_batch) / (processing_time / 1000),
            "avg_latency_ms": processing_time / len(blog_batch),
            "engine_throughput": result["combined_throughput_ops_per_second"]
        }
    
    return results


if __name__ == "__main__":
    # Ejecutar benchmark si se ejecuta directamente
    async def main():
        
    """main function."""
print("🧪 Ejecutando benchmark de performance del modelo blog...")
        benchmark_results = await run_performance_benchmark()
        
        print("\n📊 RESULTADOS DEL BENCHMARK:")
        print("=" * 50)
        
        for test_name, metrics in benchmark_results.items():
            print(f"\n{test_name.upper()}:")
            print(f"  Tamaño del lote: {metrics['batch_size']}")
            print(f"  Tiempo de procesamiento: {metrics['processing_time_ms']:.2f} ms")
            print(f"  Throughput: {metrics['throughput_ops_per_second']:.0f} ops/s")
            print(f"  Latencia promedio: {metrics['avg_latency_ms']:.3f} ms")
        
        print("\n✅ Benchmark completado exitosamente!")
    
    asyncio.run(main())