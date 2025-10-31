from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ...core.interfaces import INLPAnalyzer, IMonitor, IHealthChecker
from ...core.entities import (
        import time
from typing import Any, List, Dict, Optional
import logging
"""
🎯 USE CASES - Application Logic
===============================

Casos de uso ultra-modulares del sistema NLP.
"""


    TextInput, AnalysisResult, BatchResult, 
    AnalysisType, OptimizationTier
)


@dataclass
class AnalyzeSentimentRequest:
    """Request para análisis de sentimiento."""
    texts: List[str]
    use_cache: bool = True
    optimization_tier: Optional[OptimizationTier] = None


@dataclass
class AnalyzeQualityRequest:
    """Request para análisis de calidad."""
    texts: List[str]
    use_cache: bool = True
    optimization_tier: Optional[OptimizationTier] = None


@dataclass
class BatchAnalysisRequest:
    """Request para análisis en lote."""
    texts: List[str]
    include_sentiment: bool = True
    include_quality: bool = True
    max_concurrency: int = 10


class AnalyzeSentimentUseCase:
    """🎯 Caso de uso: Analizar sentimiento."""
    
    def __init__(self, nlp_analyzer: INLPAnalyzer, monitor: IMonitor):
        
    """__init__ function."""
self.nlp_analyzer = nlp_analyzer
        self.monitor = monitor
    
    async def execute(self, request: AnalyzeSentimentRequest) -> BatchResult:
        """Ejecutar análisis de sentimiento."""
        # Convert texts to TextInput objects
        inputs = [
            TextInput(content=text, id=f"text_{i}")
            for i, text in enumerate(request.texts)
        ]
        
        # Perform analysis
        result = await self.nlp_analyzer.analyze(inputs, AnalysisType.SENTIMENT)
        
        return result


class AnalyzeQualityUseCase:
    """📊 Caso de uso: Analizar calidad."""
    
    def __init__(self, nlp_analyzer: INLPAnalyzer, monitor: IMonitor):
        
    """__init__ function."""
self.nlp_analyzer = nlp_analyzer
        self.monitor = monitor
    
    async def execute(self, request: AnalyzeQualityRequest) -> BatchResult:
        """Ejecutar análisis de calidad."""
        # Convert texts to TextInput objects
        inputs = [
            TextInput(content=text, id=f"text_{i}")
            for i, text in enumerate(request.texts)
        ]
        
        # Perform analysis
        result = await self.nlp_analyzer.analyze(inputs, AnalysisType.QUALITY)
        
        return result


class BatchAnalysisUseCase:
    """⚡ Caso de uso: Análisis en lote."""
    
    def __init__(self, nlp_analyzer: INLPAnalyzer, monitor: IMonitor):
        
    """__init__ function."""
self.nlp_analyzer = nlp_analyzer
        self.monitor = monitor
    
    async def execute(self, request: BatchAnalysisRequest) -> Dict[str, Any]:
        """Ejecutar análisis en lote."""
        # Convert texts to TextInput objects
        inputs = [
            TextInput(content=text, id=f"text_{i}")
            for i, text in enumerate(request.texts)
        ]
        
        results = {}
        tasks = []
        
        # Add sentiment analysis if requested
        if request.include_sentiment:
            tasks.append(
                ('sentiment', self.nlp_analyzer.analyze(inputs, AnalysisType.SENTIMENT))
            )
        
        # Add quality analysis if requested
        if request.include_quality:
            tasks.append(
                ('quality', self.nlp_analyzer.analyze(inputs, AnalysisType.QUALITY))
            )
        
        # Execute all tasks concurrently
        if tasks:
            task_results = await asyncio.gather(*[task[1] for task in tasks])
            
            for i, (task_name, _) in enumerate(tasks):
                results[task_name] = task_results[i]
        
        return {
            'results': results,
            'total_texts': len(request.texts),
            'analyses_performed': len(tasks),
            'concurrency_used': min(request.max_concurrency, len(tasks))
        }


class GetSystemStatusUseCase:
    """🏥 Caso de uso: Obtener estado del sistema."""
    
    def __init__(self, health_checker: IHealthChecker):
        
    """__init__ function."""
self.health_checker = health_checker
    
    async def execute(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema."""
        # Get system status
        status = await self.health_checker.check_health()
        
        # Get dependencies
        dependencies = await self.health_checker.check_dependencies()
        
        # Run diagnostic if needed
        diagnostic = None
        if status.get_health_status() != "healthy":
            diagnostic = await self.health_checker.run_diagnostic()
        
        return {
            'status': status,
            'dependencies': dependencies,
            'diagnostic': diagnostic,
            'health': status.get_health_status()
        }


class BenchmarkPerformanceUseCase:
    """🧪 Caso de uso: Benchmark de rendimiento."""
    
    def __init__(self, nlp_analyzer: INLPAnalyzer):
        
    """__init__ function."""
self.nlp_analyzer = nlp_analyzer
    
    async def execute(self, num_texts: int = 100) -> Dict[str, Any]:
        """Ejecutar benchmark de rendimiento."""
        
        # Create test data
        test_texts = [f"Test text number {i} for benchmarking." for i in range(num_texts)]
        inputs = [
            TextInput(content=text, id=f"bench_{i}")
            for i, text in enumerate(test_texts)
        ]
        
        # Benchmark sentiment analysis
        start_time = time.perf_counter()
        sentiment_result = await self.nlp_analyzer.analyze(inputs, AnalysisType.SENTIMENT)
        sentiment_time = time.perf_counter() - start_time
        
        # Benchmark quality analysis
        start_time = time.perf_counter()
        quality_result = await self.nlp_analyzer.analyze(inputs, AnalysisType.QUALITY)
        quality_time = time.perf_counter() - start_time
        
        return {
            'benchmark_config': {
                'num_texts': num_texts,
                'test_data_type': 'synthetic'
            },
            'sentiment_analysis': {
                'total_time_ms': sentiment_time * 1000,
                'avg_time_per_text_ms': (sentiment_time * 1000) / num_texts,
                'throughput_ops_per_second': num_texts / sentiment_time,
                'success_rate': sentiment_result.success_rate,
                'average_score': sentiment_result.average_score
            },
            'quality_analysis': {
                'total_time_ms': quality_time * 1000,
                'avg_time_per_text_ms': (quality_time * 1000) / num_texts,
                'throughput_ops_per_second': num_texts / quality_time,
                'success_rate': quality_result.success_rate,
                'average_score': quality_result.average_score
            },
            'overall': {
                'total_operations': num_texts * 2,
                'total_time_ms': (sentiment_time + quality_time) * 1000,
                'combined_throughput': (num_texts * 2) / (sentiment_time + quality_time)
            }
        }


class OptimizeSystemUseCase:
    """⚡ Caso de uso: Optimizar sistema."""
    
    def __init__(self, nlp_analyzer: INLPAnalyzer, monitor: IMonitor):
        
    """__init__ function."""
self.nlp_analyzer = nlp_analyzer
        self.monitor = monitor
    
    async def execute(self, target_tier: OptimizationTier) -> Dict[str, Any]:
        """Optimizar sistema al tier especificado."""
        # This would trigger re-initialization with new tier
        # For now, return current optimization info
        
        current_metrics = self.monitor.get_metrics()
        system_status = self.monitor.get_system_status()
        
        return {
            'current_tier': system_status.optimization_tier.value,
            'target_tier': target_tier.value,
            'optimization_needed': system_status.optimization_tier != target_tier,
            'current_performance': {
                'latency_ms': current_metrics.latency_ms,
                'throughput_ops_per_second': current_metrics.throughput_ops_per_second,
                'optimization_factor': current_metrics.optimization_factor
            },
            'available_optimizers': system_status.available_optimizers
        } 