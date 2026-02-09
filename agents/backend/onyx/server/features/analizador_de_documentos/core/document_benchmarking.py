"""
Document Benchmarking - Benchmarking de Modelos
================================================

Sistema de benchmarking para comparar diferentes modelos y configuraciones.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import time
import statistics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Resultado de benchmark."""
    model_name: str
    configuration: Dict[str, Any]
    total_documents: int
    average_time: float
    average_accuracy: float
    throughput: float  # documentos por segundo
    memory_usage_mb: float
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkComparison:
    """Comparación de benchmarks."""
    benchmarks: List[BenchmarkResult]
    best_model: Optional[str] = None
    fastest_model: Optional[str] = None
    most_accurate_model: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


class ModelBenchmarker:
    """Benchmarker de modelos."""
    
    def __init__(self, analyzer):
        """Inicializar benchmarker."""
        self.analyzer = analyzer
        self.benchmark_results: List[BenchmarkResult] = []
    
    async def benchmark_model(
        self,
        model_name: str,
        test_documents: List[str],
        configuration: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Hacer benchmark de un modelo.
        
        Args:
            model_name: Nombre del modelo
            test_documents: Lista de documentos de prueba
            configuration: Configuración del modelo
        
        Returns:
            BenchmarkResult con resultados
        """
        times = []
        accuracies = []
        errors = 0
        
        start_memory = self._get_memory_usage()
        
        for doc_content in test_documents:
            try:
                start_time = time.time()
                
                # Analizar documento
                result = await self.analyzer.analyze_document(document_content=doc_content)
                
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                # Calcular accuracy (simplificado)
                if hasattr(result, 'confidence'):
                    accuracies.append(result.confidence)
                else:
                    accuracies.append(0.8)  # Default
                    
            except Exception as e:
                errors += 1
                logger.error(f"Error en benchmark: {e}")
        
        end_memory = self._get_memory_usage()
        memory_usage = end_memory - start_memory
        
        total_time = sum(times)
        avg_time = statistics.mean(times) if times else 0
        avg_accuracy = statistics.mean(accuracies) if accuracies else 0
        throughput = len(test_documents) / total_time if total_time > 0 else 0
        error_rate = errors / len(test_documents) if test_documents else 0
        
        result = BenchmarkResult(
            model_name=model_name,
            configuration=configuration or {},
            total_documents=len(test_documents),
            average_time=avg_time,
            average_accuracy=avg_accuracy,
            throughput=throughput,
            memory_usage_mb=memory_usage,
            error_rate=error_rate,
            details={
                "min_time": min(times) if times else 0,
                "max_time": max(times) if times else 0,
                "median_time": statistics.median(times) if times else 0
            }
        )
        
        self.benchmark_results.append(result)
        
        return result
    
    async def compare_models(
        self,
        models: List[Dict[str, Any]],
        test_documents: List[str]
    ) -> BenchmarkComparison:
        """
        Comparar múltiples modelos.
        
        Args:
            models: Lista de configuraciones de modelos
            test_documents: Documentos de prueba
        
        Returns:
            BenchmarkComparison con resultados
        """
        benchmarks = []
        
        for model_config in models:
            model_name = model_config.get("name", "unknown")
            config = model_config.get("config", {})
            
            result = await self.benchmark_model(model_name, test_documents, config)
            benchmarks.append(result)
        
        # Determinar mejores modelos
        best_model = max(benchmarks, key=lambda b: b.average_accuracy).model_name if benchmarks else None
        fastest_model = min(benchmarks, key=lambda b: b.average_time).model_name if benchmarks else None
        most_accurate_model = max(benchmarks, key=lambda b: b.average_accuracy).model_name if benchmarks else None
        
        # Generar recomendaciones
        recommendations = []
        if benchmarks:
            avg_throughput = statistics.mean([b.throughput for b in benchmarks])
            fastest = max(benchmarks, key=lambda b: b.throughput)
            
            recommendations.append(f"Modelo más rápido: {fastest.model_name} ({fastest.throughput:.2f} docs/s)")
            recommendations.append(f"Modelo más preciso: {most_accurate_model} ({max(benchmarks, key=lambda b: b.average_accuracy).average_accuracy:.2%})")
        
        return BenchmarkComparison(
            benchmarks=benchmarks,
            best_model=best_model,
            fastest_model=fastest_model,
            most_accurate_model=most_accurate_model,
            recommendations=recommendations
        )
    
    def _get_memory_usage(self) -> float:
        """Obtener uso de memoria (simplificado)."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    def get_benchmark_history(self) -> List[BenchmarkResult]:
        """Obtener historial de benchmarks."""
        return self.benchmark_results


__all__ = [
    "ModelBenchmarker",
    "BenchmarkResult",
    "BenchmarkComparison"
]


